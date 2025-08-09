import os
import torch
import json
import argparse
import time
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from data.datasets import FlickrImageTextDataset
from data.transforms import get_image_transform
from models.matcher import DualEncoderModel
from engine.train import train, set_optimizer_and_scheduler, freeze_encoder_layers
from engine.evaluate import evaluate

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(tokenizer, batch_size=32, num_workers=2):
    ann_file = 'data/flickr8k_aim3/dataset_flickr8k.json'
    img_dir = 'data/flickr8k_aim3/images'
    train_list = 'data/flickr8k_aim3/train_list.txt'
    val_list = 'data/flickr8k_aim3/val_list.txt'
    
    transform_train = get_image_transform(image_size=224, train=True)
    transform_val = get_image_transform(image_size=224, train=False)

    train_dataset = FlickrImageTextDataset(ann_file, img_dir, tokenizer, transform_train, train_list)
    val_dataset = FlickrImageTextDataset(ann_file, img_dir, tokenizer, transform_val, val_list)
    
    # 从环境变量中获取批次大小和工作进程数，如果有的话
    env_batch_size = os.environ.get('BATCH_SIZE')
    if env_batch_size and env_batch_size.isdigit():
        batch_size = int(env_batch_size)
        print(f"使用环境变量中的批次大小: {batch_size}")
    
    env_workers = os.environ.get('DATA_WORKERS')
    if env_workers and env_workers.isdigit():
        num_workers = int(env_workers)
        print(f"使用环境变量中的工作进程数: {num_workers}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False,  # 关闭pin_memory以减少内存使用
        prefetch_factor=2 if num_workers > 0 else None,  # 保持较低的预取因子
        persistent_workers=False,  # 关闭持久化工作进程
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False,  # 关闭pin_memory
        prefetch_factor=2 if num_workers > 0 else None  # 保持较低的预取因子
    )
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, 
                num_epochs, save_path, epoch_callback=None, use_amp=True):

    best_mean_recall = 0
    best_epoch = 0
    
    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc_t2i': [], 'train_acc_i2t': [],
        'val_t2i_r1': [], 'val_t2i_r5': [], 'val_t2i_r10': [],
        'val_i2t_r1': [], 'val_i2t_r5': [], 'val_i2t_r10': [],
        'val_mean_r1': [], 'epoch_time': []
    }
    
    # 导入需要的模块
    import os
    import json
    
    print(f"开始训练 - 总共{num_epochs}个Epoch, {'启用' if use_amp else '不启用'}混合精度")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'\n=== Epoch {epoch+1}/{num_epochs} ===')
        
        if epoch_callback:
            epoch_callback(epoch, model)
        
        # 训练阶段
        train_metrics = train(model, train_loader, optimizer, device, 
                              scheduler=scheduler, print_freq=100, 
                              use_amp=use_amp, epoch=epoch)
        
        # 记录训练指标
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc_t2i'].append(train_metrics['acc_t2i'])
        history['train_acc_i2t'].append(train_metrics['acc_i2t'])
        
        # 评估阶段
        val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)
        
        # 获取评估指标
        t2i_r1 = val_metrics['text2image']['R@1']
        t2i_r5 = val_metrics['text2image']['R@5']
        t2i_r10 = val_metrics['text2image']['R@10']
        
        i2t_r1 = val_metrics['image2text']['R@1']
        i2t_r5 = val_metrics['image2text']['R@5']
        i2t_r10 = val_metrics['image2text']['R@10']
        
        # 计算平均R@1
        mean_recall = (t2i_r1 + i2t_r1) / 2.0
        
        # 记录验证指标
        history['val_t2i_r1'].append(t2i_r1)
        history['val_t2i_r5'].append(t2i_r5)
        history['val_t2i_r10'].append(t2i_r10)
        history['val_i2t_r1'].append(i2t_r1)
        history['val_i2t_r5'].append(i2t_r5)
        history['val_i2t_r10'].append(i2t_r10)
        history['val_mean_r1'].append(mean_recall)
        
        # 如果性能提升，保存模型
        if mean_recall > best_mean_recall:
            best_mean_recall = mean_recall
            best_epoch = epoch + 1
            # 保存模型
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mean_recall': best_mean_recall,
                'history': history
            }, save_path)
            print(f'>>> 新的最佳模型已保存 Mean R@1={best_mean_recall:.4f}')
        
        # 记录每个epoch的时间
        epoch_time = time.time() - epoch_start
        history['epoch_time'].append(epoch_time)
        print(f"Epoch {epoch+1} 完成，耗时 {epoch_time:.2f} 秒")
        
        # 打印当前训练状态总结
        print(f"训练损失: {train_metrics['loss']:.4f}, "
              f"训练准确率: {(train_metrics['acc_t2i'] + train_metrics['acc_i2t'])/2:.4f}")
        print(f"验证 Mean R@1: {mean_recall:.4f}, "
              f"最佳 Mean R@1: {best_mean_recall:.4f} (Epoch {best_epoch})")
        
        # 仅在CPU模式或每3个epoch清理一次内存，避免频繁清理导致性能下降
        if device.type == 'cpu' or (epoch + 1) % 3 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        # 每3个epoch或最后一个epoch保存一次临时历史记录
        if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(f"{os.path.dirname(save_path)}/history_temp.json", 'w') as f:
                    json.dump(history, f, indent=2)
            except Exception as e:
                print(f"保存失败: {str(e)}")
    
    return best_mean_recall, history

def run_ablation_experiment(config_name, model_config, num_epochs=10, force_clean=False):
    """运行单个消融实验配置"""
    set_seed(42)  # 确保可重复性
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n==== 开始实验: {config_name} ====")
    print(f"使用设备: {device} ({'GPU' if device.type == 'cuda' else 'CPU'})")
    
    output_dir = f"ablation_results/{config_name.replace(' ', '_')}"
    
    if force_clean and os.path.exists(output_dir):
        print(f"清除现有实验目录: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n实验详情:")
    print(f"- 图像编码器: {model_config['img_encoder_cfg']['model_name']}")
    print(f"- 文本编码器: {model_config['txt_encoder_cfg']['model_name']}")
    print(f"- 池化策略: {model_config['txt_encoder_cfg'].get('pool_type', 'cls')}")
    print(f"- 困难负样本挖掘: {model_config.get('hard_negative', False)}")
    print(f"- 嵌入维度: {model_config['embed_dim']}")
    print(f"- 投影层数: {model_config['proj_layers']}")
    print("其他配置:", {k: v for k, v in model_config.items() 
                   if k not in ['img_encoder_cfg', 'txt_encoder_cfg', 'embed_dim', 'proj_layers', 'hard_negative']})
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json_config = {k: v if not isinstance(v, dict) else dict(v) for k, v in model_config.items()}
        json.dump(json_config, f, indent=2)
    
    output_dir = f"ablation_results/{config_name.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 首先尝试从Hugging Face在线下载
        print("尝试从Hugging Face在线下载BERT分词器...")
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            try:
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                                        use_fast=True,
                                                        local_files_only=False)
                success = True
                print("成功在线下载BERT分词器")
                break
            except Exception as e:
                print(f"在线下载尝试 {attempt+1}/{max_retries} 失败: {str(e)}")
                print("等待5秒后重试")
                import time
                time.sleep(5)
        
        # 如果在线下载失败，尝试使用本地模型
        if not success:
            print("在线下载失败，尝试使用本地BERT模型")
            cache_dir = os.path.join("models", "bert_cache")
            local_tokenizer_path = os.path.join(cache_dir, "tokenizer")
            
            if os.path.exists(local_tokenizer_path):
                print(f"使用本地BERT分词器: {local_tokenizer_path}")
                tokenizer = BertTokenizer.from_pretrained(local_tokenizer_path)
                success = True
            
        if not success:
            print("\n下载失败")
            return 0
    
    except Exception as e:
        print(f"加载BERT分词器时出错: {str(e)}")
        return 0
        
    train_loader, val_loader = get_data_loaders(tokenizer)
    
    # 创建模型 - 不使用本地BERT模型，使用在线模型
    try:
        # 从模型配置中移除local_bert_path，确保使用在线模型
        if 'txt_encoder_cfg' in model_config and 'local_bert_path' in model_config['txt_encoder_cfg']:
            print("注意：移除local_bert_path，强制使用在线BERT模型")
            model_config['txt_encoder_cfg'].pop('local_bert_path', None)
        
        hard_negative = False
        if 'hard_negative' in model_config:
            # 将hard_negative提取出来，不直接传给DualEncoderModel
            hard_negative = model_config.pop('hard_negative', False)
            print(f"使用困难负样本挖掘: {hard_negative}")
        
        model = DualEncoderModel(**model_config).to(device)
        model.hard_negative = hard_negative
        print(f"模型配置: hard_negative = {model.hard_negative}")
    except Exception as e:
        print(f"创建模型时出错: {str(e)}")
        print(f"错误详情: {e.__class__.__name__}: {str(e)}")
        print(f"模型配置: {model_config}")
        return 0
    
    total_steps = num_epochs * len(train_loader)
    optimizer, scheduler = set_optimizer_and_scheduler(
        model, 
        lr=3e-5,
        weight_decay=0.01,
        warmup_steps=len(train_loader),
        total_steps=total_steps
    )
    
    # 冻结层设置
    if config_name == "基线" or config_name == "+ ResNet101" or config_name == "+ Mean_Max池化":
        # 简单冻结
        freeze_encoder_layers(model, freeze_img_layers=4, freeze_txt_layers=8)
        epoch_callback = None
    elif config_name == "+ 困难负样本挖掘":
        # 简单冻结
        freeze_encoder_layers(model, freeze_img_layers=4, freeze_txt_layers=8)
        epoch_callback = None
    elif config_name == "+ 渐进式解冻" or config_name == "完整模型":
        # 初始冻结
        freeze_encoder_layers(model, freeze_img_layers=4, freeze_txt_layers=8)
        
        # 渐进式解冻回调
        def unfreeze_callback(epoch, model):
            nonlocal optimizer
            if epoch == 3:  # 第4个epoch
                print("\n阶段2: 开始逐步解冻更多编码器层")
                freeze_encoder_layers(model, freeze_img_layers=3, freeze_txt_layers=6)
                
            elif epoch == 6:  # 第7个epoch
                print("\n阶段3: 进一步解冻编码器层")
                freeze_encoder_layers(model, freeze_img_layers=1, freeze_txt_layers=2)
                
            elif epoch == 10:  # 第11个epoch
                print("\n阶段4: 完全解冻所有层")
                for param in model.parameters():
                    param.requires_grad = True
                
                # 降低学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.7
                    
        epoch_callback = unfreeze_callback
    
    # GPU内存优化设置
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        
        # 导入需要的模块
        import gc
        
        # 清理GPU内存，为模型加载腾出空间
        torch.cuda.empty_cache()
        gc.collect()
    
    # 打印模型的关键配置，确保它们被正确设置
    print("\n模型实际配置:")
    print(f"- 图像编码器实际类型: {type(model.image_encoder).__name__}")
    print(f"- 文本编码器实际类型: {type(model.text_encoder).__name__}")
    print(f"- 文本池化策略: {model.text_encoder.pool_type}")
    print(f"- 困难负样本挖掘: {getattr(model, 'hard_negative', False)}")
    print(f"- 嵌入维度: {model.image_encoder.fc.out_features}")
    print(f"- 文本编码器池化类型: {model.text_encoder.pool_type}")
    print(f"- 困难负样本挖掘: {getattr(model, 'hard_negative', False)}")
    
    # 训练模型 (内存优化配置)
    save_path = f"{output_dir}/best_model.pth"
    best_recall, history = train_model(
        model, 
        train_loader, 
        val_loader,
        optimizer, 
        scheduler,
        device,
        num_epochs=num_epochs,
        save_path=save_path,
        epoch_callback=epoch_callback,
        use_amp=True 
    )
    
    with open(f"{output_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    summary = {
        "config_name": config_name,
        "best_mean_r1": best_recall,
        "model_config": model_config,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n==== 实验 {config_name} 完成 ====")
    print(f"最佳 Mean R@1: {best_recall:.4f}")
    print(f"结果保存在: {output_dir}")
    
    return best_recall

def define_experiments():
    
    base_config = {
        "img_encoder_cfg": {'model_name': 'resnet50', 'pretrained': True, 'embed_dim': 768},
        "txt_encoder_cfg": {
            'model_name': 'bert-base-uncased', 
            'pretrained': True, 
            'embed_dim': 768,
            'pool_type': 'cls'  # 使用CLS池化
        },
        "embed_dim": 512,
        "proj_layers": 2,
        "proj_hidden": 1024,
        "proj_dropout": 0.1,
        "proj_ln": True,
        "l2_normalize": True,
        "sim_type": 'cosine'
    }
    
    # 实验1: ResNet50 + BERT-CLS
    exp1_config = base_config.copy()
    
    # 实验2: 使用ResNet101
    exp2_config = base_config.copy()
    exp2_config["img_encoder_cfg"] = {'model_name': 'resnet101', 'pretrained': True, 'embed_dim': 768}
    
    # 实验3: 使用Mean_Max池化
    exp3_config = exp2_config.copy()  # 基于ResNet101
    exp3_config["txt_encoder_cfg"] = {
        'model_name': 'bert-base-uncased', 
        'pretrained': True, 
        'embed_dim': 768,
        'pool_type': 'mean_max'  
    }
    
    # 实验4: 加入困难负样本挖掘 
    exp4_config = exp3_config.copy()  
    exp4_config["hard_negative"] = True
    
    # 实验5: 实施渐进式解冻 
    exp5_config = exp4_config.copy()  
    
    # 完整模型
    full_model_config = exp5_config.copy()
    
    return {
        "基线": exp1_config,
        "+ ResNet101": exp2_config,
        "+ Mean_Max池化": exp3_config,
        "+ 困难负样本挖掘": exp4_config,
        "+ 渐进式解冻": exp5_config,
        "完整模型": full_model_config
    }

def run_all_experiments(epochs_per_exp=10, force_clean=True):

    if force_clean and os.path.exists("ablation_results"):
        print("清除所有现有消融实验结果...")
        import shutil
        shutil.rmtree("ablation_results")
    
    os.makedirs("ablation_results", exist_ok=True)
    
    experiments = define_experiments()
    
    start_time = time.time()
    
    results = {}
    
    for name, config in experiments.items():
        print(f"开始实验: {name}")
  
        best_recall = run_ablation_experiment(name, config, num_epochs=epochs_per_exp, force_clean=force_clean)
        results[name] = best_recall
        
        with open("ablation_results/progress.json", 'w') as f:
            json.dump({
                "completed": list(results.keys()),
                "results": results,
                "remaining": list(set(experiments.keys()) - set(results.keys())),
                "elapsed_time": time.time() - start_time
            }, f, indent=2)

    total_time = time.time() - start_time
    baseline = results["基线"]
    
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_hours": total_time / 3600,
        "epochs_per_experiment": epochs_per_exp,
        "results": {
            name: {
                "Mean_R@1": value,
                "absolute_improvement": value - baseline,
                "relative_improvement_percent": (value - baseline) / baseline * 100
            }
            for name, value in results.items()
        }
    }
    
    # 保存报告
    with open("ablation_results/final_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印最终结果
    print("消融实验完成")
    print(f"总运行时间: {total_time/3600:.2f} 小时")
    print("\n实验结果摘要:")
    print(f"{'配置':<20} | {'Mean R@1':<10} | {'相对提升':<10}")
    
    for name, value in results.items():
        if name == "基线":
            rel_imp = "-"
        else:
            rel_imp = f"+{(value - baseline) / baseline * 100:.1f}%"
        print(f"{name:<20} | {value:<10.4f} | {rel_imp:<10}")
    
    print("-----------------------------------------")
    print(f"\n完整报告保存在: ablation_results/final_report.json")

def run_single_experiment(exp_name, epochs=10, force_clean=False):
    experiments = define_experiments()
    if exp_name not in experiments:
        print(f"错误: 找不到实验 '{exp_name}'")
        print(f"可用实验: {list(experiments.keys())}")
        return
    
    config = experiments[exp_name]
    run_ablation_experiment(exp_name, config, num_epochs=epochs, force_clean=force_clean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态图文检索系统消融实验")
    parser.add_argument("--run", choices=["all", "single"], default="all",
                        help="运行所有实验或单个实验")
    parser.add_argument("--experiment", type=str, default=None,
                        help="如果run=single, 指定要运行的实验名称")
    parser.add_argument("--epochs", type=int, default=10,
                        help="每个实验运行的epoch数")
    parser.add_argument("--clean", action="store_true",
                        help="清除现有实验结果")
    
    args = parser.parse_args()
    
    if args.run == "all":
        run_all_experiments(epochs_per_exp=args.epochs, force_clean=args.clean)
    elif args.run == "single":
        if args.experiment is None:
            print("错误: 运行单个实验时必须指定experiment参数")
            experiments = define_experiments()
            print(f"可用实验: {list(experiments.keys())}")
        else:
            run_single_experiment(args.experiment, epochs=args.epochs, force_clean=args.clean)

