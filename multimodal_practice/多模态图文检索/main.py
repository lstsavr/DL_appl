import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data.datasets import FlickrImageTextDataset
from data.transforms import get_image_transform
from models.matcher import DualEncoderModel
from engine.train import train, set_optimizer_and_scheduler, freeze_encoder_layers
from engine.evaluate import evaluate
import random
import numpy as np
import os
import time
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_train_val(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=10, 
               save_path='best_model.pth', epoch_callback=None, use_amp=True, eval_interval=1):
    
    best_recall = 0
    best_mean_recall = 0
    best_epoch = 0
    
    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc_t2i': [], 'train_acc_i2t': [],
        'val_t2i_r1': [], 'val_t2i_r5': [], 'val_t2i_r10': [],
        'val_i2t_r1': [], 'val_i2t_r5': [], 'val_i2t_r10': [],
        'val_mean_r1': [], 'epoch_time': []
    }
    
    # 检查是否支持混合精度训练
    if use_amp and device.type != 'cuda':
        print("无法进行混合精度训练，现在使用的是标准精度")
        use_amp = False
    
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
        
        # 根据评估间隔决定是否进行验证
        run_eval = (epoch % eval_interval == 0) or (epoch == num_epochs - 1)
        
        if run_eval:
            # 评估阶段
            val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)
            
            t2i_r1 = val_metrics['text2image']['R@1']
            t2i_r5 = val_metrics['text2image']['R@5']
            t2i_r10 = val_metrics['text2image']['R@10']
            t2i_mrr = val_metrics['text2image']['MRR']
            
            i2t_r1 = val_metrics['image2text']['R@1']
            i2t_r5 = val_metrics['image2text']['R@5']
            i2t_r10 = val_metrics['image2text']['R@10']
            i2t_mrr = val_metrics['image2text']['MRR']
            
            # 计算平均R@1
            mean_recall = val_metrics.get('mean_r1', (t2i_r1 + i2t_r1) / 2.0)
            
            # 记录验证指标
            history['val_t2i_r1'].append(t2i_r1)
            history['val_t2i_r5'].append(t2i_r5)
            history['val_t2i_r10'].append(t2i_r10)
            history['val_i2t_r1'].append(i2t_r1)
            history['val_i2t_r5'].append(i2t_r5)
            history['val_i2t_r10'].append(i2t_r10)
            history['val_mean_r1'].append(mean_recall)
            
            # 使用平均R@1作为保存模型的标准
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
            
            # 额外保存检查点，用于恢复训练
            if (epoch + 1) % 5 == 0:
                ckpt_path = save_path.replace('.pth', f'_epoch{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_mean_recall': best_mean_recall,
                    'history': history
                }, ckpt_path)
                print(f'>>> Epoch {epoch+1} 检查点已保存')
        
        # 记录每个epoch的时间
        epoch_time = time.time() - epoch_start
        history['epoch_time'].append(epoch_time)
        print(f"Epoch {epoch+1} 完成，耗时 {epoch_time:.2f} 秒")
        
        # 打印当前训练状态总结
        print(f"训练损失: {train_metrics['loss']:.4f}, "
              f"训练准确率: {(train_metrics['acc_t2i'] + train_metrics['acc_i2t'])/2:.4f}")
        
        if run_eval:
            print(f"验证 Mean R@1: {mean_recall:.4f}, "
                  f"最佳 Mean R@1: {best_mean_recall:.4f} (Epoch {best_epoch})")
        
        import json
        with open('training_history.json', 'w') as f:
            json.dump(history, f)
    
    final_path = save_path.replace('.pth', '_final.pth')
    torch.save({
        'epoch': num_epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_mean_recall': best_mean_recall,
        'history': history
    }, final_path)
    print(f'>>> 最终模型已保存 (Epoch {num_epochs})')
    
    # 最终打印最佳结果
    print(f'\n训练完成')
    print(f'最佳 Mean R@1: {best_mean_recall:.4f} (Epoch {best_epoch})')

if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device} {'(GPU)' if torch.cuda.is_available() else '(CPU)'}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    ann_file = 'data/flickr8k_aim3/dataset_flickr8k.json'
    img_dir = 'data/flickr8k_aim3/images'
    train_list = 'data/flickr8k_aim3/train_list.txt'
    val_list = 'data/flickr8k_aim3/val_list.txt'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transform_train = get_image_transform(image_size=224, train=True)
    transform_val = get_image_transform(image_size=224, train=False)

    train_dataset = FlickrImageTextDataset(ann_file, img_dir, tokenizer, transform_train, train_list)
    val_dataset = FlickrImageTextDataset(ann_file, img_dir, tokenizer, transform_val, val_list)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64,  
        shuffle=True, 
        num_workers=8,  
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64,  
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    img_encoder_cfg = {'model_name': 'resnet101', 'pretrained': True, 'embed_dim': 768} 
    txt_encoder_cfg = {
        'model_name': 'bert-base-uncased', 
        'pretrained': True, 
        'embed_dim': 768,  # 与BERT隐藏层维度一致，减少信息损失
        'pool_type': 'mean_max'  # 使用平均+最大池化的组合，效果更好
    }
    
    model = DualEncoderModel(
        img_encoder_cfg=img_encoder_cfg, 
        txt_encoder_cfg=txt_encoder_cfg,
        embed_dim=512, 
        proj_layers=2,  
        proj_hidden=1024,  
        proj_dropout=0.1,  
        proj_ln=True,    
        l2_normalize=True,
        sim_type='cosine'
    ).to(device)

    num_epochs = 20  
    total_steps = num_epochs * len(train_loader)
    
    optimizer, scheduler = set_optimizer_and_scheduler(
        model, 
        lr=3e-5, 
        weight_decay=0.01,  
        warmup_steps=len(train_loader),  
        total_steps=total_steps
    )

    print("阶段1: 冻结部分预训练编码器，训练上层和投影头...")
    freeze_encoder_layers(model, freeze_img_layers=4, freeze_txt_layers=8)
    
    def unfreeze_callback(epoch, model):
        if epoch == 3:  
            print("\n阶段2: 解冻更多编码器层")
            freeze_encoder_layers(model, freeze_img_layers=3, freeze_txt_layers=6)
            
        elif epoch == 6:  
            print("\n阶段3: 进一步解冻")
            freeze_encoder_layers(model, freeze_img_layers=1, freeze_txt_layers=2)
            
        elif epoch == 10: 
            print("\n阶段4: 完全解冻所有层，开始全局微调")
            for param in model.parameters():
                param.requires_grad = True
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.7

    run_train_val(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        device, 
        num_epochs=num_epochs, 
        save_path='best_model.pth',
        epoch_callback=lambda epoch, model: unfreeze_callback(epoch, model),
        use_amp=True,  
        eval_interval=1  

    )
