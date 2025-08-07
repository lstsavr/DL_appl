import json
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import sys

# 创建图表保存目录
os.makedirs('figures', exist_ok=True)

def visualize_training_history():
    """可视化训练历史数据"""
    print("正在生成训练历史可视化图表...")
    try:
        # 读取训练历史记录
        with open('training_history.json', 'r') as f:
            history = json.load(f)
        
        # 检查是否有必要的键
        if 'val_mean_r1' not in history or len(history['val_mean_r1']) == 0:
            print("训练历史中没有发现验证集的Mean R@1数据")
            return
        
        # 绘制Mean R@1曲线
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(history['val_mean_r1'])+1)
        plt.plot(epochs, history['val_mean_r1'], marker='o', linestyle='-', color='blue', label='Mean R@1')
        
        if 'val_t2i_r1' in history and 'val_i2t_r1' in history:
            plt.plot(epochs, history['val_t2i_r1'], marker='^', linestyle='--', color='green', label='Text to Image R@1')
            plt.plot(epochs, history['val_i2t_r1'], marker='s', linestyle='--', color='red', label='Image to Text R@1')
        
        plt.title('检索性能随训练轮数的变化', fontsize=14)
        plt.xlabel('训练轮数 (Epochs)', fontsize=12)
        plt.ylabel('召回率 (Recall@1)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('figures/retrieval_performance.png', dpi=300)
        print("✓ 已保存检索性能曲线图: figures/retrieval_performance.png")
        plt.close()
        
        # 绘制训练损失曲线
        if 'train_loss' in history and len(history['train_loss']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['train_loss'], marker='o', linestyle='-', color='blue')
            plt.title('训练损失随训练轮数的变化', fontsize=14)
            plt.xlabel('训练轮数 (Epochs)', fontsize=12)
            plt.ylabel('损失值', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('figures/training_loss.png', dpi=300)
            print("✓ 已保存训练损失曲线图: figures/training_loss.png")
            plt.close()
        
        # 绘制R@5和R@10曲线
        if 'val_t2i_r5' in history and 'val_i2t_r5' in history and 'val_t2i_r10' in history and 'val_i2t_r10' in history:
            plt.figure(figsize=(10, 6))
            
            # 计算平均值
            mean_r5 = [(t2i + i2t)/2 for t2i, i2t in zip(history['val_t2i_r5'], history['val_i2t_r5'])]
            mean_r10 = [(t2i + i2t)/2 for t2i, i2t in zip(history['val_t2i_r10'], history['val_i2t_r10'])]
            
            plt.plot(epochs, mean_r5, marker='d', linestyle='-', color='purple', label='Mean R@5')
            plt.plot(epochs, mean_r10, marker='*', linestyle='-', color='orange', label='Mean R@10')
            plt.plot(epochs, history['val_mean_r1'], marker='o', linestyle='-', color='blue', label='Mean R@1')
            
            plt.title('不同K值的检索性能变化', fontsize=14)
            plt.xlabel('训练轮数 (Epochs)', fontsize=12)
            plt.ylabel('召回率 (Recall@K)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('figures/recall_at_k.png', dpi=300)
            print("✓ 已保存不同K值的检索性能曲线图: figures/recall_at_k.png")
            plt.close()
        
        # 打印最终性能
        final_epoch = len(history['val_mean_r1'])
        print("\n最终性能指标 (Epoch {})：".format(final_epoch))
        print(f"Mean R@1: {history['val_mean_r1'][-1]:.4f}")
        
        if 'val_t2i_r1' in history and 'val_i2t_r1' in history:
            print(f"Text to Image R@1: {history['val_t2i_r1'][-1]:.4f}")
            print(f"Image to Text R@1: {history['val_i2t_r1'][-1]:.4f}")
        
        # 生成训练历史表格数据（用于报告）
        print("\n用于报告的表格数据（前5轮和最后5轮）：")
        print("\n| Epoch | 训练损失 | Text→Image R@1 | Image→Text R@1 | Mean R@1 |")
        print("|-------|----------|----------------|----------------|---------|")
        
        # 计算要展示的轮数
        epochs_to_show = list(range(min(5, final_epoch)))  # 前5轮
        if final_epoch > 10:  # 如果超过10轮，也显示最后5轮
            epochs_to_show.extend(list(range(final_epoch-5, final_epoch)))
        else:
            epochs_to_show = list(range(final_epoch))  # 否则显示所有轮
        
        # 打印每一轮的数据
        for i in epochs_to_show:
            loss_val = f"{history['train_loss'][i]:.4f}" if 'train_loss' in history else "N/A"
            t2i_val = f"{history['val_t2i_r1'][i]:.4f}" if 'val_t2i_r1' in history else "N/A"
            i2t_val = f"{history['val_i2t_r1'][i]:.4f}" if 'val_i2t_r1' in history else "N/A"
            mean_val = f"{history['val_mean_r1'][i]:.4f}" if 'val_mean_r1' in history else "N/A"
            
            print(f"| {i+1} | {loss_val} | {t2i_val} | {i2t_val} | {mean_val} |")
        
        return True
    except Exception as e:
        print(f"生成训练历史图表时出错: {str(e)}")
        return False

def visualize_examples():
    """可视化几个检索例子，从训练好的模型中提取并展示"""
    import torch
    from torch.utils.data import DataLoader
    import json
    from PIL import Image
    import matplotlib.pyplot as plt
    import os
    from models.matcher import DualEncoderModel
    from data.datasets import FlickrImageTextDataset
    from transformers import BertTokenizer
    from data.transforms import get_image_transform
    from tqdm import tqdm
    
    print("\n开始生成检索示例可视化...")
    
    # 创建结果目录
    os.makedirs('retrieval_examples', exist_ok=True)
    
    try:
        # 1. 加载模型和数据
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 配置模型
        img_encoder_cfg = {'model_name': 'resnet101', 'pretrained': False, 'embed_dim': 768}
        txt_encoder_cfg = {
            'model_name': 'bert-base-uncased', 
            'pretrained': False, 
            'embed_dim': 768,
            'pool_type': 'mean_max'
        }
        
        # 创建模型
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
        )
        
        # 选择最佳模型文件
        model_files = [f for f in os.listdir('.') if f.startswith('best_model') and f.endswith('.pth')]
        if not model_files:
            print("错误: 找不到训练好的模型文件")
            return False
        
        # 优先选择best_model.pth或best_model_final.pth
        if 'best_model.pth' in model_files:
            model_path = 'best_model.pth'
        elif 'best_model_final.pth' in model_files:
            model_path = 'best_model_final.pth'
        else:
            model_path = model_files[0]
        
        print(f"加载模型: {model_path}")
        
        # 加载训练好的权重
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"成功加载模型 (Epoch {checkpoint.get('epoch', 'unknown')}, Mean R@1: {checkpoint.get('best_mean_recall', 'unknown'):.4f})")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            # 尝试不严格加载
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("已使用非严格模式加载模型")
        
        model.to(device)
        model.eval()
        
        # 准备数据
        ann_file = 'data/flickr8k_aim3/dataset_flickr8k.json'
        img_dir = 'data/flickr8k_aim3/images'
        
        # 优先使用测试集，如果没有则使用验证集
        if os.path.exists('data/flickr8k_aim3/test_list.txt'):
            data_list = 'data/flickr8k_aim3/test_list.txt'
            split_name = "测试集"
        else:
            data_list = 'data/flickr8k_aim3/val_list.txt'
            split_name = "验证集"
        
        print(f"使用{split_name}进行示例检索")
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        transform = get_image_transform(image_size=224, train=False)
        
        # 创建数据集
        eval_dataset = FlickrImageTextDataset(ann_file, img_dir, tokenizer, transform, data_list)
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=64,
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        
        print(f"数据集加载完成，共 {len(eval_dataset)} 个样本")
        
        # 2. 运行模型获取嵌入向量
        print("开始提取特征向量...")
        img_embeds_list, txt_embeds_list = [], []
        img_names, captions = [], []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="特征提取"):
                images, texts, batch_img_names, batch_captions = batch
                images = images.to(device)
                texts = {k: v.to(device) for k, v in texts.items()}
                
                img_embeds, txt_embeds = model(images, texts)
                
                img_embeds_list.append(img_embeds.cpu())
                txt_embeds_list.append(txt_embeds.cpu())
                img_names.extend(batch_img_names)
                captions.extend(batch_captions)
        
        # 合并所有批次
        img_embeds = torch.cat(img_embeds_list, dim=0).numpy()
        txt_embeds = torch.cat(txt_embeds_list, dim=0).numpy()
        
        # 确保向量已经L2归一化
        import numpy as np
        img_embeds = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
        txt_embeds = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)
        
        # 3. 计算相似度矩阵
        print("计算相似度矩阵...")
        sim_t2i = np.matmul(txt_embeds, img_embeds.T)  # 文本到图像相似度
        sim_i2t = np.matmul(img_embeds, txt_embeds.T)  # 图像到文本相似度
        
        # 4. 找检索例子
        print("查找检索示例...")
        
        # 文本到图像检索例子
        t2i_examples = []
        for i in range(len(captions)):
            t2i_indices = np.argsort(-sim_t2i[i])  # 降序排序
            rank = np.where(t2i_indices == i)[0][0]  # 正确图像的排名
            
            t2i_examples.append({
                'caption': captions[i],
                'gt_image': img_names[i],
                'retrieved_images': [img_names[j] for j in t2i_indices[:5]],  # 前5个结果
                'similarity_scores': [float(sim_t2i[i, j]) for j in t2i_indices[:5]],
                'is_correct_r1': (rank == 0),  # 是否在R@1中
                'is_correct_r5': (rank < 5),   # 是否在R@5中
                'rank': int(rank) + 1          # 排名从1开始
            })
        
        # 图像到文本检索例子
        i2t_examples = []
        for i in range(len(img_names)):
            i2t_indices = np.argsort(-sim_i2t[i])  # 降序排序
            rank = np.where(i2t_indices == i)[0][0]  # 正确文本的排名
            
            i2t_examples.append({
                'image': img_names[i],
                'gt_caption': captions[i],
                'retrieved_captions': [captions[j] for j in i2t_indices[:5]],  # 前5个结果
                'similarity_scores': [float(sim_i2t[i, j]) for j in i2t_indices[:5]],
                'is_correct_r1': (rank == 0),  # 是否在R@1中
                'is_correct_r5': (rank < 5),   # 是否在R@5中
                'rank': int(rank) + 1          # 排名从1开始
            })
        
        # 5. 筛选有代表性的例子
        # 成功案例: R@1正确的
        t2i_success = [ex for ex in t2i_examples if ex['is_correct_r1']]
        # 一般案例: R@1错但R@5正确的
        t2i_medium = [ex for ex in t2i_examples if not ex['is_correct_r1'] and ex['is_correct_r5']]
        # 失败案例: R@5都不正确的
        t2i_failure = [ex for ex in t2i_examples if not ex['is_correct_r5']]
        
        # 随机选择不同类型的示例，确保多样性
        import random
        random.seed(42)  # 固定随机种子
        
        # 筛选表现不同的样例
        selected_success = random.sample(t2i_success, min(3, len(t2i_success)))
        selected_medium = random.sample(t2i_medium, min(2, len(t2i_medium)))
        selected_failure = random.sample(t2i_failure, min(2, len(t2i_failure)))
        
        # 6. 可视化展示
        print("生成可视化图表...")
        
        def visualize_t2i_example(example, idx, example_type):
            """可视化文本到图像检索的例子"""
            # 创建网格布局: 1行6列，第一列放文本，后5列放检索结果
            fig = plt.figure(figsize=(15, 4))
            
            # 文本部分
            plt.subplot(1, 6, 1)
            plt.text(0.5, 0.5, f"Query:\n{example['caption']}", 
                    ha='center', va='center', wrap=True, fontsize=10)
            plt.axis('off')
            
            # 检索结果 (5张图片)
            for i in range(5):
                plt.subplot(1, 6, i+2)
                
                # 加载图像
                img_path = os.path.join(img_dir, example['retrieved_images'][i])
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    plt.imshow(img)
                else:
                    plt.text(0.5, 0.5, "Image not found", ha='center', va='center')
                
                # 标记是否为正确的ground truth
                if example['retrieved_images'][i] == example['gt_image']:
                    plt.title(f"✓ GT (Score: {example['similarity_scores'][i]:.3f})", 
                             color='green', fontsize=9)
                else:
                    plt.title(f"Rank {i+1} (Score: {example['similarity_scores'][i]:.3f})", 
                             fontsize=9)
                plt.axis('off')
            
            plt.suptitle(f"{example_type} Case (GT Rank: {example['rank']})", fontsize=14)
            plt.tight_layout()
            
            # 保存图像
            os.makedirs('figures', exist_ok=True)
            filename = f"figures/t2i_{example_type.lower()}_example_{idx}.png"
            plt.savefig(filename, dpi=200)
            plt.close()
            return filename
        
        # 可视化不同类型的例子
        success_files = []
        medium_files = []
        failure_files = []
        
        for idx, example in enumerate(selected_success):
            filename = visualize_t2i_example(example, idx, "Success")
            success_files.append(filename)
            
        for idx, example in enumerate(selected_medium):
            filename = visualize_t2i_example(example, idx, "Medium")
            medium_files.append(filename)
            
        for idx, example in enumerate(selected_failure):
            filename = visualize_t2i_example(example, idx, "Failure")
            failure_files.append(filename)
        
        # 7. 保存检索结果
        # 保存示例数据以便报告使用
        with open('retrieval_examples.json', 'w') as f:
            json.dump({
                'text2image': {
                    'success_examples': [e for e in selected_success],
                    'medium_examples': [e for e in selected_medium],
                    'failure_examples': [e for e in selected_failure]
                }
            }, f, indent=2)
        
        # 8. 打印结果总结，用于报告
        print("\n检索示例生成完成！")
        print(f"成功案例: {len(selected_success)} 个")
        for file in success_files:
            print(f"  - {file}")
        
        print(f"一般案例: {len(selected_medium)} 个")
        for file in medium_files:
            print(f"  - {file}")
            
        print(f"失败案例: {len(selected_failure)} 个")
        for file in failure_files:
            print(f"  - {file}")
        
        print(f"\n检索结果摘要已保存至 retrieval_examples.json")
        
        # 9. 创建一个HTML页面展示所有结果
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>图文检索示例</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                .gallery { display: flex; flex-wrap: wrap; }
                .case { margin: 10px; box-shadow: 0 0 5px #ccc; padding: 10px; }
                img { max-width: 100%; }
            </style>
        </head>
        <body>
            <h1>多模态图文检索示例</h1>
            
            <h2>成功案例 (R@1正确)</h2>
            <div class="gallery">
        """
        
        for file in success_files:
            html_content += f'<div class="case"><img src="{file}" alt="Success Case"></div>\n'
        
        html_content += """
            </div>
            
            <h2>一般案例 (R@5内正确)</h2>
            <div class="gallery">
        """
        
        for file in medium_files:
            html_content += f'<div class="case"><img src="{file}" alt="Medium Case"></div>\n'
            
        html_content += """
            </div>
            
            <h2>失败案例 (R@5外)</h2>
            <div class="gallery">
        """
        
        for file in failure_files:
            html_content += f'<div class="case"><img src="{file}" alt="Failure Case"></div>\n'
            
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open('figures/retrieval_examples.html', 'w') as f:
            f.write(html_content)
            
        print(f"已创建可视化展示页面: figures/retrieval_examples.html")
        return True
        
    except Exception as e:
        import traceback
        print(f"生成检索示例时出错: {str(e)}")
        traceback.print_exc()
        return False
    
def create_confusion_matrix():
    """创建一个消融实验结果可视化图表"""
    print("\n正在生成消融实验可视化...")
    
    try:
        # 导入matplotlib并设置中文支持
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        # 尝试设置中文字体，解决中文显示问题
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'SimSun', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        except:
            print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")
        
        # 消融实验数据（假设的数据，您可以替换为实际数据）
        configurations = [
            '基线\n(ResNet50+BERT)', 
            '+ ResNet101', 
            '+ Mean_Max\n池化', 
            '+ 困难负样本\n挖掘', 
            '+ 渐进式\n解冻',
            '最终\n优化模型'
        ]
        
        mean_r1_scores = [0.159, 0.183, 0.208, 0.226, 0.242, 0.267]
        relative_improvements = [0.0, 15.1, 30.8, 42.1, 52.2, 67.9]
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(
            configurations, 
            mean_r1_scores, 
            color='skyblue', 
            edgecolor='black',
            width=0.6  # 调整柱子宽度
        )
        
        # 添加值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            # 在柱子顶部添加数值标签
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                height + 0.005,
                f'{height:.3f}', 
                ha='center', 
                va='bottom',
                fontsize=11, 
                fontweight='bold'
            )
            
            # 添加相对提升百分比
            if i > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    height / 2,
                    f'+{relative_improvements[i]:.1f}%', 
                    ha='center', 
                    va='center',
                    fontsize=12, 
                    color='black', 
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')  # 添加背景框
                )
        
        # 设置标题和标签
        ax.set_title('模型配置的消融实验结果', fontsize=16, pad=15)
        ax.set_ylabel('Mean R@1', fontsize=14, labelpad=10)
        ax.set_xlabel('模型配置', fontsize=14, labelpad=10)
        
        # 优化网格线和刻度
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)  # 网格线置于图形之下
        
        # 设置坐标轴刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # 调整Y轴范围，留出顶部空间显示数字
        ax.set_ylim(0, max(mean_r1_scores) * 1.1)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存高质量图像
        plt.savefig('figures/ablation_study.png', dpi=300, bbox_inches='tight')
        print("✓ 已保存消融实验结果图: figures/ablation_study.png")
        
        # 同时保存SVG矢量格式，便于论文使用
        try:
            plt.savefig('figures/ablation_study.svg', format='svg', bbox_inches='tight')
            print("✓ 已同时保存SVG矢量图: figures/ablation_study.svg")
        except:
            print("保存SVG格式失败，可能需要安装额外依赖")
        
        plt.close()
        return True
    except Exception as e:
        import traceback
        print(f"生成消融实验图表时出错: {str(e)}")
        traceback.print_exc()
        return False

def get_required_packages():
    """检查并列出运行此脚本所需的包"""
    required_packages = [
        "torch", "torchvision", "transformers", "pillow",
        "matplotlib", "numpy", "tqdm"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("pillow", "PIL"))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def print_help():
    """打印帮助信息"""
    print("\n=== 多模态图文检索模型可视化工具 ===")
    print("\n此脚本用于生成以下可视化内容:")
    print("1. 训练历史曲线 - 可视化模型训练过程中的性能变化")
    print("   - 生成Mean R@1、召回率和训练损失曲线图")
    print("   - 保存位置: figures/retrieval_performance.png, figures/training_loss.png, figures/recall_at_k.png")
    
    print("\n2. 检索例子可视化 - 展示模型的文本到图像检索结果")
    print("   - 生成成功案例、一般案例和失败案例的检索结果图")
    print("   - 保存位置: figures/t2i_*.png 和 figures/retrieval_examples.html")
    
    print("\n3. 消融实验结果 - 比较不同模型配置的性能差异")
    print("   - 生成条形图展示不同配置的性能对比")
    print("   - 保存位置: figures/ablation_study.png")
    
    print("\n用法:")
    print("  python visualize.py [选项]")
    print("选项:")
    print("  all    - 生成所有可视化")
    print("  1      - 只生成训练历史曲线")
    print("  2      - 只生成检索例子可视化")
    print("  3      - 只生成消融实验结果")
    print("  help   - 显示此帮助信息")
    
    print("\n示例:")
    print("  python visualize.py all")
    print("  python visualize.py 2")

if __name__ == "__main__":
    print("=== 多模态图文检索模型可视化工具 ===")
    
    # 检查是否有缺少的依赖包
    missing_packages = get_required_packages()
    if missing_packages:
        print(f"\n警告: 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请使用以下命令安装所需依赖:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\n安装依赖后再次运行此脚本。")
        
        # 如果缺少关键包，则退出
        if any(p in ["torch", "PIL", "matplotlib", "numpy"] for p in missing_packages):
            print("缺少关键依赖，无法继续运行。")
            sys.exit(1)
    
    # 创建figures目录
    os.makedirs('figures', exist_ok=True)
    
    # 处理命令行参数
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == 'all':
            print("\n将依次生成所有可视化内容...")
            visualize_training_history()
            visualize_examples()
            create_confusion_matrix()
            print("\n所有可视化已完成！请查看figures目录下的结果。")
        
        elif arg == '1':
            print("\n生成训练历史曲线...")
            visualize_training_history()
        
        elif arg == '2':
            print("\n生成检索例子可视化...")
            visualize_examples()
            
        elif arg == '3':
            print("\n生成消融实验结果...")
            create_confusion_matrix()
            
        elif arg in ['help', '-h', '--help']:
            print_help()
            
        else:
            print(f"未知参数: {arg}")
            print_help()
    else:
        # 交互式菜单
        print("\n请选择要生成的可视化类型:")
        print("1. 训练历史曲线 - 展示训练过程中的性能变化")
        print("2. 检索例子可视化 - 展示模型检索结果示例")
        print("3. 消融实验结果 - 比较不同模型配置的性能")
        print("4. 全部生成")
        print("5. 显示帮助信息")
        
        try:
            choice = input("\n请输入选项编号 (1-5): ")
            
            if choice == '1':
                visualize_training_history()
                print("\n训练历史曲线已生成，请查看figures目录下的结果。")
                
            elif choice == '2':
                visualize_examples()
                print("\n检索例子可视化已生成，请查看figures目录下的结果。")
                
            elif choice == '3':
                create_confusion_matrix()
                print("\n消融实验结果已生成，请查看figures目录下的结果。")
                
            elif choice == '4':
                print("\n将依次生成所有可视化内容...")
                visualize_training_history()
                visualize_examples()
                create_confusion_matrix()
                print("\n所有可视化已完成！请查看figures目录下的结果。")
                
            elif choice == '5':
                print_help()
                
            else:
                print("无效的选择，请输入1到5之间的数字。")
        except KeyboardInterrupt:
            print("\n操作已取消。")
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            print("请确保已安装所有必要的依赖。")
