import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 导入数据集处理模块
from data.flickr8k_dataset import (
    Vocabulary, build_vocab, Flickr8kDataset, get_dataloader
)

# 导入模型
from models.dual_encoder import DualEncoder, InfoNCELoss


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Flickr8K图文检索双流模型')
    
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='模型保存路径')
    parser.add_argument('--device', type=str, default=None, help='训练设备')
    
    args = parser.parse_args()
    
    # 如果未指定设备，则自动选择
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args


def eval_and_compute_recall(model, dataloader, device):
    """
    在验证集上评估模型并计算Recall@K指标
    
    Args:
        model: 双流编码器模型
        dataloader: 数据加载器
        device: 计算设备
        
    Returns:
        metrics: 包含各项指标的字典
    """
    model.eval()
    
    # 存储所有的嵌入向量
    all_img_embeds = []
    all_txt_embeds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["images"].to(device)
            captions = batch["captions"].to(device)
            
            # 前向传播
            img_embeds, txt_embeds = model(images, captions)
            
            # 保存嵌入向量（移至CPU）
            all_img_embeds.append(img_embeds.detach().cpu())
            all_txt_embeds.append(txt_embeds.detach().cpu())
    
    # 连接所有嵌入向量
    all_img_embeds = torch.cat(all_img_embeds, dim=0)
    all_txt_embeds = torch.cat(all_txt_embeds, dim=0)
    
    # 计算余弦相似度矩阵
    sim_matrix = torch.matmul(all_img_embeds, all_txt_embeds.t())
    
    # 图像检索文本 (i2t)
    i2t_ranks = []
    for i in range(sim_matrix.size(0)):
        # 获取第i个图像与所有文本的相似度
        similarities = sim_matrix[i]
        # 对相似度进行排序（降序）
        _, indices = torch.sort(similarities, descending=True)
        # 找到正确匹配的排名（假设对角线是正确匹配）
        rank = torch.where(indices == i)[0].item()
        i2t_ranks.append(rank)
    
    # 计算i2t的Recall@K
    i2t_ranks = torch.tensor(i2t_ranks)
    i2t_r1 = (i2t_ranks < 1).float().mean().item() * 100
    i2t_r5 = (i2t_ranks < 5).float().mean().item() * 100
    i2t_r10 = (i2t_ranks < 10).float().mean().item() * 100
    
    # 文本检索图像 (t2i)
    t2i_ranks = []
    for i in range(sim_matrix.size(1)):
        # 获取第i个文本与所有图像的相似度
        similarities = sim_matrix[:, i]
        # 对相似度进行排序（降序）
        _, indices = torch.sort(similarities, descending=True)
        # 找到正确匹配的排名（假设对角线是正确匹配）
        rank = torch.where(indices == i)[0].item()
        t2i_ranks.append(rank)
    
    # 计算t2i的Recall@K
    t2i_ranks = torch.tensor(t2i_ranks)
    t2i_r1 = (t2i_ranks < 1).float().mean().item() * 100
    t2i_r5 = (t2i_ranks < 5).float().mean().item() * 100
    t2i_r10 = (t2i_ranks < 10).float().mean().item() * 100
    
    # 计算整体平均R@1
    mean_r1 = (i2t_r1 + t2i_r1) / 2
    
    # 构建结果字典
    metrics = {
        "i2t": {"R1": i2t_r1, "R5": i2t_r5, "R10": i2t_r10},
        "t2i": {"R1": t2i_r1, "R5": t2i_r5, "R10": t2i_r10},
        "meanR": mean_r1
    }
    
    return metrics


def train(args):
    """
    训练主函数
    
    Args:
        args: 命令行参数
    """
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 打印参数
    print("训练参数:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载或构建词汇表
    vocab_path = "data/vocab.json"
    if os.path.exists(vocab_path):
        print(f"加载词汇表: {vocab_path}")
        vocab = Vocabulary.load(vocab_path)
    else:
        print("构建词汇表...")
        captions_file = "data/raw/Flickr8k_text/Flickr8k.token.txt"
        vocab = build_vocab(captions_file, freq_threshold=5)
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        vocab.save(vocab_path)
        print(f"词汇表已保存至: {vocab_path}")
    
    print(f"词汇表大小: {len(vocab)}")
    
    # 数据加载器
    print("创建数据加载器...")
    root_img_dir = "data/raw/Flicker8k_Dataset"
    flickr_text_dir = "data/raw/Flickr8k_text"
    
    train_loader = get_dataloader(
        split="train",
        root_img_dir=root_img_dir,
        flickr_text_dir=flickr_text_dir,
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = get_dataloader(
        split="val",
        root_img_dir=root_img_dir,
        flickr_text_dir=flickr_text_dir,
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"训练集大小: {len(train_loader.dataset)} 样本")
    print(f"验证集大小: {len(val_loader.dataset)} 样本")
    
    # 创建模型
    print("创建模型...")
    model = DualEncoder(vocab_size=len(vocab), embed_dim=args.embed_dim)
    model = model.to(device)
    
    # 创建损失函数和优化器
    loss_fn = InfoNCELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 训练循环
    best_mean_r = 0.0
    
    print("开始训练...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        # 训练一个epoch
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = batch["images"].to(device)
            captions = batch["captions"].to(device)
            
            # 前向传播
            img_embeds, txt_embeds = model(images, captions)
            
            # 计算损失
            loss = loss_fn(img_embeds, txt_embeds)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # 更新学习率
        lr_scheduler.step()
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        
        # 在验证集上评估
        metrics = eval_and_compute_recall(model, val_loader, device)
        
        # 打印训练和验证结果
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Image to Text: R@1: {metrics['i2t']['R1']:.2f}, R@5: {metrics['i2t']['R5']:.2f}, R@10: {metrics['i2t']['R10']:.2f}")
        print(f"  Text to Image: R@1: {metrics['t2i']['R1']:.2f}, R@5: {metrics['t2i']['R5']:.2f}, R@10: {metrics['t2i']['R10']:.2f}")
        print(f"  Mean R@1: {metrics['meanR']:.2f}")
        
        # 保存最佳模型（基于meanR指标）
        if metrics['meanR'] > best_mean_r:
            best_mean_r = metrics['meanR']
            checkpoint_path = os.path.join(args.checkpoint_dir, "best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'vocab_stoi': vocab.stoi
            }, checkpoint_path)
            print(f"  保存最佳模型至: {checkpoint_path} (Mean R@1: {best_mean_r:.2f})")
        
        # 保存最新模型
        checkpoint_path = os.path.join(args.checkpoint_dir, "latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'vocab_stoi': vocab.stoi
        }, checkpoint_path)
    
    print("训练完成!")
    print(f"最佳平均R@1: {best_mean_r:.2f}")


def main():
    """主函数入口"""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main() 