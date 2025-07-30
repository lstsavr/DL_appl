import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# 导入数据集处理模块
from data.flickr8k_dataset import (
    Vocabulary, build_vocab, get_dataloader
)

# 导入模型
from models.cross_attention import CrossAttentionModel
from models.dual_encoder import InfoNCELoss


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Flickr8K交叉注意力模型')
    
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=48, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--embed_dim', type=int, default=768, help='嵌入维度')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints_ca', help='模型保存路径')
    parser.add_argument('--device', type=str, default=None, help='训练设备')
    parser.add_argument('--warmup', type=int, default=500, help='预热步数')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='权重衰减')
    
    args = parser.parse_args()
    
    # 如果未指定设备，则自动选择
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args


def eval_recall(model, loader, device):
    """
    评估模型的Recall指标
    
    Args:
        model: 交叉注意力模型
        loader: 数据加载器
        device: 计算设备
        
    Returns:
        metrics: 包含各项指标的字典
    """
    model.eval()
    img_e, txt_e = [], []
    
    # 收集所有嵌入向量
    with torch.no_grad():
        for imgs, caps in loader:
            imgs, caps = imgs.to(device), caps.to(device)
            img_emb, txt_emb = model(imgs, caps)
            img_e.append(img_emb.cpu())
            txt_e.append(txt_emb.cpu())
    
    # 合并所有嵌入向量
    I = torch.cat(img_e)
    T = torch.cat(txt_e)
    
    # 计算相似度矩阵
    S = I @ T.T
    
    # 计算Recall@K
    def recall(mat, k):
        # 获取每行前k个最大值的索引
        _, indices = mat.topk(k, dim=1)
        # 检查对角线元素（正确匹配）是否在前k个中
        correct = indices.eq(torch.arange(mat.size(0)).unsqueeze(1))
        # 计算Recall@K
        return correct.any(dim=1).float().mean().item() * 100
    
    # 计算各项指标
    metrics = {
        "i2t": {
            "R1": recall(S, 1),
            "R5": recall(S, 5),
            "R10": recall(S, 10)
        },
        "t2i": {
            "R1": recall(S.T, 1),
            "R5": recall(S.T, 5),
            "R10": recall(S.T, 10)
        },
        "meanR": (recall(S, 1) + recall(S.T, 1)) / 2
    }
    
    return metrics


def train(args):
    """
    训练主函数
    
    Args:
        args: 命令行参数
    """
    # 创建检查点目录
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
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
    model = CrossAttentionModel(vocab_size=len(vocab), d_model=args.embed_dim)
    model = model.to(device)
    
    # 创建损失函数和优化器
    loss_fn = InfoNCELoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.0,
        total_iters=args.warmup
    )
    
    # 训练循环
    best_mean_r = 0.0
    global_step = 0
    
    print("开始训练...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
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
            
            # 更新学习率
            if global_step < args.warmup:
                scheduler.step()
            
            global_step += 1
            running_loss += loss.item()
        
        # 计算平均训练损失
        avg_loss = running_loss / len(train_loader)
        
        # 在验证集上评估
        print("在验证集上评估...")
        metrics = eval_recall(model, val_loader, device)
        
        # 打印训练和验证结果
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Image to Text: R@1: {metrics['i2t']['R1']:.2f}, R@5: {metrics['i2t']['R5']:.2f}, R@10: {metrics['i2t']['R10']:.2f}")
        print(f"  Text to Image: R@1: {metrics['t2i']['R1']:.2f}, R@5: {metrics['t2i']['R5']:.2f}, R@10: {metrics['t2i']['R10']:.2f}")
        print(f"  Mean R@1: {metrics['meanR']:.2f}")
        
        # 保存最佳模型
        if metrics['meanR'] > best_mean_r:
            best_mean_r = metrics['meanR']
            checkpoint_path = os.path.join(args.ckpt_dir, "best_ca.pth")
            torch.save({
                'model': model.state_dict(),
                'vocab': vocab.stoi
            }, checkpoint_path)
            print(f"  保存最佳模型至: {checkpoint_path} (Mean R@1: {best_mean_r:.2f})")
    
    print("训练完成!")
    print(f"最佳平均R@1: {best_mean_r:.2f}")


def main():
    """主函数入口"""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main() 