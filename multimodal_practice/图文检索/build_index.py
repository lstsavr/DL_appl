import os
import json
import glob
import argparse
import torch
import faiss
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

from models.dual_encoder import DualEncoder
from data.flickr8k_dataset import Vocabulary


def load_components(ckpt_path, vocab_json=None, device="cuda", embed_dim=512):
    """
    加载模型和词汇表
    
    Args:
        ckpt_path: 模型检查点路径
        vocab_json: 词汇表JSON文件路径，若为None则使用检查点中的词汇表
        device: 计算设备
        embed_dim: 嵌入维度
        
    Returns:
        model: 加载好的模型
        vocab: 词汇表对象
    """
    # 检查设备
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        device = "cpu"
    
    device = torch.device(device)
    
    # 加载检查点
    print(f"从 {ckpt_path} 加载模型...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 加载词汇表
    if vocab_json and os.path.exists(vocab_json):
        print(f"从 {vocab_json} 加载词汇表...")
        vocab = Vocabulary.load(vocab_json)
    else:
        print("使用检查点中的词汇表...")
        vocab = Vocabulary()
        vocab.stoi = checkpoint['vocab_stoi']
        vocab.itos = list(vocab.stoi.keys())
    
    # 创建并加载模型
    model = DualEncoder(vocab_size=len(vocab), embed_dim=embed_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, vocab


def encode_images_batch(model, img_dir, device, batch_size=256):
    """
    批量编码图像
    
    Args:
        model: 双流编码器模型
        img_dir: 图像目录
        device: 计算设备
        batch_size: 批处理大小
        
    Returns:
        img_embeddings: 图像嵌入向量 [N, embed_dim]
        img_paths: 图像路径列表
    """
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 获取所有图像路径
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    if not img_paths:
        raise ValueError(f"在 {img_dir} 中未找到图像")
    
    print(f"找到 {len(img_paths)} 张图像")
    
    # 批量处理图像
    all_img_embeddings = []
    
    for i in tqdm(range(0, len(img_paths), batch_size), desc="编码图像"):
        batch_paths = img_paths[i:i+batch_size]
        batch_tensors = []
        
        # 加载并转换图像
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                batch_tensors.append(img_tensor)
            except Exception as e:
                print(f"无法加载图像 {img_path}: {e}")
                # 使用零张量代替
                batch_tensors.append(torch.zeros(3, 224, 224))
        
        # 堆叠图像
        batch_images = torch.stack(batch_tensors, dim=0).to(device)
        
        # 编码图像
        with torch.no_grad():
            batch_embeddings, _ = model(batch_images, None)
            all_img_embeddings.append(batch_embeddings.cpu())
    
    # 合并所有图像嵌入
    img_embeddings = torch.cat(all_img_embeddings, dim=0)
    
    print(f"已编码 {len(img_paths)} 张图像，嵌入形状: {img_embeddings.shape}")
    
    return img_embeddings, img_paths


def encode_captions_batch(model, vocab, cap_file, device, batch_size=256):
    """
    批量编码文本
    
    Args:
        model: 双流编码器模型
        vocab: 词汇表对象
        cap_file: 包含所有caption的文件路径
        device: 计算设备
        batch_size: 批处理大小
        
    Returns:
        cap_embeddings: 文本嵌入向量 [M, embed_dim]
        captions: 文本列表
    """
    import re
    
    # 读取所有caption
    captions = []
    with open(cap_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                continue
                
            caption = parts[1]
            captions.append(caption)
    
    if not captions:
        raise ValueError(f"在 {cap_file} 中未找到caption")
    
    print(f"找到 {len(captions)} 个caption")
    
    # 批量处理文本
    all_cap_embeddings = []
    
    for i in tqdm(range(0, len(captions), batch_size), desc="编码文本"):
        batch_captions = captions[i:i+batch_size]
        
        # 分词并转换为索引
        tokenized = []
        for sentence in batch_captions:
            # 使用正则表达式分词
            sentence = sentence.lower()
            words = re.findall(r'\b[a-z]+\b', sentence)
            # 转换为索引序列
            indices = vocab.to_indices(" ".join(words))
            tokenized.append(torch.tensor(indices))
        
        # 对序列进行填充
        padded = pad_sequence(tokenized, batch_first=True, padding_value=0)
        
        # 将张量移至指定设备
        padded = padded.to(device)
        
        # 编码文本
        with torch.no_grad():
            _, batch_embeddings = model(None, padded)
            all_cap_embeddings.append(batch_embeddings.cpu())
    
    # 合并所有文本嵌入
    cap_embeddings = torch.cat(all_cap_embeddings, dim=0)
    
    print(f"已编码 {len(captions)} 个caption，嵌入形状: {cap_embeddings.shape}")
    
    return cap_embeddings, captions


def build_faiss_index(embeddings, index_path):
    """
    构建FAISS索引
    
    Args:
        embeddings: 嵌入向量 [N, D]
        index_path: 索引保存路径
        
    Returns:
        index: FAISS索引
    """
    # 转换为numpy数组
    embeddings_np = embeddings.numpy().astype(np.float32)
    
    # 创建FAISS索引 (使用内积，因为向量已经L2归一化)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    
    # 添加向量到索引
    index.add(embeddings_np)
    
    # 保存索引
    faiss.write_index(index, index_path)
    
    print(f"FAISS索引已保存至: {index_path}")
    
    return index


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建FAISS索引用于高效检索')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best.pth', help='模型检查点路径')
    parser.add_argument('--vocab', type=str, default='data/vocab.json', help='词汇表路径')
    parser.add_argument('--device', type=str, default=None, help='计算设备 (cuda或cpu)')
    parser.add_argument('--img_dir', type=str, default='data/raw/Flicker8k_Dataset', help='图像目录')
    parser.add_argument('--cap_file', type=str, default='data/raw/Flickr8k_text/Flickr8k.token.txt', help='包含所有caption的文件路径')
    parser.add_argument('--index_dir', type=str, default='indexes', help='索引保存目录')
    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--batch_size', type=int, default=256, help='批处理大小')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建索引目录
    os.makedirs(args.index_dir, exist_ok=True)
    
    # 加载模型和词汇表
    model, vocab = load_components(args.ckpt, args.vocab, args.device, args.embed_dim)
    
    # 编码图像
    img_embeddings, img_paths = encode_images_batch(model, args.img_dir, args.device, args.batch_size)
    
    # 编码文本
    cap_embeddings, captions = encode_captions_batch(model, vocab, args.cap_file, args.device, args.batch_size)
    
    # 保存嵌入和路径/文本
    torch.save(img_embeddings, os.path.join(args.index_dir, 'img_embs.pt'))
    torch.save(cap_embeddings, os.path.join(args.index_dir, 'cap_embs.pt'))
    
    with open(os.path.join(args.index_dir, 'paths.json'), 'w') as f:
        json.dump(img_paths, f)
    
    with open(os.path.join(args.index_dir, 'captions.json'), 'w') as f:
        json.dump(captions, f)
    
    print("嵌入和路径/文本已保存")
    
    # 构建FAISS索引
    build_faiss_index(img_embeddings, os.path.join(args.index_dir, 'img.index'))
    build_faiss_index(cap_embeddings, os.path.join(args.index_dir, 'cap.index'))
    
    print("索引构建完成!")


if __name__ == "__main__":
    main() 