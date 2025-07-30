import os
import re
import json
import time
import glob
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

from data.flickr8k_dataset import Vocabulary


def load_components(model_type, ckpt_path, vocab_json=None, device="cuda"):
    """
    加载模型和词汇表
    
    Args:
        model_type: 模型类型 ('dual' 或 'cross')
        ckpt_path: 模型检查点路径
        vocab_json: 词汇表JSON文件路径，若为None则使用检查点中的词汇表
        device: 计算设备
        
    Returns:
        model: 加载好的模型
        vocab: 词汇表对象
        embed_dim: 嵌入维度
    """
    # 检查设备
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        device = "cpu"
    
    device = torch.device(device)
    
    # 根据模型类型选择相应的模块和嵌入维度
    if model_type == "dual":
        from models.dual_encoder import DualEncoder
        embed_dim = 512
        print("使用双流编码器模型 (embed_dim=512)")
    else:
        from models.cross_attention import CrossAttentionModel as DualEncoder
        embed_dim = 768
        print("使用交叉注意力模型 (embed_dim=768)")
    
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
        if 'vocab_stoi' in checkpoint:
            vocab.stoi = checkpoint['vocab_stoi']
        elif 'vocab' in checkpoint:
            vocab.stoi = checkpoint['vocab']
        else:
            raise ValueError("检查点中未找到词汇表信息")
        vocab.itos = list(vocab.stoi.keys())
    
    # 创建并加载模型
    model = DualEncoder(vocab_size=len(vocab), embed_dim=embed_dim)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        raise ValueError("检查点中未找到模型权重")
    
    model = model.to(device)
    model.eval()
    
    return model, vocab, embed_dim


def encode_text(sentences, model, vocab, device):
    """
    编码文本
    
    Args:
        sentences: 文本列表
        model: 模型
        vocab: 词汇表对象
        device: 计算设备
        
    Returns:
        text_embeddings: 文本嵌入向量 [N, embed_dim]
    """
    # 分词并转换为索引
    tokenized = []
    for sentence in sentences:
        # 使用正则表达式分词，与build_vocab保持一致
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
        _, text_embeddings = model(None, padded)
    
    return text_embeddings


def encode_image(img_paths, model, device):
    """
    编码图像
    
    Args:
        img_paths: 图像路径列表
        model: 模型
        device: 计算设备
        
    Returns:
        image_embeddings: 图像嵌入向量 [N, embed_dim]
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
    
    # 加载并转换图像
    images = []
    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"无法加载图像 {img_path}: {e}")
            # 使用零张量代替
            images.append(torch.zeros(3, 224, 224))
    
    # 堆叠图像
    images = torch.stack(images, dim=0).to(device)
    
    # 编码图像
    with torch.no_grad():
        image_embeddings, _ = model(images, None)
    
    return image_embeddings


def compute_similarity(A, B):
    """
    计算余弦相似度矩阵
    
    Args:
        A: 第一组嵌入向量 [N, D]
        B: 第二组嵌入向量 [M, D]
        
    Returns:
        similarity: 相似度矩阵 [N, M]
    """
    # 计算余弦相似度
    similarity = torch.matmul(A, B.t())
    
    return similarity


def retrieve(sim_matrix, k=5):
    """
    检索最相似的项
    
    Args:
        sim_matrix: 相似度矩阵 [N, M]
        k: 返回的结果数量
        
    Returns:
        results: 列表，每个元素是(索引, 分数)元组的列表
    """
    results = []
    
    # 对每个查询
    for i in range(sim_matrix.shape[0]):
        similarities = sim_matrix[i]
        # 获取前k个最相似的项
        values, indices = torch.topk(similarities, k=min(k, len(similarities)))
        
        # 转换为列表
        items = [(idx.item(), val.item()) for idx, val in zip(indices, values)]
        results.append(items)
    
    return results


def print_results(results, mode, query, target_items):
    """
    打印检索结果
    
    Args:
        results: 检索结果列表
        mode: 检索模式 ('t2i' 或 'i2t')
        query: 查询文本或图像路径
        target_items: 目标项列表 (图像路径或文本)
    """
    print("\n" + "="*50)
    print(f"检索模式: {'文本→图像' if mode == 't2i' else '图像→文本'}")
    print(f"查询: {query}")
    print("="*50)
    
    for i, result_list in enumerate(results):
        print(f"\n查询 {i+1} 的结果:")
        for rank, (idx, score) in enumerate(result_list):
            if mode == 't2i':
                print(f"  Rank {rank+1}: {os.path.basename(target_items[idx])} (Score: {score:.4f})")
            else:
                print(f"  Rank {rank+1}: \"{target_items[idx]}\" (Score: {score:.4f})")
    
    print("\n" + "="*50)


def save_gallery(results, mode, query, target_items, save_dir="results"):
    """
    保存HTML结果画廊
    
    Args:
        results: 检索结果列表
        mode: 检索模式 ('t2i' 或 'i2t')
        query: 查询文本或图像路径
        target_items: 目标项列表 (图像路径或文本)
        save_dir: 保存目录
    
    Returns:
        html_path: 保存的HTML文件路径
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(save_dir, f"{mode}_{timestamp}.html")
    
    # 创建HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>检索结果 - {mode}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                margin-bottom: 20px;
                border-radius: 5px;
            }}
            .query {{
                background-color: #f1f1f1;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
                border-left: 5px solid #2196F3;
            }}
            .results {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }}
            .result-item {{
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                width: 300px;
            }}
            .result-item img {{
                max-width: 100%;
                height: auto;
                border-radius: 3px;
            }}
            .score {{
                color: #2196F3;
                font-weight: bold;
            }}
            .caption {{
                margin-top: 10px;
                font-size: 14px;
            }}
            .query-image {{
                max-width: 300px;
                max-height: 300px;
                display: block;
                margin-bottom: 20px;
                border-radius: 5px;
            }}
            .result-list {{
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                max-width: 800px;
            }}
            .result-text {{
                padding: 10px;
                margin: 5px 0;
                background-color: #f9f9f9;
                border-left: 3px solid #2196F3;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>检索结果</h1>
            <p>模式: {mode} ({'文本→图像' if mode == 't2i' else '图像→文本'})</p>
            <p>时间: {timestamp}</p>
        </div>
    """
    
    # 添加查询信息
    if mode == 't2i':
        html_content += f"""
        <div class="query">
            <h2>查询文本:</h2>
            <p>"{query}"</p>
        </div>
        """
    else:
        html_content += f"""
        <div class="query">
            <h2>查询图像:</h2>
            <img src="file:///{os.path.abspath(query)}" class="query-image" alt="查询图像">
        </div>
        """
    
    # 添加结果
    if mode == 't2i':
        html_content += '<div class="results">'
        for result_list in results:
            for rank, (idx, score) in enumerate(result_list):
                img_path = target_items[idx]
                html_content += f"""
                <div class="result-item">
                    <h3>Rank {rank+1}</h3>
                    <img src="file:///{os.path.abspath(img_path)}" alt="结果图像">
                    <p class="score">相似度: {score:.4f}</p>
                    <p class="caption">{os.path.basename(img_path)}</p>
                </div>
                """
        html_content += '</div>'
    else:
        html_content += '<div class="result-list">'
        for result_list in results:
            for rank, (idx, score) in enumerate(result_list):
                caption = target_items[idx]
                html_content += f"""
                <div class="result-text">
                    <h3>Rank {rank+1} (相似度: {score:.4f})</h3>
                    <p>"{caption}"</p>
                </div>
                """
        html_content += '</div>'
    
    html_content += """
    </body>
    </html>
    """
    
    # 保存HTML文件
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"结果已保存至: {html_path}")
    return html_path


def text_to_image(text, model, vocab, device, image_dir, k=5, save_html=False, save_dir="results"):
    """
    文本到图像检索
    
    Args:
        text: 查询文本
        model: 模型
        vocab: 词汇表对象
        device: 计算设备
        image_dir: 图像目录
        k: 返回结果数量
        save_html: 是否保存HTML结果
        save_dir: HTML保存目录
        
    Returns:
        results: 检索结果
    """
    # 获取所有图像路径
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    if not image_paths:
        print(f"在 {image_dir} 中未找到图像")
        return []
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 编码文本
    text_embeddings = encode_text([text], model, vocab, device)
    
    # 批量处理图像
    batch_size = 64
    all_image_embeddings = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="编码图像"):
        batch_paths = image_paths[i:i+batch_size]
        batch_embeddings = encode_image(batch_paths, model, device)
        all_image_embeddings.append(batch_embeddings.cpu())
    
    # 合并所有图像嵌入
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    
    # 计算相似度
    similarity = compute_similarity(text_embeddings.cpu(), image_embeddings)
    
    # 检索结果
    results = retrieve(similarity, k=k)
    
    # 打印结果
    print_results(results, 't2i', text, image_paths)
    
    # 保存HTML结果
    if save_html:
        save_gallery(results, 't2i', text, image_paths, save_dir)
    
    return results


def image_to_text(image_path, model, vocab, device, caption_file, k=5, save_html=False, save_dir="results"):
    """
    图像到文本检索
    
    Args:
        image_path: 查询图像路径
        model: 模型
        vocab: 词汇表对象
        device: 计算设备
        caption_file: 包含所有caption的文件路径
        k: 返回结果数量
        save_html: 是否保存HTML结果
        save_dir: HTML保存目录
        
    Returns:
        results: 检索结果
    """
    # 读取所有caption
    captions = []
    with open(caption_file, 'r', encoding='utf-8') as f:
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
        print(f"在 {caption_file} 中未找到caption")
        return []
    
    print(f"找到 {len(captions)} 个caption")
    
    # 编码图像
    image_embeddings = encode_image([image_path], model, device)
    
    # 批量处理文本
    batch_size = 128
    all_text_embeddings = []
    
    for i in tqdm(range(0, len(captions), batch_size), desc="编码文本"):
        batch_captions = captions[i:i+batch_size]
        batch_embeddings = encode_text(batch_captions, model, vocab, device)
        all_text_embeddings.append(batch_embeddings.cpu())
    
    # 合并所有文本嵌入
    text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    # 计算相似度
    similarity = compute_similarity(image_embeddings.cpu(), text_embeddings)
    
    # 检索结果
    results = retrieve(similarity, k=k)
    
    # 打印结果
    print_results(results, 'i2t', image_path, captions)
    
    # 保存HTML结果
    if save_html:
        save_gallery(results, 'i2t', image_path, captions, save_dir)
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一检索接口')
    parser.add_argument('--model_type', type=str, choices=['dual', 'cross'], required=True, help='模型类型: 双流编码器(dual)或交叉注意力(cross)')
    parser.add_argument('--mode', type=str, choices=['t2i', 'i2t'], required=True, help='检索方向: 文本到图像(t2i)或图像到文本(i2t)')
    parser.add_argument('--query', type=str, required=True, help='查询文本或图像路径')
    parser.add_argument('--k', type=int, default=5, help='返回的结果数量')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best.pth', help='模型检查点路径')
    parser.add_argument('--vocab', type=str, default='data/vocab.json', help='词汇表路径')
    parser.add_argument('--device', type=str, default=None, help='计算设备 (cuda或cpu)')
    parser.add_argument('--html', action='store_true', help='保存HTML结果')
    parser.add_argument('--save_dir', type=str, default='results', help='HTML结果保存目录')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型和词汇表
    model, vocab, _ = load_components(args.model_type, args.ckpt, args.vocab, args.device)
    
    # 执行检索
    if args.mode == 't2i':
        image_dir = "data/raw/Flicker8k_Dataset"
        text_to_image(args.query, model, vocab, args.device, image_dir, args.k, args.html, args.save_dir)
    else:
        caption_file = "data/raw/Flickr8k_text/Flickr8k.token.txt"
        image_to_text(args.query, model, vocab, args.device, caption_file, args.k, args.html, args.save_dir)


if __name__ == "__main__":
    main() 