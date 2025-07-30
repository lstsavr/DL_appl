import os
import re
import json
import torch
import faiss
import tempfile
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import gradio as gr
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


def encode_text(sentences, model, vocab, device):
    """
    编码文本
    
    Args:
        sentences: 文本列表
        model: 双流编码器模型
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
        model: 双流编码器模型
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


def load_faiss_indexes(index_dir="indexes"):
    """
    加载FAISS索引和相关数据
    
    Args:
        index_dir: 索引目录
        
    Returns:
        img_index: 图像索引
        cap_index: 文本索引
        img_paths: 图像路径列表
        captions: 文本列表
        img_embeddings: 图像嵌入向量
        cap_embeddings: 文本嵌入向量
    """
    print("加载FAISS索引和相关数据...")
    
    # 加载索引
    img_index_path = os.path.join(index_dir, "img.index")
    cap_index_path = os.path.join(index_dir, "cap.index")
    
    if not os.path.exists(img_index_path) or not os.path.exists(cap_index_path):
        raise FileNotFoundError(f"索引文件不存在，请先运行 build_index.py 构建索引")
    
    img_index = faiss.read_index(img_index_path)
    cap_index = faiss.read_index(cap_index_path)
    
    # 加载路径和文本
    with open(os.path.join(index_dir, "paths.json"), "r") as f:
        img_paths = json.load(f)
    
    with open(os.path.join(index_dir, "captions.json"), "r") as f:
        captions = json.load(f)
    
    # 加载嵌入向量
    img_embeddings = torch.load(os.path.join(index_dir, "img_embs.pt"))
    cap_embeddings = torch.load(os.path.join(index_dir, "cap_embs.pt"))
    
    print(f"已加载 {len(img_paths)} 张图像和 {len(captions)} 个文本")
    
    return img_index, cap_index, img_paths, captions, img_embeddings, cap_embeddings


def search_img(query_emb, k, img_index):
    """
    搜索图像
    
    Args:
        query_emb: 查询嵌入向量
        k: 返回结果数量
        img_index: 图像索引
        
    Returns:
        indices: 结果索引
        scores: 相似度分数
    """
    # 转换为numpy数组
    query_np = query_emb.cpu().numpy().astype(np.float32)
    
    # 搜索
    scores, indices = img_index.search(query_np, k)
    
    return indices[0], scores[0]


def search_cap(query_emb, k, cap_index):
    """
    搜索文本
    
    Args:
        query_emb: 查询嵌入向量
        k: 返回结果数量
        cap_index: 文本索引
        
    Returns:
        indices: 结果索引
        scores: 相似度分数
    """
    # 转换为numpy数组
    query_np = query_emb.cpu().numpy().astype(np.float32)
    
    # 搜索
    scores, indices = cap_index.search(query_np, k)
    
    return indices[0], scores[0]


def t2i_predict(text, k, model, vocab, device, img_index, img_paths):
    """
    文本到图像检索
    
    Args:
        text: 查询文本
        k: 返回结果数量
        model: 模型
        vocab: 词汇表
        device: 设备
        img_index: 图像索引
        img_paths: 图像路径列表
        
    Returns:
        result_images: 结果图像列表
        result_scores: 相似度分数列表
    """
    # 编码文本
    text_embedding = encode_text([text], model, vocab, device)[0].unsqueeze(0)
    
    # 搜索图像
    indices, scores = search_img(text_embedding, k, img_index)
    
    # 获取结果
    result_images = [(img_paths[idx], f"Score: {score:.4f}") for idx, score in zip(indices, scores)]
    
    return result_images


def i2t_predict(image, k, model, device, cap_index, captions):
    """
    图像到文本检索
    
    Args:
        image: 查询图像
        k: 返回结果数量
        model: 模型
        device: 设备
        cap_index: 文本索引
        captions: 文本列表
        
    Returns:
        result_texts: 结果文本列表
        result_scores: 相似度分数列表
    """
    # 保存图像到临时文件
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        image.save(tmp.name)
        
        # 编码图像
        image_embedding = encode_image([tmp.name], model, device)[0].unsqueeze(0)
    
    # 搜索文本
    indices, scores = search_cap(image_embedding, k, cap_index)
    
    # 获取结果
    result_texts = []
    for idx, score in zip(indices, scores):
        caption = captions[idx]
        result_texts.append((caption, f"Score: {score:.4f}"))
    
    return result_texts


def create_demo():
    """
    创建Gradio演示界面
    
    Returns:
        demo: Gradio演示界面
    """
    # 加载模型和词汇表
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_components(
        ckpt_path="checkpoints/best.pth",
        vocab_json="data/vocab.json",
        device=device
    )
    
    # 加载索引和数据
    img_index, cap_index, img_paths, captions, img_embeddings, cap_embeddings = load_faiss_indexes()
    
    # 文本到图像界面
    def t2i_interface(text, k):
        return t2i_predict(text, k, model, vocab, device, img_index, img_paths)
    
    # 图像到文本界面
    def i2t_interface(image, k):
        if image is None:
            return []
        return i2t_interface_process(image, k)
    
    def i2t_interface_process(image, k):
        results = i2t_predict(image, k, model, device, cap_index, captions)
        # 转换为Gradio HighlightedText格式
        formatted_results = []
        for text, score in results:
            formatted_results.append((text, score))
        return formatted_results
    
    # 创建文本到图像标签页
    with gr.Blocks() as t2i_tab:
        gr.Markdown("## 文本到图像检索")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="输入查询文本", lines=3, placeholder="输入描述图像的文本...")
                k_slider_t2i = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="返回结果数量")
                search_button_t2i = gr.Button("搜索")
            with gr.Column():
                gallery_output = gr.Gallery(label="检索结果", show_label=True, columns=3, height=600)
        
        search_button_t2i.click(
            fn=t2i_interface,
            inputs=[text_input, k_slider_t2i],
            outputs=gallery_output
        )
    
    # 创建图像到文本标签页
    with gr.Blocks() as i2t_tab:
        gr.Markdown("## 图像到文本检索")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图像")
                k_slider_i2t = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="返回结果数量")
                search_button_i2t = gr.Button("搜索")
            with gr.Column():
                text_output = gr.HighlightedText(
                    label="检索结果",
                    combine_adjacent=True,
                    show_legend=True,
                    color_map={"Score: ": "blue"}
                )
        
        search_button_i2t.click(
            fn=i2t_interface,
            inputs=[image_input, k_slider_i2t],
            outputs=text_output
        )
    
    # 创建标签页界面
    demo = gr.TabbedInterface(
        [t2i_tab, i2t_tab],
        ["文本到图像", "图像到文本"],
        theme="base",
        title="Flickr8K 跨模态检索演示"
    )
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860) 