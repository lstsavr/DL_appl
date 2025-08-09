import torch
import numpy as np
from tqdm import tqdm
import time

def recall_at_k(sim_matrix, k):
    
    N = sim_matrix.shape[0]
    ranks = np.zeros(N)
    
    for i in range(N):
        inds = np.argsort(-sim_matrix[i])
        # 找到ground truth的排名
        rank = np.where(inds == i)[0][0]
        ranks[i] = rank
    
    # 计算多个K值的召回率
    recall = [(ranks < x).mean() for x in [1, 5, 10]]
    # 计算MRR (Mean Reciprocal Rank)
    mrr = (1.0 / (ranks + 1)).mean()
    
    return recall, mrr, ranks

def evaluate(model, dataloader, device, save_topk_path=None, topk=10, use_amp=True):
    """评估模型性能，计算图文检索相关指标
    
        model: 待评估的模型
        dataloader: 数据加载器
        device: 计算设备
        save_topk_path: 保存top-k检索结果的路径
        topk: 检索结果保存的数量
        use_amp: 是否使用自动混合精度
    """
    model.eval()
    start_time = time.time()
    
    img_embeds_list, txt_embeds_list = [], []
    img_names, captions = [], []
    
    # 使用torch.no_grad()减少内存使用
    with torch.no_grad():
        if use_amp and device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
            from torch.cuda.amp import autocast
            
            for batch in tqdm(dataloader, desc='Eval'):
                images, texts, batch_img_names, batch_captions = batch
                images = images.to(device, non_blocking=True)
                texts = {k: v.to(device, non_blocking=True) for k, v in texts.items()}
                
                with autocast():
                    img_embeds, txt_embeds = model(images, texts)
                
                img_embeds_list.append(img_embeds.cpu())
                txt_embeds_list.append(txt_embeds.cpu())
                img_names.extend(batch_img_names)
                captions.extend(batch_captions)
        else:
            # 标准评估流程
            for batch in tqdm(dataloader, desc='Eval'):
                images, texts, batch_img_names, batch_captions = batch
                images = images.to(device, non_blocking=True)
                texts = {k: v.to(device, non_blocking=True) for k, v in texts.items()}
                img_embeds, txt_embeds = model(images, texts)
                img_embeds_list.append(img_embeds.cpu())
                txt_embeds_list.append(txt_embeds.cpu())
                img_names.extend(batch_img_names)
                captions.extend(batch_captions)
    
    # 拼接嵌入向量
    img_embeds = torch.cat(img_embeds_list, dim=0).numpy()
    txt_embeds = torch.cat(txt_embeds_list, dim=0).numpy()
    
    # 确保归一化
    img_embeds = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    txt_embeds = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)
    
    # 计算相似度矩阵
    print("计算相似度矩阵...")
    sim_t2i = np.matmul(txt_embeds, img_embeds.T)
    sim_i2t = np.matmul(img_embeds, txt_embeds.T)
    
    # 计算评估指标
    print("计算Text2Image检索指标...")
    recall_t2i, mrr_t2i, ranks_t2i = recall_at_k(sim_t2i, k=10)
    print("计算Image2Text检索指标...")
    recall_i2t, mrr_i2t, ranks_i2t = recall_at_k(sim_i2t, k=10)
    
    results = {
        'text2image': {'R@1': recall_t2i[0], 'R@5': recall_t2i[1], 'R@10': recall_t2i[2], 'MRR': mrr_t2i},
        'image2text': {'R@1': recall_i2t[0], 'R@5': recall_i2t[1], 'R@10': recall_i2t[2], 'MRR': mrr_i2t},
        'mean_r1': (recall_t2i[0] + recall_i2t[0]) / 2,  # 添加Mean R@1指标
        'eval_time': time.time() - start_time,
    }
    
    print(f"Text2Image: R@1={recall_t2i[0]:.4f}, R@5={recall_t2i[1]:.4f}, R@10={recall_t2i[2]:.4f}, MRR={mrr_t2i:.4f}")
    print(f"Image2Text: R@1={recall_i2t[0]:.4f}, R@5={recall_i2t[1]:.4f}, R@10={recall_i2t[2]:.4f}, MRR={mrr_i2t:.4f}")
    print(f"Mean R@1: {results['mean_r1']:.4f}")
    print(f"Evaluation time: {results['eval_time']:.2f} seconds")
    
    if save_topk_path is not None:
        import json
        print(f"保存Top-{topk}检索结果到 {save_topk_path}...")
        topk_results = []
        for i in range(len(captions)):
            inds = np.argsort(-sim_t2i[i])[:topk]
            top_imgs = [img_names[j] for j in inds]
            # 添加是否正确匹配的指示
            is_correct_r1 = (i in inds[:1])
            is_correct_r5 = (i in inds[:5])
            
            topk_results.append({
                'caption': captions[i], 
                'topk_images': top_imgs, 
                'gt': img_names[i],
                'correct_r1': is_correct_r1,
                'correct_r5': is_correct_r5
            })
            
        with open(save_topk_path, 'w', encoding='utf-8') as f:
            json.dump(topk_results, f, ensure_ascii=False, indent=2)
    

    return results
