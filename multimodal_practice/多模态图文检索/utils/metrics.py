import numpy as np

def compute_recall(similarity_matrix, K_list=[1, 5, 10]):
    """
    similarity_matrix: [N, N]，每行i为文本i与所有图像的相似度
    K_list: list of int, 计算top-K
    返回: {'text2img_R@1':..., 'img2text_R@5':...}
    
    旧版接口，保留用于兼容性，建议使用compute_metrics
    """
    N = similarity_matrix.shape[0]
    results = {}
    # text->image
    ranks_t2i = np.zeros(N)
    for i in range(N):
        inds = np.argsort(-similarity_matrix[i])  # 分数降序
        rank = np.where(inds == i)[0][0]
        ranks_t2i[i] = rank
    for k in K_list:
        recall = (ranks_t2i < k).mean()
        results[f'text2img_R@{k}'] = recall
    # image->text
    ranks_i2t = np.zeros(N)
    for j in range(N):
        inds = np.argsort(-similarity_matrix[:, j])
        rank = np.where(inds == j)[0][0]
        ranks_i2t[j] = rank
    for k in K_list:
        recall = (ranks_i2t < k).mean()
        results[f'img2text_R@{k}'] = recall
    return results
    
def compute_metrics(txt_embeds, img_embeds, K_list=[1, 5, 10]):
    """
    计算多模态检索指标
    
    参数:
    - txt_embeds: 文本特征矩阵 [N, D]
    - img_embeds: 图像特征矩阵 [N, D]
    - K_list: 计算R@K的K值列表
    
    返回:
    - 包含text2image和image2text指标的字典
    {
        'text2image': {'R@1': x, 'R@5': y, 'R@10': z, 'MRR': m},
        'image2text': {'R@1': a, 'R@5': b, 'R@10': c, 'MRR': d}
    }
    """
    # 归一化特征向量
    txt_norm = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)
    img_norm = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    
    # 计算相似度矩阵
    sim_t2i = np.matmul(txt_norm, img_norm.T)  # [N, N]
    sim_i2t = np.matmul(img_norm, txt_norm.T)  # [N, N]
    
    # 计算text->image的指标
    N = sim_t2i.shape[0]
    ranks_t2i = np.zeros(N)
    for i in range(N):
        inds = np.argsort(-sim_t2i[i])
        rank = np.where(inds == i)[0][0]
        ranks_t2i[i] = rank
    
    recalls_t2i = [(ranks_t2i < k).mean() for k in K_list]
    mrr_t2i = (1.0 / (ranks_t2i + 1)).mean()
    
    # 计算image->text的指标
    ranks_i2t = np.zeros(N)
    for i in range(N):
        inds = np.argsort(-sim_i2t[i])
        rank = np.where(inds == i)[0][0]
        ranks_i2t[i] = rank
    
    recalls_i2t = [(ranks_i2t < k).mean() for k in K_list]
    mrr_i2t = (1.0 / (ranks_i2t + 1)).mean()
    
    # 构建结果字典
    results = {
        'text2image': {f'R@{K_list[i]}': recalls_t2i[i] for i in range(len(K_list))},
        'image2text': {f'R@{K_list[i]}': recalls_i2t[i] for i in range(len(K_list))}
    }
    results['text2image']['MRR'] = mrr_t2i
    results['image2text']['MRR'] = mrr_i2t
    
    return results
    
def detailed_metrics_evaluation(txt_embeds, img_embeds, verbose=True):
    """
    计算并输出详细的检索评估指标
    
    参数:
    - txt_embeds: 文本特征矩阵 [N, D]
    - img_embeds: 图像特征矩阵 [N, D]
    - verbose: 是否打印详细信息
    
    返回:
    - 包含详细指标的字典
    """
    # 计算标准指标
    metrics = compute_metrics(txt_embeds, img_embeds, K_list=[1, 5, 10])
    
    # 归一化特征向量
    txt_norm = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)
    img_norm = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    
    # 计算相似度矩阵
    sim_t2i = np.matmul(txt_norm, img_norm.T)  # [N, N]
    
    # 计算平均排名
    N = sim_t2i.shape[0]
    ranks_t2i = np.zeros(N)
    for i in range(N):
        inds = np.argsort(-sim_t2i[i])
        rank = np.where(inds == i)[0][0]
        ranks_t2i[i] = rank
    
    # 计算中位数排名
    median_rank_t2i = np.median(ranks_t2i) + 1
    
    # 添加到指标字典
    metrics['text2image']['median_rank'] = median_rank_t2i
    metrics['text2image']['mean_rank'] = np.mean(ranks_t2i) + 1
    
    # 对称的计算image2text
    sim_i2t = np.matmul(img_norm, txt_norm.T)
    ranks_i2t = np.zeros(N)
    for i in range(N):
        inds = np.argsort(-sim_i2t[i])
        rank = np.where(inds == i)[0][0]
        ranks_i2t[i] = rank
    
    metrics['image2text']['median_rank'] = np.median(ranks_i2t) + 1
    metrics['image2text']['mean_rank'] = np.mean(ranks_i2t) + 1
    
    # 添加综合指标
    metrics['rsum'] = (
        metrics['text2image']['R@1'] + 
        metrics['text2image']['R@5'] + 
        metrics['text2image']['R@10'] +
        metrics['image2text']['R@1'] + 
        metrics['image2text']['R@5'] + 
        metrics['image2text']['R@10']
    )
    
    # 打印结果
    if verbose:
        print("\n===== 详细评估指标 =====")
        print(f"Text-to-Image:")
        print(f"  R@1: {metrics['text2image']['R@1']:.4f}")
        print(f"  R@5: {metrics['text2image']['R@5']:.4f}")
        print(f"  R@10: {metrics['text2image']['R@10']:.4f}")
        print(f"  MRR: {metrics['text2image']['MRR']:.4f}")
        print(f"  Median Rank: {metrics['text2image']['median_rank']}")
        print(f"  Mean Rank: {metrics['text2image']['mean_rank']:.2f}")
        
        print(f"Image-to-Text:")
        print(f"  R@1: {metrics['image2text']['R@1']:.4f}")
        print(f"  R@5: {metrics['image2text']['R@5']:.4f}")
        print(f"  R@10: {metrics['image2text']['R@10']:.4f}")
        print(f"  MRR: {metrics['image2text']['MRR']:.4f}")
        print(f"  Median Rank: {metrics['image2text']['median_rank']}")
        print(f"  Mean Rank: {metrics['image2text']['mean_rank']:.2f}")
        
        print(f"总体:")
        print(f"  rsum: {metrics['rsum']:.4f}")
        
    return metrics