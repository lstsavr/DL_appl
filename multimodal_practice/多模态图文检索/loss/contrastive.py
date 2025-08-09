import torch
import torch.nn.functional as F

def contrastive_loss(image_embeds, text_embeds, temperature=0.07, margin=0.2, hard_negative=True, 
                    hard_negative_weight=0.5, label_smoothing=0.1, epoch=None):

    batch_size = image_embeds.size(0)
    device = image_embeds.device
    
    # 计算相似度矩阵
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) / temperature
    logits_per_image = torch.matmul(image_embeds, text_embeds.t()) / temperature
    
    # 标签 (对角线元素为正样本)
    labels = torch.arange(batch_size, device=device)
    
    # 正负样本掩码
    pos_mask = torch.eye(batch_size, device=device).bool()
    neg_mask = ~pos_mask
    
    # 应用margin来增大正负样本的差距
    if margin is not None:
        logits_per_text = logits_per_text - margin * neg_mask.float()
        logits_per_image = logits_per_image - margin * neg_mask.float()
    
    # 提取正样本对的相似度
    pos_t2i = logits_per_text.masked_select(pos_mask).reshape(batch_size, 1)
    pos_i2t = logits_per_image.masked_select(pos_mask).reshape(batch_size, 1)
    
    # 获取负样本对的相似度
    neg_t2i = logits_per_text.masked_select(neg_mask).reshape(batch_size, -1)
    neg_i2t = logits_per_image.masked_select(neg_mask).reshape(batch_size, -1)
    

    if epoch is not None and hard_negative_weight > 0:
        adaptive_weight = min(0.7, hard_negative_weight + (epoch * 0.01))
    else:
        adaptive_weight = hard_negative_weight
    
    if hard_negative:
        # 使用适量的困难负样本
        k = max(1, int(neg_t2i.size(1) * 0.10))  # 使用前10%作为困难负样本
        
        # 获取top-k困难负样本
        hard_neg_t2i, _ = neg_t2i.topk(k, dim=1)
        hard_neg_i2t, _ = neg_i2t.topk(k, dim=1)
        
        # 特别对待最困难的负样本（前2个）
        hardest_neg_t2i = hard_neg_t2i[:, :2].mean(dim=1)
        hardest_neg_i2t = hard_neg_i2t[:, :2].mean(dim=1)
        
        # 计算正样本对相对于所有样本的InfoNCE损失
        loss_t2i_all = F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)
        loss_i2t_all = F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        
        # 计算正样本与困难负样本的对比损失 (triple loss形式)
        hard_loss_t2i = (hard_neg_t2i.mean(dim=1) - pos_t2i.squeeze(1) + 0.08).clamp(min=0).mean()
        hard_loss_i2t = (hard_neg_i2t.mean(dim=1) - pos_i2t.squeeze(1) + 0.08).clamp(min=0).mean()
        
        # 对最困难的负样本使用适中的惩罚
        hardest_loss_t2i = (hardest_neg_t2i - pos_t2i.squeeze(1) + 0.12).clamp(min=0).mean()
        hardest_loss_i2t = (hardest_neg_i2t - pos_i2t.squeeze(1) + 0.12).clamp(min=0).mean()
        
        # 结合多种损失，使用更均衡的权重
        loss_t2i = loss_t2i_all + adaptive_weight * hard_loss_t2i + 0.3 * adaptive_weight * hardest_loss_t2i
        loss_i2t = loss_i2t_all + adaptive_weight * hard_loss_i2t + 0.3 * adaptive_weight * hardest_loss_i2t
    else:
        # 标准InfoNCE损失
        loss_t2i = F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)
        loss_i2t = F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
    
    # 双向损失平均
    loss = (loss_t2i + loss_i2t) / 2
    
    # 计算Recall@1
    pred_t2i = logits_per_text.argmax(dim=1)
    acc_t2i = (pred_t2i == labels).float().mean().item()
    pred_i2t = logits_per_image.argmax(dim=1)
    acc_i2t = (pred_i2t == labels).float().mean().item()
    
    logs = {
        'loss': loss.item(),
        'loss_t2i': loss_t2i.item(),
        'loss_i2t': loss_i2t.item(),
        'acc_t2i': acc_t2i,
        'acc_i2t': acc_i2t,
        'temperature': temperature,
        'margin': margin,
        'hard_negative': hard_negative,
        'hard_neg_weight': adaptive_weight if hard_negative else 0
    }

    return loss, logs
