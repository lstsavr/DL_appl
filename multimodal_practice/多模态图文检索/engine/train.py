import torch
from tqdm import tqdm
from loss.contrastive import contrastive_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import time

from torch.cuda.amp import autocast, GradScaler

def set_optimizer_and_scheduler(model, lr=1e-4, weight_decay=0.01, warmup_steps=500, total_steps=10000):
    optimizer = AdamW([
        {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay}
    ])
    
    # 余弦退火+warmup调度器
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # 使用更平滑的余弦退火
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1. + torch.cos(torch.tensor(progress * 3.1415926535))))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

def freeze_encoder_layers(model, freeze_img_layers=0, freeze_txt_layers=0):
    # 冻结部分预训练层，避免过拟合
    if freeze_img_layers > 0 and hasattr(model.image_encoder, 'backbone'):
        if hasattr(model.image_encoder.backbone, 'children'):
            children = list(model.image_encoder.backbone.children())
            for layer in children[:freeze_img_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        # 特殊处理BatchNorm层，固定统计量
        for m in model.image_encoder.backbone.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.eval() 
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    
    # 处理文本编码器
    if freeze_txt_layers > 0 and hasattr(model.text_encoder, 'bert'):
        # 冻结指定层数的Transformer层
        for layer in model.text_encoder.bert.encoder.layer[:freeze_txt_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # 冻结嵌入层和第一层，它们通常只需要少量微调
        if freeze_txt_layers > 2:
            for param in model.text_encoder.bert.embeddings.parameters():
                param.requires_grad = False

def train(model, dataloader, optimizer, device, scheduler=None, print_freq=100, grad_clip=1.0, use_amp=True, epoch=None):

    model.train()
    total_loss = 0
    total_acc_t2i = 0
    total_acc_i2t = 0
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    
    end = time.time()
    
    for step, batch in enumerate(tqdm(dataloader, desc='Train')):
        # 测量数据加载时间
        data_time.update(time.time() - end)
        
        # 获取批次数据
        images, texts, _, _ = batch
        images = images.to(device, non_blocking=True)  # 添加non_blocking=True提高并行效率
        texts = {k: v.to(device, non_blocking=True) for k, v in texts.items()}
        
        # 清除之前的梯度
        optimizer.zero_grad(set_to_none=True)  # set_to_none=True比set_to_zero更高效
        
        if use_amp and device.type == 'cuda':
            # 混合精度训练
            with autocast():
                # 前向传播
                img_embeds, txt_embeds = model(images, texts)
                loss, logs = contrastive_loss(img_embeds, txt_embeds, epoch=epoch)
                
            # 使用缩放器进行反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪 (在scale后应用)
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准FP32训练
            img_embeds, txt_embeds = model(images, texts)
            loss, logs = contrastive_loss(img_embeds, txt_embeds, epoch=epoch)
            loss.backward()
            
            # 梯度裁剪
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            # 更新参数
            optimizer.step()
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        losses.update(loss.item(), images.size(0))
        total_acc_t2i += logs['acc_t2i']
        total_acc_i2t += logs['acc_i2t']
        
        # 测量时间
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (step + 1) % print_freq == 0:
            avg_acc_t2i = total_acc_t2i / (step + 1)
            avg_acc_i2t = total_acc_i2t / (step + 1)
            lr = optimizer.param_groups[0]['lr']
            
            print(f"Step [{step+1}/{len(dataloader)}] "
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                  f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                  f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                  f"T2I R@1 {avg_acc_t2i:.4f} I2T R@1 {avg_acc_i2t:.4f} "
                  f"LR {lr:.6f}")
    
    return {
        'loss': losses.avg,
        'acc_t2i': total_acc_t2i / len(dataloader),
        'acc_i2t': total_acc_i2t / len(dataloader)
    }


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'

        return fmtstr.format(**self.__dict__)
