import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, model_name='resnet50', embed_dim=256, pretrained=True):
        super().__init__()
        self.model_name = model_name
        
        # 处理模型更新的兼容性问题
        weights = 'IMAGENET1K_V1' if pretrained else None
        
        if model_name == 'resnet50':
            backbone = models.resnet50(weights=weights)
            modules = list(backbone.children())[:-1]  # 去掉fc层
            self.backbone = nn.Sequential(*modules)
            feat_dim = backbone.fc.in_features
        elif model_name == 'resnet101':  # 添加ResNet101支持
            backbone = models.resnet101(weights=weights)
            modules = list(backbone.children())[:-1]  # 去掉fc层
            self.backbone = nn.Sequential(*modules)
            feat_dim = backbone.fc.in_features
        elif model_name == 'resnet152':  # 添加ResNet152支持
            backbone = models.resnet152(weights=weights)
            modules = list(backbone.children())[:-1]  # 去掉fc层
            self.backbone = nn.Sequential(*modules)
            feat_dim = backbone.fc.in_features
        elif model_name == 'vit_b_16':
            backbone = models.vit_b_16(weights=weights)
            self.backbone = backbone
            feat_dim = backbone.heads.head.in_features
        else:
            raise ValueError(f'Unsupported model: {model_name}')
        self.fc = nn.Linear(feat_dim, embed_dim)

    def forward(self, x):
        """
        参数:
            x: 输入图像张量 [B, 3, H, W]
            
        返回:
            embed: 图像特征向量 [B, embed_dim]
        """
        # 图像特征提取
        if self.model_name.startswith('resnet'):  # 处理所有ResNet模型
            feat = self.backbone(x).squeeze(-1).squeeze(-1)  # [B, feat_dim]
        elif self.model_name.startswith('vit'):   # 处理所有ViT模型
            feat = self.backbone(x)  # [B, feat_dim]
        else:
            raise ValueError(f"Unsupported model for forward pass: {self.model_name}")
        
        # 特征映射
        embed = self.fc(feat)  # [B, embed_dim]
        
        # 注意：不在这里做L2归一化，统一在matcher中处理
        return embed