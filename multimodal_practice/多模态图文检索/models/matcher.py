import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=None, dropout=0.1, use_ln=True):
        super().__init__()
        layers = []
        last_dim = in_dim
        for i in range(num_layers-1):
            layers.append(nn.Linear(last_dim, hidden_dim or out_dim))
            if use_ln:
                layers.append(nn.LayerNorm(hidden_dim or out_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim or out_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.proj = nn.Sequential(*layers)
    def forward(self, x):
        return self.proj(x)

class DualEncoderModel(nn.Module):
    def __init__(self, img_encoder_cfg, txt_encoder_cfg, embed_dim=256, proj_layers=2, proj_hidden=None, proj_dropout=0.1, proj_ln=True, share_proj=False, sim_type='cosine', l2_normalize=True):

        super().__init__()
        
        # 初始化编码器
        self.image_encoder = ImageEncoder(**img_encoder_cfg)
        self.text_encoder = TextEncoder(**txt_encoder_cfg)
        
        # 编码器输出的特征维度 
        img_feat_dim = img_encoder_cfg.get('embed_dim', 256)
        txt_feat_dim = txt_encoder_cfg.get('embed_dim', 256)
        
        print(f"图像编码器输出维度: {img_feat_dim}")
        print(f"文本编码器输出维度: {txt_feat_dim}")
        
        # 投影头配置
        if share_proj:
            if img_feat_dim != txt_feat_dim:
                raise ValueError(f"Cannot share projection head with different feature dimensions: "
                                f"image={img_feat_dim}, text={txt_feat_dim}")
            self.proj = ProjectionHead(img_feat_dim, embed_dim, proj_layers, proj_hidden, proj_dropout, proj_ln)
            self.img_proj = self.proj
            self.txt_proj = self.proj
        else:
            self.img_proj = ProjectionHead(img_feat_dim, embed_dim, proj_layers, proj_hidden, proj_dropout, proj_ln)
            self.txt_proj = ProjectionHead(txt_feat_dim, embed_dim, proj_layers, proj_hidden, proj_dropout, proj_ln)
            
        assert sim_type in ['cosine', 'dot']
        self.sim_type = sim_type
        self.l2_normalize = l2_normalize

    def forward(self, images, texts):
        
        # 获取图像和文本特征
        img_feats = self.image_encoder(images)   # [B, embed_dim]
        txt_feats = self.text_encoder(texts)     # [B, embed_dim]
        
        # 应用投影头
        img_embeds = self.img_proj(img_feats)    # [B, embed_dim]
        txt_embeds = self.txt_proj(txt_feats)    # [B, embed_dim]
        
        # L2归一化 
        img_embeds = F.normalize(img_embeds, p=2, dim=-1)
        txt_embeds = F.normalize(txt_embeds, p=2, dim=-1)

        return img_embeds, txt_embeds
