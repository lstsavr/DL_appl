import torch
import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', embed_dim=256, rnn_type=None, pretrained=True, pool_type='cls'):
        """
        文本编码器
        
        参数:
            model_name: 预训练模型名称
            embed_dim: 嵌入维度
            rnn_type: 可选的RNN类型
            pretrained: 是否使用预训练权重
            pool_type: 池化类型 ('cls', 'mean', 'max')
        """
        super().__init__()
        self.model_name = model_name
        self.rnn_type = rnn_type
        self.pool_type = pool_type
        if model_name.startswith('bert'):
            self.bert = BertModel.from_pretrained(model_name) if pretrained else BertModel()
            feat_dim = self.bert.config.hidden_size
            self.rnn = None
        else:
            # 假设输入为embedding后的向量
            self.bert = None
            feat_dim = 768
            if rnn_type == 'gru':
                self.rnn = nn.GRU(feat_dim, feat_dim, batch_first=True, bidirectional=True)
                feat_dim = feat_dim * 2
            elif rnn_type == 'lstm':
                self.rnn = nn.LSTM(feat_dim, feat_dim, batch_first=True, bidirectional=True)
                feat_dim = feat_dim * 2
            else:
                raise ValueError('Unsupported text encoder')
        self.fc = nn.Linear(feat_dim, embed_dim)

    def forward(self, x):
        """
        参数:
            x: 输入字典 (包含 input_ids, attention_mask 等)
            
        返回:
            embed: 文本嵌入向量 [B, embed_dim]
        """
        # 文本特征提取
        if self.bert is not None:
            try:
                outputs = self.bert(input_ids=x['input_ids'], attention_mask=x['attention_mask'], return_dict=True)
                
                # 根据池化类型选择不同的特征提取方式
                if self.pool_type == 'cls':
                    # 使用[CLS]标记的表示
                    feat = outputs.pooler_output  # [B, hidden]
                    
                elif self.pool_type == 'mean':
                    # 使用所有token的平均值
                    last_hidden = outputs.last_hidden_state  # [B, seq_len, hidden]
                    mask = x['attention_mask'].unsqueeze(-1).float()  # [B, seq_len, 1]
                    feat = (last_hidden * mask).sum(1) / mask.sum(1)  # [B, hidden]
                    
                elif self.pool_type == 'max':
                    # 使用所有token的最大值池化
                    last_hidden = outputs.last_hidden_state  # [B, seq_len, hidden]
                    mask = x['attention_mask'].unsqueeze(-1).float()  # [B, seq_len, 1]
                    # 将padding位置设为很小的值，确保不会被选为最大值
                    masked = last_hidden * mask - 1e10 * (1 - mask)
                    feat = torch.max(masked, dim=1)[0]  # [B, hidden]
                    
                elif self.pool_type == 'mean_max':
                    # 结合平均池化和最大池化的优点 (BERT-whitening论文中证明这种方式效果很好)
                    last_hidden = outputs.last_hidden_state  # [B, seq_len, hidden]
                    mask = x['attention_mask'].unsqueeze(-1).float()  # [B, seq_len, 1]
                    
                    # 平均池化
                    mean_pooled = (last_hidden * mask).sum(1) / mask.sum(1)  # [B, hidden]
                    
                    # 最大池化
                    masked = last_hidden * mask - 1e10 * (1 - mask)
                    max_pooled = torch.max(masked, dim=1)[0]  # [B, hidden]
                    
                    # 加权合并平均池化和最大池化，而不是简单拼接
                    # 这样可以保持原有特征维度，避免增加额外参数
                    feat = 0.6 * mean_pooled + 0.4 * max_pooled  # [B, hidden]
                    
                else:
                    # 如果pool_type无效，默认使用CLS
                    print(f"警告: 不支持的池化类型 {self.pool_type}，使用默认的CLS池化")
                    feat = outputs.pooler_output
            except Exception as e:
                # 如果新API失败，回退到旧版本API
                print(f"警告: 使用新版BERT API失败，回退到旧版本: {str(e)}")
                outputs = self.bert(input_ids=x['input_ids'], attention_mask=x['attention_mask'])
                feat = outputs[1]  # 使用pooler输出
        else:
            # x: (B, seq, feat)
            out, _ = self.rnn(x['embeddings'])
            feat = out[:, -1, :]  # 取最后一个时刻
            
        # 特征映射
        embed = self.fc(feat)  # [B, embed_dim]
        
        # 注意：不在这里做L2归一化，统一在matcher中处理
        return embed