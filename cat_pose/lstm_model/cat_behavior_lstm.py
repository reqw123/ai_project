"""
猫咪行为LSTM模型定义
4类分类：舔拭、搔抓、甩头、一般
"""
import torch
import torch.nn as nn

class CatBehaviorLSTM(nn.Module):
    """
    猫咪行为分类LSTM模型
    输入: (batch, seq_len, 51) - 17关键点×3(x,y,conf)
    输出: (batch, 4) - 4类行为概率
    """
    def __init__(self, input_size=51, hidden_size=128, num_layers=2, 
                 num_classes=4, dropout=0.3):
        super(CatBehaviorLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 特征投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, 51)
        返回: (logits, attention_weights)
        """
        batch_size = x.size(0)
        
        # 投影
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # 注意力加权
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 加权求和
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # 分类
        logits = self.classifier(context)
        
        return logits, attn_weights.squeeze(-1)


def count_parameters(model):
    """统计参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试
if __name__ == "__main__":
    model = CatBehaviorLSTM(
        input_size=51,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
        dropout=0.3
    )
    
    # 测试输入
    x = torch.randn(8, 30, 51)  # batch=8, seq=30, features=51
    logits, attn = model(x)
    
    print(f"模型参数量: {count_parameters(model):,}")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"注意力权重形状: {attn.shape}")
