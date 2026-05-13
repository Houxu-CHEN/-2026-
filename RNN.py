import torch
import torch.nn as nn

class EEGLSTM(nn.Module):
    def __init__(
        self,
        chans=20,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        bidirectional=True,
        grad_clip=1.0,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.grad_clip = grad_clip

        self.lstm = nn.LSTM(
            input_size=chans, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x 原始 shape: (B, C, T)
        x =x.transpose(1,2) 

        # TODO-3: 正确接收 LSTM 返回的 hidden states
        out, (h_n, c_n)= self.lstm(x)

        if self.bidirectional:
            feat = torch.cat((h_n[-2], h_n[-1]), dim=1) 
        else:
            feat = h_n[-1]  

        logits = self.classifier(feat)
        return logits

    def clip_gradients(self):
        # 梯度爆炸处理（训练侧）：backward 后执行
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0) 


class EEGGRU(nn.Module):
    def __init__(
        self,
        chans=20,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        bidirectional=True,
        grad_clip=1.0,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.grad_clip = grad_clip

        self.gru = nn.GRU(
            input_size=chans,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, h_n = self.gru(x)  #只返回 h_n
        
        if self.bidirectional:
            feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            feat = h_n[-1]
            
        return self.classifier(feat)

    def clip_gradients(self):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)