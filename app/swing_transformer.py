"""
Swing Transformer Model — Single Binary Output

Processes raw M5 OHLCV candles and predicts: is this bar a swing point?
One output only: swing_prob (0.0 = no swing, 1.0 = swing point)

Direction (high vs low) is determined separately at inference time
by comparing the bar's high/low structure to neighbors.

Architecture:
  Input:  [batch, seq_len, 6] — normalized OHLCV + ATR
  Output: [batch, 1] — swing_prob (sigmoid)
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SwingTransformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, seq_len: int = 50):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.input_proj = nn.Linear(6, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.transformer(h)
        h = self.norm(h)

        last = h[:, -1, :]
        swing_prob = self.head(last).squeeze(-1)
        return swing_prob

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            prob = self.forward(x.unsqueeze(0) if x.dim() == 2 else x)
        return prob.item() if prob.dim() == 0 else prob[0].item()

    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
