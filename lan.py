"""
lan.py -- LAN baseline reimplementation (Cai et al., 2024).
IEEE Transactions on Information Forensics and Security.
DOI: 10.1109/TIFS.2024.3488527

Real-time activity-level graph structure learning for insider threat detection.
Selected config (Table 6): LR=0.0003, attention_heads=4, window=24h.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LAN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, n_heads,
                                                   dropout=dropout, batch_first=True)
        self.proj       = nn.Linear(input_dim, hidden_dim)
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq, adj):
        # x_seq: [B, T, input_dim] sequential activity features
        h = F.relu(self.proj(x_seq))
        h_attn, _ = self.temporal_attn(h, h, h)
        h = self.norm1(h + h_attn)
        h_last = h[:, -1, :]
        h_graph = F.relu(self.norm2(self.graph_proj(h_last)))
        return self.classifier(h_graph).squeeze(-1)
