"""
graphch.py -- GraphCH baseline reimplementation (Roy and Chen, 2024).
IEEE Transactions on Dependable and Secure Computing.
DOI: 10.1109/TDSC.2024.3353929

Cyber-human GNN combining host activity logs with psychological survey features.
Selected config (Table 6): LR=0.0003, GNN layers=2, hidden_dim=128.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphCH(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout    = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        # x: [N, input_dim], adj: [N, N] normalized adjacency
        h = x
        for layer in self.layers:
            h = torch.mm(adj, h)
            h = F.relu(self.norm(layer(h)))
            h = self.dropout(h)
        return self.classifier(h).squeeze(-1)
