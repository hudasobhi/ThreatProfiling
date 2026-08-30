"""
double_layer.py -- Double Layer Detection baseline (He et al., 2024).
IEEE Transactions on Information Forensics and Security.
DOI: 10.1109/TIFS.2024.3372771

Two-stage detection separating phishing-mediated and behavioral threats.
Selected config (Table 6): LR=0.0001, hidden_dim=128, dropout=0.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleLayerDetection(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        # Layer 1: phishing-mediated (email/web channel)
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        # Layer 2: purely behavioral
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        combined = torch.cat([h1, h2], dim=-1)
        return self.classifier(combined).squeeze(-1)
