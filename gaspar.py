"""
gaspar.py -- Gaspar et al. baseline reimplementation (2024).
IEEE Access. DOI: 10.1109/ACCESS.2024.3368377

LIME and SHAP applied to a multi-layer perceptron anomaly detector.
Flat feature vector, no relational or temporal structure.
Selected config (Table 6): LR=0.0003, MLP layers=2, hidden_units=64.
"""

import torch
import torch.nn as nn


class GasparMLP(nn.Module):
    def __init__(self, input_dim, n_layers=2, hidden_units=64, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_units), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units),
                       nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
