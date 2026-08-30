"""
exai.py -- EXAI baseline reimplementation (Arreche et al., 2024).
IEEE Access. DOI: 10.1109/ACCESS.2024.3365140

Black-box explainable AI evaluation framework applied as a non-relational baseline.
Selected config (Table 6): LR=0.001, MLP layers=3, hidden_units=128.
"""

import torch
import torch.nn as nn


class EXAI(nn.Module):
    def __init__(self, input_dim, n_layers=3, hidden_units=128, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_units), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units),
                       nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def get_feature_vector(self, x):
        """Return the penultimate layer representation for SHAP attribution."""
        h = x
        for layer in list(self.net.children())[:-1]:
            h = layer(h)
        return h
