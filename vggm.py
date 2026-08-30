"""
vggm.py -- VGGM baseline reimplementation (Zhang et al., 2024).
IEEE Transactions on Information Forensics and Security.
DOI: 10.1109/TIFS.2024.3377548

Variational graph Gaussian mixture model for unsupervised change-point detection.
Selected config (Table 6): LR=0.001, GMM components=4, latent_dim=64.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGM(nn.Module):
    def __init__(self, input_dim, latent_dim=64, n_components=4):
        super().__init__()
        self.n_components = n_components
        self.encoder_mu    = nn.Linear(input_dim, latent_dim)
        self.encoder_logvar= nn.Linear(input_dim, latent_dim)
        self.component_mu  = nn.Parameter(torch.randn(n_components, latent_dim))
        self.component_logvar = nn.Parameter(torch.zeros(n_components, latent_dim))
        self.classifier    = nn.Linear(latent_dim, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu     = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        z      = self.reparameterize(mu, logvar)
        # Anomaly score: negative log-likelihood under closest GMM component
        dists  = torch.stack([
            ((z - self.component_mu[k]) ** 2).sum(-1) for k in range(self.n_components)
        ], dim=1)
        min_dist = dists.min(dim=1).values
        return min_dist  # higher = more anomalous
