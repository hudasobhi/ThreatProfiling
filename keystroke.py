"""
keystroke.py — Keystroke dynamics proxy encoder (Section 4.1, Eqs. 11-13).

In the absence of raw hardware keystroke telemetry, dwell time (d_i) and
flight time (l_i) are approximated from inter-event timestamps of logon-session
activity records in the CERT r6.2 logs.

Feature vector k_t ∈ R^8 = [μ_d, σ_d, P50_d, P90_d, μ_l, σ_l, P50_l, P90_l]
"""

import torch
import torch.nn as nn


class KeystrokeEncoder(nn.Module):
    """
    Projects the 8-dimensional keystroke proxy feature vector into a
    learned embedding of dimension proj_dim through a two-layer MLP.
    """

    def __init__(self, input_dim: int = 8, proj_dim: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, proj_dim),
            nn.ReLU(),
        )

    def forward(self, kt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kt : FloatTensor [batch, 8]  — proxy keystroke statistics
        Returns:
            FloatTensor [batch, proj_dim]
        """
        return self.proj(kt)
