"""
mouse.py — Mouse movement entropy proxy encoder (Section 4.1, Eq. 14).

Shannon entropy H_v is computed over the device-event frequency distribution
as a proxy for pointer motion entropy; raw pointer-coordinate streams are not
available in CERT r6.2.

Feature vector m_t ∈ R^6 = [H_v, click_mean, click_std, iei_mean, iei_std, iei_p90]
"""

import torch
import torch.nn as nn


class MouseEncoder(nn.Module):
    """
    Projects the 6-dimensional mouse proxy feature vector into a
    learned embedding of dimension proj_dim through a two-layer MLP.
    """

    def __init__(self, input_dim: int = 6, proj_dim: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, proj_dim),
            nn.ReLU(),
        )

    def forward(self, mt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mt : FloatTensor [batch, 6]  — proxy mouse interaction statistics
        Returns:
            FloatTensor [batch, proj_dim]
        """
        return self.proj(mt)
