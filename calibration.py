"""
calibration.py -- Platt scaling, PSI drift detection, alert fusion (Section 5, Eqs. 31-35).

Eq. 31: q_u^t = sigmoid(A * z_u^t + B)           -- Platt calibration
Eq. 32: PSI   = sum_i (q_i - p_i) ln(q_i / p_i)  -- population stability index
Eq. 33: tau_t = tau_0 + lambda * PSI_t            -- adaptive threshold
Eq. 34: w_c   = softmax(v_c)                      -- role-conditioned channel weights
Eq. 35: R_u^t = sum_c w_c * r_{u,c}^t            -- fused risk score
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit
from scipy.optimize import minimize


class PlattCalibration:
    """Platt scaling fitted on a labeled calibration set (Eq. 31)."""

    def __init__(self):
        self.A = 1.0
        self.B = 0.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit A and B by maximum likelihood on calibration data."""
        def nll(params):
            A, B = params
            probs = expit(A * logits + B)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

        result = minimize(nll, x0=[self.A, self.B], method="L-BFGS-B")
        self.A, self.B = result.x

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply Eq. 31 to raw logits."""
        return expit(self.A * logits + self.B)


class DriftDetector:
    """PSI-based behavioral drift detection and adaptive threshold (Eqs. 32-33)."""

    def __init__(self, n_bins: int = 10, tau_0: float = 0.5, drift_lambda: float = 0.05):
        self.n_bins       = n_bins
        self.tau_0        = tau_0
        self.drift_lambda = drift_lambda
        self.ref_hist     = None

    def fit_reference(self, ref_scores: np.ndarray) -> None:
        """Compute reference score histogram from training-time calibrated scores."""
        hist, self.bin_edges = np.histogram(ref_scores, bins=self.n_bins,
                                             range=(0.0, 1.0), density=True)
        self.ref_hist = hist / hist.sum() + 1e-8

    def compute_psi(self, current_scores: np.ndarray) -> float:
        """Compute PSI between current deployment distribution and reference (Eq. 32)."""
        cur_hist, _ = np.histogram(current_scores, bins=self.bin_edges, density=True)
        cur_hist    = cur_hist / (cur_hist.sum() + 1e-8) + 1e-8
        psi = float(np.sum((cur_hist - self.ref_hist) * np.log(cur_hist / self.ref_hist)))
        return psi

    def adaptive_threshold(self, current_scores: np.ndarray) -> float:
        """Return tau_t adapted for observed drift (Eq. 33)."""
        psi = self.compute_psi(current_scores)
        return self.tau_0 + self.drift_lambda * psi


class MultiChannelAlertFusion(nn.Module):
    """
    Role-conditioned channel weight fusion (Eqs. 34-35).
    Trainable logits v_c are learned jointly with the rest of the network.
    """

    def __init__(self, n_channels: int = 3):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(n_channels))

    def forward(self, channel_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel_scores : FloatTensor [B, n_channels]
        Returns:
            R_u^t          : FloatTensor [B]  -- fused risk score (Eq. 35)
        """
        w = torch.softmax(self.v, dim=0)              # Eq. 34
        return (channel_scores * w.unsqueeze(0)).sum(dim=-1)   # Eq. 35
