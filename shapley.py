"""
shapley.py -- KernelSHAP attribution layer (Section 4.3, Eqs. 25-30).

Configuration (Table 3, selected via sensitivity analysis on validation partition):
    background_samples : 100 benign training windows (fixed)
    coalition_samples  : 512 per flagged user
    stabilization      : fidelity change < 0.003, top-3 recall change < 0.01

Shapley values phi_i satisfy additivity (Eq. 26):
    z_u^t = phi_0 + sum_i phi_i

Fidelity metric (Eq. 30):
    Fid = 1 - (1/N) sum_u | g_theta(u) - (phi_0 + sum_i phi_i) |

Observed fidelity on test partition: 0.943 +/- 0.003 (Table 7).
"""

import numpy as np
import torch
import shap


class ShapleyAttributionLayer:
    """
    Wraps KernelSHAP around the trained scoring function g_theta to
    decompose each risk logit z_u^t into per-feature contributions Phi_u^t.

    The complete player set is the d=128 fused behavioral feature dimensions
    plus one aggregated player representing the neighborhood embedding.
    The neighborhood embedding is treated as a single coalition player
    (approximation source 3, see manuscript Section 4.3).
    """

    def __init__(self, model_fn, background_data: np.ndarray,
                 n_coalition_samples: int = 512):
        """
        Args:
            model_fn            : callable(x: np.ndarray) -> np.ndarray
                                  Maps feature matrix to logit scores.
            background_data     : np.ndarray [100, d]
                                  Fixed benign training windows (Table 3, Panel A).
            n_coalition_samples : int  -- 512 per flagged user (Table 3, Panel B).
        """
        self.n_coalition_samples = n_coalition_samples
        self.explainer = shap.KernelExplainer(model_fn, background_data)

    def explain(self, x: np.ndarray, top_k: int = 3):
        """
        Compute Shapley attribution vector Phi_u^t (Eq. 27) and return
        the top-k contributing feature indices K_u^t (Eq. 28).

        Args:
            x     : np.ndarray [1, d]  -- fused feature vector for one flagged user
            top_k : int                -- number of top contributors to return

        Returns:
            phi_values : np.ndarray [d]  -- per-feature Shapley values
            top_k_idx  : np.ndarray [k]  -- indices of top-k contributors (Eq. 28)
            fidelity   : float           -- single-sample approximation fidelity
        """
        phi_values = self.explainer.shap_values(
            x, nsamples=self.n_coalition_samples, silent=True
        )
        if isinstance(phi_values, list):
            phi_values = phi_values[1]   # binary: take positive class
        phi_values = phi_values.flatten()

        # Top-k ranking (Eq. 28): argsort descending by absolute contribution
        top_k_idx = np.argsort(np.abs(phi_values))[::-1][:top_k]

        # Fidelity for this sample (Eq. 30)
        model_output  = self.explainer.expected_value + phi_values.sum()
        actual_output = self.explainer.model(x).flatten()[0]
        fidelity      = float(1.0 - abs(actual_output - model_output))

        return phi_values, top_k_idx, fidelity

    @staticmethod
    def compute_population_fidelity(g_theta_outputs: np.ndarray,
                                    phi0: float,
                                    phi_sums: np.ndarray) -> float:
        """
        Compute population-level attribution fidelity metric (Eq. 30).

        Args:
            g_theta_outputs : np.ndarray [N]  -- raw logits from g_theta
            phi0            : float           -- expected model output (phi_0)
            phi_sums        : np.ndarray [N]  -- sum of Shapley values per user
        Returns:
            Fid : float  -- mean fidelity across N flagged users
        """
        residuals = np.abs(g_theta_outputs - (phi0 + phi_sums))
        return float(1.0 - residuals.mean())
