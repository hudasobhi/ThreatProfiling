"""
regularization.py -- Temporal smoothness regularizer (Section 4.2, Eq. 24).

L_smooth = sum_u sum_t || h_u^t - h_u^{t-1} ||_2^2

Discourages abrupt embedding changes between adjacent windows in the
absence of genuine behavioral change, contributing to drift robustness
shown in Figure 9.
"""

import torch


def temporal_smoothness_loss(embeddings_t, embeddings_t_prev):
    """
    Args:
        embeddings_t      : FloatTensor [B, d] -- embeddings at window t
        embeddings_t_prev : FloatTensor [B, d] -- embeddings at window t-1
    Returns:
        scalar smoothness loss (Eq. 24)
    """
    return torch.sum((embeddings_t - embeddings_t_prev) ** 2)
