"""
shadowdetectai.py -- Main ShadowDetectAI model class.

Implements Algorithm 2: Temporal Graph Detection with Shapley Attribution.
Combines the multi-channel encoder, TGN, classification head, and
multi-channel alert fusion into a single end-to-end trainable module.

Training loss (Eq. 36):
    L = L_bce + beta * L_smooth + gamma * ||theta||_2^2
"""

import torch
import torch.nn as nn

from src.encoder         import MultiChannelEncoder
from src.tgn             import MemoryModule, TemporalAttention, temporal_smoothness_loss
from src.calibration     import MultiChannelAlertFusion


class ShadowDetectAI(nn.Module):
    """
    End-to-end ShadowDetectAI insider threat detection model.

    Architecture (Figure 3 in manuscript):
        1. MultiChannelEncoder  -- fuses keystroke/mouse/app proxies -> f_u^t
        2. MemoryModule         -- maintains and updates per-node memory states
        3. TemporalAttention    -- aggregates neighborhood memory -> h_u^{multi-t}
        4. Classification head  -- logit z_u^t = W_o * h_u^{multi-t} + b_o (Eq. 22)
        5. MultiChannelFusion   -- role-conditioned alert score R_u^t (Eq. 35)

    Shapley attribution is applied post-inference via ShapleyAttributionLayer
    (see src/shapley.py) when z_u^t exceeds the calibrated threshold.
    """

    def __init__(self, n_nodes: int, vocab_size: int, cfg: dict):
        super().__init__()
        d      = cfg["model"]["embedding_dim"]
        n_heads = cfg["model"]["attention_heads"]
        msg_dim = cfg["model"].get("mlp_hidden_dim", 256)
        dropout = cfg["model"].get("dropout", 0.1)

        # Module 1: multi-channel behavioral encoder (Section 4.1)
        self.encoder = MultiChannelEncoder(
            vocab_size    = vocab_size,
            embedding_dim = d,
            ks_proj_dim   = 32,
            mouse_proj_dim= 32,
            app_embed_dim = cfg["encoder"]["application"]["embed_dim"],
            app_gru_dim   = cfg["encoder"]["application"]["gru_hidden_dim"],
        )

        # Module 2: TGN memory (Section 4.2, Eqs. 17-18)
        self.memory = MemoryModule(
            n_nodes    = n_nodes,
            memory_dim = d,
            msg_dim    = msg_dim,
            feat_dim   = d,
        )

        # Module 3: temporal attention (Section 4.2, Eqs. 19-21)
        self.attention = TemporalAttention(
            memory_dim = d,
            n_heads    = n_heads,
            dropout    = dropout,
        )

        # Classification head (Eq. 22)
        self.classifier = nn.Linear(d, 1)

        # Multi-channel alert fusion (Eqs. 34-35)
        self.fusion = MultiChannelAlertFusion(n_channels=3)

        self.beta  = cfg["training"].get("beta",  0.01)
        self.gamma = cfg["training"].get("gamma", 0.0001)

    def reset_memory(self):
        """Cold-start memory reset at partition boundary (Section 6.1)."""
        self.memory.reset_memory()

    def forward(self, batch: dict, neighbor_memories: torch.Tensor,
                prev_embeddings: torch.Tensor = None):
        """
        Forward pass implementing Algorithm 2 (without the Shapley branch,
        which is applied externally via ShapleyAttributionLayer).

        Args:
            batch             : dict with keys from data/dataset.py collate_fn
            neighbor_memories : FloatTensor [B, max_neighbors, memory_dim]
            prev_embeddings   : FloatTensor [B, d] or None -- for L_smooth

        Returns:
            logits     : FloatTensor [B]   -- z_u^t (Eq. 22)
            probs      : FloatTensor [B]   -- p_u^t = sigma(z_u^t) (Eq. 23)
            embeddings : FloatTensor [B, d]-- h_u^{multi-t} for attribution
            loss_smooth: scalar            -- L_smooth (Eq. 24) or 0.0
        """
        # Encode behavioral channels (Eq. 16)
        f_t = self.encoder(
            kt          = batch["keystroke_features"],
            mt          = batch["mouse_features"],
            app_seq     = batch["app_sequence"],
            seq_lengths = batch["seq_lengths"],
        )  # [B, d]

        # Update memory for interacting nodes (Eqs. 17-18)
        user_ids   = batch["user_idx"]
        timestamps = batch["window_idx"].float()
        # Use placeholder dst as self for isolated window inference
        self.memory.update(user_ids, user_ids, timestamps, f_t.detach())

        # Retrieve updated memory and compute temporal embedding (Eqs. 19-21)
        mem_u       = self.memory.get_memory(user_ids)          # [B, d]
        h_multi     = self.attention(mem_u, neighbor_memories)  # [B, d]

        # Classification logit and probability (Eqs. 22-23)
        logits = self.classifier(h_multi).squeeze(-1)           # [B]
        probs  = torch.sigmoid(logits)                          # [B]

        # Temporal smoothness loss (Eq. 24)
        loss_smooth = torch.tensor(0.0, device=logits.device)
        if prev_embeddings is not None:
            loss_smooth = temporal_smoothness_loss(h_multi, prev_embeddings)

        return logits, probs, h_multi, loss_smooth

    def compute_loss(self, logits, labels, loss_smooth):
        """Composite training loss (Eq. 36)."""
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float()
        )
        l2 = sum(p.pow(2).sum() for p in self.parameters())
        return bce + self.beta * loss_smooth + self.gamma * l2
