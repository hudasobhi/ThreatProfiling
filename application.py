"""
application.py — Application usage sequence encoder (Section 4.1, Eq. 15).

Each application/resource access event is mapped to a learned categorical
embedding combined with sinusoidal positional encoding; the sequence is
aggregated by a single-layer GRU into a fixed-length vector s_t.
"""

import math
import torch
import torch.nn as nn


class ApplicationEncoder(nn.Module):
    """
    Encodes a variable-length sequence of resource-access events into a
    fixed-length vector s_t ∈ R^{gru_hidden_dim}.

    Eq. 15:  s_i = Embed(c_i) + PosEnc(i)
    Sequence aggregated by GRU -> s_t.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 gru_hidden_dim: int = 64, max_seq_len: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, gru_hidden_dim, batch_first=True)

        # Sinusoidal positional encoding (fixed, not learned)
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float()
                             * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, app_seq: torch.Tensor,
                seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            app_seq    : LongTensor [batch, L]   — padded resource index sequences
            seq_lengths: LongTensor [batch]       — actual lengths before padding
        Returns:
            FloatTensor [batch, gru_hidden_dim]   — s_t per session
        """
        B, L = app_seq.shape
        embedded = self.embedding(app_seq)                  # [B, L, embed_dim]
        pos_enc  = self.pe[:L].unsqueeze(0).expand(B, -1, -1)
        embedded = embedded + pos_enc                       # Eq. 15

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        return h_n.squeeze(0)                               # [B, gru_hidden_dim]
