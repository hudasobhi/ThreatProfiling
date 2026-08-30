"""
attention.py -- Temporal attention and multi-head embedding (Section 4.2, Eqs. 19-21).

Eq. 19: h_u^t = sum_{v in N(u,t)} alpha_{uv} W_v m_v^t
Eq. 20: alpha_{uv} computed via LeakyReLU attention
Eq. 21: h_u^{multi-t} = concat of K heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, memory_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads    = n_heads
        self.memory_dim = memory_dim
        self.head_dim   = memory_dim // n_heads
        assert memory_dim % n_heads == 0

        self.W  = nn.Linear(memory_dim, memory_dim, bias=False)
        self.a  = nn.Parameter(torch.FloatTensor(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mem_u, mem_neighbors):
        """
        Args:
            mem_u         : FloatTensor [B, memory_dim]           -- target node memory
            mem_neighbors : FloatTensor [B, max_neighbors, memory_dim]
        Returns:
            h_multi : FloatTensor [B, memory_dim]  -- concatenated K-head embedding (Eq. 21)
        """
        B, N, D = mem_neighbors.shape
        H, Dh   = self.n_heads, self.head_dim

        Wm_u = self.W(mem_u).view(B, H, Dh)                    # [B, H, Dh]
        Wm_n = self.W(mem_neighbors).view(B, N, H, Dh)          # [B, N, H, Dh]

        Wm_u_exp = Wm_u.unsqueeze(1).expand(B, N, H, Dh)
        cat       = torch.cat([Wm_u_exp, Wm_n], dim=-1)         # [B, N, H, 2*Dh]
        e         = self.leaky((cat * self.a).sum(-1))           # [B, N, H]  Eq. 20
        alpha     = F.softmax(e, dim=1)                          # [B, N, H]
        alpha     = self.dropout(alpha)

        # Eq. 19 per head
        out_per_head = (alpha.unsqueeze(-1) * Wm_n).sum(dim=1)  # [B, H, Dh]
        return out_per_head.view(B, H * Dh)                     # Eq. 21: [B, memory_dim]
