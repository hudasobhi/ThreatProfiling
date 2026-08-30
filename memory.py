"""
memory.py -- TGN memory update module (Section 4.2, Eqs. 17-18).

Eq. 17: msg_u^t = phi_msg(m_u^{t-}, m_v^{t-}, Delta_t, f_u^t)
Eq. 18: m_u^t   = GRU(m_u^{t-}, msg_u^t)
"""

import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    def __init__(self, n_nodes, memory_dim, msg_dim, feat_dim, time_enc_dim=16):
        super().__init__()
        self.n_nodes    = n_nodes
        self.memory_dim = memory_dim
        self.time_encoder = nn.Linear(1, time_enc_dim)
        phi_input_dim = 2 * memory_dim + time_enc_dim + feat_dim
        self.phi_msg = nn.Sequential(
            nn.Linear(phi_input_dim, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, msg_dim),
        )
        self.gru = nn.GRUCell(msg_dim, memory_dim)
        self.register_buffer("memory",      torch.zeros(n_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(n_nodes, dtype=torch.float32))

    def reset_memory(self):
        self.memory.fill_(0.0)
        self.last_update.fill_(0.0)

    def get_memory(self, node_ids):
        return self.memory[node_ids]

    def update(self, src_ids, dst_ids, timestamps, src_feat):
        m_src   = self.memory[src_ids]
        m_dst   = self.memory[dst_ids]
        delta_t = (timestamps - self.last_update[src_ids]).unsqueeze(-1)
        t_enc   = torch.cos(self.time_encoder(delta_t))
        msg     = self.phi_msg(torch.cat([m_src, m_dst, t_enc, src_feat], dim=-1))
        new_mem = self.gru(msg, m_src)
        self.memory[src_ids]     = new_mem.detach()
        self.last_update[src_ids] = timestamps
