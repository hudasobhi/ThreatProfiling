"""test_tgn.py -- Unit tests for TGN memory and attention modules."""
import torch, pytest
from src.tgn import MemoryModule, TemporalAttention, temporal_smoothness_loss

def test_memory_reset():
    mem = MemoryModule(n_nodes=100, memory_dim=64, msg_dim=128, feat_dim=64)
    mem.memory.fill_(1.0)
    mem.reset_memory()
    assert (mem.memory == 0).all(), "Memory not zeroed after reset"

def test_memory_update_shape():
    mem = MemoryModule(n_nodes=50, memory_dim=32, msg_dim=64, feat_dim=32)
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([3, 4, 5])
    ts  = torch.tensor([1.0, 2.0, 3.0])
    ft  = torch.zeros(3, 32)
    mem.update(src, dst, ts, ft)
    assert mem.memory.shape == (50, 32)

def test_temporal_attention_output():
    attn = TemporalAttention(memory_dim=32, n_heads=4)
    B, N = 4, 6
    mu  = torch.randn(B, 32)
    mn  = torch.randn(B, N, 32)
    out = attn(mu, mn)
    assert out.shape == (B, 32), f"Expected (4,32), got {out.shape}"

def test_smoothness_loss_zero():
    a = torch.ones(4, 32)
    assert temporal_smoothness_loss(a, a).item() == 0.0
