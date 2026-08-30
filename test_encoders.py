"""test_encoders.py -- Unit tests for multi-channel encoder."""
import torch, pytest
from src.encoder import MultiChannelEncoder

def test_encoder_output_shape():
    enc = MultiChannelEncoder(vocab_size=500, embedding_dim=128)
    B = 4
    kt   = torch.zeros(B, 8)
    mt   = torch.zeros(B, 6)
    seq  = torch.ones(B, 10, dtype=torch.long)
    lens = torch.full((B,), 10)
    out  = enc(kt, mt, seq, lens)
    assert out.shape == (B, 128), f"Expected (4,128), got {out.shape}"

def test_encoder_zero_input():
    enc = MultiChannelEncoder(vocab_size=100, embedding_dim=64)
    B = 2
    kt   = torch.zeros(B, 8)
    mt   = torch.zeros(B, 6)
    seq  = torch.zeros(B, 5, dtype=torch.long)
    lens = torch.full((B,), 5)
    out  = enc(kt, mt, seq, lens)
    assert not torch.isnan(out).any(), "NaN in encoder output"

def test_encoder_variable_seq_len():
    enc = MultiChannelEncoder(vocab_size=200, embedding_dim=64)
    kt   = torch.zeros(3, 8)
    mt   = torch.zeros(3, 6)
    seq  = torch.ones(3, 15, dtype=torch.long)
    lens = torch.tensor([15, 10, 5])
    out  = enc(kt, mt, seq, lens)
    assert out.shape == (3, 64)
