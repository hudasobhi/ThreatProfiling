"""test_baselines.py -- Unit tests for all six baseline models."""
import torch, pytest
from src.baselines.graphch     import GraphCH
from src.baselines.lan         import LAN
from src.baselines.double_layer import DoubleLayerDetection
from src.baselines.vggm        import VGGM
from src.baselines.exai        import EXAI
from src.baselines.gaspar      import GasparMLP

B, D = 8, 64

def test_graphch():
    model = GraphCH(input_dim=D)
    x   = torch.randn(B, D)
    adj = torch.eye(B) / B
    out = model(x, adj)
    assert out.shape == (B,)

def test_lan():
    model = LAN(input_dim=D)
    x   = torch.randn(B, 5, D)
    adj = torch.eye(B)
    out = model(x, adj)
    assert out.shape == (B,)

def test_double_layer():
    model = DoubleLayerDetection(input_dim=D)
    x = torch.randn(B, D)
    assert model(x).shape == (B,)

def test_vggm():
    model = VGGM(input_dim=D)
    x = torch.randn(B, D)
    assert model(x).shape == (B,)

def test_exai():
    model = EXAI(input_dim=D)
    x = torch.randn(B, D)
    assert model(x).shape == (B,)

def test_gaspar():
    model = GasparMLP(input_dim=D)
    x = torch.randn(B, D)
    assert model(x).shape == (B,)
