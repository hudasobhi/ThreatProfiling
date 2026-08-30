"""test_shapley.py -- Unit tests for Shapley attribution layer."""
import numpy as np, pytest
from src.shapley import ShapleyAttributionLayer

def dummy_model(x):
    return (x.sum(axis=1, keepdims=True) > 0).astype(float)

def test_fidelity_metric():
    g = np.array([0.8, 0.6, 0.9])
    phi0 = 0.5
    phi_sums = np.array([0.3, 0.1, 0.4])
    fid = ShapleyAttributionLayer.compute_population_fidelity(g, phi0, phi_sums)
    assert 0.0 <= fid <= 1.0, f"Fidelity out of range: {fid}"

def test_top_k_shape():
    bg = np.zeros((100, 10))
    layer = ShapleyAttributionLayer(dummy_model, bg, n_coalition_samples=16)
    x = np.ones((1, 10))
    phi, top_k, fid = layer.explain(x, top_k=3)
    assert len(top_k) == 3
    assert phi.shape == (10,)
