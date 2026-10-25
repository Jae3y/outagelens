"""Guarantees of the margin-gated Bayes + GNN fusion rule."""
import numpy as np

from core.ensemble import TAU, W_MIN, combine, fusion_weight


def _dirichlet(rng, n, sharp=1.0):
    return rng.dirichlet(np.full(n, sharp))


def test_output_is_a_distribution():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 40))
        p = combine(_dirichlet(rng, n, 0.3), _dirichlet(rng, n, 0.3))
        assert p.shape == (n,) and np.all(np.isfinite(p)) and np.all(p >= 0)
        assert abs(p.sum() - 1.0) < 1e-9


def test_decisive_bayes_passes_through():
    pb = np.array([0.80, 0.15, 0.05])
    pg = np.array([0.05, 0.05, 0.90])  # GNN disagrees strongly but Bayes margin 0.65 >= TAU
    assert fusion_weight(pb) == 1.0
    assert np.allclose(combine(pb, pg), pb, atol=1e-9)


def test_tie_lets_gnn_arbitrate():
    pb = np.array([0.45, 0.45, 0.10])
    pg = np.array([0.10, 0.80, 0.10])
    assert abs(fusion_weight(pb) - W_MIN) < 1e-12
    assert int(np.argmax(combine(pb, pg))) == 1


def test_weight_is_continuous_and_monotone_in_margin():
    margins = np.linspace(0.0, 0.9, 181)
    eps = 1e-3  # probe vectors whose top-1 minus top-2 gap is exactly m
    weights = [fusion_weight(np.array([(1 + m) / 2 - eps, (1 - m) / 2 - eps, 2 * eps])) for m in margins]
    assert all(W_MIN - 1e-12 <= w <= 1.0 for w in weights)
    assert all(b >= a - 1e-12 for a, b in zip(weights, weights[1:]))
    assert max(abs(b - a) for a, b in zip(weights, weights[1:])) < (1 - W_MIN) / TAU * 0.006
