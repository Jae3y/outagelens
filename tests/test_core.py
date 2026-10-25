"""Sanity tests: physics, gradient correctness, and that OutageLens beats the baseline."""
import numpy as np
import pytest

from core.bayesian_locator import BayesianLocator
from core.evaluator import M_BASELINE, M_BAYES, M_ENSEMBLE, headline, run_benchmark
from core.feeder_topology import ieee13, ieee34
from core.gnn_locator import TreeGNN, build_training_set, train_gnn
from core.opendss_engine import DistFlowEngine


@pytest.mark.parametrize("fn", [ieee13, ieee34])
def test_powerflow_physics(fn):
    f = fn()
    eng = DistFlowEngine(f)
    base = eng.solve()
    assert base.converged
    assert base.head_kw > f.p_kw.sum()  # losses are positive
    assert 0.9 < base.v_pu.min() and base.v_pu.max() < 1.1
    for e in range(f.n_edge):
        r = eng.solve(open_edge=e)
        assert np.all(r.v_pu[f.downstream[e]] == 0.0)
        # Shedding inductive load raises voltages upstream; shedding a capacitor bank (848) can lower them.
        if f.q_kvar[f.downstream[e]].sum() > 0 and np.all(f.q_kvar[f.downstream[e]] >= 0):
            assert np.all(r.v_pu[~f.downstream[e]] >= base.v_pu[~f.downstream[e]] - 1e-9)
        assert r.head_kw < base.head_kw + 1e-6


def test_gnn_gradients_match_finite_differences():
    f = ieee13()
    eng = DistFlowEngine(f)
    X, y = build_training_set(f, eng, n=6, seed=1)
    model = TreeGNN(f, hidden=6, seed=3)
    _, grads, _ = model.loss_and_grads(X, y)
    rng = np.random.default_rng(0)
    for name in ["W0", "W1", "a", "bv", "b0"]:
        p = model.params[name]
        for _ in range(4):
            idx = tuple(rng.integers(s) for s in p.shape)
            old = p[idx]
            p[idx] = old + 1e-5
            lp, _, _ = model.loss_and_grads(X, y)
            p[idx] = old - 1e-5
            lm, _, _ = model.loss_and_grads(X, y)
            p[idx] = old
            num = (lp - lm) / 2e-5
            assert abs(num - grads[name][idx]) < 1e-5 + 1e-3 * abs(num), name


@pytest.mark.parametrize("fn", [ieee13, ieee34])
def test_outagelens_beats_baseline(fn):
    f = fn()
    eng = DistFlowEngine(f)
    bayes = BayesianLocator(f, eng)
    gnn = train_gnn(f, eng, n_scenarios=300, epochs=30)
    df = run_benchmark(f, eng, bayes, gnn, rates=(0.4, 0.85), n_per_rate=30)
    h = headline(df).set_index("method")
    assert h.loc[M_BAYES, "top1"] > h.loc[M_BASELINE, "top1"]
    assert h.loc[M_ENSEMBLE, "top3"] >= h.loc[M_BASELINE, "top3"]
    assert h.loc[M_BAYES, "patrol_km"] < h.loc[M_BASELINE, "patrol_km"]
