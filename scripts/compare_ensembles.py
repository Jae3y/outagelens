"""Score fusion strategies side by side on identical simulated outages.

    python scripts/compare_ensembles.py

Each strategy sees the same Bayesian and GNN outputs, so differences come only from the fusion rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bayesian_locator import BayesianLocator  # noqa: E402
from core.ensemble import combine, log_linear_pool  # noqa: E402
from core.evaluator import DEFAULT_RATES, patrol_km  # noqa: E402
from core.feeder_topology import ieee13, ieee34  # noqa: E402
from core.gnn_locator import load_model, train_gnn  # noqa: E402
from core.opendss_engine import DistFlowEngine  # noqa: E402
from core.sparsity_simulator import ObservabilityConfig, simulate  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / ".cache"

STRATEGIES = {
    "Bayes only": lambda pb, pg: pb,
    "GNN only": lambda pb, pg: pg,
    "Log-linear 0.65 (previous)":lambda pb, pg: log_linear_pool(pb, pg, 0.65),
    "Arithmetic mean 0.65": lambda pb, pg: 0.65 * pb + 0.35 * pg,
    "OutageLens combine() (margin-gated)": combine,
}


def ece(conf: np.ndarray, hit: np.ndarray, bins=(0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0001)) -> float:
    idx = np.digitize(conf, bins) - 1
    return float(sum(abs(conf[idx == b].mean() - hit[idx == b].mean()) * (idx == b).sum()
                     for b in np.unique(idx)) / len(conf))


def main(n_per_rate: int = 40, seed: int = 4242) -> None:
    pd.set_option("display.width", 160)
    for build in (ieee13, ieee34):
        f = build()
        eng = DistFlowEngine(f)
        bayes = BayesianLocator(f, eng)
        gnn = load_model(f, CACHE / f"gnn_{f.key}.npz") or train_gnn(f, eng)
        rng = np.random.default_rng(seed)
        rec = {name: {"hit1": [], "hit3": [], "conf": [], "km": [], "nll": []} for name in STRATEGIES}
        for rate in DEFAULT_RATES:
            cfg = ObservabilityConfig(meter_dropout=rate)
            for fault in np.resize(rng.permutation(f.n_edge), n_per_rate):
                sc = simulate(f, eng, int(fault), cfg, seed=int(rng.integers(1 << 31)))
                pb, pg = bayes.locate(sc).probs, gnn.predict(sc)
                for name, fn in STRATEGIES.items():
                    p = np.asarray(fn(pb, pg), dtype=float)
                    p = p / p.sum()
                    r = np.argsort(-p, kind="stable")
                    rec[name]["hit1"].append(r[0] == fault)
                    rec[name]["hit3"].append(fault in r[:3])
                    rec[name]["conf"].append(p[r[0]])
                    rec[name]["km"].append(patrol_km(f, r, int(fault))[0])
                    rec[name]["nll"].append(-np.log(p[fault] + 1e-12))
        rows = []
        for name, d in rec.items():
            hit1 = np.array(d["hit1"], dtype=float)
            rows.append({"strategy": name, "top1 %": round(100 * hit1.mean(), 1),
                         "top3 %": round(100 * np.mean(d["hit3"]), 1),
                         "ECE pts": round(100 * ece(np.array(d["conf"]), hit1), 1),
                         "NLL": round(float(np.mean(d["nll"])), 3),
                         "patrol km": round(float(np.mean(d["km"])), 2)})
        n = len(DEFAULT_RATES) * n_per_rate
        print(f"\n=== {f.name} — {n} outages, 10-95 % meters missing ===")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
