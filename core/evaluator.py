"""Monte Carlo benchmark and operational impact metrics.

Every method returns a ranked list of sections. A crew patrols sections in that
order until it finds the fault, so ranking quality maps directly onto
kilometres driven and minutes of customer outage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline_locator import locate_baseline
from .bayesian_locator import BayesianLocator
from .feeder_topology import Feeder
from .gnn_locator import TreeGNN, ensemble
from .sparsity_simulator import ObservabilityConfig, simulate

PATROL_KMH = 8.0  # average along-line patrol speed (vehicle + foot inspection)
SECTION_OVERHEAD_MIN = 12.0  # isolate, inspect and report per section visited
DEFAULT_RATES = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95)

M_BAYES = "OutageLens Bayesian + Physics"
M_ENSEMBLE = "OutageLens Ensemble (Bayes + GNN)"
M_GNN = "GNN only"
M_BASELINE = "Naive LCA Baseline"
METHOD_ORDER = [M_ENSEMBLE, M_BAYES, M_GNN, M_BASELINE]


def patrol_km(feeder: Feeder, ranking: np.ndarray, true_edge: int) -> tuple[float, int]:
    """Kilometres inspected and sections visited before reaching the faulted section."""
    pos = int(np.flatnonzero(ranking == true_edge)[0])
    visited = ranking[: pos + 1]
    return float(feeder.edge_patrol_km[visited].sum()), pos + 1


def conventional_patrol_km(feeder: Feeder, true_edge: int) -> tuple[float, int]:
    """No analytics: the crew walks the feeder out from the substation, backbone first."""
    return patrol_km(feeder, np.array(feeder.dfs_edge_order()), true_edge)


def patrol_minutes(km: float, sections: int) -> float:
    return km / PATROL_KMH * 60.0 + SECTION_OVERHEAD_MIN * sections


def score(feeder: Feeder, ranking: np.ndarray, true_edge: int) -> dict:
    km, n = patrol_km(feeder, ranking, true_edge)
    return {
        "top1": float(ranking[0] == true_edge),
        "top3": float(true_edge in ranking[:3]),
        "hop_error": float(feeder.edge_hops(int(ranking[0]), true_edge)),
        "patrol_km": km,
        "sections_visited": n,
        "patrol_min": patrol_minutes(km, n),
    }


def run_benchmark(feeder: Feeder, engine, bayes: BayesianLocator, gnn: TreeGNN | None,
                  rates=DEFAULT_RATES, n_per_rate: int = 20, sensor_coverage: float = 0.25,
                  false_alarm: float = 0.02, seed: int = 2026, progress=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    total = len(rates) * n_per_rate
    done = 0
    for rate in rates:
        cfg = ObservabilityConfig(meter_dropout=float(rate), sensor_coverage=sensor_coverage,
                                  false_alarm=false_alarm)
        faults = np.resize(rng.permutation(feeder.n_edge), n_per_rate)
        for fault in faults:
            sc = simulate(feeder, engine, int(fault), cfg, seed=int(rng.integers(1 << 31)))
            rb = bayes.locate(sc)
            bl = locate_baseline(feeder, sc)
            outputs = {M_BAYES: (rb.probs, rb.ranking), M_BASELINE: (bl.probs, bl.ranking)}
            if gnn is not None and gnn.trained:
                pg = gnn.predict(sc)
                pe = ensemble(rb.probs, pg)
                outputs[M_GNN] = (pg, np.argsort(-pg, kind="stable"))
                outputs[M_ENSEMBLE] = (pe, np.argsort(-pe, kind="stable"))
            conv_km, conv_n = conventional_patrol_km(feeder, int(fault))
            for method, (probs, ranking) in outputs.items():
                rows.append({"method": method, "missing_rate": rate, "fault_edge": int(fault),
                             "confidence": float(probs[ranking[0]]),
                             "conventional_km": conv_km, "conventional_min": patrol_minutes(conv_km, conv_n),
                             **score(feeder, ranking, int(fault))})
            done += 1
            if progress is not None and (done % 5 == 0 or done == total):
                progress(done / total, f"Scenario {done}/{total}  (missing meters {rate:.0%})")
    return pd.DataFrame(rows)


ABLATIONS = {
    "Smart meters only": dict(use_meters=True, use_voltage=False, use_head=False),
    "Meters + voltage sensors": dict(use_meters=True, use_voltage=True, use_head=False),
    "Meters + voltage + feeder-head SCADA": dict(use_meters=True, use_voltage=True, use_head=True),
}


def run_ablation(feeder: Feeder, engine, bayes: BayesianLocator, rates=(0.40, 0.70, 0.95), n_per_rate: int = 30,
                 sensor_coverage: float = 0.25, false_alarm: float = 0.02, seed: int = 11) -> pd.DataFrame:
    """Which evidence stream carries the localisation as meter data disappears?"""
    rng = np.random.default_rng(seed)
    rows = []
    for rate in rates:
        cfg = ObservabilityConfig(meter_dropout=float(rate), sensor_coverage=sensor_coverage,
                                  false_alarm=false_alarm)
        for fault in np.resize(rng.permutation(feeder.n_edge), n_per_rate):
            sc = simulate(feeder, engine, int(fault), cfg, seed=int(rng.integers(1 << 31)))
            rows.append({"variant": M_BASELINE, "missing_rate": rate,
                         **score(feeder, locate_baseline(feeder, sc).ranking, int(fault))})
            for name, kw in ABLATIONS.items():
                rows.append({"variant": name, "missing_rate": rate,
                             **score(feeder, bayes.locate(sc, **kw).ranking, int(fault))})
    df = pd.DataFrame(rows)
    out = df.groupby(["variant", "missing_rate"], as_index=False).agg(top1=("top1", "mean"), top3=("top3", "mean"))
    out[["top1", "top3"]] *= 100
    return out


def run_placement_study(feeder: Feeder, engine, bayes: BayesianLocator, coverages=(0.15, 0.25, 0.40),
                        rates=(0.5, 0.85, 0.95), n_per_rate: int = 24, seed: int = 21) -> pd.DataFrame:
    """Same sensor budget, two layouts: random installs vs OutageLens-optimised placement."""
    rng = np.random.default_rng(seed)
    rows = []
    for cov in coverages:
        optimal = bayes.optimal_sensors(cov)
        for rate in rates:
            cfg = ObservabilityConfig(meter_dropout=float(rate), sensor_coverage=float(cov))
            for fault in np.resize(rng.permutation(feeder.n_edge), n_per_rate):
                s = int(rng.integers(1 << 31))
                for layout, sensor in (("Random placement", None), ("Optimised placement", optimal)):
                    sc = simulate(feeder, engine, int(fault), cfg, seed=s, sensor=sensor)
                    rows.append({"layout": layout, "coverage": cov,
                                 **score(feeder, bayes.locate(sc).ranking, int(fault))})
    df = pd.DataFrame(rows)
    out = df.groupby(["layout", "coverage"], as_index=False).agg(top1=("top1", "mean"), patrol_km=("patrol_km", "mean"))
    out["top1"] *= 100
    return out


CALIBRATION_BINS =(0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0001)


def calibration(df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, float]:
    """Reliability table of top-1 confidence vs. hit rate, plus expected calibration error (ECE)."""
    d = df[df["method"] == method].copy()
    d["bin"] = pd.cut(d["confidence"], CALIBRATION_BINS, right=False)
    t = d.groupby("bin", observed=True).agg(confidence=("confidence", "mean"), accuracy=("top1", "mean"),
                                            n=("top1", "size")).reset_index(drop=True)
    ece = float((t["n"] * (t["confidence"] - t["accuracy"]).abs()).sum() / max(t["n"].sum(), 1))
    return t, ece


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "missing_rate"], as_index=False).agg(
        top1=("top1", "mean"), top3=("top3", "mean"), hop_error=("hop_error", "mean"),
        patrol_km=("patrol_km", "mean"), patrol_min=("patrol_min", "mean"),
        conventional_km=("conventional_km", "mean"), n=("top1", "size"))
    g["top1"] *= 100
    g["top3"] *= 100
    return g


def headline(df: pd.DataFrame) -> pd.DataFrame:
    """One row per method averaged over all sparsity levels."""
    g = df.groupby("method", as_index=False).agg(
        top1=("top1", "mean"), top3=("top3", "mean"), hop_error=("hop_error", "mean"),
        patrol_km=("patrol_km", "mean"), patrol_min=("patrol_min", "mean"),
        conventional_km=("conventional_km", "mean"), conventional_min=("conventional_min", "mean"))
    g["top1"] *= 100
    g["top3"] *= 100
    g["km_reduction_pct"] = 100 * (1 - g["patrol_km"] / g["conventional_km"])
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    return g.sort_values("method", key=lambda s: s.map(order)).reset_index(drop=True)
