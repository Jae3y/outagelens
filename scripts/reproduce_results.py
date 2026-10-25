"""Reproduce the headline numbers quoted in the README (fixed seeds, DistFlow surrogate).

    python scripts/reproduce_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bayesian_locator import BayesianLocator  # noqa: E402
from core.evaluator import (M_BASELINE, M_ENSEMBLE, calibration, headline, run_ablation,  # noqa: E402
                            run_benchmark, run_placement_study, summarize)
from core.feeder_topology import ieee13, ieee34  # noqa: E402
from core.gnn_locator import train_gnn  # noqa: E402
from core.opendss_engine import DistFlowEngine  # noqa: E402

pd.set_option("display.width", 200)

for build in (ieee13, ieee34):
    f = build()
    eng = DistFlowEngine(f)
    bayes = BayesianLocator(f, eng)
    gnn = train_gnn(f, eng)
    df = run_benchmark(f, eng, bayes, gnn, n_per_rate=40, seed=2026)
    print(f"\n=== {f.name}: {len(df) // df['method'].nunique()} scenarios, 25% random sensors, 2% glitches ===")
    print(headline(df)[["method", "top1", "top3", "hop_error", "patrol_km", "patrol_min", "conventional_km",
                        "km_reduction_pct"]].round(1).to_string(index=False))
    s = summarize(df)
    print("\nTop-1 % by missing-meter rate")
    print(s.pivot(index="missing_rate", columns="method", values="top1")[[M_ENSEMBLE, M_BASELINE]].round(0).to_string())
    print("\nECE (pts):", {m: round(calibration(df, m)[1] * 100, 1) for m in df["method"].unique() if m != M_BASELINE})
    print("\nAblation (Top-1 %)")
    print(run_ablation(f, eng, bayes).pivot(index="variant", columns="missing_rate", values="top1").round(0).to_string())
    print("\nSensor placement (Top-1 %)")
    print(run_placement_study(f, eng, bayes).pivot(index="layout", columns="coverage", values="top1").round(0).to_string())
