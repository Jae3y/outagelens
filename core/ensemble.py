"""How OutageLens fuses the Bayesian physics model with the GNN.

Both inputs are probability vectors over the same feeder sections (each sums to 1).

* Bayesian + physics: the stronger model (Top-1 ~91 % on 13-bus, ~77 % on 34-bus), interpretable.
* GNN: weaker on its own (~78 % / ~62 %) but learned purely from data, so it is an independent opinion.

Rule: margin-gated log-linear pooling. Bayes leads; the GNN only gets a say when Bayes itself is
undecided between its top two sections (the typical case is two sections in series that look alike
to the meters). The Bayes weight is

    w = W_MIN + (1 - W_MIN) * min(margin / TAU, 1),   margin = P_bayes(#1) - P_bayes(#2)

so a decisive Bayes call (margin >= TAU) passes through unchanged and a near tie is pooled with the
GNN at weight 1 - W_MIN. The weight is continuous in the margin, so the output never jumps.

TAU and W_MIN were chosen on two simulation seeds that are disjoint from the evaluation seeds in
scripts/compare_ensembles.py and scripts/reproduce_results.py: it beat a fixed 0.65 pool on 13-bus
NLL (0.185 vs 0.199, 0.146 vs 0.160) and tied it on the 34-bus.
"""
from __future__ import annotations

import numpy as np

TAU = 0.5  # Bayes top-1/top-2 margin at which the GNN stops being consulted
W_MIN = 0.65  # Bayes weight when its top two sections are exactly tied


def log_linear_pool(p_bayes: np.ndarray, p_gnn: np.ndarray, w_bayes: float = 0.65) -> np.ndarray:
    """Weighted geometric mean, renormalised: a section must convince both models to rank highly."""
    logp = w_bayes * np.log(p_bayes + 1e-12) + (1 - w_bayes) * np.log(p_gnn + 1e-12)
    z = np.exp(logp - logp.max())
    return z / z.sum()


def fusion_weight(p_bayes: np.ndarray, tau: float = TAU, w_min: float = W_MIN) -> float:
    """Bayes weight for this outage: 1.0 when Bayes is decisive, ``w_min`` at a dead heat."""
    second, first = np.sort(p_bayes)[-2:]
    return float(w_min + (1.0 - w_min) * min((first - second) / tau, 1.0))


def combine(p_bayes: np.ndarray, p_gnn: np.ndarray, tau: float = TAU, w_min: float = W_MIN) -> np.ndarray:
    """Fuse the two models into one probability vector over sections (margin-gated pooling)."""
    return log_linear_pool(p_bayes, p_gnn, fusion_weight(p_bayes, tau, w_min))
