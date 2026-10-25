"""Model 1: naive Lowest-Common-Ancestor (LCA) / downstream-vote baseline.

This mirrors the rule of thumb many outage management systems start from:
"the device that failed is the one just upstream of every customer who called
in". It only looks at outage pings (S_i = 0). It ignores silent meters,
energized reports that contradict a candidate, voltage telemetry and feeder-head
SCADA, so a single false alarm or a single surviving leaf ping misleads it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .feeder_topology import Feeder
from .sparsity_simulator import Scenario


@dataclass
class BaselineResult:
    probs: np.ndarray  # vote share per section (not a calibrated probability)
    ranking: np.ndarray
    votes: np.ndarray
    lca_bus: str | None
    n_pings: int


def locate_baseline(feeder: Feeder, sc: Scenario) -> BaselineResult:
    pings = sc.meter == 0
    votes = feeder.downstream[:, pings].sum(axis=1).astype(float)
    depth = np.array([feeder.depth[e.v] for e in feeder.edges], dtype=float)

    if votes.max() > 0:
        # Most votes first; among sections that explain every ping, the deepest one is the LCA edge.
        ranking = np.lexsort((-depth, -votes))
        probs = votes / votes.sum()
        lca_bus = feeder.edges[ranking[0]].v
    else:
        # No pings at all: patrol from the substation outward.
        ranking = np.lexsort((np.arange(feeder.n_edge), depth))
        probs = np.full(feeder.n_edge, 1.0 / feeder.n_edge)
        lca_bus = None
    return BaselineResult(probs, ranking, votes, lca_bus, int(pings.sum()))
