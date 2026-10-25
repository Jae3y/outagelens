"""Dispatch agent: turns a fault-probability ranking into a closed-loop crew search.

The localisers say where the fault probably is. The agent decides what the utility does next, observes
the result, updates its beliefs and decides again, until the crew is standing at the fault:

    plan    pick the action with the lowest expected time-to-find (one-step lookahead):
              INSPECT a section, ordered by P(fault) / minutes to inspect it (optimal search
              index rule: for perfect inspections it minimises expected search time), or
              PING a silent smart meter on demand, when the answer is worth its minutes
    act     send the crew, or ping the meter
    observe the crew reports found / not found (crews miss a fault with probability 1 - detect);
            the meter answers energized / dead, or not at all
    update  Bayes' rule on the section posterior, then plan again

Once the fault is found the agent closes the loop with an isolation and restoration step.
For comparison the same simulator runs a "follow the ranking" crew (inspect in probability order,
no updates) under the same crew luck.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluator import PATROL_KMH, SECTION_OVERHEAD_MIN
from .feeder_topology import Feeder
from .sparsity_simulator import MISSING, Scenario

DETECT = 0.95  # chance a crew spots the fault when it inspects the right section
PING_RESPONSE = 0.7  # chance a silent meter answers an on-demand ping
PING_MIN = 2.0  # minutes for a head-end ping round trip
MAX_STEPS = 80


@dataclass
class Step:
    n: int
    action: str  # INSPECT | PING | RESTORE
    target: str  # section name or bus
    label: str
    minutes: float
    elapsed: float
    reason: str
    observation: str
    top_section: str
    top_p: float


@dataclass
class Episode:
    policy: str
    found: bool
    found_section: str | None
    minutes: float
    inspections: int
    pings: int
    steps: list[Step] = field(default_factory=list)


class DispatchAgent:
    def __init__(self, feeder: Feeder, detect: float = DETECT, ping_response: float = PING_RESPONSE,
                 ping_min: float = PING_MIN):
        self.f = feeder
        self.detect = detect
        self.ping_response = ping_response
        self.ping_min = ping_min
        self.cost = feeder.edge_patrol_km / PATROL_KMH * 60.0 + SECTION_OVERHEAD_MIN  # minutes per section

    # ----------------------------------------------------------------- planning
    def expected_search_minutes(self, p: np.ndarray) -> float:
        """Expected minutes to find the fault inspecting by the index rule p / cost (perfect inspections)."""
        order = np.argsort(-(p / self.cost), kind="stable")
        return float((p[order] * np.cumsum(self.cost[order])).sum())

    def _after_miss(self, p: np.ndarray, e: int) -> np.ndarray:
        q = p.copy()
        q[e] *= 1.0 - self.detect
        return q / q.sum()

    def _inspect_value(self, p: np.ndarray, e: int) -> float:
        """Cost now plus expected remaining search if the section turns out clean."""
        miss_mass = 1.0 - p[e] * self.detect
        return self.cost[e] + miss_mass * self.expected_search_minutes(self._after_miss(p, e))

    def _best_inspect_value(self, p: np.ndarray) -> float:
        return self._inspect_value(p, int(np.argmax(p / self.cost)))

    def _ping_value(self, p: np.ndarray, bus_idx: int) -> tuple[float, float]:
        """Ping now, then inspect optimally given the answer: compared on the same footing as inspecting now."""
        down = self.f.downstream[:, bus_idx]
        p_dead = float(p[down].sum())
        if p_dead <= 1e-3 or p_dead >= 1 - 1e-3:
            return np.inf, 0.0  # the answer is as good as known: no information worth minutes
        dead = np.where(down, p, 0.0) / p_dead
        live = np.where(down, 0.0, p) / (1 - p_dead)
        r = self.ping_response
        expected = (r * (p_dead * self._best_inspect_value(dead) + (1 - p_dead) * self._best_inspect_value(live))
                    + (1 - r) * self._best_inspect_value(p))
        return self.ping_min + expected, p_dead

    def plan(self, p: np.ndarray, pingable: list[int]) -> tuple[str, int, float, str]:
        e = int(np.argmax(p / self.cost))
        best = ("INSPECT", e, self._inspect_value(p, e),
                f"highest P(fault) per crew-minute ({p[e]:.0%} over {self.cost[e]:.0f} min)")
        for b in pingable:
            val, p_dead = self._ping_value(p, b)
            if val < best[2] - 0.5:  # a ping must save at least half a minute in expectation
                best = ("PING", b, val, f"an answer splits the belief {p_dead:.0%} / {1 - p_dead:.0%} and saves "
                                        f"~{self._inspect_value(p, e) - val:.0f} expected min")
        return best[0], best[1], best[2], best[3]

    # -------------------------------------------------------------- simulation
    def _top(self, p: np.ndarray) -> tuple[str, float]:
        k = int(np.argmax(p))
        return self.f.edges[k].name, float(p[k])

    def run(self, p0: np.ndarray, sc: Scenario, seed: int = 0, use_pings: bool = True) -> Episode:
        """Run the agent against a simulated outage. The agent never reads ``sc.fault_edge`` directly:
        it only receives crew reports and meter answers generated from it."""
        # Separate streams: the k-th look at the faulted section has the same outcome for every policy.
        f, crew, meters = self.f, np.random.default_rng(seed), np.random.default_rng(seed + 1_000_003)
        p = np.asarray(p0, dtype=float).copy()
        p /= p.sum()
        pingable = [i for i in range(f.n_bus) if sc.has_meter[i] and sc.meter[i] == MISSING] if use_pings else []
        ep = Episode("agent", False, None, 0.0, 0, 0)
        for n in range(1, MAX_STEPS + 1):
            action, target, _, reason = self.plan(p, pingable)
            if action == "INSPECT":
                e = target
                ep.inspections += 1
                ep.minutes += self.cost[e]
                hit = e == sc.fault_edge and crew.random() < self.detect
                name = f.edges[e].name
                if hit:
                    p = np.zeros_like(p)
                    p[e] = 1.0
                    obs = "fault found"
                else:
                    p = self._after_miss(p, e)
                    obs = "nothing found"
                top = self._top(p)
                ep.steps.append(Step(n, "INSPECT", name, f"Inspect {name.replace('-', ' → ')}",
                                     float(self.cost[e]), ep.minutes, reason, obs, *top))
                if hit:
                    ep.found, ep.found_section = True, name
                    break
            else:
                b = target
                pingable.remove(b)
                ep.pings += 1
                ep.minutes += self.ping_min
                bus = f.buses[b]
                if meters.random() < self.ping_response:
                    alive = bool(sc.energized[b])
                    down = f.downstream[:, b]
                    p = np.where(down, 0.0, p) if alive else np.where(down, p, 0.0)
                    p = p / p.sum()
                    obs = "meter answers: energized" if alive else "meter answers: no power"
                else:
                    obs = "no answer"
                top = self._top(p)
                ep.steps.append(Step(n, "PING", bus, f"Ping meter at bus {bus}", self.ping_min, ep.minutes, reason,
                                     obs, *top))
        if ep.found:
            e = f.edge_by_name[ep.found_section].idx
            kw = float(f.shed_kw[e] * sc.load_level)
            ep.steps.append(Step(len(ep.steps) + 1, "RESTORE", ep.found_section,
                                 f"Isolate {ep.found_section.replace('-', ' → ')}, repair, re-energize",
                                 0.0, ep.minutes, "fault confirmed on site",
                                 f"{kw:,.0f} kW of load back once repaired", ep.found_section, 1.0))
        return ep

    def run_ranked(self, p0: np.ndarray, sc: Scenario, seed: int = 0) -> Episode:
        """Baseline crew: walk the static ranking in order (looping on misses), no belief updates."""
        crew = np.random.default_rng(seed)
        order = np.argsort(-np.asarray(p0), kind="stable")
        ep = Episode("ranked", False, None, 0.0, 0, 0)
        for n in range(MAX_STEPS):
            e = int(order[n % len(order)])
            ep.inspections += 1
            ep.minutes += self.cost[e]
            if e == sc.fault_edge and crew.random() < self.detect:
                ep.found, ep.found_section = True, self.f.edges[e].name
                break
        return ep
