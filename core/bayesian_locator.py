"""Model 2: OutageLens topology- and physics-aware Bayesian localiser.

For each candidate section e the unnormalised log-posterior is

    log P(e | D) = log P(S | e) + log P(V_obs | e) + log P(dP_head | e) + log P(e)

* P(S | e): every reporting smart meter is a noisy witness of whether it sits
  downstream of e (de-energized) or not. Silent meters contribute log(alpha) to
  every candidate equally, so under random dropout they cancel exactly; the
  model never mistakes silence for evidence.
* P(V_obs | e): each voltage sensor's post-event reading is compared with the
  physics-predicted value if e had opened: zero if the sensor is downstream,
  otherwise its pre-event reading plus the load-drop voltage rise dV(e). The
  rise is a fingerprint of how much load was shed and where.
* P(dP_head | e): the feeder-head SCADA power drop must match the load that
  sits downstream of e.
* P(e): mild exposure prior, longer overhead sections fail more often.

Signatures dV(e) and dP(e) come from one power-flow per candidate at nominal
load, rescaled by the load level estimated from the substation SCADA reading.
Gaussian terms are wrapped in a small outlier mixture so one bad sensor cannot
veto the right answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .feeder_topology import Feeder
from .sparsity_simulator import MISSING, Scenario, fit_temperature, random_config, simulate

OUTLIER_RHO = 0.01  # probability that any single physical reading is garbage
KAPPA_V = 0.25  # relative uncertainty of a predicted voltage rise (per-bus load mismatch)
KAPPA_P = 0.12  # relative uncertainty of a predicted shed load
PRIOR_WEIGHT = 0.5


def _robust_loglik(r: np.ndarray, sigma: np.ndarray, span: float) -> np.ndarray:
    gauss = np.exp(-0.5 * (r / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    return np.log((1 - OUTLIER_RHO) * gauss + OUTLIER_RHO / span)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


@dataclass
class BayesResult:
    probs: np.ndarray
    ranking: np.ndarray
    log_meter: np.ndarray
    log_voltage: np.ndarray
    log_head: np.ndarray
    log_prior: np.ndarray
    entropy_bits: float
    load_level_hat: float
    logits: np.ndarray

    def top(self, k: int = 5) -> np.ndarray:
        return self.ranking[:k]


class BayesianLocator:
    def __init__(self, feeder: Feeder, engine, calibrate: bool = True):
        self.f = feeder
        base = engine.solve()
        self.v_nom = base.v_pu
        self.head_nom = base.head_kw
        self.dv_sig = np.zeros((feeder.n_edge, feeder.n_bus))
        self.dp_sig = np.zeros(feeder.n_edge)
        for e in range(feeder.n_edge):
            res = engine.solve(open_edge=e)
            self.dv_sig[e] = np.where(res.energized, res.v_pu - base.v_pu, 0.0)
            self.dp_sig[e] = base.head_kw - res.head_kw
        prior = PRIOR_WEIGHT * np.log(feeder.edge_patrol_km + 0.3)
        self.log_prior = prior - np.log(np.exp(prior).sum())
        self.temperature = 1.0
        if calibrate:
            self.calibrate(engine)

    def calibrate(self, engine, n: int = 300, seed: int = 5) -> float:
        """Temperature-scale the posterior on held-out simulated outages (independent of any benchmark seed)."""
        rng = np.random.default_rng(seed)
        self.temperature = 1.0
        faults = np.resize(rng.permutation(self.f.n_edge), n)
        logits = np.zeros((n, self.f.n_edge))
        for i in range(n):
            sc = simulate(self.f, engine, int(faults[i]), random_config(rng), seed=int(rng.integers(1 << 31)))
            logits[i] = self.locate(sc).logits
        self.temperature = fit_temperature(logits, faults)
        return self.temperature

    # ------------------------------------------------------- sensor planning
    def optimal_sensors(self, coverage: float, sigma: float = 0.004, cap: float = 3.0) -> np.ndarray:
        """Greedy sensor placement that maximises how well candidate sections can be told apart.

        Each bus has a physics signature per candidate section: the post-fault voltage it would read
        (0 if cut off, otherwise V + dV). Two sections are distinguishable at a sensor set if their
        signatures differ by several sigma. Greedily add the bus that most increases the (capped)
        pairwise separation summed over all section pairs; the substation is always monitored.
        """
        f = self.f
        n_total = int(np.clip(round(coverage * f.n_bus), 1, f.n_bus))
        sig = np.where(f.downstream, 0.0, self.v_nom[None, :] + self.dv_sig)  # (E, N)
        iu, ju = np.triu_indices(f.n_edge, k=1)
        diff2 = ((sig[iu] - sig[ju]) / sigma) ** 2  # (pairs, N)
        chosen = [f.bus_index[f.source]]
        acc = diff2[:, chosen].sum(axis=1)
        while len(chosen) < n_total:
            gains = np.minimum(np.sqrt(acc[:, None] + diff2), cap).sum(axis=0)
            gains[chosen] = -np.inf
            best = int(np.argmax(gains))
            chosen.append(best)
            acc = acc + diff2[:, best]
        mask = np.zeros(f.n_bus, dtype=bool)
        mask[chosen] = True
        return mask

    # ------------------------------------------------------------------ terms
    def meter_term(self, sc: Scenario) -> np.ndarray:
        eps = float(np.clip(sc.cfg.false_alarm, 0.005, 0.3))
        down = self.f.downstream
        out = sc.meter == 0
        ok = sc.meter == 1
        n_out_down = down[:, out].sum(axis=1)
        n_ok_down = down[:, ok].sum(axis=1)
        n_out_up = out.sum() - n_out_down
        n_ok_up = ok.sum() - n_ok_down
        return ((n_out_down + n_ok_up) * np.log(1 - eps) + (n_ok_down + n_out_up) * np.log(eps)).astype(float)

    def voltage_term(self, sc: Scenario, lam: float) -> np.ndarray:
        m = np.flatnonzero(sc.sensor & ~np.isnan(sc.v_post_obs))
        if m.size == 0:
            return np.zeros(self.f.n_edge)
        sv = sc.cfg.sigma_v
        down = self.f.downstream[:, m]
        dv = lam * self.dv_sig[:, m]
        expected = np.where(down, 0.0, sc.v_pre_obs[m][None, :] + dv)
        sigma = np.where(down, 3 * sv, np.sqrt(2 * sv**2 + (KAPPA_V * dv) ** 2 + 5e-4**2))
        resid = sc.v_post_obs[m][None, :] - expected
        return _robust_loglik(resid, sigma, span=1.5).sum(axis=1)

    def head_term(self, sc: Scenario, lam: float) -> np.ndarray:
        dp = lam * self.dp_sig
        sigma = np.sqrt(2 * (sc.cfg.sigma_scada * sc.head_kw_pre_obs) ** 2 + (KAPPA_P * dp) ** 2 + 5.0**2)
        return _robust_loglik(sc.shed_kw_obs - dp, sigma, span=max(self.head_nom, 1.0))

    def locate(self, sc: Scenario, use_meters=True, use_voltage=True, use_head=True) -> BayesResult:
        lam = float(np.clip(sc.head_kw_pre_obs / self.head_nom, 0.3, 3.0)) if self.head_nom > 0 else 1.0
        lm = self.meter_term(sc) if use_meters else np.zeros(self.f.n_edge)
        lv = self.voltage_term(sc, lam) if use_voltage else np.zeros(self.f.n_edge)
        lh = self.head_term(sc, lam) if use_head else np.zeros(self.f.n_edge)
        logits = lm + lv + lh + self.log_prior
        probs = _softmax(logits / self.temperature)
        nz = probs[probs > 0]
        entropy = float(-(nz * np.log2(nz)).sum())
        return BayesResult(probs, np.argsort(-probs, kind="stable"), lm, lv, lh, self.log_prior, entropy, lam,
                           logits)

    # ---------------------------------------------------------- explanations
    def explain(self, sc: Scenario, res: BayesResult, e: int, probs: np.ndarray | None = None) -> dict:
        """Structured evidence for why section ``e`` ranks where it does."""
        f = self.f
        probs = res.probs if probs is None else probs
        terms = {"Topological meter evidence": res.log_meter, "Voltage physics match": res.log_voltage,
                 "Feeder-head load drop": res.log_head, "Exposure prior": res.log_prior}
        margins = {k: float(v[e] - np.median(v)) for k, v in terms.items()}
        pos = {k: max(v, 0.0) for k, v in margins.items()}
        total = sum(pos.values()) or 1.0
        shares = {k: 100.0 * v / total for k, v in pos.items()}

        down = f.downstream[e]
        edge = f.edges[e]
        bus = np.array(f.buses)
        pings = bus[down & (sc.meter == 0)].tolist()
        silent = bus[down & (sc.meter == MISSING)].tolist()
        contradict = bus[down & (sc.meter == 1)].tolist()
        hops = f._bus_hops[edge.v]
        locks = [b for b in bus[~down & (sc.meter == 1)] if hops.get(b, 99) <= 2]
        upstream_pings = bus[~down & (sc.meter == 0)].tolist()

        sensors = []
        lam = res.load_level_hat
        for i in np.flatnonzero(sc.sensor):
            b = f.buses[i]
            if b == f.source:
                continue
            sensors.append({
                "bus": b,
                "dv_obs": float(sc.v_post_obs[i] - sc.v_pre_obs[i]),
                "v_post_obs": float(sc.v_post_obs[i]),
                "dv_expected": float(-sc.v_pre_obs[i]) if down[i] else float(lam * self.dv_sig[e, i]),
                "downstream": bool(down[i]),
            })
        rising = [s for s in sensors if not s["downstream"]]
        strongest = max(rising, key=lambda s: s["dv_obs"]) if rising else None

        shed_obs = sc.shed_kw_obs
        shed_exp = lam * self.dp_sig[e]
        return {
            "edge": e,
            "label": f.edge_label(e),
            "section": f"{edge.u} → {edge.v}",
            "prob": float(probs[e]),
            "shares": shares,
            "margins": margins,
            "downstream_buses": int(down.sum()),
            "downstream_load_kw": float(f.shed_kw[e] * lam),
            "pings": pings,
            "silent": silent,
            "contradict": contradict,
            "locks": locks,
            "upstream_pings": upstream_pings,
            "sensors": sensors,
            "strongest_sensor": strongest,
            "shed_kw_obs": shed_obs,
            "shed_kw_exp": shed_exp,
            "entropy_bits": res.entropy_bits,
            "max_entropy_bits": float(np.log2(f.n_edge)),
        }


def _buses(names: list[str], limit: int = 6) -> str:
    return ", ".join(names[:limit]) + ("…" if len(names) > limit else "")


def evidence_items(ex: dict) -> list[dict]:
    """Evidence as tagged items (tag = which data stream it comes from) for the web control room."""
    items = []
    s = ex["strongest_sensor"]
    if s is not None and abs(s["dv_obs"]) > 1e-4:
        verb = "rose" if s["dv_obs"] > 0 else "fell"
        items.append({"tag": "PHYSICS", "text": f"Bus {s['bus']} voltage {verb} {s['dv_obs']:+.3f} p.u.; the model "
                                                f"predicts {s['dv_expected']:+.3f} p.u. if {ex['section']} opened."})
    items.append({"tag": "SCADA", "text": f"Feeder head lost {ex['shed_kw_obs']:,.0f} kW. This section carries "
                                          f"~{ex['shed_kw_exp']:,.0f} kW downstream."})
    if ex["pings"]:
        items.append({"tag": "METERS", "text": f"{len(ex['pings'])} last-gasp ping{'s' if len(ex['pings']) > 1 else ''} "
                                               f"from downstream buses: {_buses(ex['pings'])}."})
    else:
        items.append({"tag": "METERS", "text": "No outage pings survived from this zone; physics and boundary "
                                               "evidence carry the call."})
    topo = []
    if ex["silent"]:
        topo.append(f"{len(ex['silent'])} silent meter{'s' if len(ex['silent']) > 1 else ''} downstream read as neutral")
    if ex["locks"]:
        verb = "still reports" if len(ex["locks"]) == 1 else "still report"
        topo.append(f"bus {_buses(ex['locks'], 4)} {verb} power, locking the fault below it")
    if topo:
        text = "; ".join(topo)
        items.append({"tag": "TOPOLOGY", "text": text[0].upper() + text[1:] + "."})
    dead = [x["bus"] for x in ex["sensors"] if x["downstream"] and x["v_post_obs"] < 0.3]
    if dead:
        items.append({"tag": "PHYSICS", "text": f"Sensor{'s' if len(dead) > 1 else ''} at {_buses(dead, 4)} "
                                                f"read{'' if len(dead) > 1 else 's'} ~0 p.u., confirming "
                                                f"{'they are' if len(dead) > 1 else 'it is'} cut off."})
    if ex["upstream_pings"]:
        n = len(ex["upstream_pings"])
        items.append({"tag": "OUTLIER", "text": f"Ping{'s' if n > 1 else ''} from {_buses(ex['upstream_pings'], 4)} "
                                                f"sit{'' if n > 1 else 's'} outside the zone and "
                                                f"{'are' if n > 1 else 'is'} outvoted as comms glitch"
                                                f"{'es' if n > 1 else ''}."})
    if ex["contradict"]:
        n = len(ex["contradict"])
        items.append({"tag": "CONFLICT", "text": f"{n} downstream meter{'s' if n > 1 else ''} claim"
                                                 f"{'' if n > 1 else 's'} power ({_buses(ex['contradict'], 4)}), "
                                                 f"consistent with comms glitches."})
    return items


def narrative(ex: dict) -> list[str]:
    """Plain-language bullet points for the control-room evidence card."""
    lines = []
    n_p, n_s = len(ex["pings"]), len(ex["silent"])
    if n_p:
        shown = ", ".join(ex["pings"][:6]) + ("…" if n_p > 6 else "")
        lines.append(f"**{n_p} last-gasp ping(s)** received from downstream buses ({shown}).")
    else:
        lines.append("**No outage pings** survived from this zone; the call is carried by physics and boundary evidence.")
    if n_s:
        lines.append(f"**{n_s} silent meter(s)** downstream are inferred de-energized "
                     f"(their silence is neutral evidence, the topology fills the gap).")
    if ex["locks"]:
        verb = "still reports" if len(ex["locks"]) == 1 else "still report"
        lines.append(f"**Healthy boundary locks:** {', '.join(ex['locks'][:6])} {verb} power, "
                     f"pinning the fault below.")
    if ex["upstream_pings"]:
        lines.append(f"⚠️ Outage ping(s) from {', '.join(ex['upstream_pings'][:4])} sit outside this zone and are "
                     f"outvoted by physics and healthy neighbours, treated as comms glitches. A naive LCA rule "
                     f"would chase them upstream.")
    if ex["contradict"]:
        lines.append(f"⚠️ {len(ex['contradict'])} downstream meter(s) claim power "
                     f"({', '.join(ex['contradict'][:4])}), consistent with comms glitches.")
    s = ex["strongest_sensor"]
    if s is not None and abs(s["dv_obs"]) > 1e-4:
        verb = "rose" if s["dv_obs"] > 0 else "fell"
        lines.append(f"**Bus {s['bus']} voltage {verb} by {s['dv_obs']:+.4f} p.u.** "
                     f"(physics model predicts {s['dv_expected']:+.4f} p.u. for this section).")
    dead = [x["bus"] for x in ex["sensors"] if x["downstream"] and x["v_post_obs"] < 0.3]
    if dead:
        lines.append(f"**Voltage sensor(s) at {', '.join(dead)} read ~0 p.u.**, confirming they are cut off.")
    lines.append(f"**Feeder-head SCADA** saw {ex['shed_kw_obs']:,.0f} kW disappear; "
                 f"this section carries ~{ex['shed_kw_exp']:,.0f} kW of downstream load.")
    return lines
