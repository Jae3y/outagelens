"""Turns core simulations into the JSON the web control room renders.

One ``System`` per feeder holds the power-flow engine (OpenDSS when it validates, the NumPy sweep
otherwise), the calibrated Bayesian locator, the trained GNN and a DistFlow surrogate used for
Monte Carlo benchmarks. Systems are built once and shared by every request.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from core.baseline_locator import locate_baseline
from core.bayesian_locator import BayesianLocator, evidence_items
from core.ensemble import fusion_weight
from core.evaluator import (DEFAULT_RATES, METHOD_ORDER, M_BASELINE, calibration, conventional_patrol_km, headline,
                            patrol_km, patrol_minutes, run_ablation, run_benchmark, run_placement_study, summarize)
from core.feeder_topology import Feeder, ieee13, ieee34
from core.gnn_locator import ensemble, load_model, save_model, train_gnn
from core.opendss_engine import DistFlowEngine, get_engine
from core.sparsity_simulator import MISSING, ObservabilityConfig, fixed_sensor_layout, simulate

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"

FEEDERS = {"ieee13": ieee13, "ieee34": ieee34}
FEEDER_META = {
    "ieee13": {"short": "IEEE 13-bus", "kv": "4.16 kV"},
    "ieee34": {"short": "IEEE 34-bus", "kv": "24.9 kV"},
}
# Opening scenarios where a single glitch fools the baseline but not OutageLens.
DEMO_DEFAULTS = {"ieee13": ("632-671", 31), "ieee34": ("854-852", 6)}
KIND_NAME = {"line": "Line", "transformer": "Transformer", "regulator": "Regulator", "switch": "Switch"}
MODEL_KEYS = ("ensemble", "bayes", "gnn")
BENCH_N_PER_RATE = 22  # 7 sparsity levels x 22 = 154 scenarios for a full run


def _role(f: Feeder, idx: int) -> str:
    e = f.edges[idx]
    if e.kind != "line":
        return {"transformer": "Transformer", "regulator": "Voltage regulator", "switch": "Switch"}[e.kind]
    return "Main trunk" if e.name in f.trunk else "Lateral"


def _display(f: Feeder, idx: int) -> str:
    e = f.edges[idx]
    return f"{KIND_NAME[e.kind]} {e.u} → {e.v}"


class System:
    def __init__(self, key: str, prefer_opendss: bool = True):
        self.key = key
        self.feeder = f = FEEDERS[key]()
        self.engine, self.status = get_engine(f, prefer_opendss)
        self.surrogate = DistFlowEngine(f)
        # Temperature calibration is deterministic and the engines agree to 1e-4 p.u., so it is computed once
        # on the fast surrogate, cached to disk, and shared by both locators (cheap starts on small hosts).
        cal_path = CACHE_DIR / f"calibration_{f.key}.json"
        temperature = _read_json(cal_path).get("temperature")
        self.bayes_surrogate = BayesianLocator(f, self.surrogate, calibrate=temperature is None)
        if temperature is None:
            temperature = self.bayes_surrogate.temperature
            _write_json(cal_path, {"temperature": temperature})
        self.bayes_surrogate.temperature = temperature
        self.bayes = BayesianLocator(f, self.engine, calibrate=False)
        self.bayes.temperature = temperature
        path = CACHE_DIR / f"gnn_{f.key}.npz"
        self.gnn = load_model(f, path)
        if self.gnn is None:
            CACHE_DIR.mkdir(exist_ok=True)
            self.gnn = train_gnn(f, self.surrogate)
            save_model(self.gnn, path)
        self.conventional = [conventional_patrol_km(f, e) for e in range(f.n_edge)]
        self._sensor_cache: dict[tuple, np.ndarray] = {}

    def sensors(self, coverage: float, placement: str) -> np.ndarray:
        k = (round(coverage, 3), placement)
        if k not in self._sensor_cache:
            self._sensor_cache[k] = (self.bayes.optimal_sensors(coverage) if placement == "optimised"
                                     else fixed_sensor_layout(self.feeder, coverage))
        return self._sensor_cache[k]


CACHE_VERSION = "outagelens-cache-v1"


def _read_json(path: Path) -> dict:
    """Cached artefact, or {} when missing, unreadable or written by another version."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data.get("version") == CACHE_VERSION else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps({"version": CACHE_VERSION, **data}), encoding="utf-8")
    except OSError:
        pass  # read-only filesystem: recompute next start instead of failing the request


_systems: dict[str, System] = {}
_locks = {k: threading.Lock() for k in FEEDERS}


def system(key: str) -> System:
    if key not in FEEDERS:
        raise KeyError(key)
    with _locks[key]:
        if key not in _systems:
            _systems[key] = System(key)
        return _systems[key]


def feeders() -> list[dict]:
    out = []
    for key, build in FEEDERS.items():
        f = build()
        demo_edge, demo_seed = DEMO_DEFAULTS[key]
        out.append({"key": key, "name": f.name, **FEEDER_META[key], "km": round(f.total_km, 1),
                    "n_bus": f.n_bus, "n_edge": f.n_edge, "default_fault": f.edge_by_name[demo_edge].idx,
                    "default_seed": demo_seed,
                    "sections": [{"idx": e.idx, "label": f"{_display(f, e.idx)} · {_role(f, e.idx).lower()}"}
                                 for e in f.edges]})
    return out


# ---------------------------------------------------------------------------- scenario
def _model_summary(s: System, sc, rb, probs: np.ndarray, p_bayes: np.ndarray, p_gnn: np.ndarray,
                   p_base: np.ndarray, fault: int) -> dict:
    f = s.feeder
    ranking = np.argsort(-probs, kind="stable")
    top = int(ranking[0])
    nz = probs[probs > 0]
    km, n = patrol_km(f, ranking, fault)
    cum = np.cumsum(f.edge_patrol_km[ranking])
    exp_km = float((probs[ranking] * cum).sum())
    exp_min = float((probs[ranking] * np.array([patrol_minutes(k, i + 1) for i, k in enumerate(cum)])).sum())
    ex = s.bayes.explain(sc, rb, top, probs=probs)
    hops = f.edge_hops(top, fault)
    status = ("exact" if top == fault else "top3" if fault in ranking[:3] else "near" if hops <= 1 else "miss")
    lam = rb.load_level_hat
    sensor_pred = {f.buses[i]: (0.0 if f.downstream[top, i] else float(lam * s.bayes.dv_sig[top, i]))
                   for i in np.flatnonzero(sc.sensor)}
    candidates = []
    for rank, idx in enumerate(ranking[:5]):
        idx = int(idx)
        e = f.edges[idx]
        kw = f.shed_kw[idx] * lam
        meta = [_role(f, idx)]
        if e.kind == "line":
            meta.append(f"{e.length_km:.2f} km")
        meta.append(f"{kw:,.0f} kW")
        candidates.append({"rank": rank + 1, "idx": idx, "display": _display(f, idx), "meta": " · ".join(meta),
                           "p": float(probs[idx]), "bayes": float(p_bayes[idx]), "gnn": float(p_gnn[idx]),
                           "baseline": float(p_base[idx])})
    return {
        "top": top, "display": _display(f, top), "section": ex["section"], "confidence": float(probs[top]),
        "top3_mass": float(probs[ranking[:3]].sum()), "entropy_bits": float(-(nz * np.log2(nz)).sum()),
        "max_entropy_bits": float(np.log2(f.n_edge)), "status": status, "hops": int(hops),
        "patrol_km": km, "sections_visited": n, "patrol_min": patrol_minutes(km, n),
        "expected_km": exp_km, "expected_min": exp_min,
        "candidates": candidates, "evidence": evidence_items(ex),
        "shares": {"physics": ex["shares"]["Voltage physics match"],
                   "meters": ex["shares"]["Topological meter evidence"],
                   "scada": ex["shares"]["Feeder-head load drop"], "prior": ex["shares"]["Exposure prior"]},
        "sensor_pred": sensor_pred, "shed_kw_exp": ex["shed_kw_exp"],
    }


def scenario(key: str, fault: int | None = None, seed: int | None = None, dropout: float = 0.65,
             coverage: float = 0.25, glitch: float = 0.02, placement: str = "optimised") -> dict:
    s = system(key)
    f = s.feeder
    demo_edge, demo_seed = DEMO_DEFAULTS[key]
    fault = f.edge_by_name[demo_edge].idx if fault is None else int(fault)
    seed = demo_seed if seed is None else int(seed)
    if not 0 <= fault < f.n_edge:
        raise ValueError(f"fault must be between 0 and {f.n_edge - 1}")
    cfg = ObservabilityConfig(meter_dropout=dropout, sensor_coverage=coverage, false_alarm=glitch)
    sc = simulate(f, s.engine, fault, cfg, seed=seed, sensor=s.sensors(coverage, placement))

    rb = s.bayes.locate(sc)
    p_gnn = s.gnn.predict(sc)
    p_ens = ensemble(rb.probs, p_gnn)
    base = locate_baseline(f, sc)
    probs = {"ensemble": p_ens, "bayes": rb.probs, "gnn": p_gnn}
    models = {m: _model_summary(s, sc, rb, p, rb.probs, p_gnn, base.probs, fault) for m, p in probs.items()}

    b_top = int(base.ranking[0])
    conv_km, conv_n = s.conventional[fault]
    conv_all = s.conventional
    conv_exp_km = float(np.mean([c[0] for c in conv_all]))
    conv_exp_min = float(np.mean([patrol_minutes(*c) for c in conv_all]))
    upstream_glitches = [f.buses[i] for i in np.flatnonzero((sc.meter == 0) & ~f.downstream[fault])]

    nodes = []
    for i, b in enumerate(f.buses):
        nodes.append({
            "bus": b, "x": f.coords[b][0], "y": f.coords[b][1], "meter": int(sc.meter[i]),
            "has_meter": bool(sc.has_meter[i]), "sensor": bool(sc.sensor[i]), "energized": bool(sc.energized[i]),
            "source": b == f.source, "load_kw": float(f.p_kw[i]), "km": round(float(f.dist_km[b]), 3),
            "v_pre": float(sc.v_pre[i]), "v_post": float(sc.v_post[i]),
            "v_pre_obs": None if np.isnan(sc.v_pre_obs[i]) else float(sc.v_pre_obs[i]),
            "v_post_obs": None if np.isnan(sc.v_post_obs[i]) else float(sc.v_post_obs[i]),
        })
    edges = [{"idx": e.idx, "u": e.u, "v": e.v, "name": e.name, "display": _display(f, e.idx),
              "role": _role(f, e.idx), "kind": e.kind, "trunk": e.name in f.trunk,
              "length_km": e.length_km, "shed_kw": float(f.shed_kw[e.idx])} for e in f.edges]

    return {
        "feeder": {"key": key, "name": f.name, **FEEDER_META[key], "km": round(f.total_km, 1), "source": f.source},
        "engine": {"name": s.status["engine"], "short": s.status["short"], "detail": s.status["detail"],
                   "max_dev_pu": s.status["max_dev_pu"], "opendss": s.status["engine"].startswith("OpenDSS")},
        "scenario": {"fault": fault, "seed": seed, "dropout": dropout, "coverage": coverage, "glitch": glitch,
                     "placement": placement, "load_level": sc.load_level, "load_level_hat": rb.load_level_hat},
        "truth": {"edge": fault, "display": _display(f, fault), "section": f"{f.edges[fault].u} → {f.edges[fault].v}",
                  "interrupted_kw": float(f.shed_kw[fault] * sc.load_level),
                  "dark_buses": int(f.downstream[fault].sum())},
        "probs": {m: [float(x) for x in p] for m, p in probs.items()} | {"baseline": [float(x) for x in base.probs]},
        "models": models,
        "fusion_weight": fusion_weight(rb.probs),
        "baseline": {"top": b_top, "display": _display(f, b_top), "pings": base.n_pings,
                     "hops": int(f.edge_hops(b_top, fault)), "correct": b_top == fault,
                     "upstream_glitches": upstream_glitches},
        "conventional": {"km": conv_km, "min": patrol_minutes(conv_km, conv_n),
                         "expected_km": conv_exp_km, "expected_min": conv_exp_min},
        "counts": {"reporting": int(((sc.meter != MISSING) & sc.has_meter).sum()), "meters": int(sc.has_meter.sum()),
                   "pings": int((sc.meter == 0).sum()), "sensors": int(sc.sensor.sum())},
        "scada": {"pre_kw": sc.head_kw_pre_obs, "post_kw": sc.head_kw_post_obs, "shed_kw": sc.shed_kw_obs},
        "nodes": nodes,
        "edges": edges,
    }


def telemetry_csv(payload: dict) -> str:
    status = {1: "1 (energized)", 0: "0 (outage ping)", MISSING: "? (missing)"}
    sc = payload["scenario"]
    lines = [f"# OutageLens telemetry | feeder={payload['feeder']['key']} | seed={sc['seed']} | "
             f"dropout={sc['dropout']:.2f} | coverage={sc['coverage']:.2f} | glitch={sc['glitch']:.2f} | "
             f"head_kw_pre={payload['scada']['pre_kw']:.1f} | head_kw_post={payload['scada']['post_kw']:.1f}",
             "bus,distance_km,nominal_load_kw,meter_report,voltage_sensor,v_pre_obs_pu,v_post_obs_pu,dv_obs_pu"]
    for n in payload["nodes"]:
        meter = "SCADA" if not n["has_meter"] else status[n["meter"]]
        pre, post = n["v_pre_obs"], n["v_post_obs"]
        dv = "" if pre is None else f"{post - pre:.5f}"
        lines.append(f"{n['bus']},{n['km']:.3f},{n['load_kw']:.1f},{meter},{n['sensor']},"
                     f"{'' if pre is None else f'{pre:.5f}'},{'' if post is None else f'{post:.5f}'},{dv}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- benchmark
def _bench_payload(s: System, df, coverage: float, glitch: float) -> dict:
    summ = summarize(df)
    head = headline(df)
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    pts, ece = calibration(df, METHOD_ORDER[0])
    return {
        "n_scenarios": int(len(df) // df["method"].nunique()),
        "coverage": coverage, "glitch": glitch,
        "methods": [m for m in METHOD_ORDER if m in set(df["method"])],
        "summary": sorted(({"method": r.method, "rate": float(r.missing_rate), "top1": float(r.top1),
                            "top3": float(r.top3), "hop": float(r.hop_error), "km": float(r.patrol_km),
                            "min": float(r.patrol_min), "conv_km": float(r.conventional_km)}
                           for r in summ.itertuples()), key=lambda d: (order[d["method"]], d["rate"])),
        "headline": [{"method": r.method, "top1": float(r.top1), "top3": float(r.top3), "hop": float(r.hop_error),
                      "km": float(r.patrol_km), "min": float(r.patrol_min), "conv_km": float(r.conventional_km),
                      "conv_min": float(r.conventional_min)} for r in head.itertuples()],
        "calibration": {"points": [{"conf": float(r.confidence), "acc": float(r.accuracy), "n": int(r.n)}
                                   for r in pts.itertuples()], "ece": ece},
        "baseline_method": M_BASELINE,
    }


@lru_cache(maxsize=16)
def _studies(key: str, coverage: float, glitch: float) -> dict:
    s = system(key)
    abl = run_ablation(s.feeder, s.surrogate, s.bayes_surrogate, sensor_coverage=coverage, false_alarm=glitch)
    pl = run_placement_study(s.feeder, s.surrogate, s.bayes_surrogate)
    return {
        "ablation": [{"variant": r.variant, "rate": float(r.missing_rate), "top1": float(r.top1)}
                     for r in abl.itertuples()],
        "placement": [{"layout": r.layout, "coverage": float(r.coverage), "top1": float(r.top1)}
                      for r in pl.itertuples()],
    }


DEFAULT_BENCH = (0.25, 0.02)  # sensor coverage and glitch rate of the README / deck numbers


@lru_cache(maxsize=16)
def benchmark(key: str, coverage: float = 0.25, glitch: float = 0.02, n_per_rate: int = 40, seed: int = 2026) -> dict:
    """Default view: the same 280-scenario seeded run that scripts/reproduce_results.py reports.

    The default settings are computed at build time (scripts/warm_cache.py) and read from disk.
    """
    is_default = (round(coverage, 2), round(glitch, 2)) == DEFAULT_BENCH and (n_per_rate, seed) == (40, 2026)
    path = CACHE_DIR / f"benchmark_{key}.json"
    if is_default and (cached := _read_json(path).get("payload")):
        return cached
    s = system(key)
    df = run_benchmark(s.feeder, s.surrogate, s.bayes_surrogate, s.gnn, DEFAULT_RATES, n_per_rate, coverage, glitch,
                       seed)
    payload = _bench_payload(s, df, coverage, glitch) | _studies(key, coverage, glitch)
    if is_default:
        _write_json(path, {"payload": payload})
    return payload


def run_full_benchmark(key: str, coverage: float, glitch: float) -> dict:
    s = system(key)
    seed = int(np.random.default_rng().integers(1 << 30))
    df = run_benchmark(s.feeder, s.surrogate, s.bayes_surrogate, s.gnn, DEFAULT_RATES, BENCH_N_PER_RATE, coverage,
                       glitch, seed)
    return _bench_payload(s, df, coverage, glitch) | _studies(key, coverage, glitch) | {"seed": seed}
