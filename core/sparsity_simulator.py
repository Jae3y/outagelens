"""Outage scenario generator with realistic partial observability.

Given a ground-truth open section, the simulator produces exactly what a
utility control room would see:

* Smart-meter status per bus: 1 (energized ping / poll answered), 0 (last-gasp
  outage ping) or missing (dropout: RF collision, cellular dead zone, battery).
  Observed reports are flipped with a small false-alarm probability.
* A sparse set of voltage sensors (the substation is always monitored) giving
  noisy pre- and post-event voltage magnitudes.
* Feeder-head SCADA active power before and after the event.

Loads are never exactly the nominal planning values: every scenario draws a
time-of-day load level and per-bus variation, so the localiser has to cope with
model mismatch rather than matching a lookup table.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from .feeder_topology import Feeder

MISSING = -1  # meter report not received


@dataclass(frozen=True)
class ObservabilityConfig:
    meter_dropout: float = 0.65  # alpha: share of meters whose report never arrives
    sensor_coverage: float = 0.25  # beta: share of buses with a voltage sensor (substation always included)
    false_alarm: float = 0.02  # epsilon: comms glitch flipping an observed report
    sigma_v: float = 0.002  # voltage sensor noise, p.u.
    load_sigma: float = 0.10  # per-bus load deviation from the planning value
    load_level: tuple[float, float] = (0.80, 1.15)  # range of the system-wide load level
    sigma_scada: float = 0.003  # relative error of each feeder-head power reading


@dataclass
class Scenario:
    feeder_key: str
    fault_edge: int
    seed: int
    cfg: ObservabilityConfig
    load_level: float
    p_kw: np.ndarray
    q_kvar: np.ndarray
    v_pre: np.ndarray  # true voltages before the event
    v_post: np.ndarray  # true voltages after the event
    energized: np.ndarray  # true post-event energization
    head_kw_pre: float
    head_kw_post: float
    has_meter: np.ndarray
    meter: np.ndarray  # 1 / 0 / MISSING
    sensor: np.ndarray  # bool: bus carries a voltage sensor
    v_pre_obs: np.ndarray  # NaN where unmonitored
    v_post_obs: np.ndarray
    head_kw_pre_obs: float
    head_kw_post_obs: float
    flipped: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def dv_obs(self) -> np.ndarray:
        return self.v_post_obs - self.v_pre_obs

    @property
    def shed_kw_obs(self) -> float:
        return self.head_kw_pre_obs - self.head_kw_post_obs


def random_config(rng: np.random.Generator) -> ObservabilityConfig:
    """Observability drawn across the whole operating envelope (training / calibration corpora)."""
    return ObservabilityConfig(meter_dropout=float(rng.uniform(0.0, 0.95)),
                               sensor_coverage=float(rng.uniform(0.08, 0.6)),
                               false_alarm=float(rng.uniform(0.0, 0.08)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Post-hoc temperature scaling (Guo et al., 2017): the T minimising held-out NLL of softmax(logits / T)."""
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.5), np.log(6.0), 60)):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = float(-logp[np.arange(len(y)), y].mean())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def place_sensors(feeder: Feeder, coverage: float, rng: np.random.Generator) -> np.ndarray:
    """Substation plus a random subset of buses, ``coverage`` share of all buses in total."""
    n_total = int(np.clip(round(coverage * feeder.n_bus), 1, feeder.n_bus))
    sensor = np.zeros(feeder.n_bus, dtype=bool)
    src = feeder.bus_index[feeder.source]
    sensor[src] = True
    others = np.array([i for i in range(feeder.n_bus) if i != src])
    if n_total > 1:
        sensor[rng.choice(others, size=n_total - 1, replace=False)] = True
    return sensor


def fixed_sensor_layout(feeder: Feeder, coverage: float) -> np.ndarray:
    """Installed hardware does not move when meter dropout is re-sampled: seed placement by feeder + coverage."""
    seed = zlib.crc32(f"{feeder.key}:{coverage:.3f}".encode())
    return place_sensors(feeder, coverage, np.random.default_rng(seed))


def simulate(feeder: Feeder, engine, fault_edge: int, cfg: ObservabilityConfig, seed: int,
             sensor: np.ndarray | None = None) -> Scenario:
    rng = np.random.default_rng(seed)

    level = rng.uniform(*cfg.load_level)
    scale = level * np.clip(rng.normal(1.0, cfg.load_sigma, feeder.n_bus), 0.5, 1.5)
    p_kw, q_kvar = feeder.p_kw * scale, feeder.q_kvar * scale

    pre = engine.solve(p_kw, q_kvar)
    post = engine.solve(p_kw, q_kvar, open_edge=fault_edge)

    has_meter = np.ones(feeder.n_bus, dtype=bool)
    has_meter[feeder.bus_index[feeder.source]] = False  # substation bus is SCADA, not AMI

    truth = post.energized.astype(np.int8)
    flipped = (rng.random(feeder.n_bus) < cfg.false_alarm) & has_meter
    meter = np.where(flipped, 1 - truth, truth).astype(np.int8)
    dropped = rng.random(feeder.n_bus) < cfg.meter_dropout
    meter[dropped | ~has_meter] = MISSING

    if sensor is None:
        sensor = place_sensors(feeder, cfg.sensor_coverage, rng)
    noise_pre = rng.normal(0.0, cfg.sigma_v, feeder.n_bus)
    noise_post = rng.normal(0.0, cfg.sigma_v, feeder.n_bus)
    v_pre_obs = np.where(sensor, pre.v_pu + noise_pre, np.nan)
    v_post_obs = np.where(sensor, np.abs(post.v_pu + noise_post), np.nan)

    head_pre_obs = pre.head_kw * (1.0 + rng.normal(0.0, cfg.sigma_scada))
    head_post_obs = post.head_kw * (1.0 + rng.normal(0.0, cfg.sigma_scada))

    return Scenario(feeder.key, fault_edge, seed, cfg, float(level), p_kw, q_kvar, pre.v_pu, post.v_pu,
                    post.energized, pre.head_kw, post.head_kw, has_meter, meter, sensor, v_pre_obs, v_post_obs,
                    float(head_pre_obs), float(head_post_obs), flipped & ~dropped)
