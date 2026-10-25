"""Power-flow engines.

Two interchangeable engines solve the same positive-sequence feeder model:

* ``OpenDSSEngine`` drives EPRI OpenDSS through ``opendssdirect.py``.
* ``DistFlowEngine`` is an exact complex backward/forward sweep in pure NumPy
  (no compiled dependencies), used as a guaranteed fallback and as the fast
  surrogate for Monte Carlo benchmarking and GNN training data.

``get_engine`` prefers OpenDSS, cross-checks it against the sweep on the
healthy feeder, and falls back automatically if the import, the solve or the
cross-check fails.
"""
from __future__ import annotations

import atexit
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np

from .feeder_topology import S_BASE_MVA, Feeder


@dataclass
class PowerFlowResult:
    v_pu: np.ndarray  # voltage magnitude per bus (0 for de-energized buses)
    energized: np.ndarray  # bool per bus
    head_kw: float  # active power drawn at the substation
    head_kvar: float
    edge_kw: np.ndarray  # active power entering each section at its parent end
    converged: bool
    iterations: int


class DistFlowEngine:
    """Exact radial backward/forward sweep with constant-power loads.

    Uses the BIBC/BCBV matrix form of the sweep (Teng, IEEE Trans. Power
    Delivery, 2003): branch currents J = BIBC @ I_load (backward sweep) and bus
    voltages V = T * V0 - BCBV @ J (forward sweep), with ideal fixed-tap
    regulators folded into both matrices. Each iteration is two mat-vecs.
    """

    name = "IEEE DistFlow Physics Engine"
    short = "DistFlow (NumPy)"

    def __init__(self, feeder: Feeder, tol: float = 1e-9, max_iter: int = 50):
        self.f = feeder
        self.tol = tol
        self.max_iter = max_iter
        n, m = feeder.n_bus, feeder.n_edge
        z = np.array([complex(e.r_pu, e.x_pu) for e in feeder.edges])
        tap = np.array([e.tap for e in feeder.edges])
        self.parent = feeder.edge_parent_idx
        self.src = feeder.bus_index[feeder.source]

        path = {b: [feeder.parent_edge[a].idx for a in feeder.ancestors[b][1:] + [b]] for b in feeder.buses
                if b != feeder.source}
        self.bibc = np.zeros((m, n))  # primary-side current of edge e per unit injection at bus i
        self.bcbv = np.zeros((n, m), dtype=complex)  # voltage drop at bus i per unit current in edge e
        self.t_ratio = np.ones(n)  # no-load voltage ratio from the source to each bus
        for b, edges in path.items():
            i = feeder.bus_index[b]
            self.t_ratio[i] = float(np.prod(tap[edges]))
            for k, e in enumerate(edges):
                gain = float(np.prod(tap[edges[k:]]))
                self.bibc[e, i] = gain
                self.bcbv[i, e] = z[e] * gain

    def solve(self, p_kw: np.ndarray | None = None, q_kvar: np.ndarray | None = None,
              open_edge: int | None = None) -> PowerFlowResult:
        f = self.f
        p_kw = f.p_kw if p_kw is None else p_kw
        q_kvar = f.q_kvar if q_kvar is None else q_kvar
        energized = np.ones(f.n_bus, dtype=bool)
        if open_edge is not None:
            energized &= ~f.downstream[open_edge]
        s_load = (p_kw + 1j * q_kvar) / 1000.0 / S_BASE_MVA * energized

        v_noload = self.t_ratio * f.source_pu + 0j
        v = v_noload.copy()
        converged, it = False, 0
        for it in range(1, self.max_iter + 1):
            j = self.bibc @ np.conj(s_load / v)
            v_new = v_noload - self.bcbv @ j
            if np.max(np.abs(v_new - v)) < self.tol:
                v = v_new
                converged = True
                break
            v = v_new
        j = self.bibc @ np.conj(s_load / v)
        v = np.where(energized, v, 0.0)

        s_edge = v[self.parent] * np.conj(j) * 1000.0 * S_BASE_MVA
        s_head = complex(s_edge[self.parent == self.src].sum())
        return PowerFlowResult(np.abs(v), energized, s_head.real, s_head.imag, s_edge.real, converged, it)


class _DSSWorker:
    """Daemon thread that owns the OpenDSS native instance.

    A daemon thread (rather than ThreadPoolExecutor) keeps serving during
    interpreter shutdown, which lets the backend's exit-time callback cleanup
    run on the owning thread instead of crashing from the main thread.
    """

    def __init__(self):
        self._jobs: queue.Queue = queue.Queue()
        self._exit_hook = False
        threading.Thread(target=self._loop, name="opendss", daemon=True).start()

    def _loop(self) -> None:
        while True:
            fn, args, fut = self._jobs.get()
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn(*args))
                except BaseException as exc:  # surfaced to the caller via the future
                    fut.set_exception(exc)

    def call(self, fn, *args, timeout: float | None = None):
        fut: Future = Future()
        self._jobs.put((fn, args, fut))
        return fut.result(timeout=timeout)

    def install_exit_hook(self) -> None:
        # Registered after the backend's own atexit hook, so it runs first (LIFO); the backend's
        # unregister step is idempotent and becomes a no-op afterwards.
        if self._exit_hook:
            return
        self._exit_hook = True

        def release() -> None:
            try:
                from dss_python_backend.events import _remove_callbacks

                self.call(_remove_callbacks, timeout=5)
            except Exception:
                pass

        atexit.register(release)


_DSS_THREAD = _DSSWorker()


class OpenDSSEngine:
    """Same feeder model compiled to an OpenDSS circuit and solved by OpenDSS.

    The OpenDSS engine is a native singleton bound to the thread that first
    touches it, while Streamlit executes every rerun on a fresh thread. All
    OpenDSS calls are therefore marshalled onto one dedicated worker thread.
    """

    name = "OpenDSS Direct Engine"
    short = "OpenDSS (opendssdirect.py)"

    def __init__(self, feeder: Feeder):
        self.f = feeder
        self._elem = {}
        self._load_names: dict[str, str] = {}
        _DSS_THREAD.call(self._init_on_worker)

    def _init_on_worker(self) -> None:
        import opendssdirect as dss  # raises ImportError when unavailable

        _DSS_THREAD.install_exit_hook()
        self.dss = dss
        self._compile()

    def solve(self, p_kw: np.ndarray | None = None, q_kvar: np.ndarray | None = None,
              open_edge: int | None = None) -> PowerFlowResult:
        return _DSS_THREAD.call(self._solve_on_worker, p_kw, q_kvar, open_edge)

    def _cmd(self, text: str) -> None:
        self.dss.Text.Command(text)

    def _compile(self) -> None:
        f = self.f
        self._cmd("clear")
        src_kv = f.base_kv[f.source]
        self._cmd(f"new circuit.outagelens_{f.key} basekv={src_kv} pu={f.source_pu} phases=3 "
                  f"bus1={f.source} MVAsc3=2000000 MVAsc1=2000000")
        for e in f.edges:
            kv_u, kv_v = f.base_kv[e.u], f.base_kv[e.v]
            if e.kind == "line":
                name = f"line.l{e.idx}"
                self._cmd(f"new {name} bus1={e.u} bus2={e.v} phases=3 r1={e.r_ohm_per_km:.6f} "
                          f"x1={e.x_ohm_per_km:.6f} r0={3 * e.r_ohm_per_km:.6f} x0={3 * e.x_ohm_per_km:.6f} "
                          f"c1=0 c0=0 length={max(e.length_km, 1e-4):.6f} units=km")
            elif e.kind == "switch":
                name = f"line.s{e.idx}"
                self._cmd(f"new {name} bus1={e.u} bus2={e.v} phases=3 switch=yes")
            elif e.kind == "transformer":
                name = f"transformer.t{e.idx}"
                kva = S_BASE_MVA * 1000.0
                r_pct, x_pct = e.r_pu * 100.0, e.x_pu * 100.0  # on the 1 MVA system base
                self._cmd(f"new {name} phases=3 windings=2 buses=[{e.u} {e.v}] conns=[wye wye] "
                          f"kvs=[{kv_u} {kv_v}] kvas=[{kva} {kva}] %Rs=[{r_pct / 2:.6f} {r_pct / 2:.6f}] "
                          f"XHL={x_pct:.6f}")
            else:  # regulator: series impedance on the primary, ideal fixed tap on the secondary
                name = f"transformer.r{e.idx}"
                kva = S_BASE_MVA * 1000.0
                r_pct, x_pct = max(e.r_pu * 100.0, 1e-6), max(e.x_pu * 100.0, 1e-6)
                self._cmd(f"new {name} phases=3 windings=2 buses=[{e.u} {e.v}] conns=[wye wye] "
                          f"kvs=[{kv_u} {kv_v}] kvas=[{kva} {kva}] %Rs=[{r_pct / 2:.6f} {r_pct / 2:.6f}] "
                          f"XHL={x_pct:.6f} taps=[1.0 {e.tap}]")
            self._elem[e.idx] = name
        for b in f.buses:
            p, q = f.load_kw.get(b, 0.0), f.load_kvar.get(b, 0.0)
            if p == 0.0 and q == 0.0:
                continue
            ld = f"ld_{b}"
            self._load_names[b] = ld
            self._cmd(f"new load.{ld} bus1={b} phases=3 conn=wye kV={f.base_kv[b]} kW={p} kvar={q} "
                      f"model=1 vminpu=0.5 vmaxpu=1.5")
        kvs = sorted(set(f.base_kv.values()), reverse=True)
        self._cmd(f"set voltagebases=[{' '.join(str(k) for k in kvs)}]")
        self._cmd("calcvoltagebases")
        self._cmd("set controlmode=off")
        self._cmd("set maxiterations=100")
        self._loaded_key = f.key

    def _solve_on_worker(self, p_kw: np.ndarray | None, q_kvar: np.ndarray | None,
                         open_edge: int | None) -> PowerFlowResult:
        dss, f = self.dss, self.f
        if dss.Circuit.Name().lower() != f"outagelens_{f.key}":
            self._compile()
        p_kw = f.p_kw if p_kw is None else p_kw
        q_kvar = f.q_kvar if q_kvar is None else q_kvar
        for b, ld in self._load_names.items():
            i = f.bus_index[b]
            dss.Loads.Name(ld)
            dss.Loads.kW(float(p_kw[i]))
            dss.Loads.kvar(float(q_kvar[i]))
        if open_edge is not None:
            self._cmd(f"open {self._elem[open_edge]} term=1")
        try:
            dss.Solution.Solve()
            converged = bool(dss.Solution.Converged())
            iters = int(dss.Solution.Iterations())
            v = np.zeros(f.n_bus)
            for b in f.buses:
                dss.Circuit.SetActiveBus(b)
                mags = np.asarray(dss.Bus.puVmagAngle())[::2]
                v[f.bus_index[b]] = float(np.mean(mags)) if len(mags) else 0.0
            total = dss.Circuit.TotalPower()
            edge_kw = np.zeros(f.n_edge)
            for e in f.edges:
                dss.Circuit.SetActiveElement(self._elem[e.idx])
                powers = np.asarray(dss.CktElement.Powers())
                n = len(powers) // 2
                edge_kw[e.idx] = float(np.sum(powers[: n][::2]))
        finally:
            if open_edge is not None:
                self._cmd(f"close {self._elem[open_edge]} term=1")
        energized = np.ones(f.n_bus, dtype=bool)
        if open_edge is not None:
            energized &= ~f.downstream[open_edge]
        v[~energized] = 0.0
        return PowerFlowResult(v, energized, float(-total[0]), float(-total[1]), edge_kw, converged, iters)


def cross_check(feeder: Feeder, a, b) -> float:
    """Largest voltage disagreement (p.u.) between two engines over healthy + every single-fault case."""
    worst = float(np.max(np.abs(a.solve().v_pu - b.solve().v_pu)))
    for e in range(0, feeder.n_edge, max(1, feeder.n_edge // 6)):
        worst = max(worst, float(np.max(np.abs(a.solve(open_edge=e).v_pu - b.solve(open_edge=e).v_pu))))
    return worst


def get_engine(feeder: Feeder, prefer_opendss: bool = True):
    """Return ``(engine, status)``; status records which engine runs and why."""
    distflow = DistFlowEngine(feeder)
    status = {"engine": distflow.name, "short": distflow.short, "detail": "", "max_dev_pu": None}
    if not prefer_opendss:
        status["detail"] = "OpenDSS disabled by user; using the NumPy sweep."
        return distflow, status
    try:
        dss_engine = OpenDSSEngine(feeder)
        dev = cross_check(feeder, dss_engine, distflow)
        status["max_dev_pu"] = dev
        if not np.isfinite(dev) or dev > 0.01:
            status["detail"] = f"OpenDSS disagreed with the sweep by {dev:.4f} p.u.; using the NumPy sweep."
            return distflow, status
        status.update(engine=dss_engine.name, short=dss_engine.short,
                      detail=f"OpenDSS cross-validated against the NumPy sweep (max |ΔV| = {dev:.1e} p.u.).")
        return dss_engine, status
    except Exception as exc:  # missing binary, unsupported platform, solver failure
        status["detail"] = f"OpenDSS unavailable ({type(exc).__name__}); using the NumPy sweep."
        return distflow, status
