"""OutageLens ⚡: Streamlit control-room dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core.baseline_locator import locate_baseline
from core.bayesian_locator import BayesianLocator, narrative
from core.evaluator import (DEFAULT_RATES, M_BASELINE, M_ENSEMBLE, conventional_patrol_km, headline,
                            patrol_km, patrol_minutes, run_ablation, run_benchmark, run_placement_study, summarize)
from core.feeder_topology import FEEDERS, get_feeder
from core.ensemble import fusion_weight
from core.gnn_locator import FEATURE_NAMES, ensemble, load_model, save_model, train_gnn
from core.opendss_engine import DistFlowEngine, get_engine
from core.sparsity_simulator import MISSING, ObservabilityConfig, fixed_sensor_layout, simulate
from ui.benchmark_charts import (ablation_chart, accuracy_chart, attribution_chart, calibration_chart,
                                 candidates_chart, hop_chart, patrol_chart, placement_chart, training_chart,
                                 voltage_profile_chart)
from ui.network_plot import network_figure
from ui.theme import COLORS, card, hero, inject_css, kpi_card, kpi_row

st.set_page_config(layout="wide", page_title="OutageLens ⚡ | IEEE Distribution Outage Localization",
                   page_icon="⚡", initial_sidebar_state="expanded")
inject_css()

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
MODEL_CHOICES = ["Ensemble (Bayes + GNN)", "Bayesian + Physics", "GNN only"]
BENCH_N_PER_RATE = 22  # 7 sparsity levels x 22 = 154 Monte Carlo scenarios


# ------------------------------------------------------------------ caching
@st.cache_resource(show_spinner=False)
def load_system(feeder_name: str, prefer_opendss: bool):
    feeder = get_feeder(feeder_name)
    engine, status = get_engine(feeder, prefer_opendss)
    return engine, status, BayesianLocator(feeder, engine)


@st.cache_resource(show_spinner=False)
def load_surrogate(feeder_name: str):
    feeder = get_feeder(feeder_name)
    engine = DistFlowEngine(feeder)
    return engine, BayesianLocator(feeder, engine)


@st.cache_resource(show_spinner=False)
def gnn_store() -> dict:
    """Process-wide trained models (shared by every browser session)."""
    return {}


def load_gnn(feeder_name: str, progress=None):
    store = gnn_store()
    feeder = get_feeder(feeder_name)
    if feeder.key not in store:
        path = CACHE_DIR / f"gnn_{feeder.key}.npz"
        model = load_model(feeder, path)
        if model is None:
            model = train_gnn(feeder, DistFlowEngine(feeder), progress=progress)
            save_model(model, path)
        store[feeder.key] = model
    return store[feeder.key]


@st.cache_data(show_spinner=False)
def cached_benchmark(feeder_name: str, coverage: float, false_alarm: float, n_per_rate: int, seed: int, _gnn=None):
    feeder = get_feeder(feeder_name)
    engine, bayes = load_surrogate(feeder_name)
    return run_benchmark(feeder, engine, bayes, _gnn, DEFAULT_RATES, n_per_rate, coverage, false_alarm, seed)


@st.cache_data(show_spinner=False)
def cached_placement(feeder_name: str):
    feeder = get_feeder(feeder_name)
    engine, bayes = load_surrogate(feeder_name)
    return run_placement_study(feeder, engine, bayes)


@st.cache_data(show_spinner=False)
def cached_ablation(feeder_name: str, coverage: float, false_alarm: float):
    feeder = get_feeder(feeder_name)
    engine, bayes = load_surrogate(feeder_name)
    return run_ablation(feeder, engine, bayes, sensor_coverage=coverage, false_alarm=false_alarm)


# ------------------------------------------------------------- session state
ss = st.session_state
ss.setdefault("hide_truth", False)
ss.setdefault("mystery_count", 0)
# Opening scenarios: a glitchy upstream meter (13-bus) and a long silent trunk (34-bus) both fool the baseline.
DEMO_DEFAULTS = {"ieee13": ("632-671", 31), "ieee34": ("854-852", 6)}


def trigger_mystery(fault_key: str, seed_key: str, n_edge: int) -> None:
    rng = np.random.default_rng()
    ss[fault_key] = int(rng.integers(n_edge))
    ss[seed_key] = int(rng.integers(1, 1_000_000))
    ss["hide_truth"] = True
    ss["mystery_count"] += 1


def resample(seed_key: str) -> None:
    ss[seed_key] = int(np.random.default_rng().integers(1, 1_000_000))


def reveal() -> None:
    ss["hide_truth"] = False


def memo_fig(key: tuple, builder):
    """Charts that only change with their data (benchmark, training) are rebuilt only when it changes."""
    cache = ss.setdefault("_fig_cache", {})
    if key not in cache:
        if len(cache) > 48:
            cache.clear()
        cache[key] = builder()
    return cache[key]


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚡ OutageLens Control")
    feeder_name = st.selectbox("Feeder testbed", list(FEEDERS.keys()), key="feeder_name")
    feeder = get_feeder(feeder_name)
    fault_key, seed_key = f"fault_{feeder.key}", f"seed_{feeder.key}"
    demo_edge, demo_seed = DEMO_DEFAULTS[feeder.key]
    ss.setdefault(fault_key, feeder.edge_by_name[demo_edge].idx)
    ss.setdefault(seed_key, demo_seed)

    st.markdown("**Fault injection**")
    st.selectbox("Line section to fault", list(range(feeder.n_edge)), key=fault_key,
                 format_func=feeder.edge_label)
    st.button("\U0001f3b2  Trigger Mystery Outage", type="primary", on_click=trigger_mystery, key="mystery_btn",
              args=(fault_key, seed_key, feeder.n_edge),
              help="Random fault, ground truth hidden. Let OutageLens find it.")

    st.markdown("**Observability & chaos**")
    dropout = st.slider("Smart meter dropout rate (% missing data)", 0, 95, 65, 5)
    coverage = st.slider("Voltage sensor coverage (% monitored buses)", 10, 100, 25, 5)
    noise = st.slider("Telemetry noise / comms glitch rate (%)", 0, 10, 2, 1)

    c1, c2 = st.columns([1, 1])
    c1.button("\U0001f504 Re-sample", on_click=resample, args=(seed_key,),
              help="New random meter-dropout mask, load level and glitches on the same fault")
    c2.markdown(f"<div class='ol-caption' style='padding-top:0.55rem'>scenario seed <span class='ol-mono'>"
                f"{ss[seed_key]}</span></div>", unsafe_allow_html=True)

    st.markdown("**Model & demo**")
    model_choice = st.segmented_control("OutageLens model", MODEL_CHOICES, default=MODEL_CHOICES[0],
                                        key="model_choice") or MODEL_CHOICES[0]
    st.toggle("Hide ground truth (blind judge demo)", key="hide_truth")
    placement = st.segmented_control("Voltage-sensor placement", ["Optimised", "Random"], default="Optimised",
                                     key="placement") or "Optimised"
    prefer_dss = st.toggle("Use OpenDSS engine when available", value=True)

# ------------------------------------------------------------ build system
engine, status, bayes = load_system(feeder_name, prefer_dss)

progress_slot = st.empty()


def _gnn_progress(frac: float, msg: str) -> None:
    progress_slot.progress(min(frac, 1.0), text=f"\U0001f9e0 Training the GNN on 600 synthetic outage scenarios "
                                                f"(one-time, cached afterwards) — {msg}")


gnn = load_gnn(feeder_name, progress=_gnn_progress)
progress_slot.empty()

hero(status["engine"], status["engine"].startswith("OpenDSS"), feeder.name, status["detail"])

fault = int(ss[fault_key])
show_truth = not ss["hide_truth"]
cfg = ObservabilityConfig(meter_dropout=dropout / 100, sensor_coverage=coverage / 100, false_alarm=noise / 100)
sensors = (bayes.optimal_sensors(cfg.sensor_coverage) if placement == "Optimised"
           else fixed_sensor_layout(feeder, cfg.sensor_coverage))
sc = simulate(feeder, engine, fault, cfg, seed=int(ss[seed_key]), sensor=sensors)

rb = bayes.locate(sc)
p_gnn = gnn.predict(sc)
p_ens = ensemble(rb.probs, p_gnn)
w_fuse = fusion_weight(rb.probs)
base = locate_baseline(feeder, sc)
primary = {"Ensemble (Bayes + GNN)": p_ens, "Bayesian + Physics": rb.probs, "GNN only": p_gnn}[model_choice]
ranking = np.argsort(-primary, kind="stable")
top1 = int(ranking[0])
nz = primary[primary > 0]
entropy_bits = float(-(nz * np.log2(nz)).sum())

# ----------------------------------------------------------------- KPI ribbon
conv_km, conv_n = conventional_patrol_km(feeder, fault)
conv_min = patrol_minutes(conv_km, conv_n)
ol_km, ol_n = patrol_km(feeder, ranking, fault)
ol_min = patrol_minutes(ol_km, ol_n)
cum_km = np.cumsum(feeder.edge_patrol_km[ranking])
exp_km = float((primary[ranking] * cum_km).sum())  # expected patrol under the posterior (no truth needed)
exp_min = float((primary[ranking] * np.array([patrol_minutes(k, i + 1) for i, k in enumerate(cum_km)])).sum())
conv_exp_km = float(np.mean([conventional_patrol_km(feeder, e)[0] for e in range(feeder.n_edge)]))
conv_exp_min = float(np.mean([patrol_minutes(*conventional_patrol_km(feeder, e)) for e in range(feeder.n_edge)]))

cards = []
if show_truth:
    cards.append(kpi_card("Ground-truth faulted section", feeder.edges[fault].label,
                          f"{feeder.shed_kw[fault] * sc.load_level:,.0f} kW interrupted · "
                          f"{int(feeder.downstream[fault].sum())} buses dark", None, COLORS["crimson"]))
else:
    cards.append(kpi_card("Ground-truth faulted section", "\U0001f512 Hidden", "Blind judge demo: truth is locked",
                          ("BLIND MODE", "grey"), COLORS["grey"]))

if show_truth:
    hops = feeder.edge_hops(top1, fault)
    if top1 == fault:
        pill = ("EXACT MATCH", "green")
    elif fault in ranking[:3]:
        pill = (f"IN TOP-3 · {hops} hop(s)", "amber")
    elif hops <= 1:
        pill = ("WITHIN 1 HOP", "amber")
    else:
        pill = (f"MISS · {hops} hops", "red")
else:
    pill = (f"Top-3 holds {primary[ranking[:3]].sum():.0%} of belief", "grey")
cards.append(kpi_card("OutageLens #1 prediction", feeder.edges[top1].label,
                      f"Confidence <b>{primary[top1]:.1%}</b> · entropy {entropy_bits:.2f} bits", pill,
                      COLORS["orange"]))

b1 = int(base.ranking[0])
if show_truth:
    bh = feeder.edge_hops(b1, fault)
    bpill = ("CORRECT", "green") if b1 == fault else (f"WRONG · {bh} hop(s) off", "red")
else:
    bpill = ("BLIND MODE", "grey")
cards.append(kpi_card("Naive LCA baseline #1", feeder.edges[b1].label,
                      f"Uses only {base.n_pings} outage ping(s)", bpill, COLORS["muted"]))

if show_truth:
    red = 100 * (1 - ol_km / conv_km) if conv_km > 0 else 0.0
    cards.append(kpi_card("Search zone (patrol km)", f"{ol_km:.2f} km <span style='color:#94a3b8;font-size:0.9rem'>"
                          f"vs {conv_km:.2f} km</span>", f"{red:.1f}% less line to inspect · {ol_n} section(s) visited",
                          None, COLORS["sky"]))
    saved = conv_min - ol_min
    cards.append(kpi_card("Crew restoration time saved", f"~{max(saved, 0):,.0f} min",
                          f"{ol_min:,.0f} min vs {conv_min:,.0f} min conventional patrol", None, COLORS["emerald"]))
else:
    red = 100 * (1 - exp_km / conv_exp_km) if conv_exp_km > 0 else 0.0
    cards.append(kpi_card("Expected search zone", f"{exp_km:.2f} km <span style='color:#94a3b8;font-size:0.9rem'>"
                          f"vs {conv_exp_km:.2f} km</span>", f"{red:.1f}% reduction (posterior expectation)", None,
                          COLORS["sky"]))
    cards.append(kpi_card("Expected crew time saved", f"~{max(conv_exp_min - exp_min, 0):,.0f} min",
                          f"{exp_min:,.0f} min vs {conv_exp_min:,.0f} min average conventional", None,
                          COLORS["emerald"]))
kpi_row(cards)

if not show_truth:
    st.button("\U0001f513 Reveal ground truth", on_click=reveal)

tab1, tab2, tab3, tab4 = st.tabs(["\U0001f5fa️ Live Control Room", "\U0001f4c8 Robustness Benchmark",
                                  "⚡ Voltage Physics Inspector", "\U0001f9e0 Architecture & Math"])

# ------------------------------------------------------------- TAB 1: map
with tab1:
    left, right = st.columns([0.65, 0.35], gap="medium")
    with left:
        st.plotly_chart(network_figure(feeder, sc, primary, show_truth), width="stretch",
                        config={"displaylogo": False, "scrollZoom": True})
        n_obs = int(((sc.meter != MISSING) & sc.has_meter).sum())
        st.markdown(
            f"<div class='ol-caption'>Meters reporting: <b>{n_obs}/{int(sc.has_meter.sum())}</b> · "
            f"outage pings: <b>{int((sc.meter == 0).sum())}</b> · voltage sensors: <b>{int(sc.sensor.sum())}</b>"
            f" · load level this scenario: <b>{sc.load_level:.0%}</b> of planning load · "
            f"line width & colour = {model_choice} posterior</div>", unsafe_allow_html=True)

    with right:
        top5 = ranking[:5]
        rows = [{"rank": i + 1, "section": feeder.edges[e].name, "label": feeder.edge_label(e),
                 "outagelens": float(primary[e]), "bayes": float(rb.probs[e]), "gnn": float(p_gnn[e]),
                 "baseline": float(base.probs[e]), "kw": float(feeder.shed_kw[e] * rb.load_level_hat),
                 "km": feeder.edges[e].length_km} for i, e in enumerate(top5)]
        table = pd.DataFrame([{
            "#": r["rank"], "Section": r["label"] + (" ✅" if show_truth and top5[r["rank"] - 1] == fault else ""),
            "OutageLens": r["outagelens"], "Bayes": r["bayes"], "GNN": r["gnn"], "Baseline": r["baseline"],
            "Load kW": r["kw"], "km": r["km"]} for r in rows])
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "OutageLens": st.column_config.ProgressColumn("OutageLens", format="percent", min_value=0, max_value=1),
            "Bayes": st.column_config.NumberColumn(format="percent"),
            "GNN": st.column_config.NumberColumn(format="percent"),
            "Baseline": st.column_config.NumberColumn("Baseline vote", format="percent"),
            "Load kW": st.column_config.NumberColumn(format="%.0f"),
            "km": st.column_config.NumberColumn(format="%.2f"),
        })
        st.plotly_chart(candidates_chart(feeder, rows), width="stretch", config={"displaylogo": False})

        ex = bayes.explain(sc, rb, top1, probs=primary)
        bullets = "".join(f"<li style='margin-bottom:0.35rem'>{_md}</li>" for _md in [
            line.replace("**", "<b>", 1).replace("**", "</b>", 1) for line in narrative(ex)])
        unc = entropy_bits / ex["max_entropy_bits"]
        unc_label = "LOW" if unc < 0.25 else ("MEDIUM" if unc < 0.55 else "HIGH")
        card(f"\U0001f50e Why OutageLens picked #1: {feeder.edges[top1].label}",
             f"<ul style='padding-left:1.1rem;margin:0'>{bullets}</ul>"
             f"<div class='ol-caption' style='margin-top:0.5rem'>Uncertainty: <b>{unc_label}</b> · Shannon "
             f"entropy H(P) = {entropy_bits:.2f} of {ex['max_entropy_bits']:.2f} bits max · "
             f"GNN agrees at {p_gnn[top1]:.0%}, Bayesian at {rb.probs[top1]:.0%} · ensemble weight this "
             f"outage: Bayes {w_fuse:.2f} / GNN {1 - w_fuse:.2f}"
             f"{' (Bayes decisive, GNN not consulted)' if w_fuse >= 1.0 else ' (near tie, GNN arbitrates)'}</div>")
        exps = [bayes.explain(sc, rb, int(e), probs=primary) for e in ranking[:3]]
        st.plotly_chart(attribution_chart(exps), width="stretch", config={"displaylogo": False})

# ------------------------------------------------------- TAB 2: benchmark
with tab2:
    st.markdown("#### Does OutageLens keep working as meter data disappears?")
    st.markdown(
        "<div class='ol-caption'>Each scenario draws a random faulted section, a random load level (80–115 % of "
        "planning load with ±10 % per-bus variation), a random voltage-sensor layout and a random meter-dropout "
        "mask. Sensor coverage and glitch rate follow the sidebar. A crew then patrols sections in the order each "
        "method ranks them. Benchmarks use the NumPy DistFlow surrogate, cross-validated against OpenDSS.</div>",
        unsafe_allow_html=True)
    bench_key = f"bench_{feeder.key}_{coverage}_{noise}"
    run = st.button(f"▶️ Run Monte Carlo Sparsity Benchmark ({BENCH_N_PER_RATE * len(DEFAULT_RATES)} scenarios)",
                    type="primary")
    if run:
        bar = st.progress(0.0, text="Starting Monte Carlo…")
        eng_s, bayes_s = load_surrogate(feeder_name)
        ss[bench_key] = run_benchmark(feeder, eng_s, bayes_s, gnn, DEFAULT_RATES, BENCH_N_PER_RATE, cfg.sensor_coverage,
                                      cfg.false_alarm, seed=int(np.random.default_rng().integers(1 << 30)),
                                      progress=lambda f_, m_: bar.progress(f_, text=m_))
        bar.empty()
    if bench_key in ss:
        df = ss[bench_key]
        st.success(f"Showing your latest run: {df['fault_edge'].size // df['method'].nunique()} scenarios.")
    else:
        with st.spinner("Pre-computing the default benchmark…"):
            df = cached_benchmark(feeder_name, cfg.sensor_coverage, cfg.false_alarm, 20, 2026, _gnn=gnn)
        st.info(f"Pre-computed quick benchmark ({df['fault_edge'].size // df['method'].nunique()} scenarios). "
                f"Press the button for the full run.")
    summ = summarize(df)
    head = headline(df)
    fp = (feeder.key, len(df), round(float(df["patrol_km"].sum()), 6), round(float(df["top1"].sum()), 3))
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(memo_fig(("acc",) + fp, lambda: accuracy_chart(summ)), width="stretch",
                    config={"displaylogo": False})
    c2.plotly_chart(memo_fig(("hop",) + fp, lambda: hop_chart(summ)), width="stretch", config={"displaylogo": False})
    c3.plotly_chart(memo_fig(("patrol",) + fp, lambda: patrol_chart(summ)), width="stretch",
                    config={"displaylogo": False})

    hb = head.set_index("method")
    if M_ENSEMBLE in hb.index and M_BASELINE in hb.index:
        e_, b_ = hb.loc[M_ENSEMBLE], hb.loc[M_BASELINE]
        hi = summ[summ["missing_rate"] >= 0.85]
        e_hi = hi[hi["method"] == M_ENSEMBLE]["top3"].mean()
        b_hi = hi[hi["method"] == M_BASELINE]["top3"].mean()
        kpi_row([
            kpi_card("Top-1 accuracy (avg)", f"{e_['top1']:.0f}% <span style='color:#94a3b8;font-size:0.9rem'>vs "
                     f"{b_['top1']:.0f}%</span>", "OutageLens ensemble vs naive baseline", None, COLORS["orange"]),
            kpi_card("Top-3 accuracy (avg)", f"{e_['top3']:.0f}% <span style='color:#94a3b8;font-size:0.9rem'>vs "
                     f"{b_['top3']:.0f}%</span>", "A crew checks three sections at most", None, COLORS["sky"]),
            kpi_card("Top-3 at ≥85% missing", f"{e_hi:.0f}% <span style='color:#94a3b8;font-size:0.9rem'>vs "
                     f"{b_hi:.0f}%</span>", "When almost every meter is silent", None, COLORS["violet"]),
            kpi_card("Patrol distance", f"{e_['patrol_km']:.2f} km", f"vs {b_['patrol_km']:.2f} km baseline, "
                     f"{e_['conventional_km']:.2f} km conventional", None, COLORS["emerald"]),
            kpi_card("Crew time per outage", f"{e_['patrol_min']:.0f} min", f"vs {b_['patrol_min']:.0f} min baseline, "
                     f"{e_['conventional_min']:.0f} min conventional", None, COLORS["amber"]),
        ])
    st.dataframe(head.rename(columns={
        "method": "Method", "top1": "Top-1 %", "top3": "Top-3 %", "hop_error": "Mean hop error",
        "patrol_km": "Patrol km", "patrol_min": "Crew min", "conventional_km": "Conventional km",
        "conventional_min": "Conventional min", "km_reduction_pct": "km reduction %"}).round(2),
        hide_index=True, width="stretch")

    st.markdown("#### Which data stream carries the answer, and can the confidence be trusted?")
    abl = cached_ablation(feeder_name, cfg.sensor_coverage, cfg.false_alarm)
    d1, d2 = st.columns([0.55, 0.45])
    d1.plotly_chart(memo_fig(("abl", feeder.key, coverage, noise), lambda: ablation_chart(abl)), width="stretch",
                    config={"displaylogo": False})
    d2.plotly_chart(memo_fig(("calib",) + fp, lambda: calibration_chart(df)), width="stretch",
                    config={"displaylogo": False})
    st.markdown("<div class='ol-caption'>Left: with most meters silent, topology alone cannot separate sections in "
                "series. Voltage-rise fingerprints and the feeder-head load drop recover the answer, which is why "
                "OutageLens fuses all three. Right: points near the diagonal mean the stated confidence matches "
                "observed hit rates, so a dispatcher can trust a 90 % call more than a 40 % one.</div>",
                unsafe_allow_html=True)

    st.markdown("#### Where should the next voltage sensor go?")
    p1, p2 = st.columns([0.55, 0.45])
    pl = cached_placement(feeder_name)
    p1.plotly_chart(memo_fig(("place", feeder.key), lambda: placement_chart(pl)), width="stretch",
                    config={"displaylogo": False})
    opt_now = bayes.optimal_sensors(cfg.sensor_coverage)
    p2.markdown(
        f"<div class='ol-card'><h4>\U0001f4cd Sensor placement optimiser</h4>"
        f"OutageLens ranks buses by how much each one helps tell candidate sections apart: the expected "
        f"post-fault reading for every possible open section, compared pair by pair, added greedily. "
        f"<br><br>For a <b>{coverage}%</b> budget on this feeder it recommends: "
        f"<span class='ol-mono'>{', '.join(feeder.buses[i] for i in np.flatnonzero(opt_now))}</span>."
        f"<br><br><span class='ol-caption'>The same physics model that localises faults also tells the utility "
        f"where its next sensor buys the most visibility.</span></div>", unsafe_allow_html=True)

# --------------------------------------------------- TAB 3: voltage physics
with tab3:
    cand = top1
    exp_dv = np.where(feeder.downstream[cand], -np.nan_to_num(sc.v_pre_obs), rb.load_level_hat * bayes.dv_sig[cand])
    st.plotly_chart(voltage_profile_chart(feeder, sc, show_truth, exp_dv, feeder.edges[cand].name), width="stretch",
                    config={"displaylogo": False})
    st.markdown(
        "<div class='ol-caption'>When a section opens, the load beyond it disappears. Current on every upstream "
        "segment falls, so the I·Z drop shrinks and <b>upstream voltages rise</b>. The size of that rise at "
        "each sensor depends on how much load was lost and where. That fingerprint lets OutageLens tell a trunk fault "
        "from a lateral fault even when 90 % of downstream meters are silent.</div>", unsafe_allow_html=True)

    st.markdown("#### Telemetry snapshot")
    status_txt = {1: "1 (energized)", 0: "0 (outage ping)", MISSING: "? (missing)"}
    tel = pd.DataFrame({
        "bus": feeder.buses,
        "distance_km": [feeder.dist_km[b] for b in feeder.buses],
        "nominal_load_kw": feeder.p_kw,
        "meter_report": [("SCADA" if not sc.has_meter[i] else status_txt[int(sc.meter[i])]) for i in range(feeder.n_bus)],
        "voltage_sensor": sc.sensor,
        "v_pre_obs_pu": sc.v_pre_obs,
        "v_post_obs_pu": sc.v_post_obs,
        "dv_obs_pu": sc.dv_obs,
    })
    if show_truth:
        tel["true_v_pre_pu"] = sc.v_pre
        tel["true_v_post_pu"] = sc.v_post
        tel["truth_energized"] = sc.energized
    st.dataframe(tel.round(5), hide_index=True, width="stretch", height=320)
    meta = (f"# OutageLens telemetry | feeder={feeder.key} | seed={sc.seed} | dropout={cfg.meter_dropout:.2f} | "
            f"coverage={cfg.sensor_coverage:.2f} | glitch={cfg.false_alarm:.2f} | head_kw_pre={sc.head_kw_pre_obs:.1f} | "
            f"head_kw_post={sc.head_kw_post_obs:.1f}" + (f" | fault={feeder.edges[fault].name}" if show_truth else "") + "\n")
    st.download_button("⬇️ Download telemetry CSV", meta + tel.to_csv(index=False),
                       file_name=f"outagelens_{feeder.key}_seed{sc.seed}.csv", mime="text/csv")

# ------------------------------------------------------ TAB 4: architecture
with tab4:
    st.markdown("#### Pipeline")
    st.markdown("""
```
 IEEE 13 / 34-bus feeder  ──►  Power-flow engine  ──►  Outage scenario  ──►  Partial observability
 (topology, R+jX, kW/kVAR)     OpenDSS ⇄ BIBC/BCBV       (open section e*,     (meter dropout α, glitches ε,
                           sweep (cross-checked)      random load level)    sparse V sensors β, SCADA)
                                                                                    │
      ┌─────────────────────────────────────────────────────────────────────────────┤
      ▼                               ▼                                             ▼
  Naive LCA baseline        Bayesian topology + physics model            Tree-structured GNN (NumPy)
  (outage pings only)       P(S|e)·P(V|e)·P(ΔP|e)·P(e)                   subtree / path / neighbour messages
                                      └───────────── log-linear ensemble ──────────┘
                                                            ▼
                      Ranked sections · calibrated probabilities · evidence · patrol km & crew minutes
```""")
    a1, a2 = st.columns([0.55, 0.45], gap="large")
    with a1:
        st.markdown("#### 1 · Power flow (exact radial sweep)")
        st.latex(r"\mathbf{J} = \mathbf{BIBC}\,\overline{\left(\mathbf{S}_{load}/\mathbf{V}\right)},\qquad "
                 r"\mathbf{V} = \mathbf{T}\,V_0 - \mathbf{BCBV}\,\mathbf{J}")
        st.latex(r"V_v \approx V_u - \frac{R_{uv}P_{uv} + X_{uv}Q_{uv}}{V_0}")
        st.markdown("Shedding the load in the subtree T(e) shrinks every upstream R·P + X·Q drop, so "
                    "**ΔV(e) > 0 at upstream sensors**, scaled by how much load was lost and where.")
        st.markdown("#### 2 · Bayesian localisation")
        st.latex(r"P(e\mid\mathcal D) \propto P(S\mid e)\,P(V^{obs}\mid e)\,P(\Delta P_{head}\mid e)\,P(e)")
        st.latex(r"P(S_i{=}0\mid i\in T(e)) = P(S_i{=}1\mid i\notin T(e)) = (1-\alpha)(1-\epsilon)")
        st.latex(r"P(S_i{=}\varnothing) = \alpha \quad\text{(identical for every } e\text{, so silence cancels)}")
        st.latex(r"\log P(V^{obs}\mid e) = \sum_{m\in\mathcal M}\log\!\Big[(1-\rho)\,\mathcal N\big(V^{obs}_m;"
                 r"\,\hat V_m(e),\,\sigma_m^2(e)\big) + \rho\,\mathcal U\Big]")
        st.latex(r"\hat V_m(e) = \begin{cases}0 & m\in T(e)\\ V^{pre}_m + \hat\lambda\,\Delta V_m(e) & \text{otherwise}"
                 r"\end{cases}")
        st.latex(r"\hat\lambda = P^{obs}_{head,pre}/P^{nom}_{head},\qquad H(P) = -\sum_e P(e)\log_2 P(e)")
        st.markdown(
            "- **Silent meters cancel.** Under random dropout a missing report contributes log α to every "
            "candidate equally, so silence is never mistaken for evidence.\n"
            "- **Robust physics.** Each Gaussian term carries a 1 % outlier floor, so a single bad sensor cannot "
            "veto the correct section.\n"
            "- **Load-level aware.** Signatures are rescaled by the load level estimated from substation SCADA, so "
            "the model copes with time-of-day load swings it was never told about.")
    with a2:
        st.markdown("#### 3 · Tree-structured GNN")
        st.latex(r"H' = \mathrm{ReLU}\big(HW_0 + SHW_1 + PHW_2 + \hat A HW_3 + b\big)")
        st.latex(r"s_e = \mathbf h_v^\top \mathbf a + \mathbf h_u^\top \mathbf b + c,\qquad P_{GNN}(e) = "
                 r"\mathrm{softmax}_e(s_e)")
        st.markdown("S averages over each node's downstream subtree, P over its path to the substation, Â over its "
                    "neighbours. Two layers already see the whole feeder. Forward and backward passes are hand-written "
                    "in NumPy (gradient-checked), so there are no compiled dependencies.")
        st.dataframe(pd.DataFrame({"#": range(1, len(FEATURE_NAMES) + 1), "Node feature": FEATURE_NAMES}),
                     hide_index=True, width="stretch")
        st.plotly_chart(memo_fig(("train", feeder.key, id(gnn)), lambda: training_chart(gnn.history)),
                        width="stretch", config={"displaylogo": False})
        if st.button("\U0001f501 Retrain GNN from scratch"):
            (CACHE_DIR / f"gnn_{feeder.key}.npz").unlink(missing_ok=True)
            gnn_store().pop(feeder.key, None)
            cached_benchmark.clear()
            st.rerun()
        st.markdown("#### 4 · Margin-gated ensemble")
        st.latex(r"P_{ens}(e) \propto P_{Bayes}(e)^{\,w}\,P_{GNN}(e)^{\,1-w}")
        st.latex(r"w = 0.65 + 0.35\,\min\!\Big(\tfrac{P_{Bayes}(\#1) - P_{Bayes}(\#2)}{0.5},\,1\Big)")
        st.markdown("Bayes leads. The GNN only gets a say when Bayes is torn between its top two sections, "
                    "typically two sections in series that the meters cannot tell apart. The weight is "
                    "continuous, so the output never jumps, and was tuned on simulation seeds kept separate "
                    "from every reported benchmark.")

    st.markdown("#### Engine & data provenance")
    dev = status.get("max_dev_pu")
    st.markdown(
        f"- **Active engine:** {status['engine']}. {status['detail']}\n"
        f"- **Feeder data:** IEEE PES Distribution Test Feeders (13-bus: 3,466 kW, 4.16 kV; 34-bus: 1,769 kW, 24.9 kV, "
        f"93.9 km of line), reduced to balanced positive-sequence equivalents. Regulators use fixed taps at their "
        f"published steady-state positions.\n"
        + (f"- **Engine agreement:** OpenDSS and the NumPy sweep agree within {dev:.1e} p.u. across healthy and faulted "
           f"cases.\n" if dev is not None else ""))
    st.markdown("#### Deployment path into a utility OMS / ADMS")
    st.markdown(
        "| Today in the prototype | In a utility deployment |\n|---|---|\n"
        "| IEEE feeder models | GIS / CIM network model export (IEC 61968/61970) |\n"
        "| Simulated last-gasp pings | AMI head-end outage & restoration events, on-demand meter pings |\n"
        "| Simulated voltage sensors | Line sensors, smart-inverter telemetry, AMI voltage reads |\n"
        "| Simulated feeder-head power | Substation SCADA / RTU analogs (DNP3, IEC 61850) |\n"
        "| Streamlit dashboard | Ranked candidate list pushed to the OMS crew-dispatch work order |\n")
    st.markdown("#### Honest limitations & next steps")
    st.markdown(
        "- Balanced positive-sequence model. Next: a full unbalanced three-phase OpenDSS model so single-phase "
        "laterals show phase-specific signatures.\n"
        "- Assumes one open section per event. Next: multi-fault and protection-coordination aware hypotheses "
        "(recloser/fuse operations).\n"
        "- Dropout is modelled as random. Real last-gasp loss can correlate with geography; the Bayesian model "
        "can absorb a per-area α.\n"
        "- The GNN is trained per feeder. Next: train across many synthetic feeders so it transfers to new "
        "networks without retraining.")
