"""Plotly charts for benchmarking, evidence attribution and the voltage inspector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.evaluator import METHOD_ORDER
from core.feeder_topology import Feeder
from core.sparsity_simulator import MISSING, Scenario

from .theme import COLORS, METHOD_COLORS

AXIS = dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)", color="#cbd5e1")


def _style(fig: go.Figure, title: str, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e8f0"), x=0.01),
        height=height, margin=dict(l=10, r=10, t=46, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1", size=12),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=dict(size=10.5),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0f172a"), **kw)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


def _methods(df: pd.DataFrame) -> list[str]:
    return [m for m in METHOD_ORDER if m in set(df["method"])]


def accuracy_chart(summary: pd.DataFrame) -> go.Figure:
    traces: list = []
    for m in _methods(summary):
        d = summary[summary["method"] == m].sort_values("missing_rate")
        c = METHOD_COLORS[m]
        traces.append(go.Scatter(x=d["missing_rate"] * 100, y=d["top1"], name=f"{m} · Top-1", mode="lines+markers",
                                 line=dict(color=c, width=3), marker=dict(size=7)))
        traces.append(go.Scatter(x=d["missing_rate"] * 100, y=d["top3"], name=f"{m} · Top-3", mode="lines",
                                 line=dict(color=c, width=1.6, dash="dash"), showlegend=False,
                                 hovertemplate=f"{m} Top-3: %{{y:.0f}}%<extra></extra>"))
    fig = go.Figure(data=traces)
    _style(fig, "Accuracy vs missing meters (solid Top-1, dashed Top-3)", height=400)
    fig.update_xaxes(title="Missing smart-meter reports (%)", ticksuffix="%")
    fig.update_yaxes(title="Accuracy (%)", range=[0, 103], ticksuffix="%")
    return fig


def hop_chart(summary: pd.DataFrame) -> go.Figure:
    traces: list = []
    for m in _methods(summary):
        d = summary[summary["method"] == m].sort_values("missing_rate")
        traces.append(go.Scatter(x=d["missing_rate"] * 100, y=d["hop_error"], name=m, mode="lines+markers",
                                 line=dict(color=METHOD_COLORS[m], width=3)))
    fig = go.Figure(data=traces)
    _style(fig, "Mean graph-hop error of the #1 pick", height=400)
    fig.update_xaxes(title="Missing smart-meter reports (%)", ticksuffix="%")
    fig.update_yaxes(title="Hops from true fault", rangemode="tozero")
    return fig


def patrol_chart(summary: pd.DataFrame) -> go.Figure:
    traces: list = []
    conv = summary.groupby("missing_rate", as_index=False)["conventional_km"].mean()
    traces.append(go.Scatter(x=conv["missing_rate"] * 100, y=conv["conventional_km"], name="Conventional patrol",
                             mode="lines", line=dict(color=COLORS["crimson"], width=2, dash="dot")))
    for m in _methods(summary):
        d = summary[summary["method"] == m].sort_values("missing_rate")
        traces.append(go.Scatter(x=d["missing_rate"] * 100, y=d["patrol_km"], name=m, mode="lines+markers",
                                 line=dict(color=METHOD_COLORS[m], width=3)))
    fig = go.Figure(data=traces)
    _style(fig, "Line km patrolled before the fault is found", height=400)
    fig.update_xaxes(title="Missing smart-meter reports (%)", ticksuffix="%")
    fig.update_yaxes(title="km inspected", rangemode="tozero")
    return fig


def ablation_chart(abl: pd.DataFrame) -> go.Figure:
    order = ["Naive LCA Baseline", "Smart meters only", "Meters + voltage sensors",
             "Meters + voltage + feeder-head SCADA"]
    colors = [COLORS["muted"], "#64748b", COLORS["sky"], COLORS["orange"]]
    traces: list = []
    for v, c in zip(order, colors):
        d = abl[abl["variant"] == v].sort_values("missing_rate")
        traces.append(go.Bar(x=[f"{r:.0%} missing" for r in d["missing_rate"]], y=d["top1"], name=v, marker_color=c,
                             text=[f"{t:.0f}%" for t in d["top1"]], textposition="outside"))
    fig = go.Figure(data=traces)
    _style(fig, "Evidence ablation: Top-1 accuracy as each data stream is added", height=380, barmode="group")
    fig.update_yaxes(title="Top-1 accuracy (%)", range=[0, 112], ticksuffix="%")
    return fig


def placement_chart(pl: pd.DataFrame) -> go.Figure:
    traces = []
    for layout, color in (("Random placement", "#64748b"), ("Optimised placement", COLORS["orange"])):
        d = pl[pl["layout"] == layout].sort_values("coverage")
        traces.append(go.Bar(x=[f"{c:.0%} of buses" for c in d["coverage"]], y=d["top1"], name=layout,
                             marker_color=color, text=[f"{t:.0f}%" for t in d["top1"]], textposition="outside"))
    fig = go.Figure(data=traces)
    _style(fig, "Same sensor budget, smarter placement (Bayesian Top-1, 50–95 % meters missing)", height=360,
           barmode="group")
    fig.update_xaxes(title="Voltage-sensor budget")
    fig.update_yaxes(title="Top-1 accuracy (%)", range=[0, 112], ticksuffix="%")
    return fig


def calibration_chart(df: pd.DataFrame) -> go.Figure:
    from core.evaluator import M_BASELINE, calibration

    traces: list = []
    traces.append(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration",
                             line=dict(color="#475569", dash="dot")))
    for m in _methods(df):
        if m == M_BASELINE:
            continue
        t, ece = calibration(df, m)
        traces.append(go.Scatter(x=t["confidence"] * 100, y=t["accuracy"] * 100, mode="lines+markers",
                                 name=f"{m} (ECE {ece * 100:.1f} pts)", line=dict(color=METHOD_COLORS[m], width=2.5),
                                 marker=dict(size=6 + np.sqrt(t["n"]).clip(0, 14)), customdata=t["n"],
                                 hovertemplate="said %{x:.0f}% → right %{y:.0f}% (n=%{customdata})<extra></extra>"))
    fig = go.Figure(data=traces)
    _style(fig, "Calibration: when OutageLens says X %, is it right X % of the time?", height=380)
    fig.update_xaxes(title="Top-1 confidence (%)", range=[0, 102], ticksuffix="%")
    fig.update_yaxes(title="Observed Top-1 accuracy (%)", range=[0, 102], ticksuffix="%")
    return fig


def training_chart(history: list[dict]) -> go.Figure:
    h = pd.DataFrame(history)
    traces = [
        go.Scatter(x=h["epoch"], y=h["train_loss"], name="Train loss", line=dict(color=COLORS["violet"])),
        go.Scatter(x=h["epoch"], y=h["val_top1"] * 100, name="Validation Top-1", yaxis="y2",
                   line=dict(color=COLORS["orange"])),
        go.Scatter(x=h["epoch"], y=h["val_top3"] * 100, name="Validation Top-3", yaxis="y2",
                   line=dict(color=COLORS["orange"], dash="dash")),
    ]
    fig = go.Figure(data=traces)
    _style(fig, "GNN training curve", height=300)
    fig.update_layout(xaxis=dict(title="Epoch"), yaxis=dict(title="Cross-entropy"),
                      yaxis2=dict(title="Accuracy (%)", overlaying="y", side="right", range=[0, 102],
                                  showgrid=False, color="#cbd5e1"))
    return fig


def candidates_chart(feeder: Feeder, rows: list[dict]) -> go.Figure:
    labels = [r["section"] for r in rows][::-1]
    traces: list = []
    series = [("OutageLens", "outagelens", COLORS["orange"]), ("Bayesian", "bayes", COLORS["sky"]),
              ("GNN", "gnn", COLORS["violet"]), ("Naive baseline", "baseline", COLORS["muted"])]
    for name, key, color in series:
        traces.append(go.Bar(y=labels, x=[r[key] * 100 for r in rows][::-1], name=name, orientation="h",
                             marker_color=color, hovertemplate=f"{name}: %{{x:.1f}}%<extra></extra>"))
    fig = go.Figure(data=traces)
    _style(fig, "Top-5 candidate sections: probability by model", height=330, barmode="group", bargap=0.25)
    fig.update_xaxes(title="P(fault) / vote share (%)", ticksuffix="%", rangemode="tozero")
    fig.update_layout(legend=dict(y=-0.3))
    return fig


def attribution_chart(explanations: list[dict]) -> go.Figure:
    keys = ["Topological meter evidence", "Voltage physics match", "Feeder-head load drop", "Exposure prior"]
    colors = [COLORS["emerald"], COLORS["sky"], COLORS["orange"], COLORS["grey"]]
    labels = [f"#{i + 1} {ex['label'].split(' (')[0]}" for i, ex in enumerate(explanations)][::-1]
    traces: list = []
    for k, c in zip(keys, colors):
        traces.append(go.Bar(y=labels, x=[ex["shares"][k] for ex in explanations][::-1], name=k, orientation="h",
                             marker_color=c, hovertemplate=f"{k}: %{{x:.0f}}%<extra></extra>"))
    fig = go.Figure(data=traces)
    _style(fig, "Evidence attribution (share of positive log-evidence margin)", height=250, barmode="stack")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_layout(legend=dict(y=-0.35))
    return fig


def voltage_profile_chart(feeder: Feeder, sc: Scenario, show_truth: bool, dv_expected: np.ndarray | None,
                          candidate_label: str) -> go.Figure:
    traces: list = []  # top panel uses axes x/y, bottom panel x2/y2 (manual layout avoids make_subplots cost)
    dist = np.array([feeder.dist_km[b] for b in feeder.buses])

    def tree_lines(v: np.ndarray, mask: np.ndarray):
        xs, ys = [], []
        for e in feeder.edges:
            i, j = feeder.bus_index[e.u], feeder.bus_index[e.v]
            if mask[i] and mask[j]:
                xs += [dist[i], dist[j], None]
                ys += [v[i], v[j], None]
        return xs, ys

    all_on = np.ones(feeder.n_bus, dtype=bool)
    x, y = tree_lines(sc.v_pre, all_on)
    traces.append(go.Scatter(x=x, y=y, mode="lines", name="Pre-outage (true)", line=dict(color=COLORS["slate"], width=2)))
    if show_truth:
        x, y = tree_lines(sc.v_post, sc.energized)
        traces.append(go.Scatter(x=x, y=y, mode="lines", name="Post-outage (true, energized part)",
                                 line=dict(color=COLORS["orange"], width=2.5)))
        dead = ~sc.energized
        traces.append(go.Scatter(x=dist[dead], y=np.full(dead.sum(), 0.935), mode="markers",
                                 name="De-energized (0 p.u.)", marker=dict(symbol="x", color=COLORS["crimson"], size=9),
                                 text=np.array(feeder.buses)[dead], hovertemplate="Bus %{text}: 0 p.u.<extra></extra>"))
    s = np.flatnonzero(sc.sensor)
    names = np.array(feeder.buses)
    traces.append(go.Scatter(x=dist[s], y=sc.v_pre_obs[s], mode="markers", name="Sensor reading (pre)",
                             marker=dict(symbol="circle-open", size=12, line=dict(width=2, color=COLORS["sky"])),
                             text=names[s], hovertemplate="Bus %{text}: %{y:.4f} p.u.<extra>pre</extra>"))
    live = s[sc.v_post_obs[s] > 0.3]
    traces.append(go.Scatter(x=dist[live], y=sc.v_post_obs[live], mode="markers", name="Sensor reading (post)",
                             marker=dict(symbol="circle", size=10, color=COLORS["sky"]),
                             text=names[live], hovertemplate="Bus %{text}: %{y:.4f} p.u.<extra>post</extra>"))
    masked = np.flatnonzero((sc.meter == MISSING) & sc.has_meter)
    traces.append(go.Scatter(x=dist[masked], y=np.full(masked.size, 0.948), mode="markers+text",
                             name="Silent meter (?)", text=["?"] * masked.size, textposition="middle center",
                             textfont=dict(color="#0b1120", size=9),
                             marker=dict(symbol="circle", size=13, color="#64748b"),
                             customdata=names[masked], hovertemplate="Bus %{customdata}: meter silent<extra></extra>"))

    non_src = np.array([i for i in s if i != feeder.bus_index[feeder.source]], dtype=int)
    sensor_live = non_src[sc.v_post_obs[non_src] > 0.3]
    sensor_dead = non_src[sc.v_post_obs[non_src] <= 0.3]
    bar_w = max(feeder.total_km / 120, 0.02)
    if sensor_live.size:
        dv = sc.v_post_obs[sensor_live] - sc.v_pre_obs[sensor_live]
        traces.append(go.Bar(x=dist[sensor_live], y=dv, name="ΔV observed (energized sensors)",
                             marker_color=COLORS["sky"], xaxis="x2", yaxis="y2", width=bar_w, text=names[sensor_live],
                             textposition="none",
                             hovertemplate="Bus %{text}: ΔV %{y:+.4f} p.u.<extra></extra>"))
        if dv_expected is not None:
            traces.append(go.Scatter(x=dist[sensor_live], y=dv_expected[sensor_live],
                                     mode="markers", name=f"ΔV predicted if {candidate_label} opened",
                                     xaxis="x2", yaxis="y2",
                                     marker=dict(symbol="line-ew-open", size=22, line=dict(width=3, color=COLORS["orange"])),
                                     text=names[sensor_live],
                                     hovertemplate="Bus %{text}: predicted %{y:+.4f} p.u.<extra></extra>"))
    if sensor_dead.size:
        traces.append(go.Scatter(x=dist[sensor_dead], y=np.zeros(sensor_dead.size), mode="markers+text",
                                 name="Sensor reads ~0 p.u. (cut off)", xaxis="x2", yaxis="y2",
                                 marker=dict(symbol="x", size=13, color=COLORS["crimson"], line=dict(width=2)),
                                 text=names[sensor_dead], textposition="top center",
                                 textfont=dict(color=COLORS["crimson"], size=10),
                                 hovertemplate="Bus %{text}: de-energized<extra></extra>"))

    fig = go.Figure(data=traces)
    _style(fig, "", height=620)
    title_font = dict(size=13, color="#e2e8f0")
    fig.update_layout(
        legend=dict(y=-0.12),
        xaxis=dict(anchor="y", showticklabels=False, **AXIS),
        yaxis=dict(domain=[0.46, 0.94], title="|V| (p.u.)",
                   range=[0.925, max(1.1, float(np.nanmax(np.r_[sc.v_post, sc.v_pre_obs, sc.v_post_obs])) + 0.01)],
                   **AXIS),
        xaxis2=dict(anchor="y2", matches="x", title="Electrical distance from substation (km)", **AXIS),
        yaxis2=dict(domain=[0.0, 0.34], title="ΔV (p.u.)", **AXIS),
        annotations=[
            dict(text="Feeder voltage profile: pre-outage vs post-outage", x=0.5, y=1.0, xref="paper",
                 yref="paper", showarrow=False, font=title_font),
            dict(text="Voltage shift ΔV at monitored buses: observed vs physics prediction", x=0.5, y=0.405,
                 xref="paper", yref="paper", showarrow=False, font=title_font),
        ],
    )
    return fig
