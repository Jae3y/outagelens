"""Interactive Plotly rendering of the feeder with the fault-probability heatmap."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from core.feeder_topology import Feeder
from core.sparsity_simulator import MISSING, Scenario

from .theme import COLORS


def _blend(c1: str, c2: str, t: float) -> str:
    a = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array([int(c2[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    r, g, bl = (a + (b - a) * float(np.clip(t, 0, 1))).astype(int)
    return f"#{r:02x}{g:02x}{bl:02x}"


def _heat(p: float) -> str:
    """Slate -> amber -> crimson as probability rises."""
    t = float(np.sqrt(np.clip(p, 0, 1)))
    if t < 0.5:
        return _blend(COLORS["slate"], COLORS["amber"], t * 2)
    return _blend(COLORS["amber"], COLORS["crimson"], (t - 0.5) * 2)


def network_figure(feeder: Feeder, sc: Scenario, probs: np.ndarray, show_truth: bool,
                   height: int = 640, top_k: int = 3) -> go.Figure:
    traces: list = []
    annotations: list = []
    k = 1.0 if feeder.n_bus <= 15 else 0.7  # shrink glyphs on dense feeders
    xy = feeder.coords
    ranking = np.argsort(-probs, kind="stable")
    top = list(ranking[:top_k])
    dead_zone = feeder.downstream[sc.fault_edge] if show_truth else np.zeros(feeder.n_bus, dtype=bool)

    # Glow underlay for the top candidates.
    for rank, e in enumerate(top):
        edge = feeder.edges[e]
        (x0, y0), (x1, y1) = xy[edge.u], xy[edge.v]
        for width, alpha in ((26, 0.10), (16, 0.22)):
            traces.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color=f"rgba(239,68,68,{alpha * (1 - 0.25 * rank):.3f})",
                                               width=width)))

    # Sections coloured and scaled by posterior probability.
    for e in feeder.edges:
        (x0, y0), (x1, y1) = xy[e.u], xy[e.v]
        p = float(probs[e.idx])
        in_dead = bool(dead_zone[feeder.bus_index[e.u]]) or (show_truth and e.idx == sc.fault_edge)
        width = 2.5 + 11 * np.sqrt(p)
        dash = "dot" if (in_dead and e.idx != sc.fault_edge) else "solid"
        kind = {"transformer": " • transformer", "regulator": " • voltage regulator",
                "switch": " • switch"}.get(e.kind, "")
        hover = (f"<b>{e.label}</b>{kind}<br>P(fault) = {p:.1%}<br>Length {e.length_km:.2f} km"
                 f"<br>Downstream load {feeder.shed_kw[e.idx]:,.0f} kW")
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        traces.append(go.Scatter(x=[x0, xm, x1], y=[y0, ym, y1], mode="lines", showlegend=False,
                                 line=dict(color=_heat(p), width=width, dash=dash),
                                 hovertemplate=hover + "<extra></extra>"))

    # Probability call-outs, fanned away from the feeder's centre so neighbours do not collide.
    cx = float(np.mean([p[0] for p in xy.values()]))
    cy = float(np.mean([p[1] for p in xy.values()]))
    for rank, e in enumerate(top):
        edge = feeder.edges[e]
        (x0, y0), (x1, y1) = xy[edge.u], xy[edge.v]
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        base = np.arctan2(ym - cy, xm - cx) if abs(xm - cx) + abs(ym - cy) > 1e-6 else np.pi / 2
        ang = base + (rank - 1) * 0.9
        dist = 70 + 22 * rank
        annotations.append(dict(x=xm, y=ym, text=f"<b>#{rank + 1}</b> {edge.name} [{probs[e]:.1%}]",
                           showarrow=True, arrowhead=0, arrowcolor="rgba(248,113,113,0.8)",
                           ax=float(dist * np.cos(ang)), ay=float(-dist * np.sin(ang)),
                           font=dict(color="#fff7ed", size=12 if k == 1.0 else 11, family="JetBrains Mono"),
                           bgcolor="rgba(127,29,29,0.88)" if rank == 0 else "rgba(124,45,18,0.85)",
                           bordercolor=COLORS["crimson"] if rank == 0 else COLORS["orange"], borderwidth=1,
                           borderpad=4))

    if show_truth:
        edge = feeder.edges[sc.fault_edge]
        (x0, y0), (x1, y1) = xy[edge.u], xy[edge.v]
        traces.append(go.Scatter(x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], mode="markers", name="Actual fault (ground truth)",
                                 marker=dict(symbol="x-thin", size=24 * k, line=dict(width=4, color="#fde047")),
                                 hovertemplate=f"<b>Actual fault</b><br>{edge.label}<extra></extra>"))

    buses = feeder.buses
    bx = np.array([xy[b][0] for b in buses])
    by = np.array([xy[b][1] for b in buses])
    span_label = 0.0 if k == 1.0 else 0.4
    src = feeder.bus_index[feeder.source]

    def hover_for(i: int) -> str:
        b = buses[i]
        status = {1: "energized ping", 0: "LAST-GASP outage ping", MISSING: "silent / missing"}[int(sc.meter[i])]
        if i == src:
            status = "substation (SCADA)"
        txt = f"<b>Bus {b}</b><br>Meter: {status}<br>Load {feeder.p_kw[i]:,.0f} kW"
        if sc.sensor[i]:
            txt += (f"<br>Voltage sensor: {sc.v_pre_obs[i]:.4f} → {sc.v_post_obs[i]:.4f} p.u."
                    f"<br>ΔV = {sc.v_post_obs[i] - sc.v_pre_obs[i]:+.4f} p.u.")
        if show_truth:
            txt += f"<br>Truth: {'energized' if sc.energized[i] else 'DE-ENERGIZED'}"
        return txt

    hovers = np.array([hover_for(i) for i in range(feeder.n_bus)])

    sens = np.flatnonzero(sc.sensor)
    traces.append(go.Scatter(x=bx[sens], y=by[sens], mode="markers", name="Voltage sensor",
                             marker=dict(symbol="circle-open", size=31 * k, line=dict(width=3, color=COLORS["sky"])),
                             hoverinfo="skip"))

    groups = [
        ("Energized & reporting (S=1)", (sc.meter == 1), "circle", COLORS["emerald"], 15, None),
        ("Outage ping received (S=0)", (sc.meter == 0), "triangle-up", COLORS["amber"], 18, None),
        ("Silent / missing meter (S=?)", (sc.meter == MISSING) & sc.has_meter, "circle-open", "#94a3b8", 15, 2),
    ]
    for name, mask, symbol, color, size, lw in groups:
        idx = np.flatnonzero(mask)
        marker = dict(symbol=symbol, size=size * k, color=color, line=dict(width=lw or 1.2, color=color if lw else "#0b1120"))
        if symbol == "circle-open":
            marker["line"] = dict(width=2, color=color)
        traces.append(go.Scatter(x=bx[idx], y=by[idx], mode="markers", name=name, marker=marker,
                                 hovertemplate="%{customdata}<extra></extra>", customdata=hovers[idx]))

    traces.append(go.Scatter(x=[bx[src]], y=[by[src]], mode="markers", name="Substation",
                             marker=dict(symbol="diamond", size=26 * k, color=COLORS["violet"],
                                         line=dict(width=2, color="#ede9fe")),
                             hovertemplate="%{customdata}<extra></extra>", customdata=[hovers[src]]))

    traces.append(go.Scatter(x=bx, y=by + 0.24 * k + 0.06 * span_label, mode="text", text=buses, showlegend=False, hoverinfo="skip",
                             textfont=dict(color="#cbd5e1", size=11 if k == 1.0 else 9, family="JetBrains Mono")))

    span_x, span_y = bx.max() - bx.min(), by.max() - by.min()
    fig = go.Figure(data=traces)  # one construction: add_trace in a loop is O(n^2) in Plotly
    fig.update_layout(
        annotations=annotations,
        height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#e2e8f0")),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5,
                    font=dict(color="#cbd5e1", size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(visible=False, range=[bx.min() - 0.12 * span_x, bx.max() + 0.12 * span_x], fixedrange=False),
        yaxis=dict(visible=False, range=[by.min() - 0.14 * span_y, by.max() + 0.16 * span_y]),
        dragmode="pan",
    )
    return fig
