"""Control-room styling: CSS, KPI cards, badges and the shared colour palette."""
from __future__ import annotations

import html

import streamlit as st

COLORS = {
    "bg": "#0b1120",
    "panel": "#111a2e",
    "panel2": "#16213a",
    "border": "#24324f",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "slate": "#475569",
    "emerald": "#10b981",
    "amber": "#f59e0b",
    "crimson": "#ef4444",
    "orange": "#f97316",
    "sky": "#38bdf8",
    "violet": "#a78bfa",
    "grey": "#64748b",
}

METHOD_COLORS = {
    "OutageLens Ensemble (Bayes + GNN)": COLORS["orange"],
    "OutageLens Bayesian + Physics": COLORS["sky"],
    "GNN only": COLORS["violet"],
    "Naive LCA Baseline": COLORS["muted"],
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1600px; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0d1528 0%, #0b1120 100%);
    border-right: 1px solid {COLORS['border']}; }}
section[data-testid="stSidebar"] .stButton button {{ width: 100%; }}
.ol-hero {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
    padding: 0.2rem 0 0.9rem 0; border-bottom: 1px solid {COLORS['border']}; margin-bottom: 1rem; }}
.ol-title {{ font-size: 2.05rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1;
    background: linear-gradient(90deg, #fde68a 0%, {COLORS['orange']} 45%, {COLORS['crimson']} 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.ol-sub {{ color: {COLORS['muted']}; font-size: 0.95rem; margin-top: 0.25rem; }}
.ol-badges {{ display:flex; gap:0.5rem; flex-wrap:wrap; }}
.ol-badge {{ font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; padding: 0.28rem 0.6rem;
    border-radius: 999px; border: 1px solid {COLORS['border']}; color: {COLORS['text']};
    background: {COLORS['panel']}; white-space: nowrap; }}
.ol-badge.live::before {{ content: '\\25CF '; color: {COLORS['emerald']}; }}
.ol-badge.warn::before {{ content: '\\25CF '; color: {COLORS['amber']}; }}
.kpi-row {{ display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 0.75rem; margin-bottom: 1rem; }}
@media (max-width: 1100px) {{ .kpi-row {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
.kpi {{ background: linear-gradient(160deg, {COLORS['panel2']} 0%, {COLORS['panel']} 100%);
    border: 1px solid {COLORS['border']}; border-radius: 14px; padding: 0.85rem 1rem 0.8rem 1rem;
    position: relative; overflow: hidden; min-height: 118px; }}
.kpi::after {{ content:''; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--accent); }}
.kpi-label {{ color: {COLORS['muted']}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
    font-weight: 600; }}
.kpi-value {{ color: {COLORS['text']}; font-size: 1.28rem; font-weight: 700; margin-top: 0.35rem;
    line-height: 1.2; }}
.kpi-sub {{ color: {COLORS['muted']}; font-size: 0.8rem; margin-top: 0.3rem; }}
.pill {{ display:inline-block; font-size: 0.7rem; font-weight: 700; padding: 0.12rem 0.5rem; border-radius: 999px;
    margin-top: 0.4rem; letter-spacing: 0.03em; }}
.pill.green {{ background: rgba(16,185,129,0.15); color: {COLORS['emerald']}; border: 1px solid rgba(16,185,129,0.4); }}
.pill.amber {{ background: rgba(245,158,11,0.15); color: {COLORS['amber']}; border: 1px solid rgba(245,158,11,0.4); }}
.pill.red {{ background: rgba(239,68,68,0.15); color: {COLORS['crimson']}; border: 1px solid rgba(239,68,68,0.4); }}
.pill.grey {{ background: rgba(148,163,184,0.12); color: {COLORS['muted']}; border: 1px solid rgba(148,163,184,0.35); }}
.ol-card {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']}; border-radius: 14px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.8rem; }}
.ol-card h4 {{ margin: 0 0 0.5rem 0; font-size: 1rem; }}
.ol-mono {{ font-family: 'JetBrains Mono', monospace; }}
.ol-caption {{ color: {COLORS['muted']}; font-size: 0.8rem; }}
div[data-testid="stTabs"] button p {{ font-size: 0.98rem; font-weight: 600; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(engine_name: str, engine_ok: bool, feeder_name: str, detail: str) -> None:
    cls = "live" if engine_ok else "warn"
    st.markdown(
        f"""
<div class="ol-hero">
  <div>
    <div class="ol-title">OutageLens ⚡</div>
    <div class="ol-sub">Distribution outage localisation under partial observability: topology × voltage
    physics × incomplete smart-meter data</div>
  </div>
  <div class="ol-badges">
    <span class="ol-badge {cls}" title="{html.escape(detail)}">{html.escape(engine_name)}</span>
    <span class="ol-badge">{html.escape(feeder_name)}</span>
    <span class="ol-badge">IEEE PES test feeder data</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", pill: tuple[str, str] | None = None,
             accent: str = COLORS["orange"]) -> str:
    pill_html = f'<div><span class="pill {pill[1]}">{html.escape(pill[0])}</span></div>' if pill else ""
    return (f'<div class="kpi" style="--accent:{accent}"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div>{pill_html}</div>')


def kpi_row(cards: list[str]) -> None:
    st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="ol-card"><h4>{title}</h4>{body_html}</div>', unsafe_allow_html=True)
