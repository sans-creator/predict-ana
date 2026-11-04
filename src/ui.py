"""Visual design helpers for the Streamlit app: CSS, cards and chart styling."""
from contextlib import contextmanager

import streamlit as st

# ---------- Design tokens ----------
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GRID = "rgba(148, 163, 184, 0.10)"
ACCENT = "#818cf8"
HISTORY = "#64748b"
POSITIVE = "#34d399"
NEGATIVE = "#fb7185"

MODEL_COLORS = {
    "Actual": TEXT,
    "SARIMA": "#f59e0b",
    "Exponential Smoothing": "#f472b6",
    "Linear Regression": "#38bdf8",
    "Random Forest": "#34d399",
    "Seasonal Naive": "#94a3b8",
}
# Ordered palette for categorical charts (store types, years, families)
PALETTE = ["#818cf8", "#38bdf8", "#34d399", "#f59e0b", "#f472b6", "#a78bfa", "#94a3b8"]

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px;}
header[data-testid="stHeader"] {background: transparent;}

/* page header */
.page-head {margin-bottom: 1.4rem;}
.page-head .eyebrow {color: #818cf8; font-size: .75rem; font-weight: 600;
    letter-spacing: .12em; text-transform: uppercase;}
.page-head h1 {font-size: 2rem; font-weight: 700; margin: .25rem 0 .35rem 0; padding: 0;
    letter-spacing: -.02em;}
.page-head p {color: #94a3b8; font-size: .98rem; margin: 0; max-width: 760px;}

/* glass surfaces: KPI cards and bordered containers whose key starts with "card" */
.kpi, [class*="st-key-card"] {
    background: linear-gradient(180deg, rgba(30, 41, 59, .55) 0%, rgba(15, 23, 42, .55) 100%);
    border: 1px solid rgba(148, 163, 184, .13) !important; border-radius: 14px;
    backdrop-filter: blur(6px);
    transition: border-color .2s ease, transform .2s ease;
}
[class*="st-key-card"] {padding: 1.05rem 1.15rem 0.6rem 1.15rem;}
.kpi {padding: 1rem 1.15rem; height: 100%;}
.kpi:hover {border-color: rgba(129, 140, 248, .45) !important; transform: translateY(-1px);}
.kpi .label {color: #94a3b8; font-size: .74rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase;}
.kpi .value {color: #f8fafc; font-size: 1.65rem; font-weight: 700; margin-top: .3rem;
    letter-spacing: -.02em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.kpi .sub {color: #94a3b8; font-size: .82rem; margin-top: .15rem;}
.kpi.highlight {border-color: rgba(52, 211, 153, .55) !important;
    background: linear-gradient(180deg, rgba(16, 185, 129, .12) 0%, rgba(15, 23, 42, .55) 100%);}

/* card titles */
[data-testid="stMarkdownContainer"] p.card-title {font-size: 1rem; font-weight: 600; color: #f1f5f9;
    margin: 0;}
[data-testid="stMarkdownContainer"] p.card-caption {font-size: .85rem; line-height: 1.45; color: #94a3b8;
    margin: .15rem 0 .5rem 0;}
[data-testid="stMarkdownContainer"] p.card-caption b {color: #cbd5e1; font-weight: 600;}

/* small elements */
.badge {display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .74rem;
    font-weight: 600; background: rgba(129, 140, 248, .14); color: #a5b4fc;
    border: 1px solid rgba(129, 140, 248, .3);}
.note {color: #94a3b8; font-size: .84rem; border-left: 3px solid rgba(129, 140, 248, .6);
    padding: .2rem 0 .2rem .8rem; margin: .4rem 0 .6rem 0;}
.brand-sub {color: #94a3b8; font-size: .8rem; margin-top: .2rem;}
ul.clean {padding-left: 1.1rem; margin: .2rem 0 .4rem 0; color: #cbd5e1; font-size: .92rem;}
ul.clean li {margin-bottom: .35rem;}

/* tabs */
.stTabs [data-baseweb="tab-list"] {gap: .4rem;}
.stTabs [data-baseweb="tab"] {font-weight: 600; padding: .5rem .9rem;}

@media (max-width: 640px) {
    .page-head h1 {font-size: 1.5rem;}
    .kpi .value {font-size: 1.3rem;}
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow, title, subtitle):
    st.markdown(f'<div class="page-head"><div class="eyebrow">{eyebrow}</div>'
                f'<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, label, value, sub="", highlight=False):
    cls = "kpi highlight" if highlight else "kpi"
    col.markdown(f'<div class="{cls}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


@contextmanager
def card(key, title=None, caption=None):
    """Bordered glass container with an optional title and caption."""
    with st.container(border=True, key=f"card_{key}"):
        if title:
            st.markdown(f'<p class="card-title">{title}</p>', unsafe_allow_html=True)
        if caption:
            st.markdown(f'<p class="card-caption">{caption}</p>', unsafe_allow_html=True)
        yield


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def human(x, decimals=1):
    """1234567 -> '1.2M'."""
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= size:
            return f"{x / size:,.{decimals}f}{unit}"
    return f"{x:,.0f}"


def style(fig, height=340, legend=True, hover="x unified", left=72, right=16):
    """Apply the dashboard look. Margins are explicit because automargin does
    not reliably reserve room for tick labels inside Streamlit."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=MUTED, size=12),
        height=height, margin=dict(l=left, r=right, t=10, b=44),
        showlegend=legend, hovermode=hover,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT), title_text=""),
        hoverlabel=dict(bgcolor="#0f172a", bordercolor="#334155", font=dict(color=TEXT, size=12)),
        bargap=0.25,
    )
    axes = dict(gridcolor=GRID, zeroline=False, linecolor=GRID,
                title_font=dict(size=12), tickfont=dict(color=MUTED))
    fig.update_xaxes(**axes)
    fig.update_yaxes(**axes)
    return fig


def chart(fig, key=None):
    st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False, "displaylogo": False})
