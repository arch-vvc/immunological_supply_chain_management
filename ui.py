"""Report styling for the dashboard: one light clinical system, nine documents.

Tokens: paper, ink, rule, muted, and one red that means out of range or disruption.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

PAPER, INK, RULE, MUTED, RED = "#F7F6F2", "#16181D", "#D9D6CE", "#6B6A66", "#C42B1C"
SLATE, MIST = "#5B6B7A", "#A9B2B9"
MONO = "'IBM Plex Mono', ui-monospace, monospace"
HEAD = "'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif"
BODY = "'IBM Plex Sans', system-ui, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"], .stApp {{ font-family: {BODY}; color: {INK}; }}
.block-container {{ padding-top: 2rem; max-width: 1180px; }}
h1, h2, h3, h4, h5 {{ font-family: {HEAD} !important; font-weight: 500 !important; color: {INK} !important; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.9rem !important; }}
h4 {{ margin-top: 2rem !important; padding-top: .6rem; border-top: 1px solid {INK}; font-size: 1.05rem !important; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}
.specimen {{ font-family: {MONO}; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
  color: {MUTED}; border-bottom: 1px solid {INK}; padding-bottom: .4rem; margin-bottom: .9rem; display:flex; justify-content:space-between; }}
.verdict {{ font-family: {HEAD}; font-weight: 500; font-size: 1.55rem; line-height: 1.25; margin: .2rem 0 1.2rem; max-width: 52rem; }}
.readout {{ border-top: 0.5px solid {RULE}; padding: .55rem 0 .7rem; min-height: 4.6rem; }}
.readout .label {{ font-size: .72rem; color: {MUTED}; letter-spacing: .04em; }}
.readout .value {{ font-family: {MONO}; font-size: 1.45rem; font-weight: 500; line-height: 1.3; }}
.readout .value.flag {{ color: {RED}; }}
.readout .sub {{ font-size: .72rem; color: {MUTED}; margin-top: .1rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {INK}; }}
.stTabs [data-baseweb="tab"] {{ font-family: {HEAD}; font-size: .95rem; padding: .5rem 0; background: none; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {RED}; height: 2px; }}
.stButton > button {{ border-radius: 4px; border: 1px solid {INK}; font-family: {HEAD}; }}
.stButton > button[kind="primary"] {{ background: {INK}; color: {PAPER}; border-color: {INK}; }}
.stButton > button:hover {{ border-color: {RED}; color: {RED}; }}
.stButton > button:focus-visible {{ outline: 2px solid {RED}; outline-offset: 2px; }}
[data-testid="stExpander"] {{ border: none; border-top: 0.5px solid {RULE}; border-radius: 0; }}
[data-testid="stDataFrame"] {{ border: 0.5px solid {RULE}; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def specimen(left, right=""):
    st.markdown(f'<div class="specimen"><span>{left}</span><span>{right}</span></div>', unsafe_allow_html=True)


def verdict(text):
    st.markdown(f'<div class="verdict">{text}</div>', unsafe_allow_html=True)


def readout(label, value, sub="", flag=False):
    cls = "value flag" if flag else "value"
    st.markdown(
        f'<div class="readout"><div class="label">{label}</div>'
        f'<div class="{cls}">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True)


def chart_template():
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=12),
        colorway=[INK, SLATE, MIST, RED],
        xaxis=dict(gridcolor=RULE, zerolinecolor=RULE, linecolor=INK, ticks="outside", tickcolor=INK),
        yaxis=dict(gridcolor=RULE, zerolinecolor=RULE, linecolor=INK),
        title=dict(font=dict(family="IBM Plex Sans Condensed, sans-serif", size=14, color=INK)),
        margin=dict(l=40, r=20, t=44, b=36),
        hoverlabel=dict(bgcolor=PAPER, bordercolor=INK, font=dict(family="IBM Plex Mono, monospace", color=INK)),
    )
    pio.templates["report"] = t
    pio.templates.default = "report"


def mpl_style():
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "grid.color": RULE,
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=[INK, SLATE, MIST, RED]),
        "font.family": "sans-serif", "font.size": 10,
    })
