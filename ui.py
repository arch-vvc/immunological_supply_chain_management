"""Report styling for the dashboard: one light clinical system, nine documents.

Tokens: paper, ink, rule, muted, and one red that means out of range or disruption.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

LIGHT = dict(PAPER="#F7F6F2", INK="#16181D", RULE="#D9D6CE", MUTED="#6B6A66", RED="#C42B1C", SLATE="#5B6B7A", MIST="#A9B2B9", RAISED="#EFEDE6")
DARK  = dict(PAPER="#0E1012", INK="#E8E6E1", RULE="#2A2D31", MUTED="#8D8B86", RED="#FF5A4A", SLATE="#8FA3B5", MIST="#4A5560", RAISED="#171A1D")
PAPER, INK, RULE, MUTED, RED, SLATE, MIST, RAISED = LIGHT.values()
DARK_MODE = False
MONO = "'IBM Plex Mono', ui-monospace, monospace"
HEAD = "'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif"
BODY = "'IBM Plex Sans', system-ui, sans-serif"


def set_mode(dark):
    """Switch palette. Call once per script run, before inject()."""
    global PAPER, INK, RULE, MUTED, RED, SLATE, MIST, RAISED, DARK_MODE
    DARK_MODE = bool(dark)
    PAPER, INK, RULE, MUTED, RED, SLATE, MIST, RAISED = (DARK if dark else LIGHT).values()


def css():
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500&family=IBM+Plex+Sans:wght@400;500&display=swap');
:root {{ color-scheme: {'dark' if DARK_MODE else 'light'}; }}
html, body, [class*="css"], .stApp {{ font-family: {BODY}; color: {INK}; background: {PAPER}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] p, .stApp li,
.stApp label, .stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stAlert"] * {{ color: {INK}; }}
.stApp table, .stApp th, .stApp td {{ color: {INK}; border-color: {RULE}; }}
.stApp [data-baseweb="select"] > div, .stApp input, .stApp textarea {{ background: {RAISED}; color: {INK}; border-color: {RULE}; }}
[data-testid="stAlert"] {{ background: {RAISED}; border: 0.5px solid {RULE}; }}
.block-container {{ padding: 1.2rem 2.2rem 2rem; max-width: 100%; }}
[data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stToolbarActions"] {{ display: none !important; }}
[data-testid="stHeader"] {{ height: 2.4rem; }}
[data-testid="stElementContainer"]:has(input[aria-label="Dark mode"]) {{ position: fixed; top: 2.6rem; right: 1.3rem; z-index: 1000; width: auto; background: {PAPER}; padding: .15rem .4rem; }}
label[data-baseweb="checkbox"] p {{ white-space: nowrap; }}
h1, h2, h3, h4, h5 {{ font-family: {HEAD} !important; font-weight: 500 !important; color: {INK} !important; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.9rem !important; }}
h4 {{ margin-top: 2rem !important; padding-top: .6rem; border-top: 1px solid {INK}; font-size: 1.05rem !important; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}
.specimen {{ font-family: {MONO}; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
  color: {MUTED}; border-bottom: 1px solid {INK}; padding-bottom: .4rem; margin-bottom: .9rem; display:flex; justify-content:space-between; }}
.verdict {{ font-family: {HEAD}; font-weight: 500; font-size: 1.55rem; line-height: 1.25; margin: .2rem 0 .5rem; max-width: 52rem; color: {INK}; }}
.blurb {{ color: {MUTED}; font-size: .92rem; margin: 0 0 1.3rem; max-width: 52rem; }}
.readout {{ border-top: 0.5px solid {RULE}; padding: .55rem 0 .7rem; min-height: 4.6rem; }}
.readout .label {{ font-size: .72rem; color: {MUTED}; letter-spacing: .04em; }}
.readout .value {{ font-family: {MONO}; font-size: 1.45rem; font-weight: 500; line-height: 1.3; color: {INK}; }}
.readout .value.flag {{ color: {RED}; }}
.readout .sub {{ font-size: .72rem; color: {MUTED}; margin-top: .1rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {INK}; }}
.stTabs [data-baseweb="tab"] {{ font-family: {HEAD}; font-size: .95rem; padding: .5rem 0; background: none; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {RED}; height: 2px; }}
.stButton > button {{ border-radius: 4px; border: 1px solid {INK}; font-family: {HEAD}; background: transparent; color: {INK}; }}
.stButton > button[kind="primary"] {{ background: {INK}; color: {PAPER}; border-color: {INK}; }}
.stButton > button:hover {{ border-color: {RED}; color: {RED}; }}
.stButton > button:focus-visible {{ outline: 2px solid {RED}; outline-offset: 2px; }}
[data-testid="stExpander"] {{ border: none; border-top: 0.5px solid {RULE}; border-radius: 0; }}
[data-testid="stDataFrame"] {{ border: 0.5px solid {RULE}; {'filter: invert(1) hue-rotate(180deg);' if DARK_MODE else ''} }}
</style>
"""


def inject():
    st.markdown(css(), unsafe_allow_html=True)


def specimen(left, right=""):
    st.markdown(f'<div class="specimen"><span>{left}</span><span>{right}</span></div>', unsafe_allow_html=True)


def blurb(text):
    st.markdown(f'<div class="blurb">{text}</div>', unsafe_allow_html=True)


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
