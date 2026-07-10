"""
Immunological Supply Chain — Focused Dashboard
===============================================
PES University Capstone  PW26_RGP_01

5-tab dashboard focused on the real-time immune response system:
  Tab 1 — Supply Chain Graph    : network topology + risk heatmap
  Tab 2 — Live Stream           : real-time transaction feed + anomaly signal
  Tab 3 — Immune Memory         : FAISS historical recall results
  Tab 4 — Immune Response       : Decision Trace per anomaly event
  Tab 5 — AI Agent              : local LLM narration + chat over the live decision trace

Run:
    # Terminal 1 — data stream
    python3 src/stream_simulator.py --interval 2 --disruption 30 --multi

    # Terminal 2 — immune response engine
    python3 src/stream_consumer.py

    # Terminal 3 — (optional, for Tab 5) local LLM
    ollama serve
    ollama pull llama3.2:1b

    # Terminal 4 — this dashboard
    streamlit run immune_app.py
"""

import os, sys, pickle, json, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# pandas 3.x defaults string columns to a PyArrow-backed dtype. This pyarrow
# build (25.0.0) has a race in its compute kernels that segfaults
# (libarrow.dylib) when a sort/compare on a string column runs inside
# Streamlit's multi-threaded runtime. Reverting to legacy numpy object-dtype
# strings avoids the arrow compute path entirely. Must be set before any
# pd.read_csv() call anywhere in the process.
pd.set_option("future.infer_string", False)

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from ai_agent import ImmuneAIAgent

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Immunological Supply Chain",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(BASE, "output")
MODELS = os.path.join(BASE, "models")
STREAM = os.path.join(BASE, "data", "stream")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    .metric-card {
        background: #0f1923;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { color: #7a9cc0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-card .value { color: #e8f4ff; font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }
    .metric-card .sub   { color: #4a6a8a; font-size: 0.72rem; margin-top: 0.15rem; }
    h1, h2, h3 { color: #c8d8ff !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 500; }
    .thinking-box {
        background: #0a1520;
        border-left: 3px solid #1e6fbf;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #b0c8e0;
        margin: 0.4rem 0;
    }
    .narration-box {
        background: #0a1c14;
        border-left: 3px solid #2fa86a;
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        font-size: 0.92rem;
        line-height: 1.5;
        color: #cdeedd;
        margin: 0.5rem 0 1rem 0;
    }
    .agent-status {
        display: inline-block;
        font-size: 0.78rem;
        padding: 0.25rem 0.7rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }
    .agent-status.on  { background: #0d2b1c; color: #4ade80; border: 1px solid #1e6b3f; }
    .agent-status.off { background: #2b220d; color: #eab308; border: 1px solid #6b551e; }
    .src-badge {
        display: inline-block;
        font-size: 0.7rem;
        padding: 0.15rem 0.6rem;
        border-radius: 10px;
        margin-top: 0.35rem;
    }
    .src-badge.verified { background: #0d2b1c; color: #4ade80; border: 1px solid #1e6b3f; }
    .src-badge.ai       { background: #14213d; color: #7aa2f7; border: 1px solid #253a63; }
    .src-badge.template { background: #2b220d; color: #eab308; border: 1px solid #6b551e; }
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def render_source_badge(source, tool_used=None):
    """Confidence cue distinguishing deterministic tool output from the
    model's own composed text."""
    if source == "tool":
        st.markdown(f'<span class="src-badge verified">Verified — computed via {tool_used}()</span>', unsafe_allow_html=True)
    elif source == "llm":
        st.markdown('<span class="src-badge ai">AI-generated — cross-check against the metrics above</span>', unsafe_allow_html=True)
    elif source == "template":
        st.markdown('<span class="src-badge template">Template fallback — no local model running</span>', unsafe_allow_html=True)


def render_execution_trace(trace):
    """Execution log for a chat answer — a record of which steps actually
    ran and how long each took, not fabricated chain-of-thought."""
    if not trace:
        return
    total_ms = sum(s.get("duration_ms", 0) for s in trace)
    with st.expander(f"Execution Trace — {len(trace)} step(s), {total_ms}ms total"):
        for i, step in enumerate(trace, 1):
            st.markdown(
                f"**{i}. {step.get('step', '?')}** · {step.get('duration_ms', 0)}ms  \n"
                f"<span style='color:#8aa; font-size:0.85rem'>{step.get('detail', '')}</span>",
                unsafe_allow_html=True,
            )


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def load_graph():
    p = os.path.join(MODELS, "supplychain_graph.pkl")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_resource
def load_graph_risk():
    # cache_resource, not cache_data: st.cache_data hashes/serializes the
    # returned DataFrame via PyArrow, which segfaults in this environment
    # (libarrow.dylib crash on macOS/arm64). cache_resource just keeps the
    # object in memory without trying to serialize it.
    p = os.path.join(OUT, "graph_risk_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_resource
def load_memory_retrieval():
    p = os.path.join(OUT, "memory_retrieval.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_memory_report():
    p = os.path.join(OUT, "memory_report.txt")
    return open(p).read() if os.path.exists(p) else ""

def load_live_results():
    p = os.path.join(STREAM, "live_results.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def load_immune_decisions():
    p = os.path.join(STREAM, "immune_decisions.jsonl")
    if not os.path.exists(p):
        return []
    decisions = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        decisions.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return decisions

@st.cache_resource
def load_anomalies():
    p = os.path.join(OUT, "anomalies.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_resource
def load_agent():
    return ImmuneAIAgent()


# ── Agent tools ───────────────────────────────────────────────────────────────
# Real functions the chat agent can choose to call instead of only narrating a
# fixed cascade. Each docstring's first line is shown to the model as the
# tool's description when it decides whether to use it.

def _find_entity(name, known):
    name_l = str(name).strip().lower()
    for e in known:
        if str(e).lower() == name_l:
            return e
    partial = [e for e in known if name_l in str(e).lower()]
    return partial[0] if partial else None


def lookup_entity_risk(entity: str) -> dict:
    """Use this ONLY to report a known entity's current risk score or risk level (a lookup, not a simulation). Not for "what if X fails" questions. Returns composite risk score, graph centrality, and transaction volume for a named supply chain entity."""
    risk_df = load_graph_risk()
    if risk_df.empty:
        return {"found": False, "reason": "risk data not available"}
    match = _find_entity(entity, risk_df["entity"].tolist())
    if not match:
        return {"found": False, "queried": entity, "reason": "no matching entity in the graph"}
    row = risk_df[risk_df["entity"] == match].iloc[0]
    return {
        "found": True,
        "entity": match,
        "type": row.get("type"),
        "composite_risk_0_to_1": round(float(row.get("composite_risk", 0)), 3),
        "betweenness_centrality": round(float(row.get("betweenness_centrality", 0)), 5),
        "in_degree": int(row.get("in_degree", 0)),
        "out_degree": int(row.get("out_degree", 0)),
        "total_transactions": int(row.get("total_transactions", 0)),
    }


def simulate_node_failure(node: str) -> dict:
    """Use this for any "what happens if X fails / goes down / is removed / stops operating" question about one or more supply chain nodes (comma-separate multiple names, e.g. "A, B", to simulate simultaneous failures). Actually removes the node(s) from the network and checks whether downstream dependents can still be reached through an alternate route."""
    import networkx as nx
    G = load_graph()
    if G is None:
        return {"found": False, "reason": "supply chain graph not available"}

    queried_parts = [p.strip() for p in str(node).split(",") if p.strip()]
    matched, not_found = [], []
    for part in queried_parts:
        m = _find_entity(part, list(G.nodes()))
        (matched if m else not_found).append(m or part)

    if not matched:
        return {"found": False, "queried": node, "reason": "no matching node in the graph"}

    preds, succs = set(), set()
    for m in matched:
        preds.update(G.predecessors(m))
        succs.update(G.successors(m))
    preds -= set(matched)
    succs -= set(matched)

    G2 = G.copy()
    G2.remove_nodes_from(matched)
    succs_list = list(succs)
    preds_list = list(preds)
    reroutable, unreachable = [], []
    for s in succs_list[:15]:
        ok = s in G2 and any(p in G2 and nx.has_path(G2, p, s) for p in preds_list[:5])
        (reroutable if ok else unreachable).append(s)

    return {
        "found": True,
        "node": ", ".join(matched),
        "nodes_failed": matched,
        "not_found": not_found,
        "upstream_suppliers": len(preds_list),
        "downstream_dependents": len(succs_list),
        "reroutable_dependents": len(reroutable),
        "unreachable_dependents": len(unreachable),
        "sample_unreachable": unreachable[:5],
        "checked_sample_of": min(len(succs_list), 15),
    }


def compare_recent_incidents() -> dict:
    """Compare the two most recently recorded immune response incidents — their severity, risk rating, and recovery time. Takes no arguments."""
    decisions = [d for d in load_immune_decisions() if d.get("event_type") != "cytokine_storm"]
    if len(decisions) < 2:
        return {"found": False, "reason": "fewer than 2 recorded incidents so far"}

    def summarize(d):
        v = d.get("verdict", {})
        return {
            "timestamp": d.get("timestamp", "")[:19],
            "distributor": d.get("distributor"),
            "severity": v.get("severity"),
            "risk_rating_out_of_10": ImmuneAIAgent._risk_rating_10(d.get("z_score", 0)),
            "recovery_days": v.get("recovery_estimate_days"),
        }

    return {"found": True, "previous": summarize(decisions[-2]), "most_recent": summarize(decisions[-1])}


AGENT_TOOLS = {
    "lookup_entity_risk": lookup_entity_risk,
    "simulate_node_failure": simulate_node_failure,
    "compare_recent_incidents": compare_recent_incidents,
}


# ── Header ───────────────────────────────────────────────────────────────────
st.title("🧬 Immunological Supply Chain")
st.caption("Self-Healing Supply Chains with AI Digital Antibodies  ·  PES University  ·  PW26_RGP_01")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Supply Chain Graph",
    "Live Stream",
    "Immune Memory",
    "Immune Response",
    "AI Agent",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUPPLY CHAIN GRAPH
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Supply Chain Network Topology")
    st.caption("2,131 nodes · 23,264 edges · 20 manufacturers · 20 distributors · 2,091 retailers")

    risk_df = load_graph_risk()

    if risk_df.empty:
        st.warning("Run Stage 2 to generate graph risk scores: `python3 main.py --only 2`")
    else:
        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Total Nodes", f"{len(risk_df):,}")
        with c2:
            high_risk = int((risk_df["composite_risk"] > 0.7).sum()) if "composite_risk" in risk_df.columns else 0
            metric_card("High Risk Nodes", str(high_risk), "composite risk > 0.7")
        with c3:
            top_node = risk_df.loc[risk_df["composite_risk"].idxmax(), "entity"] if "composite_risk" in risk_df.columns else "—"
            metric_card("Highest Risk Node", top_node[:22])
        with c4:
            avg_risk = f"{risk_df['composite_risk'].mean():.3f}" if "composite_risk" in risk_df.columns else "—"
            metric_card("Avg Composite Risk", avg_risk)

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Risk Distribution by Node Type")
            if "type" in risk_df.columns and "composite_risk" in risk_df.columns:
                fig_box = px.box(
                    risk_df, x="type", y="composite_risk",
                    color="type",
                    color_discrete_map={
                        "manufacturer": "#4fc3f7",
                        "distributor":  "#ff8a65",
                        "retailer":     "#81c784",
                    },
                    template="plotly_dark",
                )
                fig_box.update_layout(
                    height=320, showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=40),
                    xaxis_title="Node Type", yaxis_title="Composite Risk",
                )
                st.plotly_chart(fig_box, use_container_width=True)

        with col_b:
            st.markdown("#### Top 15 Highest-Risk Nodes")
            if "composite_risk" in risk_df.columns:
                top15 = risk_df.nlargest(15, "composite_risk")[["entity", "type", "composite_risk", "out_volume", "out_degree"]]
                top15.columns = ["Entity", "Type", "Risk Score", "Volume", "Connections"]
                st.dataframe(top15.reset_index(drop=True), use_container_width=True, height=320)

        st.markdown("---")
        st.markdown("#### Centrality & PageRank")
        if "betweenness_centrality" in risk_df.columns and "pagerank" in risk_df.columns:
            fig_scatter = px.scatter(
                risk_df.sample(min(500, len(risk_df))),
                x="betweenness_centrality", y="pagerank",
                color="composite_risk",
                color_continuous_scale="RdYlGn_r",
                hover_data=["entity", "type"],
                size_max=10,
                template="plotly_dark",
                labels={
                    "betweenness_centrality": "Betweenness Centrality",
                    "pagerank": "PageRank",
                    "composite_risk": "Risk",
                },
            )
            fig_scatter.update_layout(
                height=350, margin=dict(l=20, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Each point is a supply chain node. High betweenness + high PageRank = critical single point of failure.")

        # Static centrality figure if available
        fig_path = os.path.join(OUT, "figures", "fig_centrality.png")
        if os.path.exists(fig_path):
            st.markdown("---")
            st.markdown("#### Network Centrality Visualisation")
            st.image(fig_path, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE STREAM
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🔴 Live Stream Monitor — Real-Time Transaction Feed")

    # This fragment re-runs on its own every 3s WITHOUT blocking the rest of
    # the app or doing a full-page st.rerun(). The previous implementation
    # called time.sleep(3) + st.rerun() directly in the tab body — since all
    # tab bodies execute on every Streamlit script run regardless of which
    # tab is visible, that froze the ENTIRE app for 3s on every single rerun,
    # forever, for every session (busy-loop: render -> freeze -> full
    # restart -> render -> freeze -> ...). That's what looked like "keeps
    # refreshing / never shows up."
    @st.fragment(run_every="3s")
    def _live_stream_fragment():
        LIVE_RESULTS_PATH    = os.path.join(STREAM, "live_results.csv")
        DISRUPTION_FLAG_PATH = os.path.join(STREAM, "disruption_active.flag")

        if not os.path.exists(LIVE_RESULTS_PATH):
            st.info("Stream is not running yet. Start it with these two commands in separate terminals:")
            st.code(
                "# Terminal 1 — emit a row every 2s, inject disruptions every 60s\n"
                "python3 src/stream_simulator.py --interval 2 --disruption 30 --multi\n\n"
                "# Terminal 2 — consume with full immune response\n"
                "python3 src/stream_consumer.py",
                language="bash"
            )
            return

        df_live = load_live_results()

        if df_live.empty:
            st.warning("Live results file exists but is empty — consumer may still be starting up.")
        else:
            # Active disruption banner
            if os.path.exists(DISRUPTION_FLAG_PATH):
                flag_text = open(DISRUPTION_FLAG_PATH).read()
                st.error(f"⚡ ACTIVE DISRUPTION DETECTED\n\n{flag_text}")

            # Metrics
            total       = len(df_live)
            n_anomaly   = int(df_live["is_anomaly"].sum())        if "is_anomaly"          in df_live.columns else 0
            n_disruption= int(df_live["disruption_injected"].sum()) if "disruption_injected" in df_live.columns else 0
            n_rerouted  = int((df_live["alternate_route"].astype(str).str.strip() != "").sum()) if "alternate_route" in df_live.columns else 0
            anomaly_rate= f"{n_anomaly/total*100:.1f}%" if total else "0%"

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: metric_card("Rows Processed",     f"{total:,}")
            with c2: metric_card("Anomalies Detected",  str(n_anomaly),    anomaly_rate)
            with c3: metric_card("Disruptions Injected",str(n_disruption))
            with c4: metric_card("Routes Rerouted",     str(n_rerouted))
            with c5:
                ppo_count = int((df_live["routing_method"] == "PPO").sum()) if "routing_method" in df_live.columns else 0
                metric_card("PPO Decisions", str(ppo_count), "vs Dijkstra fallback")

            st.markdown("---")

            # Z-score signal chart
            if "z_score" in df_live.columns and "row_index" in df_live.columns:
                st.markdown("#### Z-Score Signal — Anomaly Detection in Real Time")
                fig = go.Figure()
                # Normal points
                normal = df_live[df_live["is_anomaly"] == 0] if "is_anomaly" in df_live.columns else df_live
                anomaly_rows = df_live[df_live["is_anomaly"] == 1] if "is_anomaly" in df_live.columns else pd.DataFrame()

                fig.add_trace(go.Scatter(
                    x=normal["row_index"], y=normal["z_score"],
                    mode="lines", name="Normal",
                    line=dict(color="#4fc3f7", width=1.2),
                ))
                if not anomaly_rows.empty:
                    fig.add_trace(go.Scatter(
                        x=anomaly_rows["row_index"], y=anomaly_rows["z_score"],
                        mode="markers", name="Anomaly",
                        marker=dict(color="#ff5252", size=8, symbol="x"),
                    ))
                fig.add_hline(y=2.5, line_dash="dash", line_color="#ff8a65",
                              annotation_text="Threshold (Z=2.5)", annotation_position="top right")
                fig.update_layout(
                    template="plotly_dark", height=300,
                    margin=dict(l=40, r=20, t=20, b=40),
                    xaxis_title="Transaction Index", yaxis_title="Z-Score",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Recent anomalies table
            st.markdown("#### Recent Anomaly Events")
            if "is_anomaly" in df_live.columns:
                df_anom = df_live[df_live["is_anomaly"] == 1].tail(10)
                if df_anom.empty:
                    st.success("✅ No anomalies detected yet — supply chain is healthy.")
                else:
                    show_cols = [c for c in [
                        "timestamp", "manufacturer", "distributor", "retailer",
                        "retailer_state", "quantity", "z_score",
                        "disruption_injected", "routing_method", "routing_note"
                    ] if c in df_anom.columns]
                    st.dataframe(df_anom[show_cols].reset_index(drop=True), use_container_width=True)

            # Last 50 transactions
            st.markdown("#### All Transactions (last 50)")
            show_cols = [c for c in [
                "timestamp", "row_index", "manufacturer", "distributor",
                "retailer", "quantity", "z_score", "is_anomaly", "routing_method"
            ] if c in df_live.columns]
            st.dataframe(df_live[show_cols].tail(50).reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        if st.button("🔄 Refresh stream data now", key="manual_refresh_live_stream"):
            st.cache_data.clear()
            st.rerun()

    _live_stream_fragment()


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — IMMUNE MEMORY
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🧠 Immunological Memory — FAISS Historical Recall")
    st.caption(
        "100,000 historical disruption records indexed as 8-dimensional feature vectors. "
        "When a live anomaly occurs, the engine queries this index for the 3 most similar "
        "past disruptions to inform the response strategy."
    )
    st.info(
        "**Innate vs Adaptive Memory** — The 100K baseline vectors are pre-seeded from "
        "synthetic historical pharmaceutical supply chain patterns (innate memory, analogous to "
        "the immune system's T-cell repertoire developed before first exposure). "
        "Each resolved live event is encoded and appended via **clonal selection**, so the index "
        "grows with real operational experience (adaptive memory). "
        "The longer the system runs, the more weight shifts from innate to adaptive.",
        icon="🧬"
    )

    mem_df  = load_memory_retrieval()
    mem_rpt = load_memory_report()

    if mem_df.empty:
        st.warning("Run Stage 13 to build the FAISS index: `python3 main.py --only 13`")
    else:
        # Summary metrics from report
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Index Size", "100,000", "historical disruptions")
        with c2: metric_card("Vector Dimensions", "8", "features per disruption")
        with c3:
            avg_dist = f"{mem_df['match_distance'].mean():.4f}" if "match_distance" in mem_df.columns else "—"
            metric_card("Avg Match Distance", avg_dist, "lower = closer match")
        with c4:
            avg_rec = f"{mem_df['match_recovery_days'].mean():.1f} days" if "match_recovery_days" in mem_df.columns else "—"
            metric_card("Avg Recovery Estimate", avg_rec)

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Response Type Distribution")
            if "match_response_type" in mem_df.columns:
                resp_counts = mem_df["match_response_type"].value_counts().reset_index()
                resp_counts.columns = ["Response Type", "Count"]
                fig_resp = px.bar(
                    resp_counts, x="Response Type", y="Count",
                    color="Response Type", template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_resp.update_layout(
                    height=300, showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=80),
                )
                st.plotly_chart(fig_resp, use_container_width=True)

        with col_b:
            st.markdown("#### Recovery Days Distribution")
            if "match_recovery_days" in mem_df.columns:
                fig_hist = px.histogram(
                    mem_df, x="match_recovery_days",
                    nbins=40, template="plotly_dark",
                    color_discrete_sequence=["#4fc3f7"],
                )
                fig_hist.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=40),
                    xaxis_title="Recovery Days",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Match Distance vs Recovery Days")
        if "match_distance" in mem_df.columns and "match_recovery_days" in mem_df.columns:
            fig_scat = px.scatter(
                mem_df.sample(min(2000, len(mem_df))),
                x="match_distance", y="match_recovery_days",
                color="match_response_type" if "match_response_type" in mem_df.columns else None,
                template="plotly_dark", opacity=0.5,
                labels={
                    "match_distance": "FAISS Distance (lower = more similar)",
                    "match_recovery_days": "Recovery Days",
                },
            )
            fig_scat.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=40))
            st.plotly_chart(fig_scat, use_container_width=True)
            st.caption("Each point is a retrieved historical match. Closer matches (lower distance) should predict recovery time more accurately.")

        if mem_rpt:
            st.markdown("---")
            with st.expander("📄 Full Memory Report (Stage 13 output)"):
                st.code(mem_rpt)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — IMMUNE RESPONSE
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🦠 Immune Response Engine — Real-Time Decision Trace")
    st.caption(
        "When a disruption is detected, the engine fires 4 parallel response systems: "
        "memory recall · alternate routing · backup supplier · inventory transfer. "
        "Each anomaly event shows its full reasoning trace below."
    )

    IMMUNE_PATH = os.path.join(STREAM, "immune_decisions.jsonl")

    if not os.path.exists(IMMUNE_PATH):
        st.info("No immune response decisions yet. Start the stream to generate responses:")
        st.code(
            "python3 src/stream_simulator.py --interval 2 --disruption 30 --multi\n"
            "python3 src/stream_consumer.py",
            language="bash"
        )
        st.markdown("**Or run a one-shot test:**")
        st.code("python3 main.py --only 16", language="bash")
    else:
        decisions = load_immune_decisions()

        if not decisions:
            st.warning("Decision log is empty. Run the stream consumer to populate it.")
        else:
            # Split storm events from normal disruption events
            storm_events  = [d for d in decisions if d.get("event_type") == "cytokine_storm"]
            normal_events = [d for d in decisions if d.get("event_type") != "cytokine_storm"]

            # Cytokine storm alerts
            if storm_events:
                st.error(f"CYTOKINE STORM — {len(storm_events)} cascade event(s) detected")
                for sd in storm_events:
                    step0  = (sd.get("thinking") or [{}])[0]
                    detail = step0.get("data", {})
                    count  = detail.get("storm_event_count", "?")
                    nodes  = detail.get("affected_distributors", [])
                    repeats= detail.get("repeat_hit_nodes", [])
                    with st.expander(f"Cytokine Storm — {sd.get('timestamp','')[:19]}  |  {count} simultaneous disruptions", expanded=True):
                        st.markdown(f'<div class="thinking-box">{step0.get("reasoning","")}</div>', unsafe_allow_html=True)
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Disruptions in window", count)
                        sc2.metric("Affected nodes", len(nodes))
                        sc3.metric("Repeat-hit (overloaded)", len(repeats))
                        if repeats:
                            st.warning(f"Critical overload on: {', '.join(repeats)}")
                        constituent = detail.get("constituent_events", [])
                        if constituent:
                            st.dataframe(pd.DataFrame(constituent), use_container_width=True, hide_index=True)
                st.markdown("---")

            decisions = normal_events   # scope rest of tab to individual events

            if not decisions:
                st.info("Only cytokine storm events recorded so far.")
            else:
                # Summary metrics
                severities = [d.get("verdict", {}).get("severity", "?") for d in decisions]
                n_critical = severities.count("CRITICAL")
                n_high     = severities.count("HIGH")
                n_mod      = severities.count("MODERATE")
                avg_sig    = sum(d.get("verdict", {}).get("signals_activated", 0) for d in decisions) / len(decisions)
                avg_rec    = [d.get("verdict", {}).get("recovery_estimate_days") for d in decisions]
                avg_rec    = round(sum(x for x in avg_rec if x is not None) / max(1, sum(1 for x in avg_rec if x is not None)), 1)

                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: metric_card("Total Events",       str(len(decisions)))
                with c2: metric_card("Critical",           str(n_critical))
                with c3: metric_card("High",               str(n_high))
                with c4: metric_card("Moderate",           str(n_mod))
                with c5: metric_card("Avg Signals Active", f"{avg_sig:.1f}/4")

                st.markdown("---")

                # ── Recovery Timeline + State Heatmap ──────────────────────
                chart_l, chart_r = st.columns([3, 2])

                with chart_l:
                    tl_rows = []
                    for i, dec in enumerate(decisions):
                        vv = dec.get("verdict", {})
                        rec = vv.get("recovery_estimate_days")
                        if rec is not None:
                            tl_rows.append({
                                "Event": i + 1,
                                "Time": dec.get("timestamp", "")[:19],
                                "Recovery Days": float(rec),
                                "Severity": vv.get("severity", "?"),
                                "Z-Score": float(dec.get("z_score", 0)),
                                "Distributor": dec.get("distributor", "?")[:22],
                            })
                    if tl_rows:
                        tl_df = pd.DataFrame(tl_rows)
                        fig_tl = px.line(
                            tl_df, x="Event", y="Recovery Days",
                            color="Severity",
                            color_discrete_map={
                                "CRITICAL": "#ef5350",
                                "HIGH":     "#ff9800",
                                "MODERATE": "#ffeb3b",
                            },
                            markers=True, template="plotly_dark",
                            title="Recovery Timeline — Estimated Days per Event",
                            hover_data={"Time": True, "Z-Score": ":.2f", "Distributor": True},
                        )
                        fig_tl.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        st.plotly_chart(fig_tl, use_container_width=True)
                    else:
                        st.caption("Recovery timeline will appear once events have verdict data.")

                with chart_r:
                    state_rows = [
                        {"state": dec.get("retailer_state", ""), "severity": dec.get("verdict", {}).get("severity", "?")}
                        for dec in decisions
                        if dec.get("retailer_state", "") not in ("", "XX", None)
                    ]
                    if state_rows:
                        state_df = pd.DataFrame(state_rows)
                        state_counts = state_df.groupby("state").size().reset_index(name="Disruptions")
                        fig_map = px.choropleth(
                            state_counts,
                            locations="state", locationmode="USA-states",
                            color="Disruptions",
                            scope="usa",
                            color_continuous_scale="Reds",
                            template="plotly_dark",
                            title="Disruption Heatmap — Events by State",
                        )
                        fig_map.update_layout(
                            height=300,
                            margin=dict(l=0, r=0, t=40, b=0),
                            geo=dict(bgcolor="rgba(0,0,0,0)"),
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                    else:
                        st.caption("State heatmap will appear once events include retailer_state data.")

                st.markdown("---")

                # Event selector
                event_labels = [
                    f"#{i+1}  {d.get('timestamp','')[:19]}  |  "
                    f"{d.get('verdict',{}).get('severity','?')}  |  "
                    f"{d.get('manufacturer','?')[:18]} -> {d.get('retailer','?')[:18]}"
                    for i, d in enumerate(decisions)
                ]
                sel = st.selectbox(
                    "Select anomaly event to inspect:",
                    range(len(decisions)),
                    format_func=lambda i: event_labels[i],
                    index=len(decisions) - 1,
                )
                d = decisions[sel]
                v = d.get("verdict", {})

                # Event header
                sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡"}.get(v.get("severity"), "⚪")
                st.markdown(f"## {sev_icon} {v.get('severity','?')} Severity Event")

                h1, h2, h3, h4, h5, h6 = st.columns(6)
                h1.markdown(f"**Manufacturer**\n\n`{d.get('manufacturer','?')[:28]}`")
                h2.markdown(f"**Disrupted via**\n\n`{d.get('distributor','?')[:28]}`")
                h3.markdown(f"**Retailer**\n\n`{d.get('retailer','?')[:28]}`")
                h4.metric("Z-Score",          f"{d.get('z_score',0):.2f}")
                h5.metric("Signals Activated",f"{v.get('signals_activated',0)}/4")
                h6.metric("Est. Recovery",    f"{v.get('recovery_estimate_days','?')} days")

                st.markdown("---")

                # Decision Trace
                st.markdown("### 🧠 Decision Trace")

                PHASE_ICONS = {
                    "DETECTION":          "🔍",
                    "MEMORY RECALL":      "🧬",
                    "ALTERNATE ROUTE":    "🗺️",
                    "BACKUP SUPPLIER":    "🏭",
                    "INVENTORY TRANSFER": "📦",
                    "FINAL VERDICT":      "✅",
                    "CYTOKINE STORM":     "⚡",
                }

                for step in d.get("thinking", []):
                    phase  = step.get("phase", "")
                    icon   = PHASE_ICONS.get(phase, "▪️")
                    title  = step.get("title", "")
                    reason = step.get("reasoning", "")
                    data   = step.get("data", {})

                    with st.expander(f"{icon} Step {step.get('step','')} — {phase}: {title}", expanded=True):
                        st.markdown(f'<div class="thinking-box">{reason}</div>', unsafe_allow_html=True)

                        if data:
                            if phase == "MEMORY RECALL" and data.get("matches"):
                                st.markdown("**Closest historical matches:**")
                                mem_rows = [{
                                    "Rank":         m.get("rank"),
                                    "Distance":     m.get("distance"),
                                    "Recovery (d)": m.get("recovery_days"),
                                    "Response":     m.get("response_type"),
                                    "Disruption":   m.get("disruption_type"),
                                } for m in data["matches"]]
                                st.dataframe(pd.DataFrame(mem_rows), use_container_width=True, hide_index=True)
                                st.success(
                                    f"📅 Estimated recovery: **{data.get('avg_recovery_days')} days**  ·  "
                                    f"Recommended: **{data.get('recommended_response','?')}**"
                                )

                            elif phase == "ALTERNATE ROUTE":
                                orig = data.get("original_route","")
                                alt  = data.get("alternate_route","")
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown(f"**Original (disrupted):**")
                                    st.code(orig)
                                with c2:
                                    st.markdown(f"**Alternate route ({data.get('method','?')}):**")
                                    st.code(alt if alt else "None found")
                                decay = data.get("decay_factor", 1.0)
                                if decay >= 0.9:
                                    decay_label = "Fresh (no recent overuse)"
                                    decay_color = "green"
                                elif decay >= 0.7:
                                    decay_label = "Moderate load (used recently)"
                                    decay_color = "orange"
                                else:
                                    decay_label = "High load — confidence reduced"
                                    decay_color = "red"
                                st.markdown(
                                    f"**Route confidence decay:** "
                                    f"<span style='color:{decay_color};font-weight:bold'>{decay:.2f} — {decay_label}</span>",
                                    unsafe_allow_html=True
                                )
                                # SHAP attribution chart
                                shap_attr = {
                                    k: v for k, v in data.get("shap_attribution", {}).items()
                                    if not k.startswith("_")
                                }
                                if shap_attr:
                                    st.markdown("**PPO Feature Attribution — what drove this routing choice:**")
                                    sorted_attr = dict(sorted(shap_attr.items(), key=lambda x: x[1]))
                                    fig_shap = px.bar(
                                        x=list(sorted_attr.values()),
                                        y=list(sorted_attr.keys()),
                                        orientation="h",
                                        template="plotly_dark",
                                        color=list(sorted_attr.values()),
                                        color_continuous_scale="Blues",
                                        labels={"x": "Mean |SHAP value|", "y": "Feature"},
                                    )
                                    fig_shap.update_layout(
                                        height=220,
                                        margin=dict(l=20, r=20, t=10, b=20),
                                        showlegend=False,
                                        coloraxis_showscale=False,
                                    )
                                    st.plotly_chart(fig_shap, use_container_width=True)
                                    top_feat = max(shap_attr, key=shap_attr.get)
                                    st.caption(
                                        f"Most influential feature: **{top_feat}** "
                                        f"(score={shap_attr[top_feat]:.5f}). "
                                        f"Gradient saliency (|∇ × input|) w.r.t. chosen action's probability — "
                                        f"equivalent to SHAP attribution on the PPO actor network."
                                    )
                                if data.get("top_candidates"):
                                    st.markdown("**Candidates evaluated:**")
                                    st.dataframe(pd.DataFrame(data["top_candidates"]), use_container_width=True, hide_index=True)
                                if data.get("blacklisted"):
                                    st.warning(f"🚫 Blacklisted nodes excluded: {', '.join(data['blacklisted'][:5])}")

                            elif phase == "BACKUP SUPPLIER" and data.get("top_backups"):
                                st.markdown(f"**{data.get('affected_retailers','?')} affected retailers · "
                                            f"{data.get('alternatives_found','?')} alternatives scanned**")
                                sup_rows = [{
                                    "Supplier":     b.get("entity","?"),
                                    "Type":         b.get("type","?"),
                                    "Backup Score": b.get("backup_score","?"),
                                    "Risk":         b.get("composite_risk","?"),
                                    "Volume":       b.get("out_volume","?"),
                                } for b in data["top_backups"]]
                                st.dataframe(pd.DataFrame(sup_rows), use_container_width=True, hide_index=True)

                            elif phase == "INVENTORY TRANSFER" and data.get("top_transfers"):
                                st.markdown(
                                    f"**State: {data.get('retailer_state','?')}  ·  "
                                    f"Fuel multiplier: {data.get('fuel_multiplier','?')}x  ·  "
                                    f"{data.get('candidates_found','?')} candidates found**"
                                )
                                inv_rows = [{
                                    "Distributor":    t.get("distributor","?"),
                                    "Transfer Score": t.get("transfer_score","?"),
                                    "Risk":           t.get("composite_risk","?"),
                                    "Spare Capacity": t.get("spare_capacity","?"),
                                    "Est. Days":      t.get("estimated_days","?"),
                                } for t in data["top_transfers"]]
                                st.dataframe(pd.DataFrame(inv_rows), use_container_width=True, hide_index=True)

                            elif phase == "DETECTION" and data.get("active_blacklist"):
                                st.warning(f"🚫 Nodes currently blacklisted (recently disrupted): "
                                           f"{', '.join(data['active_blacklist'])}")

                # Ranked action plan
                st.markdown("---")
                st.markdown("### 📋 Ranked Action Plan")
                actions = d.get("actions", [])
                if not actions:
                    st.warning("No actions generated.")
                else:
                    for a in actions:
                        conf = float(a.get("confidence", 0))
                        pri_icon = {1: "🥇", 2: "🥈", 3: "🥉"}.get(a.get("priority"), "▫️")
                        col_act, col_conf = st.columns([4, 1])
                        with col_act:
                            st.markdown(
                                f"{pri_icon} **{a.get('action','')}**  \n"
                                f"-> {a.get('label','')}  \n"
                                f"<small style='color:#6a8aaa'>{a.get('detail','')[:90]}</small>",
                                unsafe_allow_html=True
                            )
                        with col_conf:
                            st.metric("Confidence", f"{conf:.0%}")
                            st.caption(f"ETA: {a.get('eta_days','?')}d · {a.get('method','?')}")
                        st.progress(conf)
                        st.markdown("")

                st.markdown("---")

                # All events table
                st.markdown("### 📊 All Response Events")
                rows = []
                for i, dec in enumerate(decisions):
                    vv   = dec.get("verdict", {})
                    acts = vv.get("actions_ranked", [])
                    rows.append({
                        "#":           i + 1,
                        "Time":        dec.get("timestamp","")[:19],
                        "Severity":    vv.get("severity","?"),
                        "Z-Score":     dec.get("z_score","?"),
                        "Signals":     f"{vv.get('signals_activated',0)}/4",
                        "Top Action":  acts[0].get("action","?") if acts else "?",
                        "Recovery(d)": vv.get("recovery_estimate_days","?"),
                        "Manufacturer":dec.get("manufacturer","?")[:22],
                        "Retailer":    dec.get("retailer","?")[:22],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        if st.button("🔄 Refresh decisions"):
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — AI AGENT
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("AI Agent — Local LLM Analyst")
    st.caption(
        "Turns the immune response engine's structured Decision Trace into plain-English "
        "briefings and answers, using a model that runs entirely on this machine — no API keys, "
        "no cloud calls, zero cost."
    )

    agent = load_agent()
    is_up = agent.available()
    if is_up:
        st.markdown(
            f'<span class="agent-status on">Local model connected — {agent.model} via Ollama</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="agent-status off">Local model offline — showing template fallback</span>',
            unsafe_allow_html=True,
        )
        with st.expander("How to enable full narration"):
            st.code("brew install ollama\nollama serve\nollama pull llama3.2:1b", language="bash")
            st.caption("Once the server is running, reopen this tab — the agent checks automatically.")

    st.markdown("---")

    # ── Latest event narration ──────────────────────────────────────────
    st.markdown("### Latest Incident Briefing")

    all_decisions = load_immune_decisions()
    normal_decisions = [d for d in all_decisions if d.get("event_type") != "cytokine_storm"]

    if not normal_decisions:
        st.info("No immune response events yet. Start the stream to generate one, then come back here.")
    else:
        latest_decision = normal_decisions[-1]
        lv = latest_decision.get("verdict", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Severity", lv.get("severity", "?"))
        m2.metric("Risk Rating", f"{ImmuneAIAgent._risk_rating_10(latest_decision.get('z_score', 0))}/10")
        m3.metric("Z-Score", f"{latest_decision.get('z_score', 0):.2f}")
        m4.metric("Signals Activated", f"{lv.get('signals_activated', 0)}/4")
        m5.metric("Est. Recovery", f"{lv.get('recovery_estimate_days', '?')} days")

        col_narr, col_btn = st.columns([5, 1])
        with col_btn:
            regenerate = st.button("Regenerate", use_container_width=True)
        cache_key = f"narration::{latest_decision.get('timestamp','')}"
        if regenerate or cache_key not in st.session_state:
            with st.spinner("Analyst is reading the trace..."):
                st.session_state[cache_key] = agent.narrate_event(latest_decision)
        with col_narr:
            st.markdown(f'<div class="narration-box">{st.session_state[cache_key]}</div>', unsafe_allow_html=True)
        render_source_badge("llm" if is_up else "template")
        st.caption("Narrative text is AI-generated from the verified metrics above — the numbers are computed by the pipeline, the prose is the model's summary of them.")

    st.markdown("---")

    # ── What-if simulation ────────────────────────────────────────────
    # A 1B local model doesn't always route "what if X fails" chat questions
    # to simulate_node_failure correctly. This control calls it directly and
    # deterministically instead — no LLM routing involved — then only uses
    # the LLM afterward to narrate the already-known result.
    st.markdown("### What If a Supplier Fails?")
    st.caption("Directly removes a node (or several, comma-separated) from the live graph and checks reachability — no chat routing involved.")

    wcol_input, wcol_btn = st.columns([3, 1])
    with wcol_input:
        whatif_node = st.text_input("Node(s) to simulate failing", key="whatif_node_input", placeholder="e.g. MIAMI-LUKEN INC, HENRY SCHEIN INC")
    with wcol_btn:
        st.markdown("<div style='height: 1.9rem'></div>", unsafe_allow_html=True)
        run_whatif = st.button("Simulate failure", use_container_width=True)

    if run_whatif and whatif_node.strip():
        with st.spinner("Removing node from the graph and checking reachability..."):
            st.session_state.whatif_result = simulate_node_failure(whatif_node.strip())
            st.session_state.pop("whatif_narration", None)

    whatif_result = st.session_state.get("whatif_result")
    if whatif_result:
        if not whatif_result.get("found"):
            st.warning(f'No matching node found for "{whatif_result.get("queried", whatif_node)}".')
        else:
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Node", whatif_result["node"][:22])
            w2.metric("Upstream Suppliers", whatif_result["upstream_suppliers"])
            w3.metric("Still Reachable", whatif_result["reroutable_dependents"])
            w4.metric("Unreachable", whatif_result["unreachable_dependents"])

            if "whatif_narration" not in st.session_state:
                with st.spinner("Analyst is interpreting the result..."):
                    verdict_hint = (
                        f"{whatif_result['reroutable_dependents']} of {whatif_result['downstream_dependents']} "
                        f"downstream dependents would REMAIN REACHABLE via an alternate route; "
                        f"{whatif_result['unreachable_dependents']} would become UNREACHABLE."
                    )
                    st.session_state.whatif_narration = agent.explain_tool_result(
                        f"What happens if {whatif_result['node']} fails? "
                        f"Key computed finding, restate this clearly and do not contradict it: {verdict_hint}",
                        "simulate_node_failure", whatif_result,
                    )
            st.markdown(f'<div class="narration-box">{st.session_state.whatif_narration}</div>', unsafe_allow_html=True)
            render_source_badge("tool", "simulate_node_failure")

    st.markdown("---")

    # ── Chat ──────────────────────────────────────────────────────────
    st.markdown("### Ask the Agent")
    st.caption("Grounded in the current risk leaderboard and the last few immune response events.")

    risk_df = load_graph_risk()
    top_risk = (
        risk_df.sort_values("composite_risk", ascending=False).head(5)[["entity", "composite_risk"]].to_dict("records")
        if not risk_df.empty and "composite_risk" in risk_df.columns else []
    )
    anomalies_df = load_anomalies()
    anomaly_summary = (
        f"{len(anomalies_df):,} transactions scored, "
        f"{int((anomalies_df['anomaly_score'] >= 2).sum()):,} flagged high-confidence"
        if not anomalies_df.empty and "anomaly_score" in anomalies_df.columns else ""
    )
    chat_context = {
        "top_risk": top_risk,
        "recent_events": normal_decisions[-5:],
        "anomaly_summary": anomaly_summary,
        "whatif_result": st.session_state.get("whatif_result"),
    }

    if "ai_agent_history" not in st.session_state:
        st.session_state.ai_agent_history = []

    for msg in st.session_state.ai_agent_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_source_badge(msg.get("source"), msg.get("tool_used"))
                render_execution_trace(msg.get("trace"))

    question = st.chat_input("Ask about current risk, a recent disruption, or a recommended action...")
    if question:
        st.session_state.ai_agent_history.append({"role": "user", "content": question, "source": None, "tool_used": None, "trace": None})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = agent.chat(question, chat_context, st.session_state.ai_agent_history, tools=AGENT_TOOLS)
            st.markdown(result["answer"])
            render_source_badge(result["source"], result["tool_used"])
            render_execution_trace(result.get("trace"))
        st.session_state.ai_agent_history.append({
            "role": "assistant", "content": result["answer"],
            "source": result["source"], "tool_used": result["tool_used"],
            "trace": result.get("trace"),
        })

    if st.session_state.ai_agent_history and st.button("Clear chat"):
        st.session_state.ai_agent_history = []
        st.rerun()
