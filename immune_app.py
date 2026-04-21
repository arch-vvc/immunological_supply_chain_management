"""
Immunological Supply Chain — Focused Dashboard
===============================================
PES University Capstone  PW26_RGP_01

4-tab dashboard focused on the real-time immune response system:
  Tab 1 — Supply Chain Graph    : network topology + risk heatmap
  Tab 2 — Live Stream           : real-time transaction feed + anomaly signal
  Tab 3 — Immune Memory         : FAISS historical recall results
  Tab 4 — Immune Response       : chain-of-thought decisions per anomaly event

Run:
    # Terminal 1 — data stream
    python3 src/stream_simulator.py --interval 2 --disruption 30 --multi

    # Terminal 2 — immune response engine
    python3 src/stream_consumer.py

    # Terminal 3 — this dashboard
    streamlit run immune_app.py
"""

import os, pickle, json, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

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
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


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

@st.cache_data
def load_graph_risk():
    p = os.path.join(OUT, "graph_risk_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
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
    else:
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
        if st.button("🔄 Refresh stream data"):
            st.cache_data.clear()
            st.rerun()

        # Auto-refresh
        time.sleep(3)
        st.rerun()


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
    st.subheader("🦠 Immune Response Engine — Real-Time Chain-of-Thought Decisions")
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

                # Chain-of-thought
                st.markdown("### 🧠 Chain-of-Thought Reasoning")

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
