"""
Immunological Supply Chain — Streamlit Dashboard
=================================================
PES University Capstone  PW26_RGP_01

Run:  streamlit run app.py
"""

import os, pickle, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Immunological Supply Chain",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

BASE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(BASE, "output")
FIGS   = os.path.join(OUT,  "figures")
MODELS = os.path.join(BASE, "models")
EXTRA  = os.path.join(BASE, "data", "supplementary")

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card .label { color: #888; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { color: #e0e0ff; font-size: 1.5rem; font-weight: 700; margin-top: 0.2rem; line-height: 1.2; }
    .metric-card .sub   { color: #666; font-size: 0.75rem; margin-top: 0.1rem; }
    h1 { color: #c8c8ff !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


# ── Data loaders (cached) ──────────────────────────────────────────────────
@st.cache_data
def load_anomalies():
    p = os.path.join(OUT, "anomalies.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_risk():
    p = os.path.join(OUT, "risk_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_gnn():
    p = os.path.join(OUT, "gnn_risk_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_stress():
    p = os.path.join(OUT, "macro_stress_scores.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    return df

@st.cache_data
def load_forecast():
    p = os.path.join(OUT, "stress_forecast.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    return df

@st.cache_data
def load_recovery_preds():
    p = os.path.join(OUT, "recovery_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_disruption_data():
    p = os.path.join(EXTRA, "disruption_processed.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_supplier_results():
    p = os.path.join(OUT, "supplier_agent_results.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_resource
def load_graph():
    p = os.path.join(MODELS, "supplychain_graph.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_rf_regressor():
    p = os.path.join(MODELS, "recovery_regressor.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_rf_classifier():
    p = os.path.join(MODELS, "recovery_classifier.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_embeddings():
    p = os.path.join(MODELS, "node_embeddings.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ── Feature names used by RF models ───────────────────────────────────────
RF_FEATURES = [
    "disruption_type_enc", "industry_enc", "supplier_region_enc",
    "supplier_size_enc",   "disruption_severity",
    "production_impact_pct", "has_backup_supplier",
]

# ── Header ─────────────────────────────────────────────────────────────────
st.title("Immunological Supply Chain")
st.caption("Self-Healing Supply Chains with AI Digital Antibodies  |  PES University  |  PW26_RGP_01")
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Overview",
    "Supply Chain Graph",
    "Anomaly Detection",
    "Risk Analysis",
    "Macro Stress & LSTM",
    "Recovery Predictor",
    "Multi-Domain Risk",
    "🔴 Live Stream",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    G    = load_graph()
    anom = load_anomalies()
    risk = load_risk()
    gnn  = load_gnn()
    fc   = load_forecast()

    n_nodes   = G.number_of_nodes()   if G    else "—"
    n_edges   = G.number_of_edges()   if G    else "—"
    n_anom    = len(anom[anom["anomaly_score"] >= 2]) if not anom.empty else "—"
    n_high    = len(risk[risk["risk_score"] > 0.7])   if not risk.empty else "—"

    forecast_level = "—"
    if not fc.empty:
        upcoming = fc[fc["type"] == "forecast"]
        if not upcoming.empty:
            forecast_level = upcoming.iloc[0]["stress_level"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Nodes in Graph",    f"{n_nodes:,}" if isinstance(n_nodes, int) else n_nodes)
    with c2: metric_card("Edges in Graph",    f"{n_edges:,}" if isinstance(n_edges, int) else n_edges)
    with c3: metric_card("High-Conf Anomalies", f"{n_anom:,}"  if isinstance(n_anom, int)  else n_anom, "score >= 2")
    with c4: metric_card("High-Risk Entities",  f"{n_high:,}"  if isinstance(n_high, int)  else n_high, "risk > 0.70")
    with c5: metric_card("Next-Week Forecast",  forecast_level, "macro stress level")

    # ── Global SHAP Explainability ─────────────────────────────────────────
    st.markdown("#### What Drives Recovery Time? (Global Model Explainability)")
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows which features the Random Forest model "
        "relies on most when predicting how long a disruption takes to recover from. "
        "Red = above median importance. Use the Recovery Predictor tab to see a breakdown for a specific scenario."
    )

    _ov_reg_path  = os.path.join(MODELS, "recovery_regressor.pkl")
    _ov_data_path = os.path.join(EXTRA,  "disruption_processed.csv")
    _OV_FEAT_NAMES = [
        "Disruption Type", "Industry", "Supplier Region",
        "Supplier Size", "Disruption Severity",
        "Production Impact %", "Has Backup Supplier",
    ]
    _OV_FEAT_COLS = [
        "disruption_type_enc", "industry_enc", "supplier_region_enc",
        "supplier_size_enc", "disruption_severity",
        "production_impact_pct", "has_backup_supplier",
    ]

    if os.path.exists(_ov_reg_path):
        try:
            import joblib as _jl
            _ov_model = _jl.load(_ov_reg_path)
            _imps = _ov_model.feature_importances_
            _fi_df = pd.DataFrame({
                "Feature":    _OV_FEAT_NAMES,
                "Importance": _imps,
            }).sort_values("Importance", ascending=True)

            _col1, _col2 = st.columns(2)

            with _col1:
                st.markdown("**Feature Importance** — which inputs the model uses most")
                _fig_fi, _ax_fi = plt.subplots(figsize=(6, 3.5))
                _fig_fi.patch.set_facecolor("#0f0f1a")
                _ax_fi.set_facecolor("#0f0f1a")
                _fi_colors = ["#e74c3c" if v > _fi_df["Importance"].median() else "#3498db"
                              for v in _fi_df["Importance"]]
                _ax_fi.barh(_fi_df["Feature"], _fi_df["Importance"], color=_fi_colors, edgecolor="none")
                _ax_fi.set_xlabel("Importance", color="white")
                _ax_fi.tick_params(colors="white")
                _ax_fi.xaxis.label.set_color("white")
                for sp in ["top", "right"]:
                    _ax_fi.spines[sp].set_visible(False)
                for sp in ["bottom", "left"]:
                    _ax_fi.spines[sp].set_color("#4a4a6a")
                plt.tight_layout()
                st.pyplot(_fig_fi)
                plt.close()

            with _col2:
                if os.path.exists(_ov_data_path):
                    st.markdown("**Direction** — does each feature increase or decrease recovery time?")
                    try:
                        import shap as _shap_ov
                        _ov_dis = pd.read_csv(_ov_data_path)
                        _ov_dis["has_backup_supplier"] = (
                            _ov_dis["has_backup_supplier"]
                            .map({True: 1, False: 0, "True": 1, "False": 0})
                            .fillna(0).astype(int)
                        )
                        _ov_dis = _ov_dis.dropna(subset=_OV_FEAT_COLS)
                        _ov_sample = _ov_dis[_OV_FEAT_COLS].sample(min(200, len(_ov_dis)), random_state=42).values
                        _ov_exp = _shap_ov.TreeExplainer(_ov_model)
                        _ov_sv  = _ov_exp.shap_values(_ov_sample)
                        _ov_signed = _ov_sv.mean(axis=0)
                        _dir_df = pd.DataFrame({
                            "Feature":         _OV_FEAT_NAMES,
                            "Avg SHAP (days)": _ov_signed.round(1),
                            "Direction":       ["↑ Longer recovery" if v > 0 else "↓ Faster recovery"
                                               for v in _ov_signed],
                        }).sort_values("Avg SHAP (days)", ascending=False)
                        st.dataframe(_dir_df, use_container_width=True, hide_index=True, height=280)
                        st.caption("Values in days. ↑ means this feature tends to increase recovery time on average.")
                    except Exception:
                        st.info("Install shap for the direction table: pip install shap")
        except Exception as _ov_e:
            st.info(f"Run Stage 8 to see model explainability. ({_ov_e})")
    else:
        st.info("Run Stage 8 first to see model explainability.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — SUPPLY CHAIN GRAPH
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    G    = load_graph()
    risk = load_risk()

    if G is None:
        st.warning("Run the pipeline first to generate the graph.")
    else:
        risk_lookup = dict(zip(risk["entity"], risk["risk_score"])) if not risk.empty else {}

        node_types = {n: d.get("type", "unknown") for n, d in G.nodes(data=True)}
        mfrs  = [n for n, t in node_types.items() if t == "manufacturer"]
        dists = [n for n, t in node_types.items() if t == "distributor"]

        # Subgraph: all manufacturers + distributors + their top-3 connected retailers
        sub_nodes = set(mfrs + dists)
        for d in dists:
            retailer_succs = sorted(
                [n for n in G.successors(d) if node_types.get(n) == "retailer"],
                key=lambda r: risk_lookup.get(r, 0), reverse=True
            )[:3]
            sub_nodes.update(retailer_succs)

        SG  = G.subgraph(sub_nodes)
        pos = nx.spring_layout(SG, seed=42, k=1.5)

        colour_map = {"manufacturer": "#f4a261", "distributor": "#4fc3f7", "retailer": "#a8d8a8"}

        edge_x, edge_y = [], []
        for u, v in SG.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.4, color="#334"),
            hoverinfo="none",
        )

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for n in SG.nodes():
            x, y = pos[n]
            node_x.append(x); node_y.append(y)
            t    = node_types.get(n, "unknown")
            rs   = risk_lookup.get(n, 0.0)
            node_text.append(f"{n}<br>Type: {t}<br>Risk: {rs:.3f}")
            node_color.append(colour_map.get(t, "#888"))
            node_size.append(14 if t == "manufacturer" else 10 if t == "distributor" else 6)

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(size=node_size, color=node_color, line=dict(width=0.5, color="#111")),
            text=node_text, hoverinfo="text",
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                margin=dict(l=10, r=10, t=40, b=10),
                title=dict(
                    text=f"Supply Chain Network  ({SG.number_of_nodes()} nodes shown)",
                    font=dict(color="#ccc", size=13),
                ),
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=560,
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        l1, l2, l3 = st.columns(3)
        with l1: st.markdown("<span style='color:#f4a261'>●</span> Manufacturer", unsafe_allow_html=True)
        with l2: st.markdown("<span style='color:#4fc3f7'>●</span> Distributor",  unsafe_allow_html=True)
        with l3: st.markdown("<span style='color:#a8d8a8'>●</span> Retailer",     unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    anom = load_anomalies()

    if anom.empty:
        st.warning("No anomaly data found. Run Stage 3 first.")
    else:
        high = anom[anom["anomaly_score"] >= 2].copy()
        susp = anom[anom["anomaly_score"] == 1].copy()

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Total Flagged",         f"{len(anom):,}")
        with c2: metric_card("High-Confidence (2+)",  f"{len(high):,}")
        with c3: metric_card("Suspect (1 signal)",    f"{len(susp):,}")

        st.markdown("#### Anomaly Score Distribution")
        score_counts = anom["anomaly_score"].value_counts().sort_index()
        fig = px.bar(
            x=score_counts.index, y=score_counts.values,
            labels={"x": "Anomaly Score", "y": "Transactions"},
            color_discrete_sequence=["#4fc3f7"],
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", height=260)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top High-Confidence Anomalies")
        cols_show = [c for c in ["manufacturer", "distributor", "retailer", "quantity", "anomaly_score"]
                     if c in high.columns]
        st.dataframe(
            high[cols_show].sort_values("anomaly_score", ascending=False).head(50)
                           .reset_index(drop=True),
            use_container_width=True, height=320,
        )

        if "date" in anom.columns:
            st.markdown("#### Anomaly Timeline")
            anom["date"] = pd.to_datetime(anom["date"], errors="coerce")
            timeline = anom.dropna(subset=["date"])
            timeline = timeline.set_index("date").resample("ME")["anomaly_score"].count().reset_index()
            timeline.columns = ["date", "count"]
            fig2 = px.line(
                timeline, x="date", y="count",
                labels={"count": "Anomalies", "date": "Month"},
                template="plotly_dark", color_discrete_sequence=["#ff6b6b"],
            )
            fig2.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", height=260)
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    risk = load_risk()
    gnn  = load_gnn()

    if risk.empty:
        st.warning("No risk scores found. Run Stage 4 first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Entities Scored", f"{len(risk):,}")
        with c2:
            top = risk.iloc[0]
            metric_card("Highest Risk Entity", top["entity"][:22], f"score {top['risk_score']:.3f}")
        with c3:
            hr = risk[risk["risk_score"] > 0.7]
            metric_card("Entities Risk > 0.70", str(len(hr)))

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Top 20 — Centrality Risk Score")
            top20 = risk.head(20)
            fig = px.bar(
                top20, x="risk_score", y="entity", orientation="h",
                color="risk_score", color_continuous_scale="Reds",
                template="plotly_dark",
                labels={"risk_score": "Risk Score", "entity": ""},
            )
            fig.update_layout(
                paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False, height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            if not gnn.empty:
                st.markdown("#### Top 20 — GNN Enhanced Risk")
                top_gnn = gnn.sort_values("enhanced_risk", ascending=False).head(20)
                fig2 = px.bar(
                    top_gnn, x="enhanced_risk", y="entity", orientation="h",
                    color="enhanced_risk", color_continuous_scale="Blues",
                    template="plotly_dark",
                    labels={"enhanced_risk": "Enhanced Risk", "entity": ""},
                )
                fig2.update_layout(
                    paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False, height=480,
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Run Stage 7 (GNN Encoder) to see enhanced risk scores.")

        if not gnn.empty:
            st.markdown("#### GNN Embedding Space")
            fig_path = os.path.join(FIGS, "fig5_gnn_embeddings.png")
            if os.path.exists(fig_path):
                st.image(fig_path, use_column_width=True)

        # ── PPO vs Dijkstra ───────────────────────────────────────────
        st.markdown("#### PPO vs Dijkstra — Risk-Aware Routing (Stage 11)")
        ppo_fig = os.path.join(FIGS, "fig9_ppo_training.png")
        ppo_txt = os.path.join(OUT, "ppo_routing_results.txt")
        col_ppo1, col_ppo2 = st.columns([1.4, 1])
        with col_ppo1:
            if os.path.exists(ppo_fig):
                st.image(ppo_fig, use_column_width=True,
                         caption="PPO agent cumulative reward over training episodes")
            else:
                st.info("Run Stage 11 (PPO Routing Agent) to generate the training figure.")
        with col_ppo2:
            if os.path.exists(ppo_txt):
                try:
                    with open(ppo_txt) as _f:
                        st.code(_f.read(), language=None)
                except Exception:
                    st.info("Could not read PPO results file.")
            else:
                st.info("Run Stage 11 to generate PPO routing results.")

        # ── Immunological Memory ──────────────────────────────────────
        st.markdown("#### Immunological Memory — FAISS Retrieval (Stage 13)")
        mem_csv = os.path.join(OUT, "memory_retrieval.csv")
        mem_txt = os.path.join(OUT, "memory_report.txt")
        if os.path.exists(mem_txt):
            try:
                with open(mem_txt) as _f:
                    st.code(_f.read(), language=None)
            except Exception:
                pass
        if os.path.exists(mem_csv):
            try:
                mem_df = pd.read_csv(mem_csv)
                if not mem_df.empty:
                    st.markdown("**Top retrieved matches (sample)**")
                    st.dataframe(mem_df.head(10), use_container_width=True)
            except Exception:
                pass
        if not os.path.exists(mem_txt) and not os.path.exists(mem_csv):
            st.info("Run Stage 13 (Immunological Memory) to see FAISS retrieval results.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — MACRO STRESS & LSTM FORECAST
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    stress   = load_stress()
    forecast = load_forecast()

    if stress.empty:
        st.warning("Run Stage 9 first to generate stress scores.")
    else:
        latest     = stress.iloc[-1]
        high_wks   = int((stress["stress_level"] == "HIGH").sum())
        med_wks    = int((stress["stress_level"] == "MEDIUM").sum())
        low_wks    = int((stress["stress_level"] == "LOW").sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Latest Score",    f"{latest['stress_score']:.3f}", str(latest["stress_level"]))
        with c2: metric_card("HIGH stress wks", str(high_wks))
        with c3: metric_card("MEDIUM stress wks", str(med_wks))
        with c4: metric_card("LOW stress wks",  str(low_wks))

        st.markdown("#### Macro Freight Stress Timeline + LSTM Forecast")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stress["date"], y=stress["stress_score"],
            mode="lines", name="Historical", line=dict(color="#4fc3f7", width=1.2),
        ))

        if not forecast.empty:
            fc_only = forecast[forecast["type"] == "forecast"]
            if not fc_only.empty:
                # Connect last historical point to first forecast point
                bridge_x = [stress["date"].iloc[-1], fc_only["date"].iloc[0]]
                bridge_y = [stress["stress_score"].iloc[-1], fc_only["stress_score"].iloc[0]]
                fig.add_trace(go.Scatter(
                    x=bridge_x, y=bridge_y,
                    mode="lines", line=dict(color="#ff6b6b", width=1.5, dash="dot"),
                    showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=fc_only["date"], y=fc_only["stress_score"],
                    mode="lines+markers", name="LSTM Forecast",
                    line=dict(color="#ff6b6b", width=2, dash="dot"),
                    marker=dict(size=8),
                ))

        fig.add_hline(y=0.65, line_dash="dash", line_color="red",   opacity=0.5, annotation_text="HIGH")
        fig.add_hline(y=0.40, line_dash="dash", line_color="orange", opacity=0.5, annotation_text="MEDIUM")

        fig.update_layout(
            paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
            xaxis=dict(color="#888"), yaxis=dict(color="#888", range=[0, 1]),
            legend=dict(bgcolor="#1a1a2e", font=dict(color="#ccc")),
            height=380, margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        if not forecast.empty:
            fc_only = forecast[forecast["type"] == "forecast"]
            if not fc_only.empty:
                st.markdown("#### 4-Week Forward Forecast")
                display = fc_only[["date", "stress_score", "stress_level"]].copy()
                display["date"] = display["date"].dt.strftime("%Y-%m-%d")
                display.columns = ["Week", "Stress Score", "Level"]
                st.dataframe(display.reset_index(drop=True), use_container_width=True, height=180)

# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — RECOVERY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    regressor  = load_rf_regressor()
    classifier = load_rf_classifier()
    dis_data   = load_disruption_data()

    if regressor is None:
        st.warning("Run Stage 8 first to train the recovery models.")
    else:
        st.markdown("#### Predict Recovery Time for a Disruption Scenario")
        st.caption("Select disruption parameters below. The Random Forest models predict "
                   "full recovery days and the most likely response strategy.")

        if not dis_data.empty:
            dis_types   = sorted(dis_data["disruption_type"].dropna().unique().tolist())
            industries  = sorted(dis_data["industry"].dropna().unique().tolist())
            regions     = sorted(dis_data["supplier_region"].dropna().unique().tolist())
            sizes       = sorted(dis_data["supplier_size"].dropna().unique().tolist())
            resp_types  = sorted(dis_data["response_type"].dropna().unique().tolist())
        else:
            dis_types  = ["Cyber Attack", "Factory Incident", "Natural Disaster", "Logistics Delay"]
            industries = ["Pharmaceuticals", "Electronics", "Automotive", "Retail"]
            regions    = ["Asia-Pacific", "Europe", "North America", "South America"]
            sizes      = ["Small", "Medium", "Large"]
            resp_types = ["Alternative Supplier", "Combined Strategy", "Customer Delay"]

        col_a, col_b = st.columns(2)
        with col_a:
            sel_type    = st.selectbox("Disruption Type",    dis_types)
            sel_ind     = st.selectbox("Industry",           industries)
            sel_region  = st.selectbox("Supplier Region",    regions)
        with col_b:
            sel_size    = st.selectbox("Supplier Size",      sizes)
            sel_sev     = st.slider("Disruption Severity",   1, 5, 3)
            sel_impact  = st.slider("Production Impact (%)", 0, 100, 40)

        sel_backup = st.checkbox("Has Backup Supplier", value=False)

        # Encode inputs using value counts from the data
        def encode(series, val):
            cats = sorted(series.dropna().unique())
            return cats.index(val) if val in cats else 0

        if not dis_data.empty:
            type_enc   = encode(dis_data["disruption_type"],   sel_type)
            ind_enc    = encode(dis_data["industry"],          sel_ind)
            reg_enc    = encode(dis_data["supplier_region"],   sel_region)
            size_enc   = encode(dis_data["supplier_size"],     sel_size)
        else:
            type_enc = ind_enc = reg_enc = size_enc = 0

        feat_vec = np.array([[
            type_enc, ind_enc, reg_enc, size_enc,
            sel_sev, sel_impact, int(sel_backup),
        ]], dtype=float)

        if st.button("Predict Recovery", type="primary"):
            days     = float(regressor.predict(feat_vec)[0])
            strategy = classifier.predict(feat_vec)[0]

            if not dis_data.empty:
                resp_cats = sorted(dis_data["response_type"].dropna().unique())
                strategy_idx = int(strategy)
                strategy_label = (resp_cats[strategy_idx]
                                  if 0 <= strategy_idx < len(resp_cats)
                                  else str(strategy))
            else:
                strategy_label = str(strategy)

            r1, r2 = st.columns(2)
            with r1: metric_card("Predicted Full Recovery", f"{days:.0f} days")
            with r2: metric_card("Recommended Strategy",    strategy_label)

            severity_labels = {1: "Minimal", 2: "Low", 3: "Moderate", 4: "High", 5: "Critical"}
            st.info(
                f"Scenario: **{sel_type}** in **{sel_ind}** | "
                f"Region: {sel_region} | Size: {sel_size} | "
                f"Severity: {severity_labels.get(sel_sev, sel_sev)} | "
                f"Backup supplier: {'Yes' if sel_backup else 'No'}"
            )

            # ── Actionable response steps ──────────────────────────────────
            st.markdown("#### Recommended Action Plan")

            STEP_PLANS = {
                "Alternative Supplier": [
                    "🔴 **Immediately** flag the disrupted supplier in the system and halt new orders",
                    "🔍 **Within 24h** — identify backup suppliers using the Supplier Agent rankings below",
                    "📞 **Within 48h** — contact top-ranked backup suppliers and confirm available capacity",
                    "🔄 **Within 72h** — reroute all pending orders through the selected backup supplier",
                    "📊 **Ongoing** — monitor delivery performance and anomaly scores on the new route",
                    f"⏱️ **Full recovery expected in ~{days:.0f} days** once rerouting is confirmed",
                ],
                "Emergency Stockpile": [
                    "🔴 **Immediately** activate emergency inventory reserves at regional distribution centres",
                    "📦 **Within 24h** — audit current stock levels and calculate days of supply remaining",
                    "🚚 **Within 48h** — expedite internal transfers from low-risk nodes to affected retailers",
                    "📋 **Within 72h** — issue demand rationing guidelines to downstream retailers if stock is tight",
                    "🔍 **Parallel** — begin sourcing from alternative suppliers as a medium-term fix",
                    f"⏱️ **Full recovery expected in ~{days:.0f} days**",
                ],
                "Combined Strategy": [
                    "🔴 **Immediately** split response across two tracks: stockpile activation AND supplier rerouting",
                    "📦 **Track 1** — release emergency inventory to cover the next 14 days of demand",
                    "🔍 **Track 2** — engage backup suppliers in parallel; do not wait for stockpile to deplete",
                    "📞 **Within 48h** — confirm capacity with at least 2 backup suppliers (redundancy)",
                    "📊 **Weekly** — review which track is performing better and scale accordingly",
                    f"⏱️ **Full recovery expected in ~{days:.0f} days**",
                ],
                "Customer Delay": [
                    "🔴 **Immediately** identify which downstream retailers will be affected and by how much",
                    "📞 **Within 24h** — proactively notify affected customers of expected delay window",
                    "📋 **Within 48h** — issue revised delivery estimates with a buffer built in",
                    "🔍 **Parallel** — investigate root cause and whether any partial fulfilment is possible",
                    "🔄 **Once supply resumes** — prioritise backlog clearance by order date",
                    f"⏱️ **Full recovery expected in ~{days:.0f} days**",
                ],
            }

            steps = STEP_PLANS.get(strategy_label, [
                "🔴 Isolate and assess the disrupted node immediately",
                "📞 Contact affected downstream partners within 24 hours",
                "🔄 Activate contingency routing or inventory plans",
                f"⏱️ Full recovery expected in ~{days:.0f} days",
            ])

            for step in steps:
                st.markdown(f"- {step}")

            # ── If strategy is Alternative Supplier, show actual backup options ──
            if "alternative supplier" in strategy_label.lower() or "combined" in strategy_label.lower():
                supplier_df = load_supplier_results()
                if not supplier_df.empty:
                    st.markdown("#### Available Backup Suppliers (from Supplier Agent)")
                    st.caption("These are the top-ranked backup suppliers identified by the Supplier Agent for the most disrupted entities in the current pipeline run.")
                    top_backups = (
                        supplier_df[supplier_df["backup_rank"] == 1]
                        [["disrupted_entity", "backup_entity", "backup_score", "composite_risk", "out_volume"]]
                        .rename(columns={
                            "disrupted_entity": "Disrupted Supplier",
                            "backup_entity":    "Recommended Backup",
                            "backup_score":     "Backup Score",
                            "composite_risk":   "Backup Risk",
                            "out_volume":       "Supply Volume",
                        })
                        .sort_values("Backup Score", ascending=False)
                        .reset_index(drop=True)
                    )
                    top_backups["Backup Score"]  = top_backups["Backup Score"].round(3)
                    top_backups["Backup Risk"]   = top_backups["Backup Risk"].round(3)
                    top_backups["Supply Volume"] = top_backups["Supply Volume"].apply(lambda x: f"{int(x):,}")
                    st.dataframe(top_backups, use_container_width=True, hide_index=True)

            # ── Why did the model predict this? (per-prediction SHAP) ─────
            st.markdown("#### Why did the model predict this?")
            st.caption(
                "Each bar shows how much a specific feature pushed the prediction "
                "**above** (red) or **below** (blue) the average recovery time. Values are in days."
            )
            try:
                import shap as _shap
                _explainer   = _shap.TreeExplainer(regressor)
                _raw = _explainer.shap_values(feat_vec)
                # TreeExplainer returns list-of-arrays for some sklearn versions
                _shap_vals = (_raw[0][0] if isinstance(_raw, list) else
                              _raw[0] if _raw.ndim == 2 else _raw)
                _ev = _explainer.expected_value
                _base = float(_ev[0]) if hasattr(_ev, "__len__") else float(_ev)

                PRED_FEATURE_NAMES = [
                    "Disruption Type", "Industry", "Supplier Region",
                    "Supplier Size", "Disruption Severity",
                    "Production Impact %", "Has Backup Supplier",
                ]
                PRED_FEATURE_VALUES = [
                    sel_type, sel_ind, sel_region,
                    sel_size, f"Severity {sel_sev}",
                    f"{sel_impact}%", "Yes" if sel_backup else "No",
                ]

                _shap_df = pd.DataFrame({
                    "Feature":       [f"{n}  ({v})" for n, v in zip(PRED_FEATURE_NAMES, PRED_FEATURE_VALUES)],
                    "SHAP (days)":   _shap_vals,
                }).sort_values("SHAP (days)")

                fig_w, ax_w = plt.subplots(figsize=(8, 4))
                fig_w.patch.set_facecolor("#0f0f1a")
                ax_w.set_facecolor("#0f0f1a")
                colors_w = ["#e74c3c" if v > 0 else "#3498db" for v in _shap_df["SHAP (days)"]]
                bars = ax_w.barh(_shap_df["Feature"], _shap_df["SHAP (days)"],
                                 color=colors_w, edgecolor="none")
                ax_w.axvline(0, color="white", linewidth=0.8, alpha=0.5)
                ax_w.set_xlabel("Impact on predicted recovery days", color="white")
                ax_w.set_title(
                    f"Prediction breakdown  —  baseline avg: {_base:.0f} days  →  predicted: {days:.0f} days",
                    color="white", fontsize=10,
                )
                ax_w.tick_params(colors="white")
                ax_w.xaxis.label.set_color("white")
                for spine in ["top", "right"]:
                    ax_w.spines[spine].set_visible(False)
                for spine in ["bottom", "left"]:
                    ax_w.spines[spine].set_color("#4a4a6a")
                for bar, val in zip(bars, _shap_df["SHAP (days)"]):
                    label = f"+{val:.1f}d" if val > 0 else f"{val:.1f}d"
                    x_pos = val + (0.3 if val > 0 else -0.3)
                    ax_w.text(x_pos, bar.get_y() + bar.get_height() / 2,
                              label, va="center",
                              ha="left" if val > 0 else "right",
                              color="white", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_w)
                plt.close()

                # Plain-English summary
                top_pos = _shap_df[_shap_df["SHAP (days)"] > 0].sort_values("SHAP (days)", ascending=False)
                top_neg = _shap_df[_shap_df["SHAP (days)"] < 0].sort_values("SHAP (days)")
                summary_lines = [f"**Baseline average recovery: {_base:.0f} days**"]
                if not top_pos.empty:
                    for _, row in top_pos.iterrows():
                        summary_lines.append(f"🔴 **{row['Feature']}** added **+{row['SHAP (days)']:.1f} days**")
                if not top_neg.empty:
                    for _, row in top_neg.iterrows():
                        summary_lines.append(f"🔵 **{row['Feature']}** saved **{row['SHAP (days)']:.1f} days**")
                summary_lines.append(f"**→ Final prediction: {days:.0f} days**")
                for line in summary_lines:
                    st.markdown(line)

            except Exception as _e:
                st.info(f"Install shap to see prediction breakdown: pip install shap  ({_e})")

        st.divider()
        st.markdown("#### Model Performance")
        metrics_path = os.path.join(OUT, "recovery_metrics.txt")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    st.code(f.read(), language=None)
            except Exception:
                st.info("Model performance metrics available — restart the app to load.")

        fig_path = os.path.join(FIGS, "fig6_recovery.png")
        if os.path.exists(fig_path):
            st.image(fig_path, use_column_width=True)

        # ── Inventory Agent ───────────────────────────────────────────
        st.divider()
        st.markdown("#### Inventory Agent — Stock Transfer Recommendations (Stage 15)")
        st.caption(
            "Identifies retailers with single-source dependency or abnormal order volume, "
            "then recommends ranked stock transfer sources ranked by capacity, safety, and fuel cost."
        )
        inv_csv = os.path.join(OUT, "inventory_agent_results.csv")
        inv_txt = os.path.join(OUT, "inventory_agent_report.txt")

        if os.path.exists(inv_txt):
            try:
                with open(inv_txt) as _f:
                    st.code(_f.read(), language=None)
            except Exception:
                pass

        if os.path.exists(inv_csv):
            try:
                inv_df = pd.read_csv(inv_csv)
                if not inv_df.empty:
                    # Show top-1 recommendations only in a clean table
                    top1 = inv_df[inv_df["transfer_rank"] == 1][[
                        "retailer", "retailer_state", "retailer_risk_score",
                        "primary_distributor", "transfer_source",
                        "transfer_score", "spare_capacity", "estimated_days"
                    ]].copy()
                    top1.columns = [
                        "Retailer", "State", "Risk Score",
                        "Primary Distributor", "Recommended Transfer Source",
                        "Transfer Score", "Spare Capacity", "Est. Days"
                    ]
                    st.markdown("**Top-1 transfer recommendation per at-risk retailer**")
                    st.dataframe(
                        top1.style.background_gradient(subset=["Transfer Score"], cmap="Greens"),
                        use_container_width=True
                    )

                    # Summary metrics
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        metric_card("At-Risk Retailers", str(len(top1)))
                    with c2:
                        metric_card("Avg Transfer Score", f"{top1['Transfer Score'].mean():.3f}")
                    with c3:
                        metric_card("Avg Est. Delivery", f"{top1['Est. Days'].mean():.1f} days")
            except Exception as _e:
                st.warning(f"Could not load inventory results: {_e}")
        else:
            st.info("Run Stage 15 (Inventory Agent) to see transfer recommendations.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 7 — MULTI-DOMAIN RISK
# ══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Multi-Domain Risk Modelling — Objective 4")
    st.caption(
        "Same XGBoost architecture trained across 5 industries — "
        "proving the framework generalises beyond a single domain."
    )

    f1_path  = "output/multi_domain_f1.csv"
    fig_path = "output/figures/fig10_multi_domain_risk.png"

    if not os.path.exists(f1_path):
        st.warning("Run Stage 12 first: python3 main.py --from 12")
    else:
        df_f1 = pd.read_csv(f1_path)
        df_f1.columns = [str(c).strip().replace(" ", "_") for c in df_f1.columns]

        if "Industry" not in df_f1.columns or "F1_Score" not in df_f1.columns:
            st.warning("multi_domain_f1.csv is missing required columns: Industry, F1_Score")
            st.dataframe(df_f1, use_container_width=True)
            if os.path.exists(fig_path):
                st.image(fig_path, use_column_width=True)
        else:
            best_pool   = df_f1[df_f1["Industry"].astype(str).str.lower() != "all domains"]
            best_row    = best_pool.sort_values("F1_Score", ascending=False).iloc[0] if not best_pool.empty else None
            overall_row = df_f1[df_f1["Industry"] == "All Domains"]
            overall_f1  = overall_row["F1_Score"].values[0] if len(overall_row) else "N/A"

            col1, col2, col3 = st.columns(3)
            col1.metric("Best Domain",    best_row["Industry"] if best_row is not None else "N/A")
            col2.metric("Best Domain F1", f"{best_row['F1_Score']:.3f}" if best_row is not None else "N/A")
            col3.metric("Overall F1",     f"{float(overall_f1):.3f}"
                                          if overall_f1 != "N/A" else "N/A")

            st.dataframe(df_f1, use_container_width=True)

            if os.path.exists(fig_path):
                st.image(fig_path, use_column_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 8 — LIVE STREAM
# ══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("Live Stream Monitor — Real-Time Sensor Feed")
    st.caption(
        "Simulates real-time IoT/sensor data arriving row by row. "
        "Run stream_simulator.py and stream_consumer.py in two terminals to activate."
    )

    LIVE_RESULTS_PATH    = "data/stream/live_results.csv"
    DISRUPTION_FLAG_PATH = "data/stream/disruption_active.flag"

    if not os.path.exists(LIVE_RESULTS_PATH):
        st.info("Stream is not running yet. Start it with these two commands in separate terminals:")
        st.code(
            "# Terminal 1 — emit rows every 2s, inject disruption at t=30s\n"
            "python3 src/stream_simulator.py --interval 2 --disruption 30 --loop\n\n"
            "# Terminal 2 — consume and detect anomalies\n"
            "python3 src/stream_consumer.py",
            language="bash"
        )
    else:
        st.caption("Auto-refreshes every 3 seconds. Keep this tab open during your demo.")

        df_live = pd.read_csv(LIVE_RESULTS_PATH)

        total_rows      = len(df_live)
        anomaly_rows    = int(df_live["is_anomaly"].sum()) if "is_anomaly" in df_live.columns else 0
        disruption_rows = int(df_live["disruption_injected"].sum()) if "disruption_injected" in df_live.columns else 0
        rerouted_rows   = int((df_live["alternate_route"].astype(str).str.strip() != "").sum()) if "alternate_route" in df_live.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows Processed",      total_rows)
        c2.metric("Anomalies Detected",  anomaly_rows,    delta=f"{anomaly_rows} flagged")
        c3.metric("Disruptions Injected", disruption_rows)
        c4.metric("Routes Rerouted",     rerouted_rows)

        if os.path.exists(DISRUPTION_FLAG_PATH):
            flag_text = open(DISRUPTION_FLAG_PATH).read()
            st.error(f"⚡ ACTIVE DISRUPTION DETECTED\n\n{flag_text}")

        st.markdown("#### Recent Anomalies")
        if "is_anomaly" in df_live.columns:
            df_anomalies = df_live[df_live["is_anomaly"] == 1].tail(10)
            if df_anomalies.empty:
                st.success("No anomalies detected yet — supply chain is healthy.")
            else:
                cols_to_show = [c for c in [
                    "timestamp", "manufacturer", "distributor", "retailer",
                    "quantity", "z_score", "disruption_injected", "routing_note", "alternate_route"
                ] if c in df_anomalies.columns]
                st.dataframe(df_anomalies[cols_to_show].reset_index(drop=True),
                             use_container_width=True)

        st.markdown("#### All Transactions (last 50)")
        cols_to_show = [c for c in [
            "timestamp", "row_index", "manufacturer", "distributor",
            "retailer", "quantity", "z_score", "is_anomaly"
        ] if c in df_live.columns]
        st.dataframe(df_live[cols_to_show].tail(50).reset_index(drop=True),
                     use_container_width=True)

        if "z_score" in df_live.columns and "row_index" in df_live.columns:
            st.markdown("#### Z-Score Signal Over Time")
            import plotly.graph_objects as go
            fig_stream = go.Figure()
            fig_stream.add_trace(go.Scatter(
                x=df_live["row_index"], y=df_live["z_score"],
                mode="lines", name="Z-Score",
                line=dict(color="#00A896", width=1.5)
            ))
            fig_stream.add_hline(y=2.5, line_dash="dash", line_color="#F4845F",
                                 annotation_text="Anomaly Threshold (Z=2.5)")
            fig_stream.update_layout(
                xaxis_title="Row Index", yaxis_title="Z-Score",
                template="plotly_dark", height=300,
                margin=dict(l=40, r=20, t=20, b=40)
            )
            st.plotly_chart(fig_stream, use_container_width=True)

        time.sleep(3)
        st.rerun()
