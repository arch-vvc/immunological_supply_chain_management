"""
Immunological Supply Chain — Streamlit Dashboard
=================================================
PES University Capstone  PW26_RGP_01

Run:  streamlit run app.py
"""

import os, sys, pickle, warnings, time, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# torch, shap, networkx are imported lazily inside tabs — faster cold start

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

@st.cache_data
def load_json(name):
    """Small JSON artifact from output/ (None if absent or unreadable)."""
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

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
def load_rf_regressor():
    p = os.path.join(MODELS, "recovery_regressor.pkl")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_resource
def load_embeddings():
    p = os.path.join(MODELS, "node_embeddings.pkl")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ── Feature names used by the recovery model ────────────────────────────────
# Must match FEATURE_COLS in src/recovery_predictor.py exactly (order and
# values). Stage 8 was rewritten to rank response strategies by predicted
# outcome instead of classifying which one a human historically picked (that
# label turned out to be near-unlearnable — see recovery_predictor.py's
# module docstring). response_type_enc is now an INPUT feature: at decision
# time we hold everything else fixed and sweep it across every strategy.
RF_FEATURES = [
    "disruption_type_enc", "supplier_size_enc", "disruption_severity",
    "production_impact_pct", "has_backup_supplier",
    "sev_x_backup", "impact_x_backup", "response_type_enc",
]

def unwrap_regressor(bundle):
    """models/recovery_regressor.pkl is now {model, feature_cols, response_map}
    (was a bare model before the Stage 8 rewrite). Handle both for safety."""
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], bundle.get("feature_cols", RF_FEATURES), bundle.get("response_map", {})
    return bundle, RF_FEATURES[:-1], {}   # legacy bare-model pickle, no response_type_enc

# ── Header ─────────────────────────────────────────────────────────────────
st.title("Immunological Supply Chain")
st.caption("Self-Healing Supply Chains with AI Digital Antibodies  |  PES University  |  PW26_RGP_01")
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────
(tab_overview, tab_graph, tab_anomaly, tab_risk, tab_recovery,
 tab_multi, tab_memory, tab_live, tab_ir) = st.tabs([
    "Overview",
    "Supply Chain Graph",
    "Anomaly Detection",
    "Risk Analysis",
    "Recovery Predictor",
    "Multi-Domain Risk",
    "Immune Memory",
    "Live Response",
    "Dataset IR",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 11 — DATASET IR (plug-and-play capability manifest)
# ══════════════════════════════════════════════════════════════════════════
with tab_ir:
    import subprocess, sys as _sys

    st.subheader("Dataset-Independent IR — Plug & Play")
    st.markdown(
        "Every dataset is mapped by a small **adapter** into one canonical "
        "Intermediate Representation (`nodes · edges · flows · events`). Models "
        "read the IR, never raw columns. A **capability manifest** records which "
        "fields each dataset actually populates, and every analytics stage runs "
        "only when its required capability is present — **a capability is never "
        "borrowed from another dataset.** That is what keeps this honest: ARCOS "
        "and SCMS light up *different* capabilities, and a future dataset that "
        "has all of them would run everything."
    )

    IR_ROOT   = os.path.join(OUT, "ir")
    ADAPTERS  = {"arcos": "src/arcos_adapter.py", "scms": "src/scms_adapter.py",
                 "dataco": "src/dataco_adapter.py"}
    CAP_ORDER = ["HAS_TRANSACTIONS", "HAS_TOPOLOGY", "HAS_DISRUPTIONS",
                 "HAS_LEAD_TIMES", "HAS_VALUE"]

    if st.button("🔄 (Re)build IR from adapters"):
        with st.spinner("Running adapters…"):
            for name, script in ADAPTERS.items():
                r = subprocess.run([_sys.executable, script], cwd=BASE,
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    st.error(f"{name} adapter failed:\n{r.stderr[-1500:]}")
                else:
                    st.success(f"{name} adapter OK")
        st.rerun()

    # Load whatever manifests exist on disk
    manifests = {}
    if os.path.isdir(IR_ROOT):
        for name in sorted(os.listdir(IR_ROOT)):
            mpath = os.path.join(IR_ROOT, name, "manifest.json")
            if os.path.exists(mpath):
                with open(mpath) as _fh:
                    manifests[name] = json.load(_fh)

    if not manifests:
        st.info("No IR built yet. Click **(Re)build IR from adapters** above.")
    else:
        # IR size summary
        size_rows = [{
            "dataset": n,
            "nodes":  m["summary"]["nodes"],
            "edges":  m["summary"]["edges"],
            "flows":  m["summary"]["flows"],
            "roles":  ", ".join(m["summary"]["roles"]),
        } for n, m in manifests.items()]
        st.markdown("**IR contents**")
        st.dataframe(pd.DataFrame(size_rows), use_container_width=True, hide_index=True)

        # Capability matrix: capabilities × datasets
        st.markdown("**Capability matrix**  (✅ = supported by the dataset's own fields)")
        cap_matrix = {"capability": CAP_ORDER}
        for n, m in manifests.items():
            caps = m["capabilities"]
            cap_matrix[n] = ["✅" if caps.get(c) else "—" for c in CAP_ORDER]
        st.table(pd.DataFrame(cap_matrix))

        # Stage-gating matrix: stages × datasets
        st.markdown("**Stage gating**  (RUN = all required capabilities present)")
        all_stages = sorted({s for m in manifests.values() for s in m["stages"]})
        gate = {"stage": all_stages}
        for n, m in manifests.items():
            col = []
            for s in all_stages:
                info = m["stages"].get(s, {})
                if info.get("enabled"):
                    col.append("RUN")
                else:
                    col.append("skip · needs " + ", ".join(info.get("missing", [])))
            gate[n] = col
        st.table(pd.DataFrame(gate))

        # ── Actually RUN the gated analytics off the IR ──────────────────
        st.markdown("**Run gated analytics on the IR**")
        st.caption(
            "The same code runs on every dataset — flow stages read only "
            "`output/ir/<ds>/flows.csv`; the two topology stages also read "
            "`nodes.csv` + `edges.csv`. A stage whose capability is missing is "
            "skipped — never fed a borrowed field from another dataset. "
            "Takes ~2 min (the PPO routing agent trains from scratch per dataset)."
        )
        if st.button("▶ Run gated analytics"):
            import ir_stages
            import importlib
            importlib.reload(ir_stages)
            for n in manifests:
                with st.spinner(f"Running gated stages on `{n}`…"):
                    try:
                        res = ir_stages.run(n)
                    except Exception as e:
                        st.error(f"{n}: {e}")
                        continue
                with open(os.path.join(IR_ROOT, n, "stage_results.json"), "w") as _fh:
                    json.dump(res, _fh, indent=2)
                st.success(f"`{n}` done — "
                           f"{sum(1 for o in res.values() if o['status'] == 'ran')} ran, "
                           f"{sum(1 for o in res.values() if o['status'] == 'skipped')} skipped")
            st.rerun()

        # ── Stage results (persisted by the run button or ir_pipeline.py) ─
        results = {}
        for n in manifests:
            rpath = os.path.join(IR_ROOT, n, "stage_results.json")
            if os.path.exists(rpath):
                with open(rpath) as _fh:
                    results[n] = json.load(_fh)

        if results:
            def _headline(stage, d):
                if "note" in d and "n_train" not in d and "n_nodes" not in d:
                    return d["note"]
                if stage in ("anomaly_detection", "disruption_detection"):
                    return (f"P={d.get('precision', 0):.3f}  R={d.get('recall', 0):.3f}  "
                            f"F1={d.get('f1', 0):.3f}")
                if stage == "safety_stock":
                    return f"${d.get('total_buffer_usd', 0):,.0f} buffer · {d.get('n_lanes_sized', 0)} lanes"
                if stage == "event_replay":
                    return f"{d.get('n_alerts', 0)} alerts / {d.get('windows_tested', 0)} windows"
                if stage == "value_at_risk":
                    return (f"${d.get('value_at_risk_usd', 0):,.0f} at risk "
                            f"({100 * d.get('late_value_share', 0):.1f}% of value)")
                if stage == "graph_risk_routing":
                    return (f"concentration lift ×{d.get('concentration_lift', 0)} · "
                            f"reroute {100 * (d.get('knockout_reroute_rate_top5') or 0):.1f}%")
                if stage == "policy_gradient_routing":
                    tot = d.get("avg_total_reward", {})
                    return (f"reward {tot.get('pg', 0):.1f} vs greedy "
                            f"{tot.get('risk_greedy', 0):.1f} · beats greedy "
                            f"{d.get('ppo_beats_greedy_pct', 0):.0f}%")
                return ""

            st.markdown("**Gated stage results**  (persisted from the last run)")
            res_rows = []
            for n, res in results.items():
                for stage, out in res.items():
                    res_rows.append({
                        "dataset": n,
                        "stage": stage,
                        "status": out["status"],
                        "headline": (_headline(stage, out) if out["status"] == "ran"
                                     else "needs " + ", ".join(out.get("missing", []))),
                    })
            st.dataframe(pd.DataFrame(res_rows), use_container_width=True, hide_index=True)

            # Topology stages get the spotlight: they are what the 3-tier SCMS
            # adapter unlocked, and the PPO-vs-baselines table is the payoff.
            graph_ds = [n for n, res in results.items()
                        if res.get("policy_gradient_routing", {}).get("status") == "ran"
                        and "avg_route_risk" in res["policy_gradient_routing"]]
            if graph_ds:
                st.markdown("**Topology stages — graph risk & cascade PPO routing**")
                cols = st.columns(len(graph_ds))
                for col, n in zip(cols, graph_ds):
                    gr  = results[n].get("graph_risk_routing", {})
                    ppo = results[n]["policy_gradient_routing"]
                    tot = ppo.get("avg_total_reward", {})
                    avg = ppo["avg_route_risk"]
                    uns = ppo.get("unserved", {})
                    with col:
                        st.markdown(f"**`{n}`** — {gr.get('n_nodes', '?')} nodes / "
                                    f"{gr.get('n_edges', '?')} supply links")
                        st.caption("top risk: " + ", ".join(gr.get("top_risk_nodes", [])[:3])
                                   + f" · cascade: {ppo.get('cascade', '')}")
                        st.dataframe(pd.DataFrame([
                            {"method": "PPO (linear, PPO-clip)",
                             "total reward": tot.get("pg"),
                             "route risk": avg.get("pg"),
                             "unserved": uns.get("pg")},
                            {"method": "Risk-greedy (myopic)",
                             "total reward": tot.get("risk_greedy"),
                             "route risk": avg.get("risk_greedy"),
                             "unserved": uns.get("risk_greedy")},
                            {"method": "Dijkstra (min-hop)",
                             "total reward": tot.get("dijkstra"),
                             "route risk": avg.get("dijkstra"),
                             "unserved": uns.get("dijkstra")},
                            {"method": "Random valid",
                             "total reward": tot.get("random"),
                             "route risk": avg.get("random"),
                             "unserved": uns.get("random")},
                        ]), use_container_width=True, hide_index=True)
                        st.caption(
                            f"beats greedy in {ppo.get('ppo_beats_greedy_pct', 0):.0f}% "
                            f"of {ppo['n_test']} episodes (greedy wins "
                            f"{ppo.get('greedy_beats_ppo_pct', 0):.0f}%) · {ppo['note']}"
                        )

        st.caption(
            "Built by `src/{arcos,scms,dataco}_adapter.py` over `src/ir_schema.py`; "
            "gated stages in `src/ir_stages.py`; headless runner "
            "`src/ir_pipeline.py` (Stage 26 in main.py). "
            "Pure stdlib — runs without pandas/torch."
        )

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab_overview:
    G    = load_graph()
    anom = load_anomalies()
    risk = load_risk()
    gnn  = load_gnn()
    fc   = load_forecast()

    n_nodes   = G.number_of_nodes()   if G    else "—"
    n_edges   = G.number_of_edges()   if G    else "—"
    n_anom    = len(anom) if not anom.empty else "—"   # ensemble-flagged (matches Stage 3 / paper)
    n_high    = len(risk[risk["risk_score"] > 0.7])   if not risk.empty else "—"

    forecast_level = "—"
    if not fc.empty:
        upcoming = fc[fc["type"] == "forecast"]
        if not upcoming.empty:
            forecast_level = upcoming.iloc[0]["stress_level"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Nodes in Graph",    f"{n_nodes:,}" if isinstance(n_nodes, int) else n_nodes)
    with c2: metric_card("Edges in Graph",    f"{n_edges:,}" if isinstance(n_edges, int) else n_edges)
    with c3: metric_card("High-Conf Anomalies", f"{n_anom:,}"  if isinstance(n_anom, int)  else n_anom, "calibrated ensemble")
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
    # Must match FEATURE_COLS in src/recovery_predictor.py exactly.
    # response_type_enc (the chosen strategy) is an input feature here — see
    # the module docstring in recovery_predictor.py for why.
    _OV_FEAT_NAMES = [
        "Disruption Type", "Supplier Size", "Disruption Severity",
        "Production Impact %", "Has Backup Supplier",
        "Severity x Backup", "Impact % x Backup", "Response Strategy",
    ]
    _OV_FEAT_COLS = [
        "disruption_type_enc", "supplier_size_enc", "disruption_severity",
        "production_impact_pct", "has_backup_supplier",
        "sev_x_backup", "impact_x_backup", "response_type_enc",
    ]

    if os.path.exists(_ov_reg_path):
        try:
            import joblib as _jl
            _ov_bundle = _jl.load(_ov_reg_path)
            _ov_model, _bundle_cols, _ = unwrap_regressor(_ov_bundle)
            if len(_bundle_cols) != len(_OV_FEAT_COLS):
                # legacy 7-feature pickle — drop the strategy column to match
                _OV_FEAT_NAMES = _OV_FEAT_NAMES[:-1]
                _OV_FEAT_COLS  = _OV_FEAT_COLS[:-1]
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
                        _ov_dis["sev_x_backup"]    = _ov_dis["disruption_severity"]   * _ov_dis["has_backup_supplier"]
                        _ov_dis["impact_x_backup"] = _ov_dis["production_impact_pct"] * _ov_dis["has_backup_supplier"]
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
with tab_graph:
    import networkx as nx
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
with tab_anomaly:
    anom = load_anomalies()

    if anom.empty:
        st.warning("No anomaly data found. Run Stage 3 first.")
    else:
        high = anom[anom["anomaly_score"] >= 2].copy()
        susp = anom[anom["anomaly_score"] == 1].copy()

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Flagged (ensemble)",      f"{len(anom):,}")
        with c2: metric_card("Corroborated (≥2 signals)", f"{len(high):,}")
        with c3: metric_card("Single-signal",           f"{len(susp):,}")

        st.caption(
            "Decision rule: calibrated logistic ensemble over the five detection "
            "signals (held-out F1 0.386; adjusted precision 0.86). All flagged rows "
            "are confirmed anomalies — the cards break them down by how many z-score "
            "signals corroborate each, not into confident vs. unconfident."
        )

        # ── Calibrated ensemble (Stage 17 held-out benchmark) ──────────────
        cal = load_json("anomaly_ensemble_calibration.json")
        if cal:
            st.markdown("#### Calibrated Ensemble — held-out benchmark (Stage 17)")
            st.caption(
                "A logistic model over the five detection signals, tuned on one "
                "injection realization (seed 42) and scored on four unseen seeds. "
                "Raw precision is capped by ~2,500 unlabeled organic anomalies in "
                "the base data; adjusted precision excludes rows the same rule "
                "already flags pre-injection."
            )
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                metric_card("Held-out F1", f"{cal.get('held_out_f1', float('nan')):.3f}",
                            "mean over 4 unseen seeds")
            with k2:
                metric_card("Precision (raw)", f"{cal.get('held_out_precision', float('nan')):.3f}",
                            f"adjusted {cal.get('held_out_precision_adjusted', float('nan')):.2f} excl. organic")
            with k3:
                metric_card("Recall", f"{cal.get('held_out_recall', float('nan')):.3f}")
            with k4:
                metric_card("Decision threshold", f"{cal.get('threshold', float('nan')):.2f}",
                            f"tuned on seed {cal.get('tuned_on_seed', '—')}")
            _feat = cal.get("features") or []
            _coef = cal.get("coef") or []
            if _feat and len(_feat) == len(_coef):
                cw1, cw2 = st.columns([3, 2])
                with cw1:
                    coef_df = pd.DataFrame({"signal": _feat, "weight": _coef})
                    fig_c = px.bar(
                        coef_df, x="weight", y="signal", orientation="h",
                        color="weight", color_continuous_scale="RdBu",
                        template="plotly_dark",
                        labels={"weight": "Logistic weight (per raw unit)", "signal": ""},
                    )
                    fig_c.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                                        coloraxis_showscale=False, height=240,
                                        margin=dict(l=10, r=10, t=10, b=30))
                    st.plotly_chart(fig_c, use_container_width=True)
                with cw2:
                    st.markdown(
                        f"**Intercept** `{cal.get('intercept', float('nan')):.2f}`  \n"
                        "Weights are per raw feature unit, so magnitudes are not "
                        "directly comparable across signals: the Isolation Forest "
                        "score lives on a narrow scale and carries a large weight, "
                        "while z-scores span several units each. Negative weights "
                        "on surge/concentration act as *vetoes* that trade recall "
                        "for precision."
                    )

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
with tab_risk:
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

        # ── GNN score: does it detect anything? (node-level injection benchmark) ──
        _gb = load_json("gnn_injection_eval.json")
        if _gb and _gb.get("summary"):
            _s = _gb["summary"]
            st.markdown("#### Is the GNN score a detector? — node-level injection benchmark")
            st.caption(
                f"On held-out seeds {_gb['seeds']}, {_gb['n_injected_per_seed']} structural anomalies "
                "are planted into a copy of the real graph (starved, flooded, or rewired nodes), the "
                "autoencoder is retrained from scratch on each copy, and its score is asked to rank the "
                "planted nodes against simple degree/volume z-score baselines. This is the only ground "
                "truth behind the 'structurally anomalous' label."
            )
            _b1, _b2, _b3, _b4 = st.columns(4)
            with _b1: metric_card("GNN AUC", f"{_s['gnn_auc']['mean']:.3f}", f"± {_s['gnn_auc']['std']:.3f} over seeds")
            with _b2: metric_card("Best baseline AUC", f"{max(_s['volume_z_auc']['mean'], _s['degree_z_auc']['mean'], _s['max_z_auc']['mean']):.3f}",
                                  "|z| volume / degree / max")
            with _b3: metric_card("GNN precision@k", f"{_s['gnn_p_at_k']['mean']:.2f}",
                                  f"baseline {_s['max_z_p_at_k']['mean']:.2f}")
            with _b4: metric_card("Recall by kind", f"{_s['gnn_recall_starve']['mean']:.0%} / {_s['gnn_recall_flood']['mean']:.0%} / {_s['gnn_recall_rewire']['mean']:.0%}",
                                  "starved / flooded / rewired")
            _rew = _s['gnn_recall_rewire']['mean']
            st.caption(
                ("The GNN ranks planted anomalies better than the degree/volume baselines" if _s['gnn_auc']['mean'] > _s['max_z_auc']['mean'] + 0.02
                 else "The GNN does not beat the degree/volume baselines")
                + (f", but it largely misses REWIRED nodes (recall {_rew:.0%}): it sees volume shocks, not changed connectivity." if _rew < 0.3 else ".")
            )
            with st.expander("Full benchmark report"):
                _gt = os.path.join(OUT, "gnn_injection_metrics.txt")
                if os.path.exists(_gt):
                    st.code(open(_gt).read(), language=None)

        # ── GNN explainer — why is this node risky? ────────────────────
        st.markdown("#### Why is this node risky? — GNN neighbourhood explainer")
        st.caption(
            "A similarity-and-tension heuristic over the node's k-hop neighbourhood "
            "(no gradients): each neighbour scores |cosine similarity| × its own GNN "
            "score, plus an embedding-tension term. Read it as *which neighbours look "
            "most implicated*, not as a causal attribution. Top drivers are the nodes "
            "to monitor or isolate first."
        )
        _G_x   = load_graph()
        _emb_x = load_embeddings()
        if gnn.empty or _G_x is None or _emb_x is None:
            st.info("Needs Stage 2 (graph) and Stage 7 (GNN embeddings + risk scores).")
        else:
            from gnn_explainer import explain_node_risk

            @st.cache_data
            def _gnn_risk_map(_n_rows: int):
                return {
                    r["entity"]: {
                        "gnn_score":   float(r.get("gnn_score", 0.0)),
                        "recon_error": float(r.get("recon_error", 0.0)),
                        "node_type":   str(r.get("node_type", "?")),
                        "rank":        int(r.get("rank", 9999)),
                    } for _, r in gnn.iterrows()
                }
            _rmap = _gnn_risk_map(len(gnn))
            _cands = (gnn.sort_values("enhanced_risk", ascending=False)["entity"]
                      .head(60).tolist())
            xc1, xc2, xc3 = st.columns([3, 1, 1])
            with xc1:
                _xnode = st.selectbox("Entity (top 60 by GNN-enhanced risk)", _cands, index=0,
                                      key="gnn_explain_node")
            with xc2:
                _xhops = st.slider("Hops", 1, 2, 1, key="gnn_explain_hops")
            with xc3:
                _xtop = st.slider("Top drivers", 3, 12, 6, key="gnn_explain_top")
            try:
                _xr = explain_node_risk(_xnode, hops=_xhops, top_n=_xtop, silent=True,
                                        artifacts=(_G_x, _emb_x, _rmap))
            except Exception as _xe:
                st.warning(f"Could not explain `{_xnode}`: {_xe}")
                _xr = None
            if _xr:
                xk1, xk2, xk3, xk4 = st.columns(4)
                with xk1: metric_card("Target", _xnode[:22], _xr["node_type"])
                with xk2: metric_card("GNN Risk", f"{_xr['node_risk']:.4f}", f"rank #{_xr['rank']}")
                with xk3: metric_card("Recon. Error", f"{_xr['recon_error']:.2e}", "autoencoder signal")
                with xk4: metric_card("Neighbours Scored", f"{len(_xr['top_drivers'])}",
                                      f"{_xhops}-hop neighbourhood")
                _xd = pd.DataFrame(_xr["top_drivers"])
                if not _xd.empty:
                    xg1, xg2 = st.columns([3, 2])
                    with xg1:
                        fig_x = px.bar(
                            _xd.sort_values("influence"), x="influence", y="neighbour",
                            orientation="h", color="type", template="plotly_dark",
                            color_discrete_map={"manufacturer": "#ab47bc",
                                                "distributor": "#4fc3f7",
                                                "retailer": "#ffb300"},
                            hover_data={"gnn_risk": ":.4f", "similarity": ":.3f"},
                            labels={"influence": "Influence on target's risk",
                                    "neighbour": "", "type": ""},
                        )
                        fig_x.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                                            height=max(240, 34 * len(_xd) + 80),
                                            margin=dict(l=10, r=20, t=30, b=40),
                                            legend=dict(orientation="h", y=1.12, x=0))
                        st.plotly_chart(fig_x, use_container_width=True)
                    with xg2:
                        st.dataframe(
                            _xd[["neighbour", "type", "gnn_risk", "similarity", "influence"]]
                              .rename(columns={"neighbour": "Neighbour", "type": "Type",
                                               "gnn_risk": "GNN risk", "similarity": "Cos. sim.",
                                               "influence": "Influence"}),
                            use_container_width=True, hide_index=True,
                            height=max(240, 36 * len(_xd) + 40))
                        _top = _xr["top_drivers"][0]
                        st.markdown(
                            f"**Primary driver:** `{_top['neighbour']}` ({_top['type']}, "
                            f"GNN risk {_top['gnn_risk']:.3f}). Its embedding is "
                            f"{'closely' if _top['similarity'] > 0.3 else 'weakly'} aligned "
                            f"with the target (cos {_top['similarity']:.2f}), so the two nodes "
                            "share the anomalous neighbourhood pattern the autoencoder "
                            "fails to reconstruct."
                        )

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

    # ── Macro-Stress Crisis Validation (Stage 27) ─────────────────────────
    st.divider()
    st.markdown("### Macro-Stress Crisis Validation — Stage 27")
    st.caption(
        "Does the Stage-9 stress composite rise at documented global crises it "
        "was never told about? Event study: mean of the 8-week event window "
        "minus the prior 12 weeks, ranked against a placebo distribution of "
        "non-event windows. DETECTED ≥ p90, PARTIAL ≥ p75. The composite is "
        "CAUSAL: each indicator is scored as a percentile rank of its own history "
        "up to that week and carried forward, never scaled by the full series. "
        "Nulls are shown, not hidden — they say what a US-macro lens cannot see."
    )
    _mv_path = os.path.join(OUT, "macro_event_validation.csv")
    if not os.path.exists(_mv_path):
        st.info("Run Stage 27 (`python3 src/macro_event_validation.py`) to generate the crisis validation.")
    else:
        mv = pd.read_csv(_mv_path, parse_dates=["start"])
        _n_det = int((mv["verdict"] == "DETECTED").sum())
        _n_par = int((mv["verdict"] == "PARTIAL").sum())
        _n_not = int(len(mv) - _n_det - _n_par)
        _mv_rpt = os.path.join(OUT, "macro_event_validation_report.txt")
        _n_placebo = "—"
        if os.path.exists(_mv_rpt):
            import re as _re
            _m = _re.search(r"(\d+)\s+non-event windows", open(_mv_rpt).read())
            if _m:
                _n_placebo = _m.group(1)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Crises Tested", f"{len(mv)}", f"{mv['start'].min().year}–{mv['start'].max().year}")
        with m2:
            metric_card("Detected (≥ p90)", f"{_n_det}", f"{_n_par} partial · {_n_not} not seen")
        with m3:
            _best = mv.loc[mv["delta"].idxmax()]
            metric_card("Largest Rise", f"+{_best['delta']:.3f}", str(_best["event"])[:26])
        with m4:
            metric_card("Placebo Windows", _n_placebo, "non-event 8-week windows")

        _vcol = {"DETECTED": "#4caf50", "PARTIAL": "#ffb300", "NOT SEEN": "#ef5350"}
        vc1, vc2 = st.columns([3, 2])
        with vc1:
            stress = load_stress()
            if not stress.empty and "stress_score" in stress.columns:
                fig_ms = go.Figure()
                fig_ms.add_trace(go.Scatter(
                    x=stress["date"], y=stress["stress_score"], mode="lines",
                    name="Macro stress", line=dict(color="#4fc3f7", width=1.6)))
                for _, r in mv.iterrows():
                    _c = _vcol.get(r["verdict"], "#999")
                    fig_ms.add_vrect(x0=r["start"], x1=r["start"] + pd.Timedelta(weeks=8),
                                     fillcolor=_c, opacity=0.18, line_width=0)
                    fig_ms.add_vline(x=r["start"], line=dict(color=_c, width=1, dash="dot"))
                    fig_ms.add_annotation(x=r["start"], y=1.0, yref="paper", showarrow=False,
                                          text=str(r["event"]).split(" ")[0], font=dict(size=10, color=_c),
                                          yanchor="bottom")
                fig_ms.update_layout(
                    template="plotly_dark", paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                    height=300, margin=dict(l=40, r=20, t=30, b=40),
                    xaxis_title="Week", yaxis_title="Composite stress",
                )
                st.plotly_chart(fig_ms, use_container_width=True)
        with vc2:
            fig_dv = px.bar(
                mv.sort_values("delta"), x="delta", y="event", orientation="h",
                color="verdict", color_discrete_map=_vcol, template="plotly_dark",
                text=mv.sort_values("delta")["placebo_percentile"].map(lambda v: f"p{v:.0f}"),
                labels={"delta": "Δ stress (event − prior 12 wk)", "event": "", "verdict": ""},
            )
            fig_dv.update_traces(textposition="outside")
            fig_dv.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                                 height=300, margin=dict(l=10, r=40, t=30, b=40),
                                 legend=dict(orientation="h", y=1.15, x=0))
            st.plotly_chart(fig_dv, use_container_width=True)

        st.dataframe(
            mv[["event", "start", "pre_mean", "window_peak", "delta", "placebo_percentile",
                "verdict", "top_driver", "driver_delta"]].rename(columns={
                "event": "Crisis", "start": "Onset", "pre_mean": "Pre (12 wk)",
                "window_peak": "Peak (8 wk)", "delta": "Δ", "placebo_percentile": "Placebo %ile",
                "verdict": "Verdict", "top_driver": "Top Driver", "driver_delta": "Driver Δ"}),
            use_container_width=True, height=230, hide_index=True)
        with st.expander("Full Stage 27 report + figure"):
            if os.path.exists(_mv_rpt):
                st.code(open(_mv_rpt).read(), language=None)
            _f11 = os.path.join(FIGS, "fig11_macro_event_validation.png")
            if os.path.exists(_f11):
                st.image(_f11, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# (Macro Stress & LSTM tab removed 2026-07-16 — LSTM did not beat naive
#  persistence on the smoothed macro-stress series; kept the module/outputs,
#  dropped the dashboard tab. The Overview 'NEXT-WEEK FORECAST' card still
#  reads output/stress_forecast via load_forecast().)
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — RECOVERY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
with tab_recovery:
    _reg_bundle = load_rf_regressor()
    dis_data    = load_disruption_data()
    regressor, _reg_cols, response_map = (
        unwrap_regressor(_reg_bundle) if _reg_bundle is not None else (None, RF_FEATURES, {})
    )
    if not response_map and not dis_data.empty:
        # legacy pickle without a bundled response_map — rebuild from data
        response_map = dict(zip(dis_data["response_type_enc"].astype(int), dis_data["response_type"]))
    HAS_STRATEGY_FEATURE = "response_type_enc" in _reg_cols

    # ── EVENT-REACTION HARNESS — watch the model respond to a shock ────────
    st.markdown("### 🎯 Event-Reaction Harness — Watch the Model Respond")
    st.caption(
        "Pick a shock and watch the immune response unfold step by step, on "
        "real SCMS data. REPLAY = a documented historical event (the detector "
        "runs blind, walk-forward, then the response ladder reacts using only "
        "pre-event knowledge). KNOCKOUT = remove a real manufacturer today and "
        "see which lanes flip to exposed, the annual value at risk, and what "
        "the system recommends."
    )
    _eh_path = os.path.join(OUT, "event_harness.json")
    if not os.path.exists(_eh_path):
        st.info("Run `python3 src/scms_event_harness.py` to build the scenarios.")
    else:
        import json as _json
        _scn = _json.load(open(_eh_path))
        _labels = [("🌍 REPLAY — " if s["type"] == "replay" else "🏭 KNOCKOUT — ")
                   + s.get("label", s.get("vendor", "")) for s in _scn]
        _pick = st.selectbox("Choose a shock to simulate", range(len(_scn)),
                             format_func=lambda i: _labels[i], key="eh_pick")
        s = _scn[_pick]

        if s["type"] == "replay":
            d = s["detection"]
            st.markdown(f"**Real event:** {s['documented']}  \n"
                        f"**Window:** {s['country']} · {s['window']}")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                metric_card("① Detection",
                            "🚨 ALERT" if d["fired"] else "no alert",
                            f"p = {d['p_value']}")
            with k2:
                metric_card("Late Rate in Window",
                            f"{d['late_rate']*100:.0f}%",
                            f"vs {d['baseline_rate']*100:.0f}% baseline")
            with k3:
                metric_card("② Disrupted Lanes", f"{s['summary']['disrupted_lanes']}",
                            f"{s['summary']['resolved']} reroutable, "
                            f"{s['summary']['exposed']} exposed")
            with k4:
                metric_card("③ Buffer Exposure",
                            f"${s['summary']['buffer_usd_total']/1e6:.2f}M",
                            "cost if absorbed by stock")
            st.markdown("**③ Per-lane reaction** (walk-forward — response uses "
                        "only pre-event data):")
            _rows = []
            for e in s["lanes"]:
                _rows.append({
                    "Molecule": e["molecule"],
                    "Disrupted": e["disrupted_vendor"],
                    "Tier": e["tier"].split("_")[0],
                    "System's Move": e["recommendation"],
                    "Coverage": (f"{e['coverage']*100:.0f}%"
                                 if e.get("coverage") is not None else ""),
                    "Buffer $": (f"${e['buffer']['buffer_usd']:,}"
                                 if e.get("buffer") else ""),
                })
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, height=300)
        else:
            sm = s["summary"]
            st.markdown(f"**Counterfactual:** what if **{s['vendor']}** went offline?")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                metric_card("Lanes Served", f"{sm['lanes_served']}")
            with k2:
                metric_card("① Still Covered", f"{sm['resilient']}",
                            "another vendor exists")
            with k3:
                metric_card("② Newly Exposed", f"{sm['newly_exposed']}",
                            "this vendor was the only one")
            with k4:
                metric_card("③ Value at Risk",
                            f"${sm['value_at_risk_usd_per_yr']/1e6:.2f}M/yr",
                            f"buffer to absorb: ${sm['buffer_usd_total']/1e6:.1f}M")
            st.markdown("**③ Newly-exposed lanes** (ranked by annual value at "
                        "risk) **and the system's recommended fix:**")
            _rows = [{
                "Molecule": e["molecule"], "Country": e["country"],
                "Value at Risk / yr": f"${e['value_at_risk_usd_per_yr']:,}",
                "System's Move": e["recommendation"],
                "Buffer $": (f"${e['buffer']['buffer_usd']:,}"
                             if e.get("buffer") else ""),
            } for e in s["lanes"]]
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, height=300)

        _eh_rpt = os.path.join(OUT, "event_harness_report.txt")
        if os.path.exists(_eh_rpt):
            with st.expander("Full step-by-step traces (all scenarios)"):
                st.code(open(_eh_rpt).read(), language=None)

    # ── SCMS SPINE — Real Unified Dataset (north-star) ─────────────────────
    st.divider()
    st.markdown("### SCMS Spine — Real Unified Dataset (north-star)")
    st.caption(
        "The rest of this project (ARCOS) has a real NETWORK but no real "
        "disruptions — recovery numbers had to be borrowed from a separate "
        "synthetic table, and the two share no entities. This section is the "
        "fix, and it leads the tab because it is the strongest: the SCMS "
        "health-commodity dataset carries the NETWORK "
        "(Vendor → Molecule → Country) AND real DISRUPTIONS "
        "(late shipments, real delay durations) in the SAME rows. No synthetic "
        "table, no cross-dataset subtraction — every number below is one "
        "coherent real dataset."
    )

    _scms_path = os.path.join(OUT, "scms_response_plan.csv")
    if not os.path.exists(_scms_path):
        st.info("Run `python3 src/scms_spine.py` to generate the SCMS analysis.")
    else:
        sdf = pd.read_csv(_scms_path)
        n_disr   = len(sdf)
        # Honest categories (mirrors src/scms_spine.py):
        #   RESOLVED = T1 proven in-country alternate (drop-in)
        #   EXPOSED  = T2 (global only, needs onboarding) + T5 (none anywhere)
        s_res = sdf[sdf["action"] == "T1_ALT_VENDOR_SAME_LANE"]
        s_exp = sdf[sdf["action"].isin(
            ["T2_ALT_VENDOR_OTHER_COUNTRY", "T5_SINGLE_SOURCED"])]
        s_cov = pd.to_numeric(s_res["volume_coverage"], errors="coerce").dropna()

        st.markdown(
            "**A disruption here is REAL:** a shipment that arrived after its "
            "scheduled date, with the delay as its actual duration. A "
            "*proven in-country alternate* is a different vendor already "
            "shipping the same molecule to the same country — a genuine "
            "drop-in backup. *Exposed* lanes have no in-country alternate "
            "(only a global onboarding candidate, or none at all) — the honest "
            "answer there is a buffer or a vendor-qualification plan, not a "
            "pretend reroute."
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Real Disruptions", f"{n_disr:,}",
                        f"median {int(sdf['delay_days'].median())}d late")
        with c2:
            res_pct = (len(s_res) / n_disr * 100) if n_disr else 0
            metric_card("Proven In-Country Alternate", f"{len(s_res):,}",
                        f"{res_pct:.0f}%")
        with c3:
            if len(s_cov):
                full = int((s_cov >= 0.999).sum())
                metric_card("Avg Volume Coverage", f"{s_cov.mean()*100:.0f}%",
                            f"{full}/{len(s_cov)} fully covered")
            else:
                metric_card("Avg Volume Coverage", "n/a")
        with c4:
            exp_pct = (len(s_exp) / n_disr * 100) if n_disr else 0
            metric_card("Exposed (no in-country)", f"{len(s_exp):,}",
                        f"{exp_pct:.0f}% · need buffer")

        st.markdown(
            "**Sample resolved disruptions** — real late shipment, real "
            "in-country alternate vendor found, with the volume it can absorb:"
        )
        _show = s_res.copy()
        _show["volume_coverage"] = pd.to_numeric(
            _show["volume_coverage"], errors="coerce"
        ).map(lambda v: f"{v*100:.0f}%" if pd.notna(v) else "—")
        s_cols = ["vendor_disrupted", "molecule", "country", "delay_days",
                  "severity", "alternate_vendor", "volume_coverage", "route"]
        s_cols = [c for c in s_cols if c in _show.columns]
        st.dataframe(
            _show[s_cols].head(200).rename(columns={
                "vendor_disrupted": "Disrupted Vendor",
                "molecule": "Molecule",
                "country": "Country",
                "delay_days": "Delay (days)",
                "severity": "Severity",
                "alternate_vendor": "In-Country Alternate",
                "volume_coverage": "Volume Coverage",
            }).reset_index(drop=True),
            use_container_width=True, height=300
        )

        _scms_rpt = os.path.join(OUT, "scms_spine_report.txt")
        if os.path.exists(_scms_rpt):
            with st.expander("Full SCMS spine report"):
                st.code(open(_scms_rpt).read(), language=None)

    # ── SCMS Exposure Answer — Sized & Priced Safety Stock ─────────────────
    st.markdown("#### 💰 The Exposure Answer — Sized & Priced Safety-Stock Buffers")
    st.caption(
        "For the exposed lanes above (no in-country alternate vendor), the "
        "correct response is a buffer, not a reroute. Sized with King's "
        "formula — SS = z·√(L·σ_d² + D²·σ_L²) — from each lane's REAL demand "
        "history and REAL vendor lead times (PO → Delivered), priced at the "
        "lane's own median pack price. Assumptions: 95% service (z=1.65), "
        "25%/yr holding cost. Sizes are planning magnitudes (demand is "
        "lumpy), not SKU-level truth."
    )
    _ss_path = os.path.join(OUT, "scms_safety_stock.csv")
    if not os.path.exists(_ss_path):
        st.info("Run `python3 src/scms_safety_stock.py` to size the buffers.")
    else:
        ssdf = pd.read_csv(_ss_path)
        _n_exposed = ""
        _sp_path = os.path.join(OUT, "scms_response_plan.csv")
        if os.path.exists(_sp_path):
            _sp = pd.read_csv(_sp_path)
            _n_exposed = _sp[_sp["action"].isin(
                ["T2_ALT_VENDOR_OTHER_COUNTRY", "T5_SINGLE_SOURCED"]
            )].groupby(["molecule", "country"]).ngroups
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            metric_card("Exposed Lanes Sized", f"{len(ssdf)}",
                        f"of {_n_exposed} flagged by the spine" if _n_exposed else
                        "flagged by the spine")
        with s2:
            metric_card("Total Buffer @95%",
                        f"${ssdf['buffer_value_usd_95'].sum()/1e6:.1f}M",
                        f"{ssdf['buffer_value_usd_95'].sum()/ssdf['annual_flow_value_usd'].sum()*100:.0f}% of annual flow")
        with s3:
            metric_card("Median Days of Cover",
                        f"{ssdf['days_of_cover_95'].median():.0f}d",
                        "≈ real PEPFAR-era stock norms")
        with s4:
            metric_card("Holding Cost / yr",
                        f"${ssdf['holding_cost_usd_per_yr'].sum()/1e6:.1f}M",
                        "the measurable price of no backup")
        st.markdown(
            "**Largest buffers** — the carrying cost of lumpy demand on long "
            "(mean 115d) single-sourced pipelines. This dollar figure is the "
            "quantified business case for qualifying a second vendor:"
        )
        _top = ssdf.sort_values("buffer_value_usd_95", ascending=False).head(10)
        st.dataframe(
            _top[["molecule", "country", "safety_stock_packs_95",
                  "days_of_cover_95", "buffer_value_usd_95",
                  "holding_cost_usd_per_yr", "lead_source"]].rename(columns={
                "safety_stock_packs_95": "SS (packs)",
                "days_of_cover_95": "Cover (days)",
                "buffer_value_usd_95": "Buffer ($)",
                "holding_cost_usd_per_yr": "Holding ($/yr)",
                "lead_source": "Lead-Time Source"}).reset_index(drop=True),
            use_container_width=True, height=260)
        _ss_rpt = os.path.join(OUT, "scms_safety_stock_report.txt")
        if os.path.exists(_ss_rpt):
            with st.expander("Full sizing report (incl. lanes not sizeable + caveats)"):
                st.code(open(_ss_rpt).read(), language=None)

    # ── SCMS Counter-Offer — Vendor Scorecard & Qualification Shortlist ────
    st.markdown("#### 🏭 The Counter-Offer — Vendor Scorecard & Qualification Shortlist")
    st.caption(
        "Instead of paying the holding cost forever, qualify a second vendor. "
        "For every exposed lane, candidates that already ship this molecule "
        "elsewhere are ranked by 0.35·reliability (on-time rate, shrunk k=5 so "
        "tiny samples can't fake perfection) + 0.35·capacity + 0.30·experience. "
        "Weights are stated, not fitted — and validated walk-forward: under "
        "the same backtest protocol this ranking matches real procurement's "
        "eventual vendor slightly BETTER than the planner rule (hit@3 90.0% "
        "vs 88.3% conditional; see src/test_vendor_scorecard.py, 18/18 "
        "assertions pass)."
    )
    _vs_path = os.path.join(OUT, "scms_vendor_scorecard.csv")
    if not os.path.exists(_vs_path):
        st.info("Run `python3 src/scms_vendor_scorecard.py` to build the shortlists.")
    else:
        vs = pd.read_csv(_vs_path)
        _delta = pd.to_numeric(vs["r1_buffer_delta_usd"], errors="coerce")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            metric_card("Lanes Shortlisted",
                        f"{int((vs['n_candidates'] > 0).sum())}",
                        f"of {len(vs)} exposed (rest: buffer-only)")
        with v2:
            metric_card("Median Candidates/Lane",
                        f"{vs.loc[vs['n_candidates'] > 0, 'n_candidates'].median():.0f}")
        with v3:
            metric_card("Potential Buffer Release",
                        f"${_delta[_delta > 0].sum()/1e6:.1f}M",
                        "what-if, top candidate's lead profile")
        with v4:
            metric_card("Candidate Pipeline Slower",
                        f"{int((_delta <= 0).sum())} lanes",
                        "negative deltas shown, not hidden")
        st.markdown("**Top qualification targets** — ranked by buffer released "
                    "if the lane's lead-time profile matched the candidate's:")
        _vt = vs[_delta > 0].assign(_d=_delta[_delta > 0]) \
                .sort_values("_d", ascending=False).head(10)
        st.dataframe(
            _vt[["molecule", "country", "r1_vendor", "r1_otif",
                 "r1_lead_mean_days", "buffer_now_usd",
                 "r1_buffer_whatif_usd", "r1_buffer_delta_usd"]].rename(columns={
                "r1_vendor": "Qualify This Vendor", "r1_otif": "OTIF",
                "r1_lead_mean_days": "Lead (days)",
                "buffer_now_usd": "Buffer Now ($)",
                "r1_buffer_whatif_usd": "Buffer What-If ($)",
                "r1_buffer_delta_usd": "Released ($)"}).reset_index(drop=True),
            use_container_width=True, height=260)
        _vs_rpt = os.path.join(OUT, "scms_vendor_scorecard_report.txt")
        if os.path.exists(_vs_rpt):
            with st.expander("Full scorecard report + honest caveats"):
                st.code(open(_vs_rpt).read(), language=None)

    # ── SCMS Validation 1 — Revealed-Preference Backtest ───────────────────
    st.markdown("#### ✅ Validation 1 — Revealed-Preference Backtest (walk-forward)")
    st.caption(
        "At each lane's FIRST late delivery, the planner builds a vendor "
        "shortlist using only data available at that moment. Ground truth = "
        "the brand-new vendor real procurement actually brought onto that "
        "lane afterwards. A hit means an independent, real-world process "
        "arrived at the same vendor the planner named in advance. Claim: "
        "consistency & feasibility — NOT causation."
    )
    _bt_path = os.path.join(OUT, "scms_backtest.csv")
    if not os.path.exists(_bt_path):
        st.info("Run `python3 src/scms_backtest.py` to generate the backtest.")
    else:
        bt = pd.read_csv(_bt_path)
        _cond = bt[bt["nameable_at_t0"] > 0]
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            metric_card("Evaluable Lanes", f"{len(bt)}",
                        "new vendor really entered")
        with b2:
            metric_card("Pool Containment", f"{len(_cond)/len(bt)*100:.0f}%",
                        "future vendor was nameable")
        with b3:
            metric_card("Planner Hit@3", f"{_cond['hit3_ours'].mean()*100:.0f}%",
                        f"random: {_cond['p_hit3_rand'].mean()*100:.0f}%")
        with b4:
            metric_card("Planner Hit@1", f"{_cond['hit1_ours'].mean()*100:.0f}%",
                        f"random: {_cond['p_hit1_rand'].mean()*100:.0f}%")
        with st.expander("Per-lane results + full protocol report"):
            _bt_rpt = os.path.join(OUT, "scms_backtest_report.txt")
            if os.path.exists(_bt_rpt):
                st.code(open(_bt_rpt).read(), language=None)
            st.dataframe(
                bt[["molecule", "country", "t0", "pool_size", "entrants",
                    "top3_ours", "hit3_ours"]].rename(columns={
                        "t0": "First Late", "pool_size": "Pool",
                        "entrants": "Vendor(s) Reality Chose",
                        "top3_ours": "Planner Top-3", "hit3_ours": "Hit@3"}),
                use_container_width=True, height=260)

    # ── SCMS Validation 2 — Real-Event Replay (Haiti 2010) ─────────────────
    st.markdown("#### ✅ Validation 2 — Real-Event Replay: Haiti Earthquake, Jan 2010")
    st.caption(
        "A blind walk-forward detector (3-month windows, binomial surprise vs "
        "each country's OWN past late-rate; fixed settings, no per-event "
        "tuning) is run over all countries and years. The pre-registered "
        "question: does it flag Haiti in 2010 without being told about the "
        "earthquake?"
    )
    _er_path = os.path.join(OUT, "scms_event_replay.csv")
    if not os.path.exists(_er_path):
        st.info("Run `python3 src/scms_event_replay.py` to generate the event replay.")
    else:
        er = pd.read_csv(_er_path)
        _al = er[er["alert"] == 1]
        _ht = _al[(_al["country"] == "Haiti")
                  & (_al["window_end"].astype(str).str.startswith(("2010", "2011")))]
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            metric_card("Windows Tested", f"{len(er):,}", "43 countries, 2006–2015")
        with e2:
            metric_card("Haiti 2010 Flagged?", "YES" if len(_ht) else "NO",
                        f"first: {_ht['window_end'].min()}" if len(_ht) else "")
        with e3:
            if len(_ht):
                _pk = _ht.loc[pd.to_numeric(_ht["p_value"]).idxmin()]
                metric_card("Peak Significance", f"p={float(_pk['p_value']):.1e}",
                            f"{int(_pk['late_window'])}/{int(_pk['n_window'])} late vs "
                            f"{float(_pk['baseline_rate'])*100:.0f}% baseline")
            else:
                metric_card("Peak Significance", "—")
        with e4:
            metric_card("Alert Windows Total", f"{len(_al)}",
                        f"≈{max(1, round(len(er)*0.001))} expected by chance")
        with st.expander("All episodes ranked by significance (full report)"):
            _er_rpt = os.path.join(OUT, "scms_event_replay_report.txt")
            if os.path.exists(_er_rpt):
                st.code(open(_er_rpt).read(), language=None)

    # ── SCMS Validation 3 — Documented Real-World Events ───────────────────
    st.markdown("#### 📜 Validation 3 — Documented Real-World Events (case table)")
    st.caption(
        "Eleven documented disruptions (Tōhoku/Renesas, KFC-DHL, Suez, COVID "
        "PPE, Texas freeze, Colonial Pipeline, …) hand-encoded and scored on "
        "two questions: would this system's exposure machinery have flagged "
        "the vulnerability EX ANTE, and does its recommended move MATCH what "
        "firms actually did? Three events were additionally found blind in "
        "our own delivery data by the Stage-21 detector — those claims are "
        "machine-cross-checked at build time. Scores are deliberately not "
        "all YES: three vocabulary gaps are surfaced as limitations. The "
        "claim is consistency with reality, never 'we would have done better'."
    )
    _ct_path = os.path.join(OUT, "case_table.csv")
    if not os.path.exists(_ct_path):
        st.info("Run `python3 src/case_table.py` to build the case table.")
    else:
        ct = pd.read_csv(_ct_path)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Events Encoded", f"{len(ct)}",
                        "3 found blind in-data")
        with c2:
            metric_card("Ex-Ante Flag",
                        f"{int((ct['ex_ante']=='YES').sum())} YES / "
                        f"{int((ct['ex_ante']=='PARTIAL').sum())} PARTIAL",
                        f"{int((ct['ex_ante']=='NO').sum())} NO — shown, not hidden")
        with c3:
            metric_card("Response Match",
                        f"{int((ct['match']=='YES').sum())} YES / "
                        f"{int((ct['match']=='PARTIAL').sum())} PARTIAL",
                        f"{int((ct['match']=='NO').sum())} NO")
        with c4:
            metric_card("Vocabulary Gaps Found", "3",
                        "routes · substitution · mode granularity")
        st.dataframe(
            ct[["event", "date", "failure_mode", "ex_ante", "match",
                "in_data"]].rename(columns={
                "event": "Event", "date": "Date",
                "failure_mode": "Failure Mode (our vocabulary)",
                "ex_ante": "Ex-Ante", "match": "Match",
                "in_data": "In-Data Evidence (Stage 21)"}),
            use_container_width=True, height=300)
        with st.expander("Full case narratives, rationales + sources"):
            _ct_rpt = os.path.join(OUT, "case_table_report.txt")
            if os.path.exists(_ct_rpt):
                st.code(open(_ct_rpt).read(), language=None)
            st.dataframe(ct, use_container_width=True, height=300)

    # ── SCMS Validation 4 — Outcome Counterfactual (Stage 28) ──────────────
    st.markdown("#### ⚖️ Validation 4 — Outcome Counterfactual: planner's pick vs procurement's real choice")
    st.caption(
        "At each lane's first late shipment, realized delivery performance over "
        "the following year for the vendor procurement actually used (A) vs the "
        "walk-forward planner's top pick (B), both scored at the same molecule "
        "scope. A random-pool vendor lands close to the planner's pick, so the "
        "gap vs the incumbent is evidence for *switching at all*, not for the "
        "exact ranking (that is Validation 1's job). Observational, not causal."
    )
    _cf_path = os.path.join(OUT, "scms_counterfactual.csv")
    if not os.path.exists(_cf_path):
        st.info("Run Stage 28 (`python3 src/scms_counterfactual.py`) to generate the counterfactual.")
    else:
        cf = pd.read_csv(_cf_path)
        _cfs = load_json("scms_counterfactual_stats.json") or {}
        _n = len(cf)
        _wins = int(cf["rec_wins_late_rate"].sum())
        _ties = int(cf["rec_ties_late_rate"].sum())
        _worse = _n - _wins - _ties
        _a_lr, _b_lr = cf["a_late_rate"].mean(), cf["b_late_rate"].mean()
        _r_lr = cf["r_late_rate"].dropna().mean()
        _saved = cf["lateness_saved_days"]
        _ci = lambda k, f: (f"95% CI [{f(_cfs[k][0])}, {f(_cfs[k][1])}]" if k in _cfs else "")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            metric_card("Comparable Lanes", f"{_n}", "both vendors observed post-t0")
        with f2:
            metric_card("Planner Pick Was Better", f"{_wins}/{_n}",
                        _ci("win_rate_ci95", lambda v: f"{v*100:.0f}%")
                        or f"{_wins/_n*100:.0f}% · tied {_ties} · worse {_worse}")
        with f3:
            metric_card("Mean Late Rate A → B", f"{_a_lr*100:.1f}% → {_b_lr*100:.1f}%",
                        (f"gap {_cfs['late_rate_gap']*100:+.1f} pts, "
                         + _ci("late_rate_gap_ci95", lambda v: f"{v*100:+.1f}"))
                        if "late_rate_gap" in _cfs else f"random pool vendor {_r_lr*100:.1f}%")
        with f4:
            metric_card("Lateness Saved / Shipment", f"{_saved.mean():+.1f} d",
                        _ci("lateness_saved_ci95", lambda v: f"{v:+.1f}")
                        or f"median {_saved.median():+.1f} d")
        if _cfs:
            st.caption(
                f"Bootstrap over lanes, {_cfs.get('n_boot', 0):,} resamples. Exact sign test on "
                f"wins vs losses (ties dropped): {_cfs.get('wins')} vs {_cfs.get('losses')}, "
                f"p = {_cfs.get('sign_test_p', float('nan')):.1e}. Random pool vendor late rate "
                f"{_r_lr*100:.1f}% · tied {_ties} · actual better {_worse}."
            )
        g1, g2 = st.columns([2, 3])
        with g1:
            _bar = pd.DataFrame({
                "vendor": ["Actual choice (A)", "Planner pick (B)", "Random pool"],
                "late_rate": [_a_lr * 100, _b_lr * 100, _r_lr * 100],
                "lateness": [cf["a_lateness_days"].mean(), cf["b_lateness_days"].mean(),
                             cf["r_lateness_days"].dropna().mean()],
            })
            fig_cf = go.Figure()
            fig_cf.add_trace(go.Bar(x=_bar["vendor"], y=_bar["late_rate"], name="Late rate (%)",
                                    marker_color=["#ef5350", "#4caf50", "#78909c"],
                                    text=_bar["late_rate"].map(lambda v: f"{v:.1f}%"),
                                    textposition="outside"))
            fig_cf.update_layout(template="plotly_dark", paper_bgcolor="#0f0f1a",
                                 plot_bgcolor="#0f0f1a", height=280, showlegend=False,
                                 yaxis_title="Late rate, year after t0 (%)",
                                 margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(fig_cf, use_container_width=True)
        with g2:
            fig_sv = px.histogram(
                cf, x="lateness_saved_days", nbins=25, template="plotly_dark",
                color_discrete_sequence=["#4fc3f7"],
                labels={"lateness_saved_days": "Lateness-days saved per shipment (B vs A)"},
            )
            fig_sv.add_vline(x=0, line=dict(color="#ef5350", dash="dash"))
            fig_sv.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", height=280,
                                 yaxis_title="Lanes", margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(fig_sv, use_container_width=True)
        if _cfs.get("strata"):
            st.markdown("**Sensitivity — does the gap survive within strata?**")
            st.caption(
                "Selection concern: the incumbent is scored while its book is in trouble. "
                "Tertiles on incumbent post-alert volume and on lane maturity at t0 hold "
                "those fixed. Where a stratum reverses, the report says so."
            )
            _srows = []
            for _grp, _lab in (("incumbent_volume", "Incumbent volume"), ("lane_tenure", "Lane tenure")):
                for s in _cfs["strata"][_grp]["rows"]:
                    _srows.append({
                        "Stratum": f"{_lab} · {s['stratum']}", "Lanes": s["n"],
                        "Win rate": f"{s['win_rate']*100:.0f}%",
                        "A late": f"{s['a_late_rate']*100:.1f}%",
                        "B late": f"{s['b_late_rate']*100:.1f}%",
                        "Gap A−B": f"{(s['a_late_rate']-s['b_late_rate'])*100:+.1f} pts",
                        "Saved d/ship": f"{s['lateness_saved']:+.1f}",
                        "Holds?": "yes" if s["a_late_rate"] > s["b_late_rate"] else "REVERSES",
                    })
            st.dataframe(pd.DataFrame(_srows), use_container_width=True, hide_index=True,
                         height=36 * len(_srows) + 40)
        with st.expander("Per-lane outcomes + full protocol report"):
            _cf_rpt = os.path.join(OUT, "scms_counterfactual_report.txt")
            if os.path.exists(_cf_rpt):
                st.code(open(_cf_rpt).read(), language=None)
            st.dataframe(
                cf[["molecule", "country", "t0", "actual_vendor", "recommended_vendor",
                    "a_n", "a_late_rate", "b_n", "b_late_rate", "lateness_saved_days"]].rename(columns={
                    "molecule": "Molecule", "country": "Country", "t0": "First Late",
                    "actual_vendor": "Actual Vendor (A)", "recommended_vendor": "Planner Pick (B)",
                    "a_n": "A ships", "a_late_rate": "A late", "b_n": "B ships",
                    "b_late_rate": "B late", "lateness_saved_days": "Days saved/ship"}),
                use_container_width=True, height=300, hide_index=True)
            _f12 = os.path.join(FIGS, "fig12_scms_counterfactual.png")
            if os.path.exists(_f12):
                st.image(_f12, use_container_width=True)

    # ── Network-Grounded Response Plan (Stage 18) ──────────────────────────
    st.divider()
    st.markdown("### Network-Grounded Response Plan — REVISED (capacity-grounded)")
    st.caption(
        "REVISED METHODOLOGY (for A/B against app.py). Unlike the abstract "
        "what-if calculator (demoted to the bottom of this tab), which has no "
        "knowledge of your graph, this section searches the real ARCOS graph "
        "for a proven alternative — a "
        "distributor already connected to both the manufacturer and the "
        "retailer — but it reports only claims the REAL data can support: "
        "whether that alternate can actually absorb the lost volume, and how "
        "many nodes are genuinely single-sourced. The fabricated "
        "'days saved' figure (a subtraction across two unrelated tables) is "
        "removed; recovery-days appears only as a labelled population prior."
    )

    _plan_path = os.path.join(OUT, "response_plan.csv")
    if not os.path.exists(_plan_path):
        st.info("Run Stage 18 (src/response_planner.py) to generate the response plan.")
    else:
        plan_df = pd.read_csv(_plan_path).reset_index(drop=True)

        # ── REVISED METHODOLOGY: compute capacity coverage from REAL data ────
        # per-distributor spare-capacity proxy = typical volume per served link
        # (out_volume / out_degree), straight from the Stage-2 graph.
        _cap = {}
        try:
            _g = pd.read_csv(os.path.join(OUT, "graph_risk_scores.csv"))
            _gd = _g[_g["type"] == "distributor"]
            for _, r in _gd.iterrows():
                od = max(float(r.get("out_degree", 1) or 1), 1.0)
                _cap[r["entity"]] = float(r.get("out_volume", 0) or 0) / od
        except Exception:
            pass

        # lost volume per anomaly (positional: plan row i ↔ anomaly row i;
        # both are 2,283 rows generated from the same anomalies.csv pass).
        _lost = None
        try:
            _a = pd.read_csv(os.path.join(OUT, "anomalies.csv"))
            if len(_a) == len(plan_df) and "quantity" in _a.columns:
                _lost = _a["quantity"].abs().values
        except Exception:
            pass

        def _coverage(i, entity):
            if _lost is None or entity not in _cap:
                return None
            lost = _lost[i]
            if not lost or lost <= 0:
                return None
            return min(1.0, _cap[entity] / lost)

        plan_df["volume_coverage"] = [
            _coverage(i, plan_df.at[i, "entity"]) for i in plan_df.index
        ]

        nb_df = plan_df[plan_df["no_listed_backup"] == True]
        # Resolved = network found a PROVEN supplier already linked to the
        # retailer (T1/T2). This replaces the old days_saved>0 filter with the
        # same set, defined structurally instead of via a fabricated number.
        PROVEN = ["T1_REROUTE_EXISTING", "T2_DEFACTO_ALT_SUPPLIER"]
        resolved_df = nb_df[nb_df["action"].isin(PROVEN)]
        # Structural exposure = no proven alternate; needs a buffer, not a reroute.
        exposed_df = nb_df[~nb_df["action"].isin(PROVEN)]

        st.markdown(
            "**What we claim vs. what we don't.** The network can *prove* two "
            "things from real transaction data: (1) whether a proven alternate "
            "exists, and (2) whether that alternate has the **capacity** to "
            "absorb the disrupted shipment's volume. It **cannot** prove a "
            "recovery-time saving — that would require joining our ARCOS graph "
            "to an unrelated synthetic disruption table, which we no longer do. "
            "Recovery-days below is a historical population prior, shown for "
            "context only and never subtracted."
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("No Listed Backup", f"{len(nb_df):,}")
        with c2:
            resolved_pct = (len(resolved_df) / len(nb_df) * 100) if len(nb_df) else 0
            metric_card("Resolved to Proven Supplier", f"{len(resolved_df):,}", f"{resolved_pct:.0f}%")
        with c3:
            _cov = resolved_df["volume_coverage"].dropna()
            if len(_cov):
                full = int((_cov >= 0.999).sum())
                metric_card("Avg Volume Coverage", f"{_cov.mean()*100:.0f}%",
                            f"{full}/{len(_cov)} fully covered")
            else:
                metric_card("Avg Volume Coverage", "n/a")
        with c4:
            metric_card("Single-Sourced Exposure", f"{len(exposed_df):,}",
                        "need safety-stock buffer")

        if len(resolved_df):
            st.markdown(
                "**Resolved cases** — no listed backup, network found a proven "
                "supplier. *Volume coverage* = share of the disrupted shipment "
                "the alternate can absorb from its own spare throughput "
                "(1.00 = fully covered). *Recovery prior* is context only."
            )
            _disp = resolved_df.copy()
            _disp["volume_coverage"] = _disp["volume_coverage"].map(
                lambda v: f"{v*100:.0f}%" if pd.notna(v) else "—"
            )
            show_cols = ["manufacturer", "entity", "retailer", "route",
                         "volume_coverage", "activation_days", "est_recovery_days"]
            show_cols = [c for c in show_cols if c in _disp.columns]
            st.dataframe(
                _disp[show_cols].rename(columns={
                    "entity": "Backup Supplier Found",
                    "volume_coverage": "Volume Coverage",
                    "activation_days": "Activation (days)",
                    "est_recovery_days": "Recovery Prior — context only (days)",
                }).reset_index(drop=True),
                use_container_width=True, height=280
            )

        if len(exposed_df):
            st.markdown(
                f"**Structural exposure** — {len(exposed_df):,} no-backup cases "
                "have *no proven alternate* in the network (only a new-link or "
                "customer-delay option). These are the genuinely single-sourced "
                "nodes: the correct answer is a sized inventory buffer, not a "
                "reroute. Sizing belongs in the backend planner — flagged here "
                "so the count isn't hidden inside a rosy 'resolved' headline."
            )

        _report_path = os.path.join(OUT, "response_plan_report.txt")
        if os.path.exists(_report_path):
            with st.expander("Full Stage 18 report"):
                st.code(open(_report_path).read(), language=None)

    if regressor is None:
        st.warning("Run Stage 8 first to train the recovery model.")
    else:
        st.divider()
        st.markdown("#### ⬇️ Baseline (demoted): Abstract What-If Strategy Calculator")
        st.warning(
            "⚠️ **This is the OLD approach, kept only for contrast.** It is an "
            "ABSTRACT what-if calculator driven by the *synthetic* disruption "
            "table — it takes the sliders below, not your real supply graph, so "
            "it can never name a real backup supplier, and its recovery numbers "
            "cross an illegitimate dataset seam. The two sections **above** — "
            "**SCMS Spine** (real unified dataset) and **Network-Grounded "
            "Response Plan** (real graph, capacity-checked) — are the actual "
            "methodology. This calculator is shown so you can see what we moved "
            "away from, not as something to rely on."
        )
        st.caption(
            "Stage 8 does not classify which strategy a human historically picked — "
            "that label turned out to be near-unlearnable (mutual information ≈ 0 for "
            "every feature except backup-supplier status; see recovery_predictor.py). "
            "Instead it predicts recovery time **under each candidate strategy** and "
            "recommends whichever is fastest. This is decision support based on "
            "historical association, not a causal guarantee — the per-case number "
            "carries real error, but the ranking is a stable effect over thousands "
            "of past disruptions."
        )
        st.caption(
            "Industry and Supplier Region are not shown here — mutual-information "
            "analysis found they carry near-zero signal for this prediction "
            "(MI ≈ 0.0002–0.0011) and the current model doesn't use them."
        )

        if not dis_data.empty:
            dis_types   = sorted(dis_data["disruption_type"].dropna().unique().tolist())
            sizes       = sorted(dis_data["supplier_size"].dropna().unique().tolist())
            resp_types  = sorted(dis_data["response_type"].dropna().unique().tolist())
        else:
            dis_types  = ["Cyber Attack", "Factory Incident", "Natural Disaster", "Logistics Delay"]
            sizes      = ["Small", "Medium", "Large"]
            resp_types = ["Alternative Supplier", "Combined Strategy", "Customer Delay"]

        col_a, col_b = st.columns(2)
        with col_a:
            sel_type    = st.selectbox("Disruption Type",    dis_types)
            sel_size    = st.selectbox("Supplier Size",      sizes)
        with col_b:
            sel_sev     = st.slider("Disruption Severity",   1, 5, 3)
            sel_impact  = st.slider("Production Impact (%)", 0, 100, 40)

        sel_backup = st.checkbox("Has Backup Supplier", value=False)

        # Encode inputs using value counts from the data
        def encode(series, val):
            cats = sorted(series.dropna().unique())
            return cats.index(val) if val in cats else 0

        if not dis_data.empty:
            type_enc   = encode(dis_data["disruption_type"],   sel_type)
            size_enc   = encode(dis_data["supplier_size"],     sel_size)
        else:
            type_enc = size_enc = 0

        backup_flag  = int(sel_backup)
        sev_x_backup    = sel_sev * backup_flag
        impact_x_backup = sel_impact * backup_flag

        base_feats = [type_enc, size_enc, sel_sev, sel_impact, backup_flag,
                      sev_x_backup, impact_x_backup]

        # Without a backup supplier, "Alternative Supplier" is not an
        # available option — mirrors the exclusion in response_planner.py.
        name_to_enc = {v: k for k, v in response_map.items()} if response_map else {}
        alt_enc = name_to_enc.get("Alternative Supplier")

        if st.button("Rank Strategies", type="primary"):
            if HAS_STRATEGY_FEATURE and response_map:
                rows = []
                for enc, name in response_map.items():
                    if not sel_backup and enc == alt_enc:
                        continue   # not an available option without a backup
                    fv = np.array([base_feats + [float(enc)]], dtype=float)
                    pred = float(regressor.predict(fv)[0])
                    rows.append({"Strategy": name, "Predicted Recovery (days)": round(pred, 1),
                                 "_enc": enc, "_feat_vec": fv})
                rank_df = pd.DataFrame(rows).sort_values("Predicted Recovery (days)").reset_index(drop=True)
                best        = rank_df.iloc[0]
                days        = float(best["Predicted Recovery (days)"])
                strategy_label = best["Strategy"]
                feat_vec    = best["_feat_vec"]
            else:
                # legacy 7-feature pickle — no strategy sweep possible
                feat_vec = np.array([base_feats], dtype=float)
                days     = float(regressor.predict(feat_vec)[0])
                strategy_label = "Alternative Supplier"
                rank_df  = None

            r1, r2 = st.columns(2)
            with r1: metric_card("Fastest Predicted Recovery", f"{days:.0f} days")
            with r2: metric_card("Recommended Strategy",       strategy_label)

            if rank_df is not None:
                st.markdown("##### All strategies ranked by predicted recovery")
                if not sel_backup:
                    st.caption("'Alternative Supplier' is excluded — not available without a backup supplier.")
                _show_df = rank_df[["Strategy", "Predicted Recovery (days)"]].copy()
                st.dataframe(_show_df, use_container_width=True, hide_index=True)

            severity_labels = {1: "Minimal", 2: "Low", 3: "Moderate", 4: "High", 5: "Critical"}
            st.info(
                f"Scenario: **{sel_type}** | Size: {sel_size} | "
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
                    "Disruption Type", "Supplier Size", "Disruption Severity",
                    "Production Impact %", "Has Backup Supplier",
                    "Severity x Backup", "Impact % x Backup",
                ]
                PRED_FEATURE_VALUES = [
                    sel_type, sel_size, f"Severity {sel_sev}",
                    f"{sel_impact}%", "Yes" if sel_backup else "No",
                    sev_x_backup, impact_x_backup,
                ]
                if HAS_STRATEGY_FEATURE:
                    PRED_FEATURE_NAMES.append("Response Strategy")
                    PRED_FEATURE_VALUES.append(strategy_label)

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
with tab_multi:
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
# TAB 8 — IMMUNOLOGICAL MEMORY (FAISS)
# ══════════════════════════════════════════════════════════════════════════
with tab_memory:
    st.subheader("Immunological Memory — FAISS Disruption Index")
    st.caption(
        "Models memory B-cells: stores 100K historical disruption signatures as "
        "8-dimensional vectors in a FAISS index. When a new anomaly is detected, "
        "the system retrieves the top-3 most similar past disruptions and surfaces "
        "the response strategies that worked. This is adaptive immunity with recall."
    )

    MEM_REPORT   = os.path.join(OUT, "memory_report.txt")
    MEM_RETRIEVAL = os.path.join(OUT, "memory_retrieval.csv")

    if not os.path.exists(MEM_RETRIEVAL):
        st.warning("Run Stage 13 first: python3 main.py --only 13")
    else:
        df_mem = pd.read_csv(MEM_RETRIEVAL)

        # ── Top metrics ──────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Memory Index Size",    "100,000 vectors")
        c2.metric("Feature Dimensions",   "8")
        c3.metric("Anomalies Queried",    f"{df_mem['anomaly_idx'].nunique():,}")
        c4.metric("Retrievals per Query", "Top 3")

        # ── Memory report ────────────────────────────────────────────────
        if os.path.exists(MEM_REPORT):
            with st.expander("Full Memory Report", expanded=False):
                st.code(open(MEM_REPORT).read(), language=None)

        st.divider()

        # ── Top response types retrieved ─────────────────────────────────
        st.markdown("#### Most Common Historical Response Strategies")
        st.caption("When the system recalls a similar past disruption, what responses did it find?")
        if "match_response_type" in df_mem.columns:
            resp_counts = df_mem["match_response_type"].value_counts().reset_index()
            resp_counts.columns = ["Response Strategy", "Count"]
            fig_resp, ax_resp = plt.subplots(figsize=(8, 3.5))
            fig_resp.patch.set_facecolor("#0f0f1a")
            ax_resp.set_facecolor("#0f0f1a")
            bars = ax_resp.barh(resp_counts["Response Strategy"],
                                resp_counts["Count"],
                                color="#00A896", edgecolor="none")
            ax_resp.tick_params(colors="white")
            ax_resp.set_xlabel("Retrieval Count", color="white")
            for sp in ["top", "right"]:
                ax_resp.spines[sp].set_visible(False)
            for sp in ["bottom", "left"]:
                ax_resp.spines[sp].set_color("#333355")
            for bar in bars:
                ax_resp.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                             f"{int(bar.get_width())}", va="center", color="white", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_resp)
            plt.close()

        # ── Top disruption types retrieved ───────────────────────────────
        st.markdown("#### Most Common Disruption Types in Memory")
        if "match_disruption_type" in df_mem.columns:
            dtype_counts = df_mem["match_disruption_type"].value_counts().reset_index()
            dtype_counts.columns = ["Disruption Type", "Count"]
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(dtype_counts, use_container_width=True, hide_index=True)
            with col2:
                if "match_recovery_days" in df_mem.columns:
                    avg_rec = (df_mem.groupby("match_disruption_type")["match_recovery_days"]
                               .mean().reset_index()
                               .rename(columns={"match_disruption_type": "Disruption Type",
                                                "match_recovery_days": "Avg Recovery Days"}))
                    avg_rec["Avg Recovery Days"] = avg_rec["Avg Recovery Days"].round(1)
                    avg_rec = avg_rec.sort_values("Avg Recovery Days", ascending=False)
                    st.markdown("**Average Recovery Days by Disruption Type**")
                    st.dataframe(avg_rec, use_container_width=True, hide_index=True)

        st.divider()

        # ── Match distance distribution ──────────────────────────────────
        st.markdown("#### Match Quality — How Close Are Retrieved Memories?")
        st.caption(
            "Lower L2 distance = more similar historical disruption. "
            "Most matches cluster below 1.0, meaning the memory index is finding "
            "genuinely relevant past events, not random noise."
        )
        if "match_distance" in df_mem.columns:
            fig_dist, ax_dist = plt.subplots(figsize=(8, 3))
            fig_dist.patch.set_facecolor("#0f0f1a")
            ax_dist.set_facecolor("#0f0f1a")
            ax_dist.hist(df_mem["match_distance"].dropna(), bins=40,
                         color="#F4845F", edgecolor="none", alpha=0.85)
            ax_dist.set_xlabel("L2 Match Distance (lower = more similar)", color="white")
            ax_dist.set_ylabel("Frequency", color="white")
            ax_dist.tick_params(colors="white")
            for sp in ["top", "right"]:
                ax_dist.spines[sp].set_visible(False)
            for sp in ["bottom", "left"]:
                ax_dist.spines[sp].set_color("#333355")
            avg_dist = df_mem["match_distance"].mean()
            ax_dist.axvline(avg_dist, color="#00A896", linestyle="--", linewidth=1.5,
                            label=f"Avg = {avg_dist:.3f}")
            ax_dist.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close()

        st.divider()

        # ── Sample retrieval table ───────────────────────────────────────
        st.markdown("#### Sample Memory Retrievals")
        st.caption("Each row is one historical disruption retrieved for a detected anomaly.")
        cols_to_show = [c for c in [
            "anomaly_idx", "manufacturer", "retailer", "anomaly_score",
            "match_rank", "match_distance", "match_recovery_days",
            "match_response_type", "match_disruption_type"
        ] if c in df_mem.columns]
        st.dataframe(df_mem[cols_to_show].head(30), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 9 — LIVE STREAM
# ══════════════════════════════════════════════════════════════════════════
# @st.fragment(run_every=...) reruns ONLY this function on its own timer —
# not the whole script. Before this fix, a bare `time.sleep(3); st.rerun()`
# at module level re-executed the ENTIRE app every 3 seconds (Streamlit has
# no notion of "inactive tab" — all tabs' code runs on every rerun), which
# meant every button click anywhere in the app (e.g. Recovery Predictor's
# "Rank Strategies") got wiped by the next auto-refresh before its result
# could ever be seen. Must be defined before tab_live calls it.
# ── Stream process management ─────────────────────────────────────────────
# The simulator and consumer are long-running scripts. They are launched here
# as detached subprocesses so a demo needs no extra terminals. PIDs are kept in
# a small file (not session_state) so a page reload or a second browser tab
# still sees — and can stop — processes started earlier.
import subprocess as _sp
import signal as _signal

STREAM_DIR      = os.path.join(BASE, "data", "stream")
STREAM_PID_FILE = os.path.join(STREAM_DIR, ".stream_pids.json")
STREAM_LOGS     = {"simulator": os.path.join(STREAM_DIR, "simulator.log"),
                   "consumer":  os.path.join(STREAM_DIR, "consumer.log")}

def _pid_alive(pid):
    """True only for a process that is still running.

    Children we spawned become zombies after they exit until someone waits on
    them, and a signal-0 probe still succeeds on a zombie — so reap first.
    waitpid raises ChildProcessError for processes that are not our children
    (e.g. after a server restart); fall back to the signal probe for those.
    """
    try:
        pid = int(pid)
    except (ValueError, TypeError):
        return False
    try:
        done_pid, _ = os.waitpid(pid, os.WNOHANG)
        if done_pid == pid:
            return False          # exited — reaped just now
        return True               # still running (our child)
    except ChildProcessError:
        pass                      # not our child: probe instead
    except OSError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def _stream_pids():
    """{'simulator': pid, 'consumer': pid} for processes that are still alive."""
    if not os.path.exists(STREAM_PID_FILE):
        return {}
    try:
        with open(STREAM_PID_FILE) as f:
            pids = json.load(f)
    except Exception:
        return {}
    return {k: v for k, v in pids.items() if _pid_alive(v)}

def _stop_stream():
    pids = _stream_pids()
    for name, pid in pids.items():
        try:
            os.kill(int(pid), _signal.SIGTERM)
        except OSError:
            pass
    # Reap so the children do not linger as zombies under the dashboard
    # process. Give them a moment to honour SIGTERM before checking.
    deadline = time.time() + 3.0
    remaining = dict(pids)
    while remaining and time.time() < deadline:
        for name, pid in list(remaining.items()):
            if not _pid_alive(pid):
                remaining.pop(name)
        if remaining:
            time.sleep(0.1)
    for name, pid in remaining.items():           # stubborn: escalate
        try:
            os.kill(int(pid), _signal.SIGKILL)
            _pid_alive(pid)
        except OSError:
            pass
    if os.path.exists(STREAM_PID_FILE):
        os.remove(STREAM_PID_FILE)
    return list(pids)

def _start_stream(interval, disruption_at, multi, domain):
    """Start consumer first (it waits for the feed), then the simulator."""
    _stop_stream()
    os.makedirs(STREAM_DIR, exist_ok=True)
    py = sys.executable
    # -u: unbuffered stdout so the log tails in the UI update line by line
    consumer_cmd = [py, "-u", os.path.join(BASE, "src", "stream_consumer.py"), "--domain", domain]
    sim_cmd = [py, "-u", os.path.join(BASE, "src", "stream_simulator.py"), "--interval", str(interval)]
    if disruption_at is not None:
        sim_cmd += ["--disruption", str(disruption_at)]
    if multi:
        sim_cmd.append("--multi")
    procs = {}
    for name, cmd in (("consumer", consumer_cmd), ("simulator", sim_cmd)):
        log = open(STREAM_LOGS[name], "w")
        procs[name] = _sp.Popen(cmd, cwd=BASE, stdout=log, stderr=_sp.STDOUT,
                                start_new_session=True).pid
    with open(STREAM_PID_FILE, "w") as f:
        json.dump(procs, f)
    return procs

def _tail(path, n=25):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""


@st.fragment(run_every=3)
def _live_stream_fragment(live_results_path, disruption_flag_path):
    st.caption("Auto-refreshes every 3 seconds. Keep this tab open during your demo.")

    _running = _stream_pids()
    if not os.path.exists(live_results_path):
        if _running:
            st.info("Stream is starting — the consumer is loading the immune engine and "
                    "waiting for its first rows. This usually takes a few seconds.")
        else:
            st.info("No stream results yet. Click **Start simulation** above.")
        return

    try:
        df_live = pd.read_csv(live_results_path)
    except Exception:
        st.info("Results file is being written — retrying on the next refresh.")
        return

    total_rows      = len(df_live)
    anomaly_rows    = int(df_live["is_anomaly"].sum()) if "is_anomaly" in df_live.columns else 0
    disruption_rows = int(df_live["disruption_injected"].sum()) if "disruption_injected" in df_live.columns else 0
    rerouted_rows   = int((df_live["alternate_route"].astype(str).str.strip() != "").sum()) if "alternate_route" in df_live.columns else 0

    caught_rows = (int(((df_live["disruption_injected"].astype(str) == "1")
                        & (df_live["is_anomaly"].astype(str) == "1")).sum())
                   if {"disruption_injected", "is_anomaly"} <= set(df_live.columns) else 0)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows Processed",      total_rows)
    c2.metric("Anomalies Detected",  anomaly_rows,    delta=f"{anomaly_rows} flagged")
    c3.metric("Disruptions Injected", disruption_rows,
              help="Rows the simulator flagged. The flag is NOT used by the detector; it only scores recall.")
    c4.metric("Injected Caught",     f"{caught_rows}/{disruption_rows}" if disruption_rows else "—",
              help="Detector recall on injected rows: real rolling z-score or the zero-quantity rule.")
    c5.metric("Routes Rerouted",     rerouted_rows)

    if os.path.exists(disruption_flag_path):
        flag_text = open(disruption_flag_path).read()
        st.error(f"⚡ ACTIVE DISRUPTION DETECTED\n\n{flag_text}")

    st.markdown("#### Recent Anomalies")
    if "is_anomaly" in df_live.columns:
        df_anomalies = df_live[df_live["is_anomaly"] == 1].tail(10)
        if df_anomalies.empty:
            st.success("No anomalies detected yet — supply chain is healthy.")
        else:
            cols_to_show = [c for c in [
                "timestamp", "manufacturer", "distributor", "retailer",
                "quantity", "z_score", "disruption_injected",
                "routing_method", "routing_note", "alternate_route"
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
    # Fragment reruns itself every 3s (run_every=3 above) — no manual
    # sleep/rerun needed, and this no longer touches the rest of the app.


with tab_live:
    st.subheader("Live Stream Monitor — Real-Time Sensor Feed")
    st.caption(
        "Simulates real-time IoT/sensor data arriving row by row. The simulator "
        "and the immune-response consumer run as background processes started "
        "from this page — no extra terminals needed."
    )

    LIVE_RESULTS_PATH    = os.path.join(STREAM_DIR, "live_results.csv")
    DISRUPTION_FLAG_PATH = os.path.join(STREAM_DIR, "disruption_active.flag")

    # ── Controls ──────────────────────────────────────────────────────────
    _running = _stream_pids()
    _domains = sorted(
        os.path.splitext(f)[0] for f in os.listdir(os.path.join(BASE, "config"))
        if f.endswith(".yaml")
    ) if os.path.isdir(os.path.join(BASE, "config")) else ["pharma"]
    with st.container(border=True):
        cc1, cc2, cc3, cc4, cc5 = st.columns([1.1, 1.1, 1.1, 1.3, 1.6])
        with cc1:
            _interval = st.number_input("Row interval (s)", 0.5, 10.0, 2.0, 0.5,
                                        disabled=bool(_running))
        with cc2:
            _dis_at = st.number_input("Disruption at (s)", 0, 300, 30, 5,
                                      disabled=bool(_running),
                                      help="Seconds after start to inject the first disruption. 0 = never.")
        with cc3:
            _multi = st.checkbox("Repeat every 60s", value=True, disabled=bool(_running),
                                 help="Keep injecting disruptions (needed to trigger a cytokine storm).")
        with cc4:
            _domain = st.selectbox("Domain config", _domains,
                                   index=_domains.index("pharma") if "pharma" in _domains else 0,
                                   disabled=bool(_running))
        with cc5:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if _running:
                if st.button("⏹ Stop simulation", type="primary", use_container_width=True):
                    _stop_stream()
                    st.rerun()
            else:
                if st.button("▶ Start simulation", type="primary", use_container_width=True):
                    _start_stream(_interval, _dis_at if _dis_at > 0 else None, _multi, _domain)
                    st.rerun()
        if _running:
            st.success(
                "Running — " + " · ".join(f"{k} pid {v}" for k, v in sorted(_running.items()))
                + ". Results below refresh every 3 s."
            )
        else:
            st.caption(
                "Or run by hand in two terminals: "
                "`python3 src/stream_simulator.py --interval 2 --disruption 30 --multi` "
                "and `python3 src/stream_consumer.py`."
            )
        with st.expander("Process logs (last 25 lines each)"):
            lc1, lc2 = st.columns(2)
            with lc1:
                st.markdown("**simulator**")
                st.code(_tail(STREAM_LOGS["simulator"]) or "(no output yet)", language=None)
            with lc2:
                st.markdown("**consumer**")
                st.code(_tail(STREAM_LOGS["consumer"]) or "(no output yet)", language=None)

    _live_stream_fragment(LIVE_RESULTS_PATH, DISRUPTION_FLAG_PATH)


# ══════════════════════════════════════════════════════════════════════════
# TAB 10 — IMMUNE RESPONSE (Chain-of-Thought Real-Time Decisions)
# ══════════════════════════════════════════════════════════════════════════
with tab_live:
    st.divider()
    st.subheader("🦠 Immune Response Engine — Real-Time Chain-of-Thought Decisions")
    st.caption(
        "When a disruption is detected in the live stream, the Immune Response Engine fires "
        "four parallel response systems: memory recall, alternate routing, backup supplier "
        "activation, and inventory transfer. Each anomaly shows its full reasoning trace here."
    )

    IMMUNE_DECISIONS_PATH = os.path.join(STREAM_DIR, "immune_decisions.jsonl")

    if not os.path.exists(IMMUNE_DECISIONS_PATH):
        st.info(
            "No immune response decisions yet. Click **Start simulation** above — "
            "decisions appear here once the first disruption is injected."
        )
        st.markdown("**Or run a one-shot test of the engine directly:**")
        st.code("python3 src/immune_response_engine.py", language="bash")
    else:
        # Load all decisions
        decisions = []
        with open(IMMUNE_DECISIONS_PATH, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        decisions.append(json.loads(_line))
                    except Exception:
                        pass

        if not decisions:
            st.warning("Decision log exists but is empty. Run the stream consumer to populate it.")
        else:
            # Separate cytokine storm events from individual disruption events
            storm_decisions  = [d for d in decisions if d.get("event_type") == "cytokine_storm"]
            normal_decisions = [d for d in decisions if d.get("event_type") != "cytokine_storm"]

            # ── Cytokine storm banners ─────────────────────────────────────
            if storm_decisions:
                st.error(
                    f"⚡ **{len(storm_decisions)} CYTOKINE STORM event(s) recorded** — "
                    "simultaneous multi-node cascade(s) detected. See details below."
                )
                for sd in storm_decisions:
                    ts_str  = sd.get("timestamp", "")[:19]
                    step0   = (sd.get("thinking") or [{}])[0]
                    detail  = step0.get("data", {})
                    affected = detail.get("affected_distributors", [])
                    repeats  = detail.get("repeat_hit_nodes", [])
                    count    = detail.get("storm_event_count", "?")
                    with st.expander(f"🔴 Cytokine Storm — {ts_str} | {count} simultaneous disruptions", expanded=False):
                        st.markdown(f"**Reasoning:** {step0.get('reasoning','')}")
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Disruptions in window", count)
                        sc2.metric("Affected nodes",        len(affected))
                        sc3.metric("Repeat-hit (overloaded)", len(repeats))
                        if affected:
                            st.markdown(f"**Affected distributors:** {', '.join(affected)}")
                        if repeats:
                            st.warning(f"Critical overload on: **{', '.join(repeats)}**")
                        constituent = detail.get("constituent_events", [])
                        if constituent:
                            import pandas as _pd2
                            st.dataframe(_pd2.DataFrame(constituent), use_container_width=True)
                st.markdown("---")

            # ── Regular event metrics ──────────────────────────────────────
            if not normal_decisions:
                st.info("Only cytokine storm events recorded so far. Individual disruption events will appear here.")
            else:
                st.markdown(f"### {len(normal_decisions)} individual disruption event(s) processed")
                severities  = [d.get("verdict", {}).get("severity", "?") for d in normal_decisions]
                n_critical  = severities.count("CRITICAL")
                n_high      = severities.count("HIGH")
                n_moderate  = severities.count("MODERATE")
                avg_signals = sum(d.get("verdict", {}).get("signals_activated", 0) for d in normal_decisions) / len(normal_decisions)

                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Total Events",       len(normal_decisions))
                mc2.metric("Critical",           n_critical,  delta=f"{n_critical}" if n_critical else None, delta_color="inverse")
                mc3.metric("High Severity",      n_high)
                mc4.metric("Moderate",           n_moderate)
                mc5.metric("Avg Signals Active", f"{avg_signals:.1f}/4")

                st.markdown("---")
                decisions = normal_decisions   # scope the rest of the tab to individual events

            # Event selector
            event_labels = [
                f"Event {i+1} | {d.get('timestamp','')[:19]} | "
                f"{d.get('verdict',{}).get('severity','?')} | "
                f"{d.get('manufacturer','?')[:20]} → {d.get('retailer','?')[:20]}"
                for i, d in enumerate(decisions)
            ]
            selected_idx = st.selectbox(
                "Select an anomaly event to inspect:",
                range(len(decisions)),
                format_func=lambda i: event_labels[i],
                index=len(decisions) - 1,   # default to latest
            )
            d = decisions[selected_idx]
            verdict = d.get("verdict", {})

            # Event header
            sev_colour = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "CYTOKINE_STORM": "⚡"}.get(verdict.get("severity"), "⚪")
            st.markdown(f"## {sev_colour} {verdict.get('severity','?')} severity disruption")
            hc1, hc2, hc3 = st.columns(3)
            hc1.markdown(f"**Manufacturer**\n\n{d.get('manufacturer','?')}")
            hc2.markdown(f"**Disrupted via**\n\n{d.get('distributor','?')}")
            hc3.markdown(f"**Retailer**\n\n{d.get('retailer','?')}")
            hc4, hc5, hc6 = st.columns(3)
            hc4.metric("Z-Score",      f"{d.get('z_score', 0):.2f}")
            hc5.metric("Signals Active", f"{verdict.get('signals_activated',0)}/4")
            hc6.metric("Est. Recovery", f"{verdict.get('recovery_estimate_days','?')} days")

            st.markdown("---")

            # Chain-of-thought steps
            st.markdown("### 🧠 Chain-of-Thought Reasoning")
            thinking = d.get("thinking", [])

            PHASE_ICONS = {
                "DETECTION":           "🔍",
                "MEMORY RECALL":       "🧬",
                "ALTERNATE ROUTE":     "🗺️",
                "BACKUP SUPPLIER":     "🏭",
                "INVENTORY TRANSFER":  "📦",
                "FINAL VERDICT":       "✅",
                "CYTOKINE STORM":      "⚡",
            }

            for step in thinking:
                phase  = step.get("phase", "")
                title  = step.get("title", "")
                reason = step.get("reasoning", "")
                data   = step.get("data", {})
                icon   = PHASE_ICONS.get(phase, "▪️")

                with st.expander(f"{icon} Step {step.get('step','')} — {phase}: {title}", expanded=True):
                    st.markdown(f"**Reasoning:**")
                    st.info(reason)

                    if data:
                        # Render specific rich data per step type
                        if phase == "MEMORY RECALL" and data.get("matches"):
                            st.markdown("**Top memory matches:**")
                            mem_rows = []
                            for m in data["matches"]:
                                mem_rows.append({
                                    "Rank":            m.get("rank"),
                                    "Distance":        m.get("distance"),
                                    "Recovery (days)": m.get("recovery_days"),
                                    "Response Type":   m.get("response_type"),
                                    "Disruption Type": m.get("disruption_type"),
                                })
                            st.dataframe(pd.DataFrame(mem_rows), use_container_width=True)
                            if data.get("avg_recovery_days"):
                                st.success(f"📅 Estimated recovery: **{data['avg_recovery_days']} days** | "
                                           f"Recommended: **{data.get('recommended_response','?')}**")

                        elif phase == "ALTERNATE ROUTE":
                            orig = data.get("original_route", "")
                            alt  = data.get("alternate_route", "")
                            if orig:
                                st.markdown(f"**Original route:** `{orig}`")
                            if alt:
                                st.success(f"**Alternate route:** `{alt}`")
                                decay = data.get("decay_factor", 1.0)
                                decay_label = (
                                    "🟢 Fresh (no recent overuse)"       if decay >= 0.9 else
                                    "🟡 Moderate load (used recently)"   if decay >= 0.7 else
                                    "🔴 High load — confidence reduced"
                                )
                                st.markdown(
                                    f"Method: `{data.get('method','?')}` | "
                                    f"Chosen node risk: `{data.get('chosen_risk','?')}` | "
                                    f"Load decay: `{decay}` {decay_label}"
                                )
                            if data.get("top_candidates"):
                                st.markdown("**Top candidate distributors evaluated:**")
                                st.dataframe(pd.DataFrame(data["top_candidates"]), use_container_width=True)

                        elif phase == "BACKUP SUPPLIER" and data.get("top_backups"):
                            st.markdown(f"**Affected retailers:** {data.get('affected_retailers','?')} | "
                                        f"Alternatives found: {data.get('alternatives_found','?')}")
                            st.markdown("**Top backup suppliers:**")
                            sup_rows = []
                            for b in data["top_backups"]:
                                sup_rows.append({
                                    "Supplier":       b.get("entity","?"),
                                    "Type":           b.get("type","?"),
                                    "Backup Score":   b.get("backup_score","?"),
                                    "Risk":           b.get("composite_risk","?"),
                                    "Volume":         b.get("out_volume","?"),
                                    "Safety Score":   b.get("safety_score","?"),
                                })
                            st.dataframe(pd.DataFrame(sup_rows), use_container_width=True)

                        elif phase == "INVENTORY TRANSFER" and data.get("top_transfers"):
                            st.markdown(f"**Retailer state:** {data.get('retailer_state','?')} | "
                                        f"Fuel multiplier: {data.get('fuel_multiplier','?')}x | "
                                        f"Candidates: {data.get('candidates_found','?')}")
                            st.markdown("**Top transfer candidates:**")
                            inv_rows = []
                            for t in data["top_transfers"]:
                                inv_rows.append({
                                    "Distributor":    t.get("distributor","?"),
                                    "Transfer Score": t.get("transfer_score","?"),
                                    "Risk":           t.get("composite_risk","?"),
                                    "Spare Capacity": t.get("spare_capacity","?"),
                                    "Fuel Cost":      t.get("fuel_cost","?"),
                                    "Est. Days":      t.get("estimated_days","?"),
                                })
                            st.dataframe(pd.DataFrame(inv_rows), use_container_width=True)

                        elif phase == "FINAL VERDICT":
                            st.markdown(f"Severity: **{data.get('severity','?')}** | "
                                        f"Signals activated: **{data.get('signals_activated','?')}/4** | "
                                        f"Estimated recovery: **{data.get('recovery_estimate_days','?')} days**")

            # Ranked action plan
            st.markdown("---")
            st.markdown("### 📋 Ranked Action Plan")
            actions = d.get("actions", [])
            if not actions:
                st.warning("No actions generated.")
            else:
                for a in actions:
                    conf = float(a.get("confidence", 0))
                    colour = "#00A896" if conf >= 0.7 else "#F4845F" if conf >= 0.4 else "#888"
                    priority_icon = {1: "🥇", 2: "🥈", 3: "🥉"}.get(a.get("priority"), "▪️")
                    with st.container():
                        ac1, ac2 = st.columns([3, 1])
                        with ac1:
                            st.markdown(
                                f"{priority_icon} **{a.get('action','')}**  \n"
                                f"→ {a.get('label','')}  \n"
                                f"*{a.get('detail','')}*"
                            )
                        with ac2:
                            st.metric("Confidence", f"{conf:.0%}")
                            st.caption(f"ETA: {a.get('eta_days','?')} day(s) | {a.get('method','?')}")
                        st.progress(conf)
                        st.markdown("")

            # All events summary table
            st.markdown("---")
            st.markdown("### 📊 All Response Events")
            summary_rows = []
            for i, dec in enumerate(decisions):
                v = dec.get("verdict", {})
                acts = v.get("actions_ranked", [])
                summary_rows.append({
                    "Event":       i + 1,
                    "Timestamp":   dec.get("timestamp","")[:19],
                    "Severity":    v.get("severity","?"),
                    "Z-Score":     dec.get("z_score","?"),
                    "Signals":     f"{v.get('signals_activated',0)}/4",
                    "Top Action":  acts[0].get("action","?") if acts else "?",
                    "Recovery (d)":v.get("recovery_estimate_days","?"),
                    "Manufacturer":dec.get("manufacturer","?")[:25],
                    "Retailer":    dec.get("retailer","?")[:25],
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.markdown("---")
        if st.button("🔄 Refresh decisions"):
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# TAB 9 (cont.) — REAL-EVENT REPLAY PLAYBACK
# Stage 21's blind walk-forward detections on real SCMS deliveries, played
# window by window, with Stage 25's per-lane response when the alert fires.
# This is a PLAYBACK of committed results, deliberately separate from the
# live engine above (which runs on the ARCOS graph and synthetic rows).
# ══════════════════════════════════════════════════════════════════════════
with tab_live:
    st.divider()
    st.subheader("🎞 Real-Event Replay — documented crises, window by window")
    st.caption(
        "A playback of the Stage-21 detector on **real** SCMS delivery data: for "
        "each country it sees only that country's own past late-rate and the "
        "shipments inside a rolling 3-month window, and fires when the binomial "
        "surprise crosses a fixed p < 0.001 (no per-event tuning). When the alert "
        "fires inside a documented crisis, the Stage-25 response ladder shows the "
        "moves the system would have made using only pre-event knowledge. This "
        "replays committed detections; it is not the live engine above, which "
        "runs on the ARCOS graph."
    )
    _rp_csv  = os.path.join(OUT, "scms_event_replay.csv")
    _rp_json = os.path.join(OUT, "event_harness.json")
    if not (os.path.exists(_rp_csv) and os.path.exists(_rp_json)):
        st.info("Run Stage 21 (`scms_event_replay.py`) and Stage 25 (`scms_event_harness.py`) first.")
    else:
        _er_all = pd.read_csv(_rp_csv)
        with open(_rp_json) as _f:
            _rp_scn = [s for s in json.load(_f) if s.get("type") == "replay"]

        _rp_pick = st.selectbox("Documented event", range(len(_rp_scn)),
                                format_func=lambda i: _rp_scn[i]["label"], key="rp_pick")
        _ev = _rp_scn[_rp_pick]
        _rows = (_er_all[_er_all["country"] == _ev["country"]]
                 .sort_values("window_end").reset_index(drop=True).copy())
        _rows["date"] = pd.to_datetime(_rows["window_end"] + "-01")
        _rows["p_value"] = pd.to_numeric(_rows["p_value"], errors="coerce")
        _rows["surprise"] = -np.log10(_rows["p_value"].clip(lower=1e-300))
        _w0, _w1 = _ev["window"].split("..")
        _w0d = pd.Timestamp(_w0 + "-01")
        _w1d = pd.Timestamp(_w1 + "-01") + pd.offsets.MonthEnd(0)
        _n = len(_rows)
        _doc_mask = (_rows["date"] >= _w0d) & (_rows["date"] <= _w1d)
        _alert_in_doc = _rows.index[(_rows["alert"] == 1) & _doc_mask].tolist()
        _first_alert = _alert_in_doc[0] if _alert_in_doc else None
        _onset_idx = int(_rows.index[_rows["date"] >= _w0d][0]) if (_rows["date"] >= _w0d).any() else 0
        _start_idx = max(0, _onset_idx - 9)          # start from the quiet run-up

        # ── playback state (reset when the event changes) ──────────────────
        if st.session_state.get("rp_event") != _rp_pick:
            st.session_state.update(rp_event=_rp_pick, rp_idx=_start_idx, rp_playing=False)
        st.session_state["rp_idx"] = int(min(max(st.session_state.get("rp_idx", _start_idx), 0), _n - 1))

        st.markdown(f"**Documented:** {_ev['documented']}  \n"
                    f"**Country / window:** {_ev['country']} · {_w0} to {_w1}  ·  "
                    f"{_n} detector windows on record")

        # controls live OUTSIDE the fragment so a click triggers a full rerun,
        # which is what re-evaluates run_every below.
        b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1.4, 2])
        with b1:
            if st.session_state["rp_playing"]:
                if st.button("⏸ Pause", use_container_width=True, key="rp_pause"):
                    st.session_state["rp_playing"] = False
            else:
                if st.button("▶ Play", type="primary", use_container_width=True, key="rp_play"):
                    if st.session_state["rp_idx"] >= _n - 1:
                        st.session_state["rp_idx"] = _start_idx
                    st.session_state["rp_playing"] = True
        with b2:
            if st.button("⏭ Step", use_container_width=True, key="rp_step"):
                st.session_state["rp_playing"] = False
                st.session_state["rp_idx"] = min(st.session_state["rp_idx"] + 1, _n - 1)
        with b3:
            if st.button("↺ Reset", use_container_width=True, key="rp_reset"):
                st.session_state["rp_playing"] = False
                st.session_state["rp_idx"] = _start_idx
        with b4:
            if st.button("🚨 Jump to alert", use_container_width=True, key="rp_jump",
                         disabled=_first_alert is None):
                st.session_state["rp_playing"] = False
                st.session_state["rp_idx"] = int(_first_alert)
        with b5:
            _rp_speed = st.select_slider("Speed", options=[0.25, 0.5, 1.0, 2.0], value=0.5,
                                         format_func=lambda v: f"{v:g} s / window", key="rp_speed")

        _rp_playing = bool(st.session_state.get("rp_playing"))

        @st.fragment(run_every=_rp_speed if _rp_playing else None)
        def _replay_view():
            idx = int(st.session_state["rp_idx"])
            if st.session_state.get("rp_playing"):
                if idx < _n - 1:
                    idx += 1
                    st.session_state["rp_idx"] = idx
                else:
                    st.session_state["rp_playing"] = False
                    st.rerun(scope="app")
            cur  = _rows.iloc[idx]
            seen = _rows.iloc[: idx + 1]
            in_doc = bool(_doc_mask.iloc[idx])
            fired_so_far = _first_alert is not None and idx >= _first_alert
            status = ("🚨 ALERT" if int(cur["alert"]) == 1 else
                      ("watching" if in_doc else "quiet"))

            k1, k2, k3, k4, k5 = st.columns(5)
            with k1: metric_card("Window ending", str(cur["window_end"]),
                                 "inside documented crisis" if in_doc else "before / after")
            with k2: metric_card("Shipments in window", f"{int(cur['n_window'])}",
                                 f"{int(cur['late_window'])} late")
            with k3: metric_card("Late rate", f"{cur['late_rate']*100:.0f}%",
                                 f"own baseline {cur['baseline_rate']*100:.1f}%")
            with k4: metric_card("Surprise", f"p = {cur['p_value']:.1e}",
                                 "threshold p < 0.001")
            with k5: metric_card("Detector", status,
                                 (f"first fired {str(_rows.iloc[_first_alert]['window_end'])}"
                                  if fired_so_far else "no alert yet"))

            # ── late-rate chart with the playhead ──────────────────────────
            fig_r = go.Figure()
            fig_r.add_vrect(x0=_w0d, x1=_w1d, fillcolor="#ffb300", opacity=0.10, line_width=0,
                            annotation_text="documented crisis", annotation_position="top left",
                            annotation_font=dict(color="#ffb300", size=10))
            fig_r.add_trace(go.Scatter(x=_rows["date"], y=_rows["baseline_rate"] * 100, mode="lines",
                                       name="country's own baseline", line=dict(color="#78909c", dash="dot", width=1.2)))
            fig_r.add_trace(go.Scatter(x=seen["date"], y=seen["late_rate"] * 100, mode="lines+markers",
                                       name="late rate (3-mo window)", line=dict(color="#4fc3f7", width=2),
                                       marker=dict(size=5)))
            _al = seen[seen["alert"] == 1]
            if not _al.empty:
                fig_r.add_trace(go.Scatter(x=_al["date"], y=_al["late_rate"] * 100, mode="markers",
                                           name="alert fired", marker=dict(color="#ef5350", size=11, symbol="x")))
            fig_r.add_vline(x=cur["date"], line=dict(color="#e0e0ff", width=1.5))
            fig_r.update_layout(template="plotly_dark", paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                                height=300, margin=dict(l=40, r=20, t=30, b=40),
                                yaxis_title="Late shipments (%)", xaxis_title="Window end",
                                legend=dict(orientation="h", y=1.14, x=0),
                                xaxis=dict(range=[_rows["date"].min(), _rows["date"].max()]))
            fig_s = go.Figure()
            fig_s.add_vrect(x0=_w0d, x1=_w1d, fillcolor="#ffb300", opacity=0.10, line_width=0)
            fig_s.add_trace(go.Bar(x=seen["date"], y=seen["surprise"], name="−log10 p",
                                   marker_color=["#ef5350" if a == 1 else "#4fc3f7" for a in seen["alert"]]))
            fig_s.add_hline(y=3, line=dict(color="#ffb300", dash="dash"),
                            annotation_text="alert threshold (p = 0.001)", annotation_position="top left",
                            annotation_font=dict(color="#ffb300", size=10))
            fig_s.add_vline(x=cur["date"], line=dict(color="#e0e0ff", width=1.5))
            fig_s.update_layout(template="plotly_dark", paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                                height=220, margin=dict(l=40, r=20, t=20, b=40), showlegend=False,
                                yaxis_title="Surprise (−log10 p)",
                                xaxis=dict(range=[_rows["date"].min(), _rows["date"].max()]))
            g1, g2 = st.columns([3, 2])
            with g1: st.plotly_chart(fig_r, use_container_width=True)
            with g2: st.plotly_chart(fig_s, use_container_width=True)

            # ── response ladder engages once the alert has fired ───────────
            if fired_so_far:
                _lag = int(_first_alert) - _onset_idx
                _d = _ev["detection"]; _sm = _ev["summary"]
                st.error(
                    f"🚨 Detector fired at window **{_rows.iloc[_first_alert]['window_end']}**, "
                    f"{_lag} window(s) after the documented onset ({_w0}), on the country's own "
                    f"history alone. Over the documented window: {_d['late_window']}/{_d['n_window']} "
                    f"late ({_d['late_rate']*100:.0f}% vs {_d['baseline_rate']*100:.1f}% baseline, "
                    f"p = {_d['p_value']}). Response ladder engaged (Stage 25, pre-event data only):"
                )
                r1, r2, r3 = st.columns(3)
                with r1: metric_card("Disrupted lanes", f"{_sm['disrupted_lanes']}", "molecule × vendor in this country")
                with r2: metric_card("Reroutable now", f"{_sm['resolved']}", f"{_sm['exposed']} exposed → qualify + buffer")
                with r3: metric_card("Buffer exposure", f"${_sm['buffer_usd_total']/1e6:.2f}M", "if absorbed by safety stock")
                _lr = [{
                    "Molecule": e["molecule"], "Disrupted vendor": e["disrupted_vendor"],
                    "Tier": e["tier"].split("_")[0], "System's move": e["recommendation"],
                    "Buffer $": (f"${e['buffer']['buffer_usd']:,}" if e.get("buffer") else ""),
                } for e in _ev["lanes"]]
                st.dataframe(pd.DataFrame(_lr), use_container_width=True, hide_index=True,
                             height=min(320, 36 * len(_lr) + 40))
            elif in_doc:
                st.warning("Inside the documented window — the detector has not crossed its "
                           "threshold yet. Keep playing.")
            else:
                st.info("Quiet run-up: the country's late rate sits on its own baseline. "
                        "Press ▶ Play or ⏭ Step to advance.")

        _replay_view()
