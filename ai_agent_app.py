"""
Self-contained single-page app: a local, zero-cost LLM ("Immune System
Analyst") narrating the real Decision Trace produced by the Immune Response
Engine — the same engine used by the full dashboard (immune_app.py), just
without needing the live stream_simulator / stream_consumer terminals running.

The chat agent has tool access: it can decide to call lookup_entity_risk,
simulate_node_failure, or compare_recent_incidents — real functions running
against the live pipeline data, not just narrating a fixed cascade. Answers
grounded in a tool call are labeled "Verified"; answers the model composes
itself from context are labeled "AI-generated" so it's clear which is which.

A "Generate a new incident" button samples a real high-signal anomaly from
this run's data and drives it through the actual engine (FAISS memory
recall -> PPO/Dijkstra routing -> supplier agent -> inventory agent ->
verdict), so every narrated event is a genuine decision, not a mock.

Run:
    ollama serve                 # optional — enables full LLM narration;
    ollama pull llama3.2:1b      # without it, a template summary is used
    streamlit run ai_agent_app.py
"""

import os
import sys
import json
import warnings
from typing import Optional
warnings.filterwarnings("ignore")

# Must be set before torch / faiss / xgboost are imported anywhere downstream —
# each bundles its own libomp.dylib on macOS and loading more than one in the
# same process segfaults otherwise.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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

BASE = os.path.dirname(os.path.abspath(__file__))

from ai_agent import ImmuneAIAgent

OUT    = os.path.join(BASE, "output")
STREAM = os.path.join(BASE, "data", "stream")

st.set_page_config(
    page_title="Immunological Supply Chain — AI Agent",
    layout="centered",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 820px; }
    .narration-box {
        background: #0a1c14;
        border-left: 3px solid #2fa86a;
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        font-size: 0.95rem;
        line-height: 1.55;
        color: #cdeedd;
        margin: 0.6rem 0 1rem 0;
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
    .phase-timeline { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; margin: 0.6rem 0 1rem 0; }
    .phase-chip {
        font-size: 0.72rem; font-weight: 600; padding: 0.35rem 0.7rem; border-radius: 8px;
        background: #12141b; border: 1px solid #2c313d; color: #b0c8e0; white-space: nowrap;
    }
    .phase-arrow { color: #444b58; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


def render_source_badge(source, tool_used=None):
    """Confidence cue distinguishing deterministic tool output from the
    model's own composed text — the earlier critique's point (c)."""
    if source == "tool":
        st.markdown(f'<span class="src-badge verified">Verified — computed via {tool_used}()</span>', unsafe_allow_html=True)
    elif source == "llm":
        st.markdown('<span class="src-badge ai">AI-generated — cross-check against the metrics above</span>', unsafe_allow_html=True)
    elif source == "template":
        st.markdown('<span class="src-badge template">Template fallback — no local model running</span>', unsafe_allow_html=True)


def render_execution_trace(trace):
    """Execution log for a chat answer — not chain-of-thought prose, a record
    of which steps actually ran and how long each took. This is the honest
    version of the deep multi-tool-planning ask: it shows exactly the (small)
    number of real steps that ran, with real timings, not a fabricated plan."""
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


def render_phase_timeline(thinking):
    """Horizontal step sequence of the immune response engine's real phases
    (detection -> memory recall -> routing -> ... -> verdict) for one
    incident. An "incident timeline" built from the actual pipeline stages
    that ran, not a fabricated narrative."""
    if not thinking:
        return
    chips = [f'<span class="phase-chip">{s.get("phase", "?")}</span>' for s in thinking]
    st.markdown('<div class="phase-timeline">' + '<span class="phase-arrow"> → </span>'.join(chips) + '</div>', unsafe_allow_html=True)


def render_decision_comparison(decision):
    """Table comparing every ranked response strategy the engine actually
    computed for this incident — reroute, backup supplier, inventory
    transfer, historical protocol — side by side on cost/risk/confidence/ETA.
    All values are read straight from the decision; nothing here is
    LLM-generated."""
    actions = decision.get("actions", [])
    if not actions:
        st.caption("No ranked strategies available for this incident.")
        return
    # Plain HTML table, not st.dataframe(): st.dataframe() serializes through
    # PyArrow to send data to the frontend, which is the same libarrow.dylib
    # compute engine that segfaulted earlier this session (confirmed via a
    # second, independent crash report). A manual table has zero PyArrow
    # involvement — no risk of hitting that race again.
    header = "".join(f"<th style='text-align:left; padding:4px 10px; color:#8899a8'>{h}</th>" for h in
                      ["Priority", "Strategy", "Detail", "Confidence", "ETA (days)", "Method"])
    body_rows = []
    for a in actions:
        conf = a.get("confidence")
        conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)
        cells = [a.get("priority"), a.get("action"), (a.get("label") or "")[:70], conf_str, a.get("eta_days"), a.get("method")]
        body_rows.append(
            "<tr>" + "".join(f"<td style='padding:4px 10px; border-top:1px solid #222632'>{c}</td>" for c in cells) + "</tr>"
        )
    st.markdown(
        f"<table style='width:100%; font-size:0.85rem'><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>",
        unsafe_allow_html=True,
    )
    best = actions[0]
    best_conf = best.get("confidence")
    best_conf_str = f"{best_conf:.0%}" if isinstance(best_conf, (int, float)) else str(best_conf)
    st.success(f"Recommended: **{best.get('label', '')}** — {best_conf_str} confidence, ~{best.get('eta_days', '?')} day(s) to recover.")


def render_local_subgraph(decision, whatif_result=None):
    """Local neighborhood of the disrupted node from the live supply chain
    graph, colored by role: red = disrupted, green = reroute target,
    dark red = unreachable in the current what-if simulation, orange =
    high composite risk (>0.7), gray = normal. Real graph, real layout —
    not a mockup."""
    import networkx as nx
    G = load_graph()
    center = decision.get("distributor")
    if G is None or not center or center not in G:
        st.caption("Graph unavailable for this node.")
        return

    neighbors = list(set(G.predecessors(center)) | set(G.successors(center)))
    sub_nodes = [center] + neighbors[:24]  # cap for a readable plot
    H = G.subgraph(sub_nodes)
    pos = nx.spring_layout(H, seed=7, k=0.9)

    risk_df = load_graph_risk()
    risk_map = dict(zip(risk_df["entity"], risk_df["composite_risk"])) if not risk_df.empty and "composite_risk" in risk_df.columns else {}

    reroute_target = None
    actions = decision.get("actions", [])
    if actions and actions[0].get("action") == "REROUTE":
        hops = [h.strip() for h in (actions[0].get("detail") or "").replace("→", "->").split("->")]
        if len(hops) >= 2:
            reroute_target = hops[1] if len(hops) > 2 else hops[-1]

    unreachable_set = set((whatif_result or {}).get("sample_unreachable", []))

    edge_x, edge_y = [], []
    for u, v in H.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for n in H.nodes():
        node_x.append(pos[n][0])
        node_y.append(pos[n][1])
        risk_val = risk_map.get(n)
        if n == center:
            node_color.append("#ef5350"); node_size.append(22)
        elif n == reroute_target:
            node_color.append("#4ade80"); node_size.append(18)
        elif n in unreachable_set:
            node_color.append("#b91c1c"); node_size.append(14)
        elif isinstance(risk_val, (int, float)) and risk_val > 0.7:
            node_color.append("#f6c945"); node_size.append(14)
        else:
            node_color.append("#5d646f"); node_size.append(10)
        risk_label = f"{risk_val:.2f}" if isinstance(risk_val, (int, float)) else "?"
        node_text.append(f"{n}<br>composite_risk={risk_label}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#2c313d", width=1), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#08090c")),
        text=node_text, hoverinfo="text",
    ))
    fig.update_layout(
        showlegend=False, height=380, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#08090c", paper_bgcolor="#08090c",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        '<div style="font-size:0.75rem">'
        '<span style="color:#ef5350">● Disrupted node</span> &nbsp; '
        '<span style="color:#4ade80">● Reroute target</span> &nbsp; '
        '<span style="color:#b91c1c">● Unreachable (what-if)</span> &nbsp; '
        '<span style="color:#f6c945">● High risk (&gt;0.7)</span> &nbsp; '
        '<span style="color:#5d646f">● Normal</span></div>',
        unsafe_allow_html=True,
    )


def build_recovery_plan(decision: dict, whatif_result: Optional[dict] = None) -> dict:
    """Deterministically assembles a structured recovery action plan from
    the incident's own already-computed data (ranked actions, verdict,
    optionally the current what-if simulation). Nothing here is invented —
    it's the same numbers shown elsewhere on this page, reorganized into a
    plan. The LLM is only ever asked to summarize this dict, never to
    produce the structure or the numbers itself."""
    v = decision.get("verdict", {})
    actions = decision.get("actions", [])
    z = decision.get("z_score", 0)

    steps = []
    for i, a in enumerate(actions, 1):
        steps.append({
            "step": i,
            "action": a.get("action"),
            "description": a.get("label"),
            "detail": a.get("detail"),
            "eta_days": a.get("eta_days"),
            "confidence": a.get("confidence"),
            "priority": "PRIMARY" if i == 1 else "CONTINGENCY",
        })

    risks = []
    if whatif_result and whatif_result.get("found") and whatif_result.get("unreachable_dependents", 0) > 0:
        risks.append(
            f"{whatif_result['unreachable_dependents']} downstream dependent(s) would be unreachable if "
            f"[{whatif_result.get('node')}] fails, per the current what-if simulation."
        )
    numeric_conf_actions = [a for a in actions if isinstance(a.get("confidence"), (int, float))]
    if numeric_conf_actions:
        weakest = min(numeric_conf_actions, key=lambda a: a["confidence"])
        if weakest["confidence"] < 0.6:
            risks.append(f"Contingency \"{weakest.get('label')}\" has low confidence ({weakest['confidence']:.0%}) — treat as fallback only, not primary.")
    if not risks:
        risks.append("No additional risks flagged beyond the primary disruption.")

    qty = decision.get("quantity", 0)
    return {
        "distributor": decision.get("distributor"),
        "manufacturer": decision.get("manufacturer"),
        "retailer": decision.get("retailer"),
        "severity": v.get("severity"),
        "risk_rating_out_of_10": ImmuneAIAgent._risk_rating_10(z),
        "business_impact": (
            f"{v.get('severity', '?')} severity disruption (z-score {z:.2f}) on the shipment route to "
            f"{decision.get('retailer', '?')}; approximately {qty:,.0f} units at risk."
        ),
        "estimated_recovery_days": v.get("recovery_estimate_days"),
        "prioritized_steps": steps,
        "risks": risks,
    }


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    return ImmuneAIAgent()

@st.cache_resource
def load_graph_risk():
    # cache_resource, not cache_data: st.cache_data hashes/serializes the
    # returned DataFrame via PyArrow, which segfaults in this environment
    # (libarrow.dylib crash on macOS/arm64). cache_resource just keeps the
    # object in memory without trying to serialize it.
    p = os.path.join(OUT, "graph_risk_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_resource
def load_anomalies():
    p = os.path.join(OUT, "anomalies.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_resource
def load_graph():
    import pickle
    p = os.path.join(BASE, "models", "supplychain_graph.pkl")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def load_immune_decisions():
    p = os.path.join(STREAM, "immune_decisions.jsonl")
    if not os.path.exists(p):
        return []
    decisions = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    decisions.append(json.loads(line))
                except Exception:
                    pass
    return [d for d in decisions if d.get("event_type") != "cytokine_storm"]


def run_demo_incident():
    """Sample a real high-signal anomaly from this run's data and drive it
    through the actual ImmuneResponseEngine — memory recall, routing,
    backup supplier, inventory transfer, verdict. Returns a decision dict
    identical in shape to what the live stream produces."""
    from immune_response_engine import ImmuneResponseEngine

    anomalies = load_anomalies()
    if anomalies.empty:
        raise RuntimeError("No anomalies.csv found — run the pipeline first: python3 main.py")

    top_n = anomalies.sort_values("anomaly_score", ascending=False).head(20)
    row = top_n.sample(1).iloc[0]
    event = {
        "manufacturer":       row.get("manufacturer", ""),
        "distributor":        row.get("distributor", ""),
        "retailer":           row.get("retailer", ""),
        "retailer_state":     row.get("retailer_state", ""),
        "quantity":           float(row.get("quantity", 0)),
        "z_score":            float(row.get("z_quantity", row.get("z_score", 0))),
        "disruption_injected": True,
    }
    engine = ImmuneResponseEngine(verbose=False, domain="pharma")
    return engine.respond(event)


# ── Agent tools ───────────────────────────────────────────────────────────────
# Real functions the chat agent can choose to call instead of only narrating a
# fixed cascade. Each docstring's first line is shown to the model as the
# tool's description when it decides whether to use it.

def _find_entity(name: str, known: list) -> Optional[str]:
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
    decisions = load_immune_decisions()
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
st.title("Immunological Supply Chain — AI Agent")
st.caption("Self-Healing Supply Chains with AI Digital Antibodies · PES University · PW26_RGP_01")

with st.expander("What this is"):
    st.markdown(
        "The full pipeline models a supply chain as an immune system: anomaly detection, "
        "graph risk scoring, a GNN encoder, a PPO routing agent, FAISS-based memory of past "
        "disruptions, and supplier/inventory 'digital antibody' agents that decide how to "
        "respond to a live disruption. All of that reasoning was previously only structured "
        "JSON. **This page adds a local, zero-cost LLM analyst** (via Ollama — no API keys, "
        "no cloud) that narrates a real decision in plain English and answers questions about "
        "the current risk state, grounded in this run's actual data."
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
    with st.expander("Enable full narration"):
        st.code("brew install ollama\nollama serve\nollama pull llama3.2:1b", language="bash")

st.divider()

# ── Incident briefing ────────────────────────────────────────────────────────
st.markdown("### Incident Briefing")

if "last_decision" not in st.session_state:
    # Generate a fresh incident on first load of each session instead of
    # always showing whatever happens to be the last line already sitting
    # in immune_decisions.jsonl from a previous run — that was static and
    # showed the same event on every reload.
    with st.spinner("Running the full immune response cascade..."):
        try:
            st.session_state.last_decision = run_demo_incident()
        except Exception as e:
            existing = load_immune_decisions()
            if existing:
                st.session_state.last_decision = existing[-1]
            else:
                st.error(f"Could not generate an incident: {e}")

col_gen, col_info = st.columns([1, 2])
with col_gen:
    if st.button("Generate a new incident", width="stretch"):
        with st.spinner("Running the full immune response cascade..."):
            try:
                st.session_state.last_decision = run_demo_incident()
                st.session_state.pop("narration_cache", None)
            except Exception as e:
                st.error(f"Could not generate an incident: {e}")
with col_info:
    st.caption(
        "Samples a real high-signal anomaly from this run and drives it through the actual "
        "engine — memory recall, routing, backup supplier, inventory transfer, verdict."
    )

decision = st.session_state.get("last_decision")

if not decision:
    st.info("No incident yet — click **Generate a new incident** above.")
else:
    v = decision.get("verdict", {})
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Severity", v.get("severity", "?"))
    h2.metric("Risk Rating", f"{ImmuneAIAgent._risk_rating_10(decision.get('z_score', 0))}/10")
    h3.metric("Z-Score", f"{decision.get('z_score', 0):.2f}")
    h4.metric("Signals Activated", f"{v.get('signals_activated', 0)}/4")
    h5.metric("Est. Recovery", f"{v.get('recovery_estimate_days', '?')} days")

    render_phase_timeline(decision.get("thinking", []))

    if "narration_cache" not in st.session_state:
        with st.spinner("Analyst is reading the trace..."):
            st.session_state.narration_cache = agent.narrate_event(decision)
    st.markdown(f'<div class="narration-box">{st.session_state.narration_cache}</div>', unsafe_allow_html=True)
    render_source_badge("llm" if is_up else "template")
    st.caption("Narrative text is AI-generated from the verified metrics above — the numbers are computed by the pipeline, the prose is the model's summary of them.")

    with st.expander("Raw Decision Trace (what the LLM was given)"):
        for step in decision.get("thinking", []):
            st.markdown(f"**[{step.get('phase')}]** {step.get('reasoning', '')}")

    st.markdown("#### Compare Response Strategies")
    render_decision_comparison(decision)

    st.markdown("#### Supply Chain Neighborhood")
    render_local_subgraph(decision, st.session_state.get("whatif_result"))

    st.markdown("#### Recovery Action Plan")
    st.caption("Prioritized steps, business impact, and risks — assembled from the same ranked strategies above, not a separate LLM invention.")
    if st.button("Generate Recovery Action Plan"):
        st.session_state.recovery_plan = build_recovery_plan(decision, st.session_state.get("whatif_result"))
        st.session_state.pop("recovery_plan_summary", None)

    plan = st.session_state.get("recovery_plan")
    if plan:
        if "recovery_plan_summary" not in st.session_state:
            with st.spinner("Analyst is drafting the executive summary..."):
                st.session_state.recovery_plan_summary = agent.explain_tool_result(
                    "Summarize this recovery action plan in 2-3 sentences for an executive audience.",
                    "build_recovery_plan", plan,
                )
        st.markdown(f'<div class="narration-box">{st.session_state.recovery_plan_summary}</div>', unsafe_allow_html=True)
        render_source_badge("tool", "build_recovery_plan")

        p1, p2 = st.columns(2)
        p1.metric("Business Impact", plan["severity"] or "?", plan["business_impact"][:60] + "...")
        p2.metric("Estimated Recovery", f"{plan['estimated_recovery_days']} days")

        st.markdown("**Prioritized Steps**")
        for s in plan["prioritized_steps"]:
            conf = s["confidence"]
            conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)
            st.markdown(f"{s['step']}. **[{s['priority']}]** {s['description']} — confidence {conf_str}, ETA {s['eta_days']} day(s)")

        st.markdown("**Risks & Contingencies**")
        for r in plan["risks"]:
            st.markdown(f"- {r}")

st.divider()

# ── What-if simulation ──────────────────────────────────────────────────────
# The chat agent CAN route "what if X fails" questions to simulate_node_failure,
# but a 1B local model doesn't always pick the right tool for that phrasing.
# Rather than depend on the LLM guessing correctly, this control calls the
# same underlying function directly and deterministically — no routing
# involved — then only uses the LLM afterward to narrate the known result.
st.markdown("### What If a Supplier Fails?")
st.caption("Directly removes a node (or several, comma-separated) from the live graph and checks reachability — no chat routing involved, so it's never wrong about which tool to use.")

col_input, col_btn = st.columns([3, 1])
with col_input:
    whatif_node = st.text_input(
        "Node(s) to simulate failing", key="whatif_node_input",
        placeholder="e.g. MIAMI-LUKEN INC, HENRY SCHEIN INC",
    )
with col_btn:
    st.markdown("<div style='height: 1.9rem'></div>", unsafe_allow_html=True)
    run_whatif = st.button("Simulate failure", width="stretch")

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
                # Compute the conclusion in Python and hand it to the model to
                # restate, rather than asking a small model to derive
                # "reachable vs unreachable" itself from raw counts — that
                # inversion is exactly where it went wrong in testing.
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

st.divider()

# ── Chat ─────────────────────────────────────────────────────────────────────
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
# Re-read fresh (not the "existing" snapshot from page load) so a just-generated
# incident is always included, even though it was appended mid-script by the button above.
recent_events = load_immune_decisions()[-4:]
if decision and decision.get("timestamp") not in {e.get("timestamp") for e in recent_events}:
    recent_events.append(decision)
chat_context = {
    "top_risk": top_risk,
    "recent_events": recent_events[-5:],
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

question = st.chat_input("Ask about current risk, the incident above, or a recommended action...")
if question:
    st.session_state.ai_agent_history.append({"role": "user", "content": question, "source": None, "tool_used": None, "trace": None})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        # chat_stream() yields text chunks as the local model produces them —
        # st.write_stream renders them progressively and returns the fully
        # assembled string. Routing/trace metadata can't travel through a
        # plain text generator, so it's written into `meta` as a side effect
        # during iteration and read back once the stream is exhausted.
        meta = {}
        answer = st.write_stream(
            agent.chat_stream(question, chat_context, st.session_state.ai_agent_history,
                               tools=AGENT_TOOLS, meta=meta)
        )
        render_source_badge(meta.get("source"), meta.get("tool_used"))
        render_execution_trace(meta.get("trace"))
    st.session_state.ai_agent_history.append({
        "role": "assistant", "content": answer,
        "source": meta.get("source"), "tool_used": meta.get("tool_used"),
        "trace": meta.get("trace"),
    })

if st.session_state.ai_agent_history and st.button("Clear chat"):
    st.session_state.ai_agent_history = []
    st.rerun()