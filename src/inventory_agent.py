"""
Stage 15 — Inventory Agent (Digital Antibody #4)
Maps to: Adaptive Immunity — transfers stock between nodes to cover shortfalls.

Immune analogy: Complement proteins redistribute resources to sites of infection.
The Inventory Agent does the same — it finds retailers with critical single-source
dependency, identifies distributors with spare capacity, and recommends stock
transfers ranked by feasibility.

What it does:
  1. Identifies at-risk retailers from anomalies.csv:
       - flag_concentration = True  → 90%+ dependent on one distributor (fragile)
       - flag_volume = True          → unusually large recent order (potential stockout)
  2. For each at-risk retailer, finds their primary (dominant) distributor
  3. Finds alternative distributors with spare capacity that aren't already
     under stress (low composite_risk, high spare_capacity_ratio)
  4. Estimates transfer feasibility:
       - spare_capacity = out_volume / out_degree  (avg throughput per link)
       - fuel_cost from routing.py's state-based lookup (re-used here)
       - transfer_score = 0.40*capacity + 0.35*safety + 0.25*(1/fuel_cost_norm)
  5. Outputs prioritised transfer recommendations

Outputs:
  output/inventory_agent_results.csv   — transfer recommendations per at-risk retailer
  output/inventory_agent_report.txt    — plain-text summary
"""

import os
import pickle
import shutil
import tempfile
import pandas as pd
import numpy as np
import networkx as nx

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_PATH   = os.path.join(ROOT, "models", "supplychain_graph.pkl")
RISK_PATH    = os.path.join(ROOT, "output", "graph_risk_scores.csv")
ANOMALY_PATH = os.path.join(ROOT, "output", "anomalies.csv")
DATA_PATH    = os.path.join(ROOT, "data",   "processed", "clean_chain.csv")
RESULTS_OUT  = os.path.join(ROOT, "output", "inventory_agent_results.csv")
REPORT_OUT   = os.path.join(ROOT, "output", "inventory_agent_report.txt")

TOP_N_RETAILERS  = 15   # at-risk retailers to analyse
TOP_N_TRANSFERS  = 3    # transfer options per retailer

# Re-use fuel cost map from routing.py
REGION_FUEL_COST = {
    "ME":1.8,"NH":1.8,"VT":1.8,"MA":1.7,"RI":1.7,"CT":1.7,
    "NY":1.5,"NJ":1.5,"PA":1.4,"DE":1.5,"MD":1.4,
    "VA":1.3,"WV":1.2,"NC":1.3,"SC":1.4,"GA":1.4,"FL":1.6,
    "AL":1.3,"MS":1.3,"TN":1.2,"KY":1.2,
    "OH":1.0,"IN":1.0,"IL":1.0,"MI":1.1,"WI":1.1,
    "MN":1.2,"IA":1.1,"MO":1.1,"ND":1.3,"SD":1.3,
    "NE":1.2,"KS":1.2,
    "TX":1.3,"OK":1.2,"AR":1.2,"LA":1.3,
    "MT":1.6,"ID":1.6,"WY":1.5,"CO":1.4,"NM":1.5,
    "AZ":1.5,"UT":1.5,"NV":1.6,
    "CA":1.8,"OR":1.8,"WA":1.8,"AK":2.5,"HI":3.0,
}
DEFAULT_FUEL = 1.3

print("=" * 55)
print("  STAGE 15 — INVENTORY AGENT (Digital Antibody #4)")
print("=" * 55)

# ── Load inputs ───────────────────────────────────────────────
for path in [GRAPH_PATH, RISK_PATH, ANOMALY_PATH, DATA_PATH]:
    if not os.path.exists(path):
        print(f"[ERROR] Required file not found: {path}")
        exit(1)

with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

risk_df    = pd.read_csv(RISK_PATH)
anomaly_df = pd.read_csv(ANOMALY_PATH)
chain_df   = pd.read_csv(DATA_PATH)

print(f"  Graph     : {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print(f"  Risk data : {len(risk_df):,} nodes scored")
print(f"  Anomalies : {len(anomaly_df):,} flagged transactions")

risk_lookup = risk_df.set_index("entity").to_dict("index")

# ── Retailer state lookup ─────────────────────────────────────
retailer_state = chain_df.groupby("retailer")["retailer_state"].first().to_dict()

# ── Identify at-risk retailers ────────────────────────────────
# Concentration-flagged = single source dependency (most fragile)
# Volume-flagged        = abnormal demand (potential stockout risk)
at_risk = (
    anomaly_df[anomaly_df["flag_concentration"] == True]
    .groupby("retailer")["anomaly_score"]
    .max()
    .reset_index()
    .rename(columns={"anomaly_score": "risk_score"})
    .sort_values("risk_score", ascending=False)
    .head(TOP_N_RETAILERS)
)

print(f"\n  At-risk retailers (concentration flag): {len(at_risk)}")

# ── Spare capacity estimation ─────────────────────────────────
# A distributor's spare capacity proxy = out_volume / out_degree
# (avg volume it handles per retailer link)
# Distributors with high spare capacity AND low risk are good transfer candidates
dist_nodes = risk_df[risk_df["type"] == "distributor"].copy()
dist_nodes["spare_capacity"] = (
    dist_nodes["out_volume"] / dist_nodes["out_degree"].clip(lower=1)
).round(1)

# Normalise for scoring
def _norm(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

dist_nodes["capacity_norm"] = _norm(dist_nodes["spare_capacity"])
dist_nodes["safety_norm"]   = _norm(1.0 - dist_nodes["composite_risk"])

dist_lookup = dist_nodes.set_index("entity").to_dict("index")

# ── Analyse each at-risk retailer ────────────────────────────
all_rows = []

for _, rrow in at_risk.iterrows():
    retailer   = rrow["retailer"]
    risk_score = rrow["risk_score"]

    if retailer not in G:
        continue

    # Current dominant distributor(s)
    current_dists = list(G.predecessors(retailer))
    if not current_dists:
        continue

    # Primary = distributor with highest edge volume to this retailer
    primary_dist = max(
        current_dists,
        key=lambda d: G[d][retailer].get("weight", 0) if G.has_edge(d, retailer) else 0
    )

    state       = str(retailer_state.get(retailer, "")).upper().strip()
    fuel_to_ret = REGION_FUEL_COST.get(state, DEFAULT_FUEL)

    # Find candidate transfer sources:
    # distributors NOT currently serving this retailer, not high-risk themselves
    # Use relative threshold: top 50% safest distributors available
    all_risks = [v.get("composite_risk", 1.0) for v in dist_lookup.values()]
    risk_threshold = float(np.percentile(all_risks, 75)) if all_risks else 0.95

    candidates = []
    for dist, attrs in dist_lookup.items():
        if dist in current_dists:
            continue
        if attrs.get("composite_risk", 1.0) > risk_threshold:
            continue      # exclude top 25% riskiest distributors

        capacity_n = float(attrs.get("capacity_norm", 0))
        safety_n   = float(attrs.get("safety_norm",   0))
        # fuel score: lower fuel cost = higher score (invert and normalise)
        fuel_score = 1.0 - (fuel_to_ret - 1.0) / 2.0   # rough 0–1 from 1–3 range
        fuel_score = max(0.0, min(1.0, fuel_score))

        transfer_score = (
            0.40 * capacity_n
          + 0.35 * safety_n
          + 0.25 * fuel_score
        )

        candidates.append({
            "transfer_source":   dist,
            "transfer_score":    round(transfer_score, 4),
            "composite_risk":    round(float(attrs.get("composite_risk", 0)), 4),
            "spare_capacity":    round(float(attrs.get("spare_capacity", 0)), 1),
            "out_volume":        int(attrs.get("out_volume", 0)),
            "fuel_cost_to_dest": round(fuel_to_ret, 2),
        })

    if not candidates:
        continue

    top = sorted(candidates, key=lambda x: x["transfer_score"], reverse=True)[:TOP_N_TRANSFERS]

    print(f"\n  Retailer: {retailer[:50]}  (state: {state}  risk: {risk_score})")
    print(f"  Primary dist: {primary_dist[:45]}  |  Transfer candidates: {len(candidates)}")
    for rank, c in enumerate(top, 1):
        print(f"    #{rank} {c['transfer_source'][:45]:<45}  "
              f"score={c['transfer_score']:.3f}  "
              f"risk={c['composite_risk']:.3f}  "
              f"capacity={int(c['spare_capacity']):,}")

    for rank, c in enumerate(top, 1):
        all_rows.append({
            "retailer":            retailer,
            "retailer_state":      state,
            "retailer_risk_score": risk_score,
            "primary_distributor": primary_dist,
            "transfer_rank":       rank,
            "transfer_source":     c["transfer_source"],
            "transfer_score":      c["transfer_score"],
            "composite_risk":      c["composite_risk"],
            "spare_capacity":      c["spare_capacity"],
            "out_volume":          c["out_volume"],
            "fuel_cost_to_dest":   c["fuel_cost_to_dest"],
            "estimated_days":      round(c["fuel_cost_to_dest"] * 1.5, 1),
        })

# ── Save outputs ──────────────────────────────────────────────
os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)

if all_rows:
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_OUT, index=False)

    best = results_df[results_df["transfer_rank"] == 1]
    avg_score    = best["transfer_score"].mean()
    avg_est_days = best["estimated_days"].mean()

    print(f"\n{'─'*55}")
    print(f"  INVENTORY AGENT SUMMARY")
    print(f"{'─'*55}")
    print(f"  At-risk retailers analysed  : {len(at_risk)}")
    print(f"  Transfer recommendations    : {len(results_df)}")
    print(f"  Avg transfer score (rank #1): {avg_score:.3f}")
    print(f"  Avg estimated delivery days : {avg_est_days:.1f}")
    print(f"  Results saved → {RESULTS_OUT}")

    _lines = [
        "STAGE 15 — INVENTORY AGENT REPORT",
        "=" * 55,
        "",
        f"At-risk retailers analysed  : {len(at_risk)}",
        f"Transfer recommendations    : {len(results_df)}",
        f"Avg transfer score (rank #1): {avg_score:.3f}",
        f"Avg estimated delivery days : {avg_est_days:.1f}",
        "",
        "Scoring weights:",
        "  Spare capacity  : 40%",
        "  Safety (1-risk) : 35%",
        "  Fuel efficiency : 25%",
        "",
        "Top-1 transfer per at-risk retailer:",
    ]
    for _, r in best.iterrows():
        _lines.append(
            f"  {r['retailer'][:38]:<38} → "
            f"{r['transfer_source'][:38]:<38} "
            f"(score={r['transfer_score']:.3f}  "
            f"~{r['estimated_days']}d)"
        )
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("\n".join(_lines))
            _tmp = tmp.name
        shutil.copy2(_tmp, REPORT_OUT)
        os.unlink(_tmp)
        print(f"  Report saved  → {REPORT_OUT}")
    except Exception as _e:
        print(f"  [WARN] Could not save inventory report: {_e}")
else:
    print("\n[WARN] No transfer recommendations generated.")
