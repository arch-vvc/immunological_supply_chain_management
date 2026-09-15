"""
Stage 14 — Supplier Agent (Digital Antibody #2)
Maps to: Adaptive Immunity — activates backup suppliers when a primary node fails.

Immune analogy: When a B-cell detects a known antigen, it triggers a targeted
antibody response. The Supplier Agent is that response for supplier failures —
it identifies the best available backup and scores it for the Coordination Agent.

What it does:
  1. Identifies disrupted/high-risk entities from anomalies.csv
  2. Queries the supply chain graph to find which downstream retailers
     depend on each disrupted entity
  3. Finds alternative distributors that already serve those same retailers
     (proven supply routes, not hypothetical)
  4. Scores alternatives on three dimensions:
       - Safety     : low composite_risk (won't itself become a disruption)
       - Capacity   : high out_volume (can absorb extra load)
       - Efficiency : high out_volume / out_degree (avg throughput per link)
  5. Outputs ranked backup recommendations per disrupted entity

Outputs:
  output/supplier_agent_results.csv    — ranked backup suppliers per disrupted entity
  output/supplier_agent_report.txt     — plain-text summary
"""

import os
import pickle
import shutil
import tempfile
import pandas as pd
import numpy as np
from isc_common import SUPPLIER_WEIGHTS   # shared with the live engine
import networkx as nx

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_PATH   = os.path.join(ROOT, "models",  "supplychain_graph.pkl")
RISK_PATH    = os.path.join(ROOT, "output",  "graph_risk_scores.csv")
ANOMALY_PATH = os.path.join(ROOT, "output",  "anomalies.csv")
RESULTS_OUT  = os.path.join(ROOT, "output",  "supplier_agent_results.csv")
REPORT_OUT   = os.path.join(ROOT, "output",  "supplier_agent_report.txt")

TOP_N_DISRUPTED  = 10   # how many disrupted entities to analyse
TOP_N_BACKUPS    = 3    # how many backup suppliers to recommend per entity

print("=" * 55)
print("  STAGE 14 — SUPPLIER AGENT (Digital Antibody #2)")
print("=" * 55)

# ── Load inputs ───────────────────────────────────────────────
for path in [GRAPH_PATH, RISK_PATH, ANOMALY_PATH]:
    if not os.path.exists(path):
        print(f"[ERROR] Required file not found: {path}")
        exit(1)

with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

risk_df   = pd.read_csv(RISK_PATH)
anomaly_df= pd.read_csv(ANOMALY_PATH)

print(f"  Graph    : {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print(f"  Risk data: {len(risk_df):,} nodes scored")
print(f"  Anomalies: {len(anomaly_df):,} flagged transactions")

# Build fast lookup: entity → node attributes
risk_lookup = risk_df.set_index("entity").to_dict("index")

# ── Identify disrupted entities ───────────────────────────────
# Use top anomalies ranked by anomaly_score.
# Focus on manufacturers and distributors (not retailers — they're the victims).
disrupted_candidates = (
    anomaly_df
    .groupby("manufacturer")["anomaly_score"]
    .max()
    .reset_index()
    .rename(columns={"manufacturer": "entity", "anomaly_score": "max_score"})
)
dist_scores = (
    anomaly_df
    .groupby("distributor")["anomaly_score"]
    .max()
    .reset_index()
    .rename(columns={"distributor": "entity", "anomaly_score": "max_score"})
)
disrupted_candidates = (
    pd.concat([disrupted_candidates, dist_scores])
    .sort_values("max_score", ascending=False)
    .drop_duplicates("entity")
    .head(TOP_N_DISRUPTED)
)

print(f"\n[ANALYSIS] Top {len(disrupted_candidates)} disrupted entities identified")

# ── Score backup suppliers ─────────────────────────────────────
def score_backup(entity, risk_lookup, G):
    """
    Score a candidate backup supplier on:
      - safety_score    : 1 - composite_risk  (lower risk = safer)
      - capacity_score  : normalised out_volume
      - efficiency_score: out_volume / out_degree  (throughput per link)
    Returns composite backup score 0–1.
    """
    if entity not in risk_lookup:
        return None
    nd = risk_lookup[entity]
    safety    = 1.0 - float(nd.get("composite_risk", 0.5))
    out_vol   = float(nd.get("out_volume", 0))
    out_deg   = max(int(nd.get("out_degree", 1)), 1)
    efficiency= out_vol / out_deg
    return {
        "entity":           entity,
        "type":             nd.get("type", "unknown"),
        "composite_risk":   round(float(nd.get("composite_risk", 0.5)), 4),
        "out_volume":       int(out_vol),
        "out_degree":       out_deg,
        "efficiency":       round(efficiency, 1),
        "safety_score":     round(safety, 4),
    }


all_rows = []

for _, drow in disrupted_candidates.iterrows():
    disrupted = drow["entity"]
    d_score   = drow["max_score"]

    # Find all retailers that this entity supplies (direct successors)
    if disrupted not in G:
        continue
    affected_retailers = set(G.successors(disrupted))
    if not affected_retailers:
        # It's a manufacturer — find its distributors' retailers
        for dist in G.successors(disrupted):
            affected_retailers.update(G.successors(dist))

    # Find alternative entities that already serve any of the affected retailers
    # AND are not the disrupted entity itself
    alternatives = set()
    for retailer in affected_retailers:
        for pred in G.predecessors(retailer):
            if pred != disrupted and pred in risk_lookup:
                nd = risk_lookup[pred]
                if nd.get("type") in ("distributor", "manufacturer"):
                    alternatives.add(pred)

    if not alternatives:
        print(f"  {disrupted[:45]:<45}  → No alternatives found in graph")
        continue

    # Score and rank alternatives
    scored = []
    for alt in alternatives:
        s = score_backup(alt, risk_lookup, G)
        if s:
            # Composite backup score: safety 50%, capacity 30%, efficiency 20%
            # (normalise capacity and efficiency relative to all alternatives)
            scored.append(s)

    if not scored:
        continue

    scored_df = pd.DataFrame(scored)

    # Min-max normalise capacity and efficiency for scoring
    for col in ["out_volume", "efficiency"]:
        mn, mx = scored_df[col].min(), scored_df[col].max()
        scored_df[f"{col}_norm"] = (scored_df[col] - mn) / (mx - mn + 1e-9)

    scored_df["backup_score"] = (
        SUPPLIER_WEIGHTS["safety"]     * scored_df["safety_score"]
      + SUPPLIER_WEIGHTS["volume"]     * scored_df["out_volume_norm"]
      + SUPPLIER_WEIGHTS["efficiency"] * scored_df["efficiency_norm"]
    ).round(4)

    top = scored_df.sort_values("backup_score", ascending=False).head(TOP_N_BACKUPS)

    print(f"\n  Disrupted: {disrupted[:50]}  (anomaly score: {d_score})")
    print(f"  Affected retailers: {len(affected_retailers)}  |  Alternatives found: {len(alternatives)}")
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        print(f"    #{rank} {r['entity'][:45]:<45}  "
              f"backup={r['backup_score']:.3f}  "
              f"risk={r['composite_risk']:.3f}  "
              f"vol={int(r['out_volume']):,}")

    for rank, (_, r) in enumerate(top.iterrows(), 1):
        all_rows.append({
            "disrupted_entity":    disrupted,
            "disrupted_score":     d_score,
            "affected_retailers":  len(affected_retailers),
            "backup_rank":         rank,
            "backup_entity":       r["entity"],
            "backup_type":         r["type"],
            "backup_score":        r["backup_score"],
            "composite_risk":      r["composite_risk"],
            "out_volume":          r["out_volume"],
            "out_degree":          r["out_degree"],
            "efficiency":          r["efficiency"],
        })

# ── Save outputs ──────────────────────────────────────────────
os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)

if all_rows:
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_OUT, index=False)

    best_backups = results_df[results_df["backup_rank"] == 1]
    avg_score    = best_backups["backup_score"].mean()
    avg_risk_reduction = (
        results_df.groupby("disrupted_entity")["composite_risk"].mean().mean()
    )

    print(f"\n{'─'*55}")
    print(f"  SUPPLIER AGENT SUMMARY")
    print(f"{'─'*55}")
    print(f"  Disrupted entities analysed : {disrupted_candidates.shape[0]}")
    print(f"  Recommendations generated   : {len(results_df)}")
    print(f"  Avg backup score (rank #1)  : {avg_score:.3f}")
    print(f"  Results saved → {RESULTS_OUT}")

    _lines = [
        "STAGE 14 — SUPPLIER AGENT REPORT",
        "=" * 55,
        "",
        f"Disrupted entities analysed : {disrupted_candidates.shape[0]}",
        f"Backup recommendations      : {len(results_df)}",
        f"Avg backup score (rank #1)  : {avg_score:.3f}",
        "",
        "Scoring weights:",
        "  Safety (1-composite_risk) : 50%",
        "  Capacity (out_volume)     : 30%",
        "  Efficiency (vol/degree)   : 20%",
        "",
        "Top-1 backup per disrupted entity:",
    ]
    for _, r in best_backups.iterrows():
        _lines.append(f"  {r['disrupted_entity'][:40]:<40} → "
                      f"{r['backup_entity'][:40]:<40} "
                      f"(score={r['backup_score']:.3f})")
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("\n".join(_lines))
            _tmp = tmp.name
        shutil.copy2(_tmp, REPORT_OUT)
        os.unlink(_tmp)
        print(f"  Report saved  → {REPORT_OUT}")
    except Exception as _e:
        print(f"  [WARN] Could not save supplier report: {_e}")
else:
    print("\n[WARN] No backup recommendations generated — check graph connectivity.")
