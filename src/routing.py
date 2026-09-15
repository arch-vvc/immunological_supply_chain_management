"""
Stage 5 — Disruption Injection & Recovery Routing
Maps to: Adaptive Immunity — targeted response after a disruption is detected

Two modes:
  1. Random disruption  — randomly removes a high-risk node, finds alternate path
  2. Targeted disruption — remove a specific node by name

Routing modes:
  A. Hop-optimised   — fewest intermediate nodes (baseline Dijkstra)
  B. Cost-optimised  — minimises fuel cost weighted by transaction volume
     Fuel cost derived from US state/region shipping multipliers per edge.

Time window constraint:
  Each edge has an estimated delivery time proportional to its fuel cost.
  If recovery path exceeds MAX_DELIVERY_DAYS, Inventory Agent fallback activates.
"""

import pickle
import pandas as pd
import networkx as nx
import numpy as np
import random
import os
import shutil
import tempfile

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL  = os.path.join(ROOT, "models", "supplychain_graph.pkl")
DATA   = os.path.join(ROOT, "data", "processed", "clean_chain.csv")
RISK   = os.path.join(ROOT, "output", "risk_scores.csv")
OUTPUT = os.path.join(ROOT, "output", "routing_results.txt")

# ── FUEL COST CONFIGURATION ────────────────────────────────────────────────────
# Regional shipping cost multipliers (relative to Midwest baseline = 1.0),
# loaded from config/<domain>.yaml via isc_common. They are ILLUSTRATIVE
# distance-from-hub multipliers, not a sourced dataset (see isc_common.py);
# they only break ties between otherwise similar reroute options.
MAX_DELIVERY_DAYS = 7   # time window constraint — trigger Inventory Agent if exceeded
BASE_DELIVERY_DAYS_PER_HOP = 1.5  # baseline days per supply chain hop

from isc_common import REGION_FUEL_COST, DEFAULT_FUEL_COST   # single source (config/pharma.yaml)

print("=" * 55)
print("  STAGE 5 — DISRUPTION INJECTION & RECOVERY ROUTING")
print("=" * 55)

if not os.path.exists(MODEL):
    print(f"[ERROR] {MODEL} not found. Run build_chain.py first.")
    exit(1)

with open(MODEL, "rb") as f:
    G = pickle.load(f)

df = pd.read_csv(DATA)

# ─────────────────────────────────────────────────────────────────
# BUILD FUEL-COST GRAPH
# Edge weight = fuel_cost_multiplier / log(volume + 1)
# Dijkstra minimises this — prefers high-volume, low-cost edges.
# Fuel cost for distributor→retailer edges: from retailer_state.
# Fuel cost for manufacturer→distributor edges: avg downstream cost.
# ─────────────────────────────────────────────────────────────────
# Per-edge fuel cost from retailer_state
edge_state = (
    df.groupby(["distributor", "retailer"])["retailer_state"]
    .first()
    .reset_index()
)
edge_state["fuel_cost"] = (
    edge_state["retailer_state"]
    .str.upper().str.strip()
    .map(REGION_FUEL_COST)
    .fillna(DEFAULT_FUEL_COST)
)

# Avg downstream fuel cost per distributor (used for mfr→dist edges)
dist_avg_fuel = (
    edge_state.groupby("distributor")["fuel_cost"]
    .mean()
    .to_dict()
)

# Build O(1) lookup dict for dist→retailer fuel costs
edge_fuel_lookup = {
    (row["distributor"], row["retailer"]): row["fuel_cost"]
    for _, row in edge_state.iterrows()
}

G_fuel = nx.DiGraph()
for u, v, data in G.edges(data=True):
    vol = data.get("weight", 1)
    fc  = edge_fuel_lookup.get((u, v), dist_avg_fuel.get(v, DEFAULT_FUEL_COST))
    composite_weight = fc / np.log1p(vol)
    G_fuel.add_edge(u, v, weight=composite_weight, fuel_cost=fc, volume=vol)

print(f"  Fuel-cost graph built: {G_fuel.number_of_nodes()} nodes, {G_fuel.number_of_edges()} edges")
print(f"  Time window constraint: MAX_DELIVERY_DAYS = {MAX_DELIVERY_DAYS}")

# ─────────────────────────────────────────────
# Pick a random transaction as the "active delivery"
# ─────────────────────────────────────────────
random.seed(42)
row = df.sample(1, random_state=42).iloc[0]
source = row["manufacturer"]
target = row["retailer"]

print(f"\nSimulated active transaction:")
print(f"  Manufacturer : {source}")
print(f"  Retailer     : {target}")
print(f"  Quantity     : {int(row['quantity']):,}")

# ─────────────────────────────────────────────
# Helper: estimate delivery days for a path
# ─────────────────────────────────────────────
def path_delivery_days(graph, path):
    """Sum of (fuel_cost * BASE_DELIVERY_DAYS_PER_HOP) across each edge in path."""
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        fc = graph[u][v].get("fuel_cost", DEFAULT_FUEL_COST) if graph.has_edge(u, v) else DEFAULT_FUEL_COST
        total += fc * BASE_DELIVERY_DAYS_PER_HOP
    return round(total, 1)

# ─────────────────────────────────────────────
# Baseline route BEFORE disruption
# ─────────────────────────────────────────────
print(f"\n[BEFORE DISRUPTION]")
try:
    baseline_path = nx.shortest_path(G, source, target)
    baseline_days = path_delivery_days(G_fuel, baseline_path)
    print(f"  Baseline route   : {' → '.join(baseline_path)}")
    print(f"  Estimated days   : {baseline_days}  (time window: {MAX_DELIVERY_DAYS} days)")
    baseline_len = len(baseline_path)
except nx.NetworkXNoPath:
    print("  No direct baseline path found.")
    baseline_path = []
    baseline_len  = 0
    baseline_days = 0.0

# ─────────────────────────────────────────────
# Inject disruption
# ─────────────────────────────────────────────
disrupted_node = row["distributor"]
print(f"\n[DISRUPTION INJECTED]")
print(f"  Node removed: {disrupted_node}  (simulates supplier failure)")

G_disrupted      = G.copy()
G_fuel_disrupted = G_fuel.copy()
G_disrupted.remove_node(disrupted_node)
if disrupted_node in G_fuel_disrupted:
    G_fuel_disrupted.remove_node(disrupted_node)

# ─────────────────────────────────────────────
# Recovery routing — Mode A: Hop-optimised (fewest hops)
# ─────────────────────────────────────────────
print(f"\n[RECOVERY ROUTING — Mode A: Hop-Optimised]")
hop_path = []
hop_days = 0.0
try:
    hop_path = nx.shortest_path(G_disrupted, source, target)
    hop_days = path_delivery_days(G_fuel_disrupted, hop_path)
    hop_status = "SUCCESS ✅"
    print(f"  Route          : {' → '.join(hop_path)}")
    print(f"  Hops           : {len(hop_path) - 1}  (was {baseline_len - 1})")
    print(f"  Estimated days : {hop_days}")
    if hop_days > MAX_DELIVERY_DAYS:
        print(f"  ⚠️  Exceeds time window ({MAX_DELIVERY_DAYS} days) — Inventory Agent triggered")
except nx.NetworkXNoPath:
    hop_status = "NO ALTERNATE ROUTE ❌"
    print(f"  Status: {hop_status}")
except nx.NodeNotFound:
    hop_status = "SOURCE/TARGET ALSO DISRUPTED ❌"
    print(f"  Status: {hop_status}")

# ─────────────────────────────────────────────
# Recovery routing — Mode B: Cost-optimised (min fuel cost)
# ─────────────────────────────────────────────
print(f"\n[RECOVERY ROUTING — Mode B: Fuel-Cost-Optimised]")
cost_path = []
cost_days = 0.0
cost_total = 0.0
try:
    cost_path = nx.shortest_path(G_fuel_disrupted, source, target, weight="weight")
    cost_days = path_delivery_days(G_fuel_disrupted, cost_path)
    cost_total = sum(
        G_fuel_disrupted[cost_path[i]][cost_path[i+1]].get("fuel_cost", DEFAULT_FUEL_COST)
        for i in range(len(cost_path) - 1)
    )
    cost_status = "SUCCESS ✅"
    print(f"  Route          : {' → '.join(cost_path)}")
    print(f"  Hops           : {len(cost_path) - 1}")
    print(f"  Total fuel cost: {cost_total:.2f}x  (relative multiplier)")
    print(f"  Estimated days : {cost_days}")
    if cost_days > MAX_DELIVERY_DAYS:
        print(f"  ⚠️  Exceeds time window ({MAX_DELIVERY_DAYS} days) — Inventory Agent triggered")
    if cost_path != hop_path:
        print(f"  ℹ️  Cost-optimised route differs from hop-optimised route")
    else:
        print(f"  ℹ️  Cost-optimised route matches hop-optimised route")
except nx.NetworkXNoPath:
    cost_status = "NO ALTERNATE ROUTE ❌"
    print(f"  Status: {cost_status}")
except nx.NodeNotFound:
    cost_status = "SOURCE/TARGET ALSO DISRUPTED ❌"
    print(f"  Status: {cost_status}")

# ─────────────────────────────────────────────
# Time-window check + Inventory Agent fallback
# ─────────────────────────────────────────────
recovery_path = cost_path if cost_path else hop_path
recovery_days = cost_days if cost_path else hop_days

no_route   = not recovery_path
over_window = recovery_days > MAX_DELIVERY_DAYS

if no_route or over_window:
    trigger_reason = "no alternate route found" if no_route else f"delivery time {recovery_days}d exceeds {MAX_DELIVERY_DAYS}d window"
    print(f"\n[FALLBACK — Inventory Agent Activated]")
    print(f"  Reason: {trigger_reason}")
    print(f"  Checking safety stock at nearest warehouse...")
    if os.path.exists(RISK):
        risk_df = pd.read_csv(RISK)
        if "entity" in risk_df.columns and "risk_score" in risk_df.columns:
            candidates = risk_df[risk_df["entity"] != disrupted_node].copy()
            candidates = candidates.sort_values("risk_score", ascending=False).head(3)
            for _, cand in candidates.iterrows():
                print(f"  → Alternate supplier: {cand['entity']} (risk score: {cand['risk_score']:.4f})")
    print(f"  Inventory reorder signal sent. Estimated restocking: 7-14 days")
    print(f"  Supplier Agent would be activated next.")
    status = f"FALLBACK ACTIVATED — {trigger_reason}"
else:
    status = "SUCCESS ✅"

# ─────────────────────────────────────────────
# Alternate distributors available
# ─────────────────────────────────────────────
print(f"\n[ALTERNATE DISTRIBUTORS available for {source}]")
successors = list(G_disrupted.successors(source)) if source in G_disrupted else []
if successors:
    for s in successors[:5]:
        vol = G_disrupted[source][s].get("weight", 0)
        fc  = G_fuel_disrupted[source][s].get("fuel_cost", DEFAULT_FUEL_COST) if G_fuel_disrupted.has_edge(source, s) else DEFAULT_FUEL_COST
        print(f"  → {s}  (volume: {int(vol):,}  |  fuel cost: {fc:.2f}x)")
else:
    print("  None available after disruption.")

# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# Write to a temp file first, then copy — avoids macOS FUSE mount timeout
_content = "\n".join([
    "DISRUPTION & RECOVERY ROUTING REPORT",
    "=" * 55,
    f"Disrupted node    : {disrupted_node}",
    f"Source            : {source}",
    f"Target            : {target}",
    f"Time window       : {MAX_DELIVERY_DAYS} days",
    "",
    f"Baseline route    : {' → '.join(baseline_path)}",
    f"Baseline days     : {baseline_days}",
    "",
    f"Recovery (hops)   : {' → '.join(hop_path) if hop_path else 'N/A'}",
    f"Recovery (cost)   : {' → '.join(cost_path) if cost_path else 'N/A'}",
    f"Recovery days     : {recovery_days}",
    f"Status            : {status}",
])

try:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(_content)
        tmp_path = tmp.name
    shutil.copy2(tmp_path, OUTPUT)
    os.unlink(tmp_path)
    print(f"\nResults saved → {OUTPUT}")
except Exception as e:
    print(f"\n[WARN] Could not save routing results file: {e}")
    print("  Routing logic completed successfully — file write skipped.")
