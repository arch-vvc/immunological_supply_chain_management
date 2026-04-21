"""
Stream Consumer — Real-Time Anomaly Detection & Full Immune Response
====================================================================
Watches data/stream/live_feed.csv for new rows written by stream_simulator.py.
For each new row:
  1. Z-score check on rolling window (innate immunity — fast, stateless)
  2. If anomaly: fires the full ImmuneResponseEngine cascade:
       Step 1 — Memory Recall   (FAISS — what happened before?)
       Step 2 — Alternate Route (PPO / Dijkstra — reroute A→C→B)
       Step 3 — Backup Supplier (SupplierAgent — who can cover the load?)
       Step 4 — Inventory Transfer (InventoryAgent — emergency stock shift)
       Step 5 — Final Verdict   (ranked action plan with confidence scores)
  3. Prints chain-of-thought reasoning to terminal
  4. Writes decisions to data/stream/immune_decisions.jsonl
  5. Writes live results to data/stream/live_results.csv

Run in a separate terminal while the simulator is running:
    python3 src/stream_consumer.py
"""

import os
import sys
import csv
import time
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Immune Response Engine ─────────────────────────────────────────────────
try:
    from immune_response_engine import ImmuneResponseEngine
    _ENGINE_AVAILABLE = True
except Exception as _e:
    print(f"[WARN] ImmuneResponseEngine not available: {_e}")
    _ENGINE_AVAILABLE = False

LIVE_FEED       = ROOT / "data" / "stream" / "live_feed.csv"
LIVE_RESULTS    = ROOT / "data" / "stream" / "live_results.csv"
DISRUPTION_FLAG = ROOT / "data" / "stream" / "disruption_active.flag"
GRAPH_MODEL     = ROOT / "models" / "supplychain_graph.pkl"
PPO_MODEL       = ROOT / "models" / "ppo_routing_agent.pth"
RISK_CSV        = ROOT / "output"  / "risk_scores.csv"
GNN_CSV         = ROOT / "output"  / "gnn_risk_scores.csv"

STREAM_DIR = ROOT / "data" / "stream"
STREAM_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE   = 50
ZSCORE_THRESH = 2.5
K             = 10
F_DIM         = 5
STATE_DIM     = K * F_DIM
ACTION_DIM    = K

# ── PPO actor (must match architecture in ppo_routing_agent.py) ───────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, 128), nn.ReLU(),
                nn.Linear(128, 64),        nn.ReLU(),
                nn.Linear(64, ACTION_DIM),
            )
        def forward(self, state, mask=None):
            logits = self.net(state)
            if mask is not None:
                logits = logits + (1.0 - mask) * (-1e9)
            return F.softmax(logits, dim=-1)

    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[WARN] PyTorch not available — PPO disabled, falling back to Dijkstra")

# ── Load graph ────────────────────────────────────────────────────────────
def load_graph():
    if not GRAPH_MODEL.exists():
        print(f"[WARN] Graph not found: {GRAPH_MODEL}")
        return None
    with open(GRAPH_MODEL, "rb") as f:
        G = pickle.load(f)
    print(f"[CON] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# ── Load risk maps ────────────────────────────────────────────────────────
def load_risk_maps():
    risk_map, gnn_map = {}, {}
    if RISK_CSV.exists():
        import pandas as pd
        df = pd.read_csv(RISK_CSV)
        risk_col = "composite_risk" if "composite_risk" in df.columns else "risk_score"
        raw = dict(zip(df["entity"], df[risk_col]))
        rmax = max(raw.values()) if raw else 1.0
        risk_map = {k: v / rmax for k, v in raw.items()}
    if GNN_CSV.exists():
        import pandas as pd
        df = pd.read_csv(GNN_CSV)
        raw = dict(zip(df["entity"], df["gnn_score"]))
        gmax = max(raw.values()) if raw else 1.0
        gnn_map = {k: v / gmax for k, v in raw.items()}
    return risk_map, gnn_map

# ── Load PPO actor weights ────────────────────────────────────────────────
def load_ppo():
    if not TORCH_OK or not PPO_MODEL.exists():
        print("[CON] PPO model not found — using Dijkstra fallback")
        return None
    try:
        actor = Actor()
        ckpt  = torch.load(PPO_MODEL, weights_only=True)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        print("[CON] PPO actor loaded ✓")
        return actor
    except Exception as e:
        print(f"[CON] PPO load failed ({e}) — using Dijkstra fallback")
        return None

# ── Node feature vector (must match ppo_routing_agent.py) ─────────────────
def node_feat(node, retailer, G, risk_map, gnn_map, in_deg, out_deg,
              max_in, max_out):
    import networkx as nx
    risk     = risk_map.get(node, 0.5)
    gnn      = gnn_map.get(node, 0.5)
    ind      = in_deg.get(node, 0) / max_in
    outd     = out_deg.get(node, 0) / max_out
    has_path = 1.0 if nx.has_path(G, node, retailer) else 0.0
    return [risk, gnn, ind, outd, has_path]

# ── PPO rerouting decision ────────────────────────────────────────────────
def ppo_reroute(actor, G, manufacturer, disrupted_dist, retailer,
                risk_map, gnn_map, in_deg, out_deg, max_in, max_out):
    import networkx as nx

    distributors = {n for n, d in G.nodes(data=True) if d.get("type") == "distributor"}
    candidates   = [
        n for n in G.successors(manufacturer)
        if n in distributors and n != disrupted_dist
        and nx.has_path(G, n, retailer)
    ]

    if not candidates:
        return None, [], "No valid candidates found"

    random.shuffle(candidates) if len(candidates) > K else None
    candidates = candidates[:K]

    state = []
    for cand in candidates:
        state.extend(node_feat(cand, retailer, G, risk_map, gnn_map,
                               in_deg, out_deg, max_in, max_out))
    while len(state) < STATE_DIM:
        state.extend([0.0] * F_DIM)

    mask = [1.0] * len(candidates) + [0.0] * (K - len(candidates))

    st_t  = torch.FloatTensor(state).unsqueeze(0)
    msk_t = torch.FloatTensor([mask])
    with torch.no_grad():
        probs = actor(st_t, msk_t)
    action = int(probs.argmax(dim=-1).item())

    if action >= len(candidates):
        return None, candidates, "PPO selected padding slot"

    chosen     = candidates[action]
    risk_score = risk_map.get(chosen, 0.5)

    try:
        path = nx.shortest_path(G, chosen, retailer)
        route_str = f"{manufacturer} → {chosen} → " + " → ".join(path[1:])
        note  = (f"PPO rerouted via {chosen} "
                 f"(risk={risk_score:.3f}, {len(path)} hops)")
        return route_str, candidates, note
    except nx.NetworkXNoPath:
        return None, candidates, f"PPO chose {chosen} but no path to retailer"

# ── Dijkstra fallback ─────────────────────────────────────────────────────
def dijkstra_reroute(G, manufacturer, disrupted_dist, retailer):
    import networkx as nx
    distributors = {n for n, d in G.nodes(data=True) if d.get("type") == "distributor"}
    G_alt = G.copy()
    if disrupted_dist in G_alt:
        G_alt.remove_node(disrupted_dist)
    try:
        path = nx.shortest_path(G_alt, manufacturer, retailer)
        return " → ".join(path), f"Dijkstra fallback ({len(path)-1} hops)"
    except Exception as e:
        return None, f"No path: {e}"

# ── Rolling Z-score ───────────────────────────────────────────────────────
def check_anomaly(quantity: float, window: deque):
    window.append(quantity)
    if len(window) < 5:
        return False, 0.0
    arr  = list(window)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))
    if std == 0:
        return False, 0.0
    z = abs((quantity - mean) / std)
    return z > ZSCORE_THRESH, round(z, 3)

# ── Results writer ────────────────────────────────────────────────────────
RESULT_FIELDS = [
    "timestamp", "row_index", "manufacturer", "distributor",
    "retailer", "retailer_state", "quantity", "z_score",
    "is_anomaly", "disruption_injected", "routing_method",
    "alternate_route", "routing_note"
]

def write_result(result: dict, header_written: bool):
    mode = "a" if header_written else "w"
    with open(LIVE_RESULTS, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not header_written:
            writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in RESULT_FIELDS})

# ── Main loop ─────────────────────────────────────────────────────────────
def run():
    import random as _random
    global random
    import random

    print("[CON] Stream consumer starting (Full Immune Response)...")
    print(f"[CON] Watching : {LIVE_FEED}")
    print(f"[CON] Results  → {LIVE_RESULTS}\n")

    # Load the full immune response engine (handles graph, PPO, FAISS, supplier, inventory)
    engine = None
    if _ENGINE_AVAILABLE:
        try:
            engine = ImmuneResponseEngine(verbose=False)
            print("[CON] ImmuneResponseEngine loaded ✓ — full chain-of-thought responses enabled\n")
        except Exception as _e:
            print(f"[CON][WARN] Engine init failed: {_e} — falling back to legacy routing\n")

    # Legacy fallback loaders (used only if engine is None)
    G        = load_graph()   if engine is None else engine.G
    actor    = load_ppo()     if engine is None and G else None
    risk_map, gnn_map = load_risk_maps() if engine is None else ({}, {})

    if G:
        import networkx as nx
        in_deg  = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        max_in  = max(in_deg.values())  or 1
        max_out = max(out_deg.values()) or 1
    else:
        in_deg = out_deg = {}
        max_in = max_out = 1

    if LIVE_RESULTS.exists():
        LIVE_RESULTS.unlink()

    quantity_window = deque(maxlen=WINDOW_SIZE)
    rows_seen       = 0
    anomalies_found = 0
    header_written  = False

    print("[CON] Waiting for stream data...\n")

    while True:
        if not LIVE_FEED.exists():
            time.sleep(0.5)
            continue

        with open(LIVE_FEED, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        new_rows = reader[rows_seen:]
        if not new_rows:
            time.sleep(0.3)
            continue

        for row in new_rows:
            rows_seen += 1
            ts = datetime.now().strftime("%H:%M:%S")

            try:
                qty = float(row.get("quantity", 0))
            except (ValueError, TypeError):
                qty = 0.0

            is_disruption = str(row.get("disruption_injected", "0")) == "1"

            if is_disruption or qty == 0:
                is_anomaly = True
                z_score    = 99.0
            else:
                is_anomaly, z_score = check_anomaly(qty, quantity_window)

            manufacturer = row.get("manufacturer", "")
            distributor  = row.get("distributor", "")
            retailer     = row.get("retailer", "")
            state        = row.get("retailer_state", "")

            alternate_route = ""
            routing_note    = "Normal flow"
            routing_method  = "none"

            if is_anomaly:
                anomalies_found += 1

                if engine is not None:
                    # ── Full Immune Response (chain-of-thought) ────────────────
                    event_dict = {
                        "manufacturer":      manufacturer,
                        "distributor":       distributor,
                        "retailer":          retailer,
                        "retailer_state":    state,
                        "quantity":          qty,
                        "z_score":           z_score,
                        "disruption_injected": is_disruption,
                    }
                    try:
                        decision = engine.respond(event_dict)
                        engine.print_response(decision)
                        verdict = decision.get("verdict", {})
                        top_action = verdict.get("actions_ranked", [{}])[0]
                        alternate_route = top_action.get("detail", "")
                        routing_note    = top_action.get("label", "Immune response fired")
                        routing_method  = "ImmuneEngine"
                    except Exception as _eng_e:
                        print(f"[CON][WARN] Engine response failed: {_eng_e} — using legacy routing")
                        engine = None   # disable for subsequent rows, fall through below

                if engine is None:
                    # ── Legacy fallback: PPO / Dijkstra only ──────────────────
                    if G and manufacturer and retailer:
                        if actor is not None:
                            route, candidates, note = ppo_reroute(
                                actor, G, manufacturer, distributor, retailer,
                                risk_map, gnn_map, in_deg, out_deg, max_in, max_out
                            )
                            if route:
                                alternate_route = route
                                routing_note    = note
                                routing_method  = "PPO"
                            else:
                                route, note = dijkstra_reroute(G, manufacturer,
                                                               distributor, retailer)
                                alternate_route = route or ""
                                routing_note    = f"[Dijkstra fallback] {note}"
                                routing_method  = "Dijkstra"
                        else:
                            route, note = dijkstra_reroute(G, manufacturer,
                                                           distributor, retailer)
                            alternate_route = route or ""
                            routing_note    = note
                            routing_method  = "Dijkstra"
                    else:
                        routing_note   = "Graph unavailable"
                        routing_method = "none"

                    print(f"[CON] [{ts}] ⚡ ANOMALY #{anomalies_found} | Row {rows_seen}")
                    print(f"         Flow   : {manufacturer[:30]} → {distributor[:25]} → {retailer[:25]}")
                    print(f"         Qty    : {qty:.0f} | Z={z_score} | Method: {routing_method}")
                    if alternate_route:
                        print(f"         Route  : {alternate_route[:90]}")
                    print()
            else:
                if rows_seen % 20 == 0:
                    print(f"[CON] [{ts}] Row {rows_seen:4d} | Normal | "
                          f"{manufacturer[:25]:25s} → {retailer[:20]:20s} | qty={qty:.0f}")

            write_result({
                "timestamp"          : datetime.now().isoformat(),
                "row_index"          : rows_seen,
                "manufacturer"       : manufacturer,
                "distributor"        : distributor,
                "retailer"           : retailer,
                "retailer_state"     : state,
                "quantity"           : qty,
                "z_score"            : z_score,
                "is_anomaly"         : int(is_anomaly),
                "disruption_injected": int(is_disruption),
                "routing_method"     : routing_method,
                "alternate_route"    : alternate_route,
                "routing_note"       : routing_note,
            }, header_written)
            header_written = True

        time.sleep(0.1)

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n[CON] Stopped by user.")
