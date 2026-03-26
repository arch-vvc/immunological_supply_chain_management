"""
Stream Consumer — Real-Time Anomaly Detection & Routing
========================================================
Watches data/stream/live_feed.csv for new rows written by stream_simulator.py.
For each new row:
  1. Checks for anomaly using Z-score on quantity (fast, stateless)
  2. If anomaly detected, queries the supply chain graph for an alternate route
  3. Writes live results to data/stream/live_results.csv
  4. Prints a live event log to terminal

Run in a separate terminal while the simulator is running:
    python3 src/stream_consumer.py

Then open the Streamlit dashboard to see live updates:
    streamlit run app.py
"""

import os
import sys
import csv
import time
import pickle
import json
from pathlib import Path
from datetime import datetime
from collections import deque

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LIVE_FEED       = ROOT / "data" / "stream" / "live_feed.csv"
LIVE_RESULTS    = ROOT / "data" / "stream" / "live_results.csv"
DISRUPTION_FLAG = ROOT / "data" / "stream" / "disruption_active.flag"
GRAPH_MODEL     = ROOT / "models" / "supplychain_graph.pkl"
STREAM_DIR      = ROOT / "data" / "stream"

STREAM_DIR.mkdir(parents=True, exist_ok=True)

# Rolling window for Z-score (stateless, last N quantities)
WINDOW_SIZE   = 50
ZSCORE_THRESH = 2.5

# ── Load graph ────────────────────────────────────────────────────────────────
def load_graph():
    if not GRAPH_MODEL.exists():
        print(f"[WARN] Graph model not found at {GRAPH_MODEL}")
        print("       Run pipeline first: python3 main.py --from 1 --to 2")
        return None
    with open(GRAPH_MODEL, "rb") as f:
        G = pickle.load(f)
    print(f"[CON] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# ── Z-score anomaly check on rolling window ───────────────────────────────────
def check_anomaly(quantity: float, window: deque) -> tuple[bool, float]:
    """Returns (is_anomaly, z_score) using rolling window."""
    window.append(quantity)
    if len(window) < 5:
        return False, 0.0
    import numpy as np
    arr  = list(window)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))
    if std == 0:
        return False, 0.0
    z = abs((quantity - mean) / std)
    return z > ZSCORE_THRESH, round(z, 3)

# ── Alternate route finder ─────────────────────────────────────────────────────
def find_alternate_route(G, manufacturer: str, retailer: str, disrupted_node: str):
    """Remove disrupted node and find shortest alternate path."""
    try:
        import networkx as nx
        G_alt = G.copy()
        if disrupted_node in G_alt:
            G_alt.remove_node(disrupted_node)
        if manufacturer not in G_alt or retailer not in G_alt:
            return None, "Nodes not in graph"
        path = nx.shortest_path(G_alt, manufacturer, retailer)
        return path, None
    except Exception as e:
        return None, str(e)

# ── Results writer ─────────────────────────────────────────────────────────────
RESULT_FIELDS = [
    "timestamp", "row_index", "manufacturer", "distributor",
    "retailer", "retailer_state", "quantity", "z_score",
    "is_anomaly", "disruption_injected",
    "alternate_route", "routing_note"
]

def write_result(result: dict, header_written: bool):
    mode = "a" if header_written else "w"
    with open(LIVE_RESULTS, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not header_written:
            writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in RESULT_FIELDS})

# ── Main consumer loop ─────────────────────────────────────────────────────────
def run():
    print("[CON] Stream consumer starting...")
    print(f"[CON] Watching: {LIVE_FEED}")
    print(f"[CON] Writing results to: {LIVE_RESULTS}")
    print("[CON] Waiting for data...\n")

    G = load_graph()

    # Reset results file
    if LIVE_RESULTS.exists():
        LIVE_RESULTS.unlink()

    quantity_window = deque(maxlen=WINDOW_SIZE)
    rows_seen       = 0
    anomalies_found = 0
    header_written  = False

    while True:
        if not LIVE_FEED.exists():
            time.sleep(0.5)
            continue

        # Read all rows, process only new ones
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

            is_disruption = row.get("disruption_injected", "0") == "1"

            # Override: zero quantity is always anomalous
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

            if is_anomaly:
                anomalies_found += 1

                # Try to find alternate route bypassing disrupted distributor
                if G and manufacturer and retailer:
                    path, err = find_alternate_route(G, manufacturer, retailer, distributor)
                    if path:
                        alternate_route = " → ".join(path)
                        routing_note    = f"Rerouted via {len(path)-1} hops (bypassed {distributor})"
                    else:
                        routing_note = f"No alternate route: {err}"
                else:
                    routing_note = "Graph unavailable — flagged for manual review"

                print(f"[CON] [{ts}] ⚡ ANOMALY #{anomalies_found} | Row {rows_seen}")
                print(f"         Flow    : {manufacturer[:30]} → {distributor[:25]} → {retailer[:25]}")
                print(f"         Qty     : {qty} | Z-score: {z_score}")
                if alternate_route:
                    print(f"         Reroute : {alternate_route[:80]}")
                print()
            else:
                if rows_seen % 20 == 0:
                    print(f"[CON] [{ts}] Row {rows_seen:4d} | Normal | "
                          f"{manufacturer[:25]:25s} → {retailer[:20]:20s} | "
                          f"qty={qty:.0f}")

            result = {
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
                "alternate_route"    : alternate_route,
                "routing_note"       : routing_note,
            }
            write_result(result, header_written)
            header_written = True

        time.sleep(0.1)

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n[CON] Stopped by user.")
