"""
Stream Simulator — Real-Time Supply Chain Sensor Feed
======================================================
Generates fresh synthetic rows on the fly using the known node pool
from the supply chain graph. Every run produces unique data — not a
replay of the same CSV.

Usage:
    python3 src/stream_simulator.py                    # 2s interval, no disruption
    python3 src/stream_simulator.py --interval 1       # faster
    python3 src/stream_simulator.py --disruption 30    # inject disruption at t=30s
    python3 src/stream_simulator.py --multi            # inject multiple disruptions

How it works:
    - Loads the supply chain graph to get real manufacturer/distributor/retailer names
    - Generates random but realistic transactions: normal quantities follow a log-normal
      distribution, timestamps increment in real time
    - At --disruption seconds, fires a shipment halt (qty=0) on a random active route
    - With --multi, fires a new disruption every 60s after the first
"""

import argparse
import csv
import os
import sys
import time
import random
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

ROOT            = Path(__file__).resolve().parent.parent
GRAPH_MODEL     = ROOT / "models" / "supplychain_graph.pkl"
STREAM_DIR      = ROOT / "data" / "stream"
LIVE_FEED       = STREAM_DIR / "live_feed.csv"
DISRUPTION_FLAG = STREAM_DIR / "disruption_active.flag"

STREAM_DIR.mkdir(parents=True, exist_ok=True)

STATES = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
          "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
          "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
          "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
          "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]

FIELDNAMES = ["date", "manufacturer", "distributor", "retailer",
              "retailer_state", "quantity", "disruption_injected"]

# ── Load graph node pools ─────────────────────────────────────────────────
def load_node_pools():
    if not GRAPH_MODEL.exists():
        print(f"[ERROR] Graph model not found: {GRAPH_MODEL}")
        print("  Run pipeline first: python3 main.py --from 1 --to 2")
        sys.exit(1)

    with open(GRAPH_MODEL, "rb") as f:
        G = pickle.load(f)

    manufacturers = [n for n, d in G.nodes(data=True) if d.get("type") == "manufacturer"]
    distributors  = [n for n, d in G.nodes(data=True) if d.get("type") == "distributor"]
    retailers     = [n for n, d in G.nodes(data=True) if d.get("type") == "retailer"]

    # Build adjacency for realistic routing
    mfr_to_dist   = {}
    dist_to_retail = {}
    for mfr in manufacturers:
        dists = [n for n in G.successors(mfr) if n in set(distributors)]
        if dists:
            mfr_to_dist[mfr] = dists
    for dist in distributors:
        rets = [n for n in G.successors(dist) if n in set(retailers)]
        if rets:
            dist_to_retail[dist] = rets

    print(f"[SIM] Graph loaded: {len(manufacturers)} mfrs, "
          f"{len(distributors)} distributors, {len(retailers)} retailers")
    return mfr_to_dist, dist_to_retail

# ── Generate one realistic transaction ────────────────────────────────────
def generate_row(mfr_to_dist: dict, dist_to_retail: dict,
                 disruption: bool = False) -> dict:
    mfr  = random.choice(list(mfr_to_dist.keys()))
    dists = mfr_to_dist[mfr]
    dist  = random.choice(dists)

    if dist in dist_to_retail:
        retailer = random.choice(dist_to_retail[dist])
    else:
        retailer = f"PHARMACY #{random.randint(1000,9999)} ({random.choice(STATES)})"

    # Infer state from retailer name if possible, else random
    state = "XX"
    for s in STATES:
        if f"({s})" in retailer:
            state = s
            break
    if state == "XX":
        state = random.choice(STATES)

    if disruption:
        qty = 0
    else:
        # Log-normal: realistic skewed quantity distribution
        qty = int(np.random.lognormal(mean=5.5, sigma=1.2))
        qty = max(1, min(qty, 50000))

    return {
        "date"               : datetime.now().strftime("%Y-%m-%d"),
        "manufacturer"       : mfr,
        "distributor"        : dist,
        "retailer"           : retailer,
        "retailer_state"     : state,
        "quantity"           : qty,
        "disruption_injected": 1 if disruption else 0,
    }

# ── Writer ────────────────────────────────────────────────────────────────
def write_row(row: dict, header_written: bool):
    mode = "a" if header_written else "w"
    with open(LIVE_FEED, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not header_written:
            writer.writeheader()
        writer.writerow(row)

# ── Main ──────────────────────────────────────────────────────────────────
def run(interval: float, disruption_after: float, multi: bool):
    mfr_to_dist, dist_to_retail = load_node_pools()

    if LIVE_FEED.exists():
        LIVE_FEED.unlink()
    if DISRUPTION_FLAG.exists():
        DISRUPTION_FLAG.unlink()

    print(f"[SIM] Streaming fresh synthetic data → {LIVE_FEED}")
    print(f"[SIM] Interval : {interval}s per row")
    if disruption_after:
        print(f"[SIM] Disruption fires at t+{disruption_after}s"
              + (" then every 60s" if multi else ""))
    print(f"[SIM] Press Ctrl+C to stop\n")

    start_time        = time.time()
    last_disruption   = -9999
    disruption_count  = 0
    header_written    = False
    row_count         = 0

    while True:
        elapsed = time.time() - start_time

        # Decide if this row should be a disruption
        fire_disruption = False
        if disruption_after:
            if elapsed >= disruption_after:
                if multi:
                    if elapsed - last_disruption >= 60:
                        fire_disruption = True
                else:
                    if disruption_count == 0:
                        fire_disruption = True

        row = generate_row(mfr_to_dist, dist_to_retail,
                           disruption=fire_disruption)
        write_row(row, header_written)
        header_written = True
        row_count += 1

        if fire_disruption:
            disruption_count += 1
            last_disruption = elapsed
            DISRUPTION_FLAG.write_text(
                f"Disruption #{disruption_count} at row {row_count}, t={elapsed:.1f}s\n"
                f"Node: {row['distributor']} → {row['retailer']}\n"
            )
            print(f"[SIM] ⚡ DISRUPTION #{disruption_count} at t={elapsed:.1f}s — "
                  f"{row['distributor']} → {row['retailer']}")

        if row_count % 10 == 0:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[SIM] [{ts}] Row {row_count:4d} | "
                  f"{row['manufacturer'][:28]:28s} → "
                  f"{row['retailer'][:22]:22s} | qty={row['quantity']}")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply Chain Stream Simulator")
    parser.add_argument("--interval",   type=float, default=2.0,
                        help="Seconds between rows (default: 2.0)")
    parser.add_argument("--disruption", type=float, default=None,
                        help="Inject disruption after N seconds")
    parser.add_argument("--multi",      action="store_true",
                        help="Keep injecting a disruption every 60s after the first")
    args = parser.parse_args()

    try:
        run(args.interval, args.disruption, args.multi)
    except KeyboardInterrupt:
        print("\n[SIM] Stopped by user.")
