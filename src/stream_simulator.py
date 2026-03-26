"""
Stream Simulator — Real-Time Supply Chain Sensor Feed
======================================================
Reads from an existing processed dataset (clean_chain.csv) and emits
rows one at a time to a shared queue file (data/stream/live_feed.csv),
simulating sensor/IoT data arriving in real time.

Usage:
    python3 src/stream_simulator.py                    # normal speed (2s interval)
    python3 src/stream_simulator.py --interval 0.5     # faster
    python3 src/stream_simulator.py --disruption 30    # inject disruption after 30s
    python3 src/stream_simulator.py --loop             # loop back to start when done

Disruption injection:
    - After --disruption seconds, one row is spiked with:
        quantity = 0, delay_flag = 1 (simulates shipment halt)
    - This is what triggers the anomaly pipeline to fire
"""

import argparse
import csv
import os
import sys
import time
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV   = ROOT / "data" / "processed" / "clean_chain.csv"
STREAM_DIR  = ROOT / "data" / "stream"
LIVE_FEED   = STREAM_DIR / "live_feed.csv"
DISRUPTION_FLAG = STREAM_DIR / "disruption_active.flag"

STREAM_DIR.mkdir(parents=True, exist_ok=True)

# ── Column definitions ────────────────────────────────────────────────────────
REQUIRED_COLS = ["date", "manufacturer", "distributor", "retailer",
                 "retailer_state", "quantity"]

def load_source_data():
    if not CLEAN_CSV.exists():
        print(f"[ERROR] {CLEAN_CSV} not found.")
        print("  Run the pipeline first: python3 main.py --from 1")
        sys.exit(1)

    rows = []
    with open(CLEAN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[SIM] Loaded {len(rows):,} rows from {CLEAN_CSV.name}")
    return rows

def inject_disruption(row: dict) -> dict:
    """Spike a row to simulate a shipment disruption."""
    spiked = row.copy()
    spiked["quantity"] = "0"
    spiked["disruption_injected"] = "1"
    spiked["date"] = datetime.now().strftime("%Y-%m-%d")
    return spiked

def write_row(row: dict, header_written: bool, fieldnames: list):
    mode = "a" if header_written else "w"
    with open(LIVE_FEED, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not header_written:
            writer.writeheader()
        writer.writerow(row)

def run(interval: float, disruption_after: float, loop: bool):
    rows = load_source_data()

    # Get all columns (some may have extras from encoding/feature engineering)
    fieldnames = list(rows[0].keys())
    if "disruption_injected" not in fieldnames:
        fieldnames.append("disruption_injected")

    # Reset live feed
    if LIVE_FEED.exists():
        LIVE_FEED.unlink()
    if DISRUPTION_FLAG.exists():
        DISRUPTION_FLAG.unlink()

    print(f"[SIM] Streaming to: {LIVE_FEED}")
    print(f"[SIM] Interval: {interval}s per row")
    if disruption_after:
        print(f"[SIM] Disruption will fire at t+{disruption_after}s")
    print(f"[SIM] Press Ctrl+C to stop\n")

    start_time      = time.time()
    disruption_fired = False
    header_written  = False
    row_count       = 0

    while True:
        if loop:
            source = rows * 999  # effectively infinite
        else:
            source = rows

        for row in source:
            elapsed = time.time() - start_time

            # Inject disruption once
            if disruption_after and not disruption_fired and elapsed >= disruption_after:
                disrupted_row = inject_disruption(row)
                disrupted_row["disruption_injected"] = "1"
                write_row(disrupted_row, header_written, fieldnames)
                header_written = True
                row_count += 1
                disruption_fired = True
                # Write flag file so consumer knows
                DISRUPTION_FLAG.write_text(
                    f"Disruption injected at row {row_count}, t={elapsed:.1f}s\n"
                    f"Node: {row.get('distributor','?')} → {row.get('retailer','?')}\n"
                )
                print(f"[SIM] ⚡ DISRUPTION INJECTED at t={elapsed:.1f}s — "
                      f"{row.get('distributor','?')} → {row.get('retailer','?')}")
            else:
                normal_row = row.copy()
                normal_row["disruption_injected"] = "0"
                write_row(normal_row, header_written, fieldnames)
                header_written = True
                row_count += 1

            if row_count % 10 == 0:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[SIM] [{ts}] Emitted {row_count} rows | "
                      f"Latest: {row.get('manufacturer','?')[:25]:25s} → "
                      f"{row.get('retailer','?')[:20]:20s} | qty={row.get('quantity','?')}")

            time.sleep(interval)

        if not loop:
            print(f"\n[SIM] Done. Emitted {row_count} rows total.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply Chain Stream Simulator")
    parser.add_argument("--interval",    type=float, default=2.0,
                        help="Seconds between rows (default: 2.0)")
    parser.add_argument("--disruption",  type=float, default=None,
                        help="Inject disruption after N seconds (default: disabled)")
    parser.add_argument("--loop",        action="store_true",
                        help="Loop back to start when source data is exhausted")
    args = parser.parse_args()

    try:
        run(args.interval, args.disruption, args.loop)
    except KeyboardInterrupt:
        print("\n[SIM] Stopped by user.")
