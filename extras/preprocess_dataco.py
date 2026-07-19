"""
Stage 1-ALT — DataCo Supply Chain Dataset Preprocessor
=======================================================
Alternative to preprocess.py when the ARCOS dataset is unavailable.
Reads DataCoSupplyChainDataset.csv and maps its columns to the same
6-column clean_chain.csv format used by every downstream stage.

Column mapping:
    date           ← order date (DateOrders)
    manufacturer   ← Department Name       (Fan Shop, Apparel, Golf …)
    distributor    ← Order Region          (Western Europe, LATAM …)
    retailer       ← Customer City + State (e.g. "Los Angeles, CA")
    retailer_state ← Customer State
    quantity       ← Sales                 (order dollar volume)

Run once instead of Stage 1:
    python3 src/preprocess_dataco.py
Then run the rest of the pipeline:
    python3 main.py --from 2
"""

import pandas as pd
import os

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT  = os.path.join(ROOT, "..", "capstone", "DataCoSupplyChainDataset.csv")
OUTPUT = os.path.join(ROOT, "data", "processed", "clean_chain.csv")

print("=" * 55)
print("  STAGE 1-ALT — DATACO PREPROCESSOR")
print("=" * 55)

if not os.path.exists(INPUT):
    print(f"[ERROR] DataCoSupplyChainDataset.csv not found at:\n  {INPUT}")
    print("  Ensure the capstone/ folder is at the same level as capstone_project/")
    exit(1)

print("Loading DataCoSupplyChainDataset.csv ...")
df = pd.read_csv(INPUT, encoding="latin-1")
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

# ── Select and rename columns ────────────────────────────────
df = df[[
    "order date (DateOrders)",
    "Department Name",
    "Order Region",
    "Customer City",
    "Customer State",
    "Sales",
]].copy()

df.columns = ["date", "manufacturer", "distributor", "retailer_city", "retailer_state", "quantity"]

# Combine city + state into a unique retailer identifier
df["retailer"] = df["retailer_city"].str.strip() + ", " + df["retailer_state"].str.strip()
df = df.drop(columns=["retailer_city"])

# Reorder to match pipeline convention
df = df[["date", "manufacturer", "distributor", "retailer", "retailer_state", "quantity"]]

# ── Clean ────────────────────────────────────────────────────
before = len(df)
df = df.dropna()

# Parse dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Drop zero/negative sales
df = df[df["quantity"] > 0]

# Strip whitespace from string columns
for col in ["manufacturer", "distributor", "retailer", "retailer_state"]:
    df[col] = df[col].str.strip()

after = len(df)
print(f"Dropped {before - after:,} null/invalid rows  ({after:,} remaining)")

# ── Summary ──────────────────────────────────────────────────
print(f"\nUnique manufacturers : {df['manufacturer'].nunique()}")
print(f"Unique distributors  : {df['distributor'].nunique()}")
print(f"Unique retailers     : {df['retailer'].nunique()}")
print(f"Unique states        : {df['retailer_state'].nunique()}")
print(f"Date range           : {df['date'].min().date()}  →  {df['date'].max().date()}")
print(f"Quantity (Sales $)   : min={df['quantity'].min():.2f}  max={df['quantity'].max():.2f}  mean={df['quantity'].mean():.2f}")

# ── Save ─────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
df.to_csv(OUTPUT, index=False)

print(f"\nSaved → {OUTPUT}")
print(f"\nNext: python3 main.py --from 2")
