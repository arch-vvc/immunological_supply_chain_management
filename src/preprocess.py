"""
Stage 1 — Universal Preprocessor
==================================
Reads config.yaml (written by auto_onboard.py) and maps whatever columns
the company's dataset has into the standard 6-column clean_chain.csv.

Falls back to ARCOS column names if no config.yaml exists.

Output always has exactly these columns:
    date, manufacturer, distributor, retailer, retailer_state, quantity
    + label-encoded variants: manufacturer_enc, distributor_enc,
                              retailer_enc, retailer_state_enc
    + engineered features:   tx_freq_per_pair, vol_concentration,
                              month, year, day_of_week

Also validates disruption_processed.csv if present in data/supplementary/.
"""

import pandas as pd
import numpy as np
import os
import sys

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config.yaml")
OUTPUT = os.path.join(ROOT, "data", "processed", "clean_chain.csv")

print("=" * 55)
print("  STAGE 1 — UNIVERSAL PREPROCESSOR")
print("=" * 55)


# ── Load config or fall back to ARCOS defaults ───────────────

def load_config():
    if HAS_YAML and os.path.exists(CONFIG):
        with open(CONFIG, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"  Config loaded from config.yaml")
        return cfg
    else:
        print("  No config.yaml found — using ARCOS defaults")
        return {
            "dataset": {
                "path"     : os.path.join(ROOT, "data", "raw", "arcos_sampled_50k.csv"),
                "separator": ",",
                "encoding" : "utf-8",
            },
            "columns": {
                "date"          : "TRANSACTION_DATE",
                "manufacturer"  : "Revised_Company_Name",
                "distributor"   : "REPORTER_NAME",
                "retailer"      : "BUYER_NAME",
                "retailer_state": "BUYER_STATE",
                "quantity"      : "QUANTITY",
            },
            "settings": {
                "retailer_combine": None,
            },
        }

cfg      = load_config()
ds       = cfg.get("dataset", {})
col_map  = cfg.get("columns", {})
settings = cfg.get("settings", {})

filepath = ds.get("path", "")
if filepath and not os.path.isabs(filepath):
    filepath = os.path.join(ROOT, filepath)
sep      = ds.get("separator", ",")
encoding = ds.get("encoding", "utf-8")

if not os.path.exists(filepath):
    print(f"[ERROR] Dataset not found: {filepath}")
    print("  Run auto-onboarding first: python3 main.py --onboard /path/to/data.csv")
    sys.exit(1)

# ── Load dataset ─────────────────────────────────────────────

print(f"\n  Loading: {os.path.basename(filepath)}")
df = pd.read_csv(filepath, sep=sep, encoding=encoding, low_memory=False, on_bad_lines="skip")
print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

# ── Map columns ───────────────────────────────────────────────

ROLES = ["date", "manufacturer", "distributor", "retailer", "retailer_state", "quantity"]
missing = [r for r in ROLES if r not in col_map or col_map[r] not in df.columns]

if missing:
    print(f"\n[ERROR] Could not find mapped columns for: {missing}")
    print("  Re-run onboarding: python3 main.py --onboard /path/to/data.csv")
    sys.exit(1)

df = df[[col_map[r] for r in ROLES]].copy()
df.columns = ROLES

# ── Combine retailer + extra column if configured ────────────

retailer_combine = settings.get("retailer_combine")
if retailer_combine:
    raw = pd.read_csv(filepath, sep=sep, encoding=encoding, low_memory=False,
                      usecols=[retailer_combine], on_bad_lines="skip")
    if retailer_combine in raw.columns:
        df["retailer"] = (df["retailer"].astype(str).str.strip()
                          + ", "
                          + raw[retailer_combine].astype(str).str.strip())
        print(f"  Retailer combined with '{retailer_combine}' for uniqueness")

# ── Clean ─────────────────────────────────────────────────────

before = len(df)
df = df.dropna()
df["date"]     = pd.to_datetime(df["date"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df = df.dropna()
df = df[df["quantity"] > 0]

for col in ["manufacturer", "distributor", "retailer", "retailer_state"]:
    df[col] = df[col].astype(str).str.strip()

after = len(df)
print(f"  Dropped {before - after:,} invalid rows  ({after:,} remaining)")

# ── Summary ───────────────────────────────────────────────────

print(f"\n  Manufacturers : {df['manufacturer'].nunique()}")
print(f"  Distributors  : {df['distributor'].nunique()}")
print(f"  Retailers     : {df['retailer'].nunique()}")
print(f"  States        : {df['retailer_state'].nunique()}")
print(f"  Date range    : {df['date'].min().date()}  →  {df['date'].max().date()}")
print(f"  Quantity      : min={df['quantity'].min():.1f}  max={df['quantity'].max():.1f}  mean={df['quantity'].mean():.1f}")

# ── Label Encoding ────────────────────────────────────────────
# Encode categorical string columns as integer codes.
# Original string columns are preserved; _enc variants added.

print("\n  Label encoding categorical columns...")
for col in ["manufacturer", "distributor", "retailer", "retailer_state"]:
    df[f"{col}_enc"] = df[col].astype("category").cat.codes
    n_cats = df[f"{col}_enc"].nunique()
    print(f"    {col}_enc  →  {n_cats} unique categories")

# ── Feature Engineering ───────────────────────────────────────
# Add derived features useful for downstream ML stages.

print("\n  Engineering features...")

# Transaction frequency per manufacturer-retailer pair
pair_freq = (
    df.groupby(["manufacturer", "retailer"])
    .size()
    .reset_index(name="tx_freq_per_pair")
)
df = df.merge(pair_freq, on=["manufacturer", "retailer"], how="left")

# Volume concentration: this distributor's share of retailer's total supply
retailer_total = df.groupby("retailer")["quantity"].transform("sum")
dist_ret_vol   = df.groupby(["distributor", "retailer"])["quantity"].transform("sum")
df["vol_concentration"] = (dist_ret_vol / retailer_total.replace(0, np.nan)).fillna(0).round(4)

# Temporal features
df["month"]       = df["date"].dt.month
df["year"]        = df["date"].dt.year
df["day_of_week"] = df["date"].dt.dayofweek   # 0=Monday … 6=Sunday

print(f"    tx_freq_per_pair  — range: {df['tx_freq_per_pair'].min()}–{df['tx_freq_per_pair'].max()}")
print(f"    vol_concentration — mean: {df['vol_concentration'].mean():.3f}")
print(f"    month / year / day_of_week added")

# ── Disruption Dataset Validation ────────────────────────────
# If disruption_processed.csv is present, validate schema and report stats.
# This file feeds Stages 8 and 12 — we check it is intact here.

DISRUPT_PATH = os.path.join(ROOT, "data", "supplementary", "disruption_processed.csv")
REQUIRED_DISRUPTION_COLS = [
    "disruption_type_enc", "industry_enc", "supplier_region_enc",
    "supplier_size_enc", "response_type_enc", "disruption_severity",
    "production_impact_pct", "has_backup_supplier", "full_recovery_days",
]

print(f"\n  Checking supplementary disruption dataset...")
if os.path.exists(DISRUPT_PATH):
    try:
        ddf = pd.read_csv(DISRUPT_PATH, nrows=5)   # schema check only — no full load
        full_ddf = pd.read_csv(DISRUPT_PATH)
        missing_cols = [c for c in REQUIRED_DISRUPTION_COLS if c not in full_ddf.columns]
        if missing_cols:
            print(f"    [WARN] disruption_processed.csv missing columns: {missing_cols}")
        else:
            print(f"    disruption_processed.csv  ✅  {len(full_ddf):,} rows  |  {len(full_ddf.columns)} columns")
            print(f"    Industries : {full_ddf['industry_enc'].nunique() if 'industry_enc' in full_ddf else 'N/A'} unique")
            print(f"    Avg recovery days : {full_ddf['full_recovery_days'].mean():.1f}" if 'full_recovery_days' in full_ddf else "")
            sev_counts = full_ddf["disruption_severity"].value_counts().to_dict() if "disruption_severity" in full_ddf else {}
            print(f"    Severity distribution : {sev_counts}")
    except Exception as e:
        print(f"    [WARN] Could not read disruption_processed.csv: {e}")
else:
    print(f"    disruption_processed.csv not found at {DISRUPT_PATH}")
    print(f"    (Stages 8 and 12 require this file)")

# ── Save ──────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
df.to_csv(OUTPUT, index=False)
print(f"\n  Saved → {OUTPUT}")
print(f"  Columns: {list(df.columns)}")
