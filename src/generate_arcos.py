"""
Stage 0-ALT — Synthetic ARCOS Dataset Generator
================================================
Generates a realistic 50K-row synthetic pharmaceutical supply chain dataset
in ARCOS format. Use this when the real ARCOS file (datasetuc.csv) is not
available. Output is identical in structure to arcos_sampled_50k.csv.

Run once:
    python3 src/generate_arcos.py
Then run the full pipeline:
    python3 main.py
"""

import pandas as pd
import numpy as np
import os
import random

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "data", "raw", "arcos_sampled_50k.csv")

SEED     = 18
N_ROWS   = 50_000

# ── SCENARIO ────────────────────────────────────────────────
# "concentrated" : few distributors dominate, heavy bottlenecks,
#                  multiple manufacturer surges → higher anomaly count
# "distributed"  : traffic spread evenly, mild volume spikes,
#                  single small surge → lower anomaly count
# Change this alongside SEED to get structurally different runs.
SCENARIO = "distributed"   # "concentrated" | "distributed"

random.seed(SEED)
np.random.seed(SEED)

print("=" * 55)
print("  SYNTHETIC ARCOS DATASET GENERATOR")
print(f"  Scenario : {SCENARIO.upper()}   Seed : {SEED}")
print("=" * 55)

# ── Real-world entity names ─────────────────────────────────

MANUFACTURERS = [
    "PURDUE PHARMA LP", "MALLINCKRODT LLC", "ACTAVIS PHARMA INC",
    "WATSON LABORATORIES INC", "ENDO PHARMACEUTICALS INC",
    "RHODES TECHNOLOGIES", "ABBOTT LABORATORIES",
    "JANSSEN PHARMACEUTICALS INC", "TEVA PHARMACEUTICALS USA INC",
    "AMNEAL PHARMACEUTICALS LLC", "HIKMA PHARMACEUTICALS",
    "PAR PHARMACEUTICAL INC", "QUALITEST PHARMACEUTICALS",
    "MYLAN PHARMACEUTICALS INC", "SANDOZ INC",
    "LANNETT COMPANY INC", "IMPAX LABORATORIES INC",
    "SUN PHARMA GLOBAL INC", "CADISTA PHARMACEUTICALS INC",
    "VINTAGE PHARMACEUTICALS LLC",
]

DISTRIBUTORS = [
    "MCKESSON CORPORATION", "CARDINAL HEALTH INC",
    "AMERISOURCEBERGEN DRUG CORP", "MIAMI-LUKEN INC",
    "HD SMITH WHOLESALE DRUG CO", "NATIONAL WHOLESALE DRUG CO INC",
    "HENRY SCHEIN INC", "KINRAY LLC",
    "ANDA INC", "BURLINGTON DRUG COMPANY INC",
    "SMITH DRUG COMPANY", "PENN PHARMACEUTICAL SERVICES INC",
    "MORRIS & DICKSON CO LLC", "DAKOTA DRUG INC",
    "BELLCO DRUG CORP", "KEYSOURCE MEDICAL INC",
    "GULF COAST MEDICAL SUPPLY", "PRESCRIPTION SUPPLY INC",
    "DIRECT RX", "VALUEDRUGCO",
]

PHARMACY_PREFIXES = [
    "CVS PHARMACY", "WALGREENS", "RITE AID", "KROGER PHARMACY",
    "WALMART PHARMACY", "PUBLIX PHARMACY", "HEALTH MART PHARMACY",
    "GOOD NEIGHBOR PHARMACY", "WINN-DIXIE PHARMACY", "HARRIS TEETER PHARMACY",
    "FAMILY CARE PHARMACY", "COMMUNITY PHARMACY", "PEOPLES PHARMACY",
    "MEDICO PHARMACY", "SUNRISE PHARMACY", "VILLAGE PHARMACY",
    "MAIN STREET PHARMACY", "CORNER DRUG STORE", "HOMETOWN PHARMACY",
    "QUALITY CARE PHARMACY",
]

STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
]

# Weight states toward high-opioid-impact states (WV, OH, KY, PA, TN)
STATE_WEIGHTS = {s: 1 for s in STATES}
for s in ["WV","OH","KY","PA","TN","IN","MI","NC","VA","LA"]:
    STATE_WEIGHTS[s] = 6
state_list   = list(STATE_WEIGHTS.keys())
state_probs  = np.array(list(STATE_WEIGHTS.values()), dtype=float)
state_probs /= state_probs.sum()

# ── Generate pharmacy names ─────────────────────────────────

def make_pharmacy(state, idx):
    prefix = random.choice(PHARMACY_PREFIXES)
    num    = random.randint(1000, 9999)
    return f"{prefix} #{num} ({state})"

# Pre-generate fixed pool of ~500 pharmacies per state
PHARMACY_POOL = {}
for s in STATES:
    PHARMACY_POOL[s] = [make_pharmacy(s, i) for i in range(random.randint(20, 60))]

# ── Date range (ARCOS covers 2006–2014) ─────────────────────

start_date = pd.Timestamp("2006-01-01")
end_date   = pd.Timestamp("2014-12-31")
date_range = (end_date - start_date).days

# ── Scenario parameters ──────────────────────────────────────

if SCENARIO == "concentrated":
    # Heavy bottlenecks: few distributors dominate, large spikes, multiple surges
    SPIKE_MULT_LO   = 20
    SPIKE_MULT_HI   = 80
    SPIKE_CLIP_LO   = 80_000
    SPIKE_CLIP_HI   = 8_000_000
    ANOMALY_FRAC    = 0.05         # 5% volume-spike rows
    CONCENTRATION_N = 500          # flood rows for a single pharmacy
    CONC_QTY_LO     = 60_000
    CONC_QTY_HI     = 200_000
    # Three manufacturer surges
    SURGES = [
        dict(mfr="PURDUE PHARMA LP",         n=300, qty_lo=100_000, qty_hi=600_000,
             start="2012-06-01", end="2012-09-30"),
        dict(mfr="MALLINCKRODT LLC",          n=200, qty_lo=80_000,  qty_hi=400_000,
             start="2010-03-01", end="2010-07-31"),
        dict(mfr="ACTAVIS PHARMA INC",        n=150, qty_lo=60_000,  qty_hi=300_000,
             start="2013-01-01", end="2013-05-31"),
    ]
    # Skew 70% of distributor traffic to top 3
    dist_weights = np.ones(len(DISTRIBUTORS))
    for top_d in ["MCKESSON CORPORATION", "CARDINAL HEALTH INC", "AMERISOURCEBERGEN DRUG CORP"]:
        dist_weights[DISTRIBUTORS.index(top_d)] = 12
    dist_probs = dist_weights / dist_weights.sum()
else:  # "distributed"
    # Even traffic, mild spikes, single small surge
    SPIKE_MULT_LO   = 5
    SPIKE_MULT_HI   = 15
    SPIKE_CLIP_LO   = 20_000
    SPIKE_CLIP_HI   = 1_000_000
    ANOMALY_FRAC    = 0.015        # 1.5% volume-spike rows
    CONCENTRATION_N = 80           # light concentration
    CONC_QTY_LO     = 15_000
    CONC_QTY_HI     = 50_000
    SURGES = [
        dict(mfr="JANSSEN PHARMACEUTICALS INC", n=60, qty_lo=20_000, qty_hi=100_000,
             start="2011-04-01", end="2011-06-30"),
    ]
    dist_probs = None              # uniform distributor selection

# ── Build base transactions ──────────────────────────────────

print(f"Generating {N_ROWS:,} transactions...")

mfr_idx   = np.random.choice(len(MANUFACTURERS),  N_ROWS)
dist_idx  = np.random.choice(len(DISTRIBUTORS),   N_ROWS, p=dist_probs)
state_idx = np.random.choice(len(state_list),      N_ROWS, p=state_probs)

states    = [state_list[i] for i in state_idx]
retailers = [random.choice(PHARMACY_POOL[s]) for s in states]

# Quantities: log-normal (realistic pharma distribution)
quantities = np.random.lognormal(mean=5.5, sigma=1.2, size=N_ROWS).astype(int)
quantities = np.clip(quantities, 10, 100_000)

# Dates: skewed toward 2010-2013 (peak opioid years)
day_offsets = np.random.beta(a=3, b=2, size=N_ROWS) * date_range
dates = [start_date + pd.Timedelta(days=int(d)) for d in day_offsets]

# ── Inject anomalies ─────────────────────────────────────────

n_anomalies = int(N_ROWS * ANOMALY_FRAC)
anomaly_idx = np.random.choice(N_ROWS, n_anomalies, replace=False)

# Volume spike anomalies — scenario-specific multiplier
quantities[anomaly_idx] = (quantities[anomaly_idx] *
    np.random.uniform(SPIKE_MULT_LO, SPIKE_MULT_HI, n_anomalies)).astype(int)
quantities[anomaly_idx] = np.clip(quantities[anomaly_idx], SPIKE_CLIP_LO, SPIKE_CLIP_HI)

# Concentration anomaly — one distributor floods a single pharmacy in WV
concentration_n     = CONCENTRATION_N
concentration_dist  = "MIAMI-LUKEN INC"
concentration_pharm = "TUG VALLEY PHARMACY #1 (WV)"
conc_idx = np.random.choice(N_ROWS, concentration_n, replace=False)
for i in conc_idx:
    dist_idx[i]  = DISTRIBUTORS.index(concentration_dist)
    states[i]    = "WV"
    retailers[i] = concentration_pharm
    quantities[i] = int(np.random.uniform(CONC_QTY_LO, CONC_QTY_HI))

# Temporal surges — scenario-specific number and scale
total_surge_n = 0
for surge in SURGES:
    s_start = pd.Timestamp(surge["start"])
    s_end   = pd.Timestamp(surge["end"])
    s_days  = (s_end - s_start).days
    s_idx   = np.random.choice(N_ROWS, surge["n"], replace=False)
    for i in s_idx:
        mfr_idx[i]   = MANUFACTURERS.index(surge["mfr"])
        quantities[i] = int(np.random.uniform(surge["qty_lo"], surge["qty_hi"]))
        dates[i]     = s_start + pd.Timedelta(days=int(random.uniform(0, s_days)))
    total_surge_n += surge["n"]

# ── Assemble DataFrame ───────────────────────────────────────

df = pd.DataFrame({
    "TRANSACTION_DATE"      : [d.strftime("%m/%d/%Y") for d in dates],
    "Revised_Company_Name"  : [MANUFACTURERS[i]  for i in mfr_idx],
    "REPORTER_NAME"         : [DISTRIBUTORS[i]   for i in dist_idx],
    "BUYER_NAME"            : retailers,
    "BUYER_STATE"           : states,
    "QUANTITY"              : quantities,
    # Filler columns to reach realistic column count
    "REPORTER_DEA_NO"       : [f"R{random.randint(100000,999999)}" for _ in range(N_ROWS)],
    "BUYER_DEA_NO"          : [f"B{random.randint(100000,999999)}" for _ in range(N_ROWS)],
    "DRUG_NAME"             : np.random.choice(
                                ["OXYCODONE","HYDROCODONE","FENTANYL",
                                 "MORPHINE","CODEINE","METHADONE",
                                 "OXYMORPHONE","BUPRENORPHINE"], N_ROWS),
    "TRANSACTION_CODE"       : np.random.choice(["S","P","T","X"], N_ROWS),
    "UNIT"                   : np.random.choice(["GM","MCG","ML"], N_ROWS),
    "ACTION_INDICATOR"       : np.random.choice(["S","T"], N_ROWS),
    "ORDER_FORM_NO"          : [f"OF{random.randint(1000000,9999999)}" for _ in range(N_ROWS)],
    "CORRECTION_NO"          : np.zeros(N_ROWS, dtype=int),
    "STRENGTH"               : np.random.choice([5, 10, 15, 20, 30, 40, 80], N_ROWS),
    "TRANSACTION_ID"         : range(1, N_ROWS + 1),
})

# Shuffle rows
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
df.to_csv(OUTPUT, index=False)

total_injected = n_anomalies + concentration_n + total_surge_n
print(f"Rows generated    : {len(df):,}")
print(f"Anomalies injected: {total_injected:,} (~{total_injected/N_ROWS*100:.1f}%)")
print(f"Date range        : {df['TRANSACTION_DATE'].iloc[0]}  →  synthetic 2006-2014")
print(f"Manufacturers     : {df['Revised_Company_Name'].nunique()}")
print(f"Distributors      : {df['REPORTER_NAME'].nunique()}")
print(f"Pharmacies        : {df['BUYER_NAME'].nunique()}")
print(f"States            : {df['BUYER_STATE'].nunique()}")
print(f"\nSaved → {OUTPUT}")
print(f"\nNext: python3 main.py   (runs full pipeline from Stage 1)")

# Write config.yaml so preprocess.py uses ARCOS columns
import yaml
config = {
    "dataset": {
        "path"     : OUTPUT,
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
    "settings": {"retailer_combine": None},
    "thresholds": {
        "volume_zscore": 3.0,
        "frequency_zscore": 2.5,
        "temporal_surge_zscore": 2.5,
        "concentration_pct": 0.90,
    },
}
config_path = os.path.join(ROOT, "config.yaml")
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print(f"Config written → {config_path}")