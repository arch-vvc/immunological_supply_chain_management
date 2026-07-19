"""
Stage 9 — Macro Freight Risk Scorer
=====================================
Maps to: Innate Immunity — making anomaly detection thresholds dynamic
         by reading the real-world supply chain stress environment.

Reads real US government freight indicators (2019-2026) and computes
a weekly macro stress score from 5 key signals:
  1. Diesel prices          — high price = supply chain cost pressure
  2. Truck spot rates       — high rates = capacity crunch
  3. Ships awaiting berths  — high count = port congestion
  4. Freight Services Index — low index  = overall freight slowdown
  5. Inventory/Sales ratio  — high ratio = demand falling faster than supply

The stress score (0-1) is saved as a time-series lookup table.
anomaly_detection.py reads this table and tightens thresholds on
high-stress dates — fewer false negatives when the environment is bad.

Outputs:
    output/macro_stress_scores.csv        — weekly stress scores 2019-2026
    output/figures/fig7_macro_stress.png  — stress timeline with key events
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "supplementary", "Supply_Chain_and_Freight_Indicators.csv")
OUT_CSV   = os.path.join(ROOT, "output",  "macro_stress_scores.csv")
FIG_OUT   = os.path.join(ROOT, "output",  "figures", "fig7_macro_stress.png")
CONFIG    = os.path.join(ROOT, "config.yaml")

os.makedirs(os.path.join(ROOT, "output", "figures"), exist_ok=True)

print("=" * 55)
print("  STAGE 9 — MACRO FREIGHT RISK SCORER")
print("=" * 55)

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Supply_Chain_and_Freight_Indicators.csv not found at:\n  {DATA_PATH}")
    sys.exit(1)

# ── Load data ─────────────────────────────────────────────────
print("  Loading freight indicators...")
df = pd.read_csv(DATA_PATH)
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE", "VALUE1"])
df["VALUE1"] = pd.to_numeric(df["VALUE1"], errors="coerce")
df = df.dropna(subset=["VALUE1"])
print(f"  Loaded {len(df):,} records  |  {df['INDICATOR'].nunique()} indicators")
print(f"  Date range: {df['DATE'].min().date()} → {df['DATE'].max().date()}")

# ── Define stress indicators ──────────────────────────────────
# (indicator substring, direction: +1=higher is worse, -1=lower is worse, weight)
STRESS_INDICATORS = [
    ("Diesel Sales Prices",                        +1, 0.25),
    ("Truck Spot Rates",                           +1, 0.25),
    ("Containerships Awaiting Berths",             +1, 0.20),
    ("Freight Transportation Services Index",      -1, 0.15),
    ("Inventory to Sales Ratio",                   +1, 0.15),
]

def get_indicator(df, keyword):
    mask = df["INDICATOR"].str.contains(keyword, case=False, na=False)
    sub  = df[mask][["DATE", "VALUE1"]].copy()
    sub  = sub.dropna().sort_values("DATE")
    # Aggregate to weekly mean
    sub  = sub.set_index("DATE").resample("W")["VALUE1"].mean().reset_index()
    sub.columns = ["date", "value"]
    return sub

# ── Build per-indicator stress scores (0-1) ───────────────────
print("\n  Computing stress signals:")
series_list = []   # list of DataFrames
col_names   = []   # parallel list of plain Python strings (column names)
weights_list = []  # parallel list of weights

for keyword, direction, weight in STRESS_INDICATORS:
    s = get_indicator(df, keyword)
    if len(s) < 5:
        print(f"    ⚠ Skipped (insufficient data): {keyword}")
        continue

    vmin, vmax = s["value"].min(), s["value"].max()
    if vmax == vmin:
        s["stress"] = 0.5
    else:
        normalized = (s["value"] - vmin) / (vmax - vmin)
        s["stress"] = normalized if direction == +1 else (1 - normalized)

    series_list.append(s)
    col_names.append(keyword[:20].replace(" ", "_"))  # plain string
    weights_list.append(weight)
    print(f"    ✓ {keyword[:50]:<50}  weight={weight}")

if not series_list:
    print("[ERROR] No valid stress indicators found.")
    sys.exit(1)

# ── Combine into weekly stress score ─────────────────────────
all_dates = pd.date_range(
    start=min(s["date"].min() for s in series_list),
    end  =max(s["date"].max() for s in series_list),
    freq ="W"
)

stress_df = pd.DataFrame({"date": all_dates})

target_ts = stress_df["date"].values.astype(np.int64)   # nanoseconds

for s, col in zip(series_list, col_names):
    src_ts = s["date"].values.astype(np.int64)
    src_v  = s["stress"].values.astype(float)
    # Interpolate source series onto target date grid
    interp_vals = np.interp(target_ts, src_ts, src_v)
    stress_df[col] = interp_vals

# Weighted average
stress_cols = col_names                                 # plain list of strings
weights     = np.array(weights_list)
weights     = weights / weights.sum()   # normalise to sum to 1

stress_matrix         = stress_df[stress_cols].values
stress_df["stress_score"] = stress_matrix @ weights

# Smooth with 4-week rolling average
stress_df["stress_score"] = stress_df["stress_score"].rolling(4, min_periods=1).mean()

# Classify stress level
def classify(v):
    if v >= 0.65: return "HIGH"
    if v >= 0.40: return "MEDIUM"
    return "LOW"

stress_df["stress_level"] = stress_df["stress_score"].apply(classify)

# ── Save ──────────────────────────────────────────────────────
# Component columns are exported too: the LSTM forecaster (Stage 10) uses the
# raw indicators as multivariate input instead of only the smoothed composite.
stress_df[["date", "stress_score", "stress_level"] + stress_cols].to_csv(OUT_CSV, index=False)
print(f"\n  Stress scores saved → {OUT_CSV}")
print(f"  Weeks computed : {len(stress_df):,}")
print(f"  Score range    : {stress_df['stress_score'].min():.3f} → {stress_df['stress_score'].max():.3f}")
print(f"  HIGH stress weeks  : {(stress_df['stress_level']=='HIGH').sum()}")
print(f"  MEDIUM stress weeks: {(stress_df['stress_level']=='MEDIUM').sum()}")
print(f"  LOW stress weeks   : {(stress_df['stress_level']=='LOW').sum()}")

# ── Threshold impact preview ──────────────────────────────────
print("\n  Dynamic threshold preview (how anomaly Z-scores adjust):")
print(f"  {'Stress Level':<12} {'Multiplier':<12} {'Volume Z':<10} {'Freq Z':<10}")
print("  " + "─" * 44)
for level, mult in [("LOW", 1.20), ("MEDIUM", 1.0), ("HIGH", 0.75)]:
    print(f"  {level:<12} {mult:<12.2f} {3.0 * mult:<10.2f} {2.5 * mult:<10.2f}")

# ── Visualisation ─────────────────────────────────────────────
print("\n  Generating Fig 7: Macro Stress Timeline...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
fig.patch.set_facecolor("#0f0f1a")
for ax in [ax1, ax2]:
    ax.set_facecolor("#0f0f1a")

dates  = stress_df["date"]
scores = stress_df["stress_score"]

# Colour bands by stress level
ax1.fill_between(dates, scores, where=scores >= 0.65,
                 color="#e74c3c", alpha=0.4, label="HIGH stress")
ax1.fill_between(dates, scores, where=(scores >= 0.40) & (scores < 0.65),
                 color="#f39c12", alpha=0.4, label="MEDIUM stress")
ax1.fill_between(dates, scores, where=scores < 0.40,
                 color="#2ecc71", alpha=0.4, label="LOW stress")

ax1.plot(dates, scores, color="white", linewidth=1.2, alpha=0.9)
ax1.axhline(0.65, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.6)
ax1.axhline(0.40, color="#f39c12", linewidth=0.8, linestyle="--", alpha=0.6)

# Annotate known events
events = [
    ("2020-03-01", "COVID-19", "#e74c3c"),
    ("2021-03-01", "Suez Canal", "#f39c12"),
    ("2021-11-01", "Port Congestion", "#f39c12"),
    ("2022-03-01", "Ukraine War", "#e74c3c"),
]
for date_str, label, color in events:
    try:
        dt = pd.Timestamp(date_str)
        if dates.min() <= dt <= dates.max():
            ax1.axvline(dt, color=color, linewidth=1.2, linestyle=":", alpha=0.8)
            ax1.text(dt, scores.max() * 0.92, label,
                     color=color, fontsize=7, rotation=90, va="top", ha="right")
    except Exception:
        pass

ax1.set_ylabel("Macro Stress Score (0–1)", color="white")
ax1.set_title("Supply Chain Macro Stress Score — Weekly (2019–2026)\n"
              "Derived from US DoT Freight Indicators",
              color="white", fontsize=11, pad=10)
ax1.tick_params(colors="white")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, loc="upper left")
for spine in ax1.spines.values():
    spine.set_color("#4a4a6a")
ax1.set_ylim(0, 1)

# Bottom panel: stress level bar
level_colors = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
bar_colors   = [level_colors[l] for l in stress_df["stress_level"]]
ax2.bar(dates, [1] * len(dates), color=bar_colors, width=7, alpha=0.85)
ax2.set_yticks([])
ax2.set_xlabel("Date", color="white")
ax2.set_title("Stress Level", color="white", fontsize=8)
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_color("#4a4a6a")

patches = [mpatches.Patch(color=v, label=k) for k, v in level_colors.items()]
ax2.legend(handles=patches, facecolor="#1a1a2e", labelcolor="white",
           fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {FIG_OUT}")
print("\n  Stage 9 complete.")
print("  anomaly_detection.py will now auto-adjust thresholds using these scores.")
