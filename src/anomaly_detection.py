"""
Stage 3 — Multi-Dimensional Anomaly Detection (Enhanced)
Maps to: Innate Immunity — fast, broad detection of suspicious signals

Five detection dimensions:
  1. Volume Anomaly     — Z-score on transaction quantity
  2. Frequency Anomaly — unusually high number of transactions on a pair
  3. Temporal Surge    — sudden spike in a manufacturer's monthly volume
  4. Concentration Risk — single distributor supplying >90% of a retailer
  5. Isolation Forest  — ML-based joint-distribution outlier detection (5th signal)

A transaction flagged on 2+ dimensions is classified as a high-confidence anomaly.

Evaluation note: detection quality (P/R/F1) is NOT measured here — flags
derived from the detectors cannot also serve as ground truth. The independent
benchmark lives in anomaly_eval_injection.py (Stage 3b), which injects known
synthetic anomalies and scores the same detectors against that label.
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import shutil
import tempfile

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT  = os.path.join(ROOT, "data", "processed", "clean_chain.csv")
OUTPUT = os.path.join(ROOT, "output", "anomalies.csv")
CONFIG = os.path.join(ROOT, "config.yaml")

# Load thresholds from config.yaml if present, else use defaults
_thresholds = {}
if HAS_YAML and os.path.exists(CONFIG):
    with open(CONFIG) as f:
        _cfg = yaml.safe_load(f)
    _thresholds = _cfg.get("thresholds", {})

VOL_Z    = _thresholds.get("volume_zscore",         3.0)
FREQ_Z   = _thresholds.get("frequency_zscore",      2.5)
SURGE_Z  = _thresholds.get("temporal_surge_zscore", 2.5)
CONC_PCT = _thresholds.get("concentration_pct",     0.90)

# ── Macro stress adjustment (per-week) ────────────────────────
# If Stage 9 has run, each TRANSACTION's z-thresholds are scaled by the
# multiplier of ITS OWN calendar week's macro stress — tighten in high-stress
# weeks (catch more), relax in low-stress weeks (fewer false alarms). This is
# the paper's theta_adj = lambda * theta_base applied per week rather than as
# one run-level average. Transactions outside indicator coverage fall back to
# the coverage-mean multiplier (equivalent to the old static behaviour).
MACRO_STRESS = os.path.join(ROOT, "output", "macro_stress_scores.csv")

def _stress_lambda(score):
    # HIGH stress (>=0.65) -> 0.75 (lower thresholds = more sensitive)
    # LOW  stress (<0.40)  -> 1.20 (higher thresholds = fewer false positives)
    if score >= 0.65:
        return 0.75
    elif score >= 0.40:
        return 1.00
    return 1.20

def get_weekly_multipliers(date_series):
    """Per-transaction multiplier Series aligned to date_series, plus
    (share of rows inside indicator coverage, fallback multiplier)."""
    if not os.path.exists(MACRO_STRESS):
        return pd.Series(1.0, index=date_series.index), 0.0, 1.0
    try:
        stress_df = (pd.read_csv(MACRO_STRESS, parse_dates=["date"])
                       .sort_values("date"))
        fallback = _stress_lambda(stress_df["stress_score"].mean())
        d = (date_series.rename("date").reset_index()
                        .sort_values("date"))
        merged = pd.merge_asof(d, stress_df[["date", "stress_score"]],
                               on="date", direction="backward",
                               tolerance=pd.Timedelta(days=14))
        in_cov = float(merged["stress_score"].notna().mean())
        lam = merged["stress_score"].apply(
            lambda s: fallback if pd.isna(s) else _stress_lambda(s))
        lam.index = merged["index"]
        return lam.sort_index(), in_cov, fallback
    except Exception:
        return pd.Series(1.0, index=date_series.index), 0.0, 1.0

print("=" * 55)
print("  STAGE 3 — MULTI-DIMENSIONAL ANOMALY DETECTION")
print("=" * 55)
print(f"  Base thresholds: volume Z>{VOL_Z}  freq Z>{FREQ_Z}  surge Z>{SURGE_Z}  conc>{CONC_PCT:.0%}")

if not os.path.exists(INPUT):
    print(f"[ERROR] {INPUT} not found. Run preprocess.py first.")
    exit(1)

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df):,} transactions.")

# Apply PER-WEEK macro stress multipliers to thresholds
df["macro_lambda"], _in_cov, _fallback = get_weekly_multipliers(df["date"])
_lam_counts = df["macro_lambda"].value_counts().sort_index()
_lam_desc = "  ".join(f"λ={lam:g}:{cnt:,}" for lam, cnt in _lam_counts.items())
print(f"  Macro stress : per-week thresholds — {_lam_desc}")
print(f"                 {_in_cov:.0%} of transactions inside indicator coverage "
      f"(out-of-coverage rows use coverage-mean λ={_fallback:g})")
print(f"  Base thresholds x row λ: volume Z>{VOL_Z}λ  freq Z>{FREQ_Z}λ  surge Z>{SURGE_Z}λ\n")


def robust_z(x):
    """Median/MAD z-score — robust to the very outliers we are trying to detect.

    Standard (mean/std) z-scores are non-robust: a cluster of extreme values
    inflates the std so much that the extremes themselves stop clearing the
    threshold (on the injection benchmark, plain-z volume recall was only 7%).
    MAD (median absolute deviation) barely moves under a small fraction of
    outliers, so genuine spikes keep a large score.  Scaled by 1.4826 so it
    equals the ordinary z-score for normally-distributed data.
    """
    x = np.asarray(x, dtype=float)
    med   = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - med))
    if scale < 1e-9:                       # degenerate: mostly identical values
        s = x.std()
        return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)
    return (x - med) / scale


# ─────────────────────────────────────────────
# DIMENSION 1: Volume Anomaly (robust Z-score)
# Flags transactions where quantity is an extreme outlier
# ─────────────────────────────────────────────
df["z_quantity"] = robust_z(df["quantity"])
df["flag_volume"] = df["z_quantity"].abs() > VOL_Z * df["macro_lambda"]

n = df["flag_volume"].sum()
print(f"[1] Volume Anomaly       : {n} flagged  (|Z-score| > {VOL_Z}·λ_week)")

# ─────────────────────────────────────────────
# DIMENSION 2: Frequency Anomaly
# Flags manufacturer-retailer pairs that transact abnormally often
# ─────────────────────────────────────────────
pair_freq = (
    df.groupby(["manufacturer", "retailer"])
      .size()
      .reset_index(name="pair_freq")
)
df = df.merge(pair_freq, on=["manufacturer", "retailer"], how="left")

if df["pair_freq"].std() > 0:
    df["z_freq"] = robust_z(df["pair_freq"])
else:
    df["z_freq"] = 0.0

df["flag_frequency"] = df["z_freq"] > FREQ_Z * df["macro_lambda"]

n = df["flag_frequency"].sum()
print(f"[2] Frequency Anomaly    : {n} flagged  (pair transaction count Z > {FREQ_Z}·λ_week)")

# ─────────────────────────────────────────────
# DIMENSION 3: Temporal Surge
# Flags when a manufacturer ships unusually large monthly totals
# ─────────────────────────────────────────────
df["year_month"] = df["date"].dt.to_period("M")

monthly = (
    df.groupby(["manufacturer", "year_month"])["quantity"]
      .sum()
      .reset_index(name="monthly_total")
)

def safe_zscore(x):
    if len(x) < 2:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return pd.Series(robust_z(x), index=x.index)

monthly["z_surge"] = monthly.groupby("manufacturer")["monthly_total"].transform(safe_zscore)

df = df.merge(
    monthly[["manufacturer", "year_month", "z_surge"]],
    on=["manufacturer", "year_month"],
    how="left"
)
df["z_surge"] = df["z_surge"].fillna(0)
df["flag_surge"] = df["z_surge"].abs() > SURGE_Z * df["macro_lambda"]

n = df["flag_surge"].sum()
print(f"[3] Temporal Surge       : {n} flagged  (monthly volume Z > {SURGE_Z}·λ_week)")

# ─────────────────────────────────────────────
# DIMENSION 4: Concentration Risk
# Flags retailer-distributor pairs with >90% supply dependency
# ─────────────────────────────────────────────
retailer_total = (
    df.groupby("retailer")["quantity"]
      .sum()
      .reset_index(name="retailer_total")
)

dist_retailer = (
    df.groupby(["distributor", "retailer"])["quantity"]
      .sum()
      .reset_index(name="dr_total")
)

dist_retailer = dist_retailer.merge(retailer_total, on="retailer")
dist_retailer["concentration"] = dist_retailer["dr_total"] / dist_retailer["retailer_total"]

df = df.merge(
    dist_retailer[["distributor", "retailer", "concentration"]],
    on=["distributor", "retailer"],
    how="left"
)
df["concentration"] = df["concentration"].fillna(0)
df["flag_concentration"] = df["concentration"] > CONC_PCT

n = df["flag_concentration"].sum()
print(f"[4] Concentration Risk   : {n} flagged  (single supplier >{CONC_PCT:.0%} of retailer supply)")

# ─────────────────────────────────────────────
# DIMENSION 5: Isolation Forest
# ML-based approach — learns joint distribution of all 4 signal
# features and flags statistical outliers in that combined space.
# Catches anomalies that are subtle in each dimension individually
# but form an outlier pattern when viewed together.
# ─────────────────────────────────────────────
IF_FEATURES = ["z_quantity", "z_freq", "z_surge", "concentration"]
X_if = df[IF_FEATURES].fillna(0).values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_if)

# contamination: ~5% expected anomaly rate based on synthetic dataset design
iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
preds   = iforest.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
scores  = iforest.score_samples(X_scaled) # lower raw score = more anomalous

df["flag_iforest"]   = (preds == -1)
df["iforest_score"]  = -scores            # negate so higher = more anomalous

n = df["flag_iforest"].sum()
print(f"[5] Isolation Forest     : {n} flagged  (contamination=0.05, n_estimators=100)")

# ─────────────────────────────────────────────
# COMPOSITE SCORING (5 signals)
# High-confidence rule: the calibrated logistic ensemble from the injection
# benchmark (Stage 3b) when its calibration file exists — it beat every
# rule-based ensemble on held-out injections (F1 0.39 vs 0.28 for the legacy
# 2+ rule). Falls back to the legacy 2+ count on a fresh run where Stage 3b
# hasn't produced a calibration yet.
# ─────────────────────────────────────────────
zscore_flag_cols  = ["flag_volume", "flag_frequency", "flag_surge", "flag_concentration"]
all_flag_cols     = zscore_flag_cols + ["flag_iforest"]
df["anomaly_score"] = df[all_flag_cols].sum(axis=1)

CALIB = os.path.join(ROOT, "output", "anomaly_ensemble_calibration.json")
_rule = "legacy 2+ signal count"
if os.path.exists(CALIB):
    try:
        import json
        with open(CALIB) as _f:
            _cal = json.load(_f)
        _X = np.column_stack([
            df["z_quantity"].abs().values,
            df["z_freq"].values,
            df["z_surge"].abs().values,
            df["concentration"].values,
            df["iforest_score"].values,
        ])
        _logits = _X @ np.array(_cal["coef"]) + _cal["intercept"]
        df["ensemble_proba"] = 1.0 / (1.0 + np.exp(-_logits))
        df["is_anomaly"] = df["ensemble_proba"] >= _cal["threshold"]
        _rule = (f"calibrated ensemble (thr={_cal['threshold']:.2f}, "
                 f"held-out F1={_cal['held_out_f1']:.3f})")
    except Exception as _e:
        print(f"[WARN] calibration unusable ({_e}) — using legacy 2+ rule")
        df["is_anomaly"] = df["anomaly_score"] >= 2
else:
    df["is_anomaly"] = df["anomaly_score"] >= 2
if "ensemble_proba" not in df.columns:
    df["ensemble_proba"] = np.nan
df["is_suspect"] = (~df["is_anomaly"]) & (df["anomaly_score"] >= 1)

print(f"\n──────────────────────────────────────────────────")
print(f"  RESULTS")
print(f"──────────────────────────────────────────────────")
print(f"  Decision rule                             : {_rule}")
print(f"  High-confidence anomalies                 : {df['is_anomaly'].sum()}")
print(f"  Suspect transactions (signal, not flagged): {df['is_suspect'].sum()}")
print(f"  Clean transactions                        : {((~df['is_anomaly']) & (df['anomaly_score'] == 0)).sum():,}")
print(f"\n  Detection quality (P/R/F1) is measured by the independent")
print(f"  injection benchmark — run: python3 src/anomaly_eval_injection.py")

# Save metrics report
METRICS_OUT = os.path.join(ROOT, "output", "anomaly_metrics.txt")
os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
_am = "\n".join([
    "STAGE 3 — ANOMALY DETECTION METRICS",
    "=" * 50,
    "",
    f"Dataset size         : {len(df):,} transactions",
    f"Macro stress         : per-week λ ({_lam_desc}); "
    f"{_in_cov:.0%} of rows in indicator coverage, fallback λ={_fallback:g}",
    f"Base Z thresholds    : vol={VOL_Z}·λ  freq={FREQ_Z}·λ  surge={SURGE_Z}·λ",
    "",
    "Signal counts:",
    f"  [1] Volume Anomaly       : {df['flag_volume'].sum()}",
    f"  [2] Frequency Anomaly    : {df['flag_frequency'].sum()}",
    f"  [3] Temporal Surge       : {df['flag_surge'].sum()}",
    f"  [4] Concentration Risk   : {df['flag_concentration'].sum()}",
    f"  [5] Isolation Forest     : {df['flag_iforest'].sum()}",
    "",
    f"Decision rule        : {_rule}",
    f"High-confidence anomalies              : {df['is_anomaly'].sum()}",
    f"Suspect transactions (unflagged signal): {df['is_suspect'].sum()}",
    "",
    "Detection quality (Precision/Recall/F1) is evaluated against an",
    "INDEPENDENT ground truth in Stage 3b (anomaly_eval_injection.py):",
    "known synthetic anomalies are injected and the same detectors are",
    "scored against the injection label. See anomaly_injection_metrics.txt.",
    "Flags derived from the detectors are never reused as ground truth.",
])
try:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(_am)
        _tmp = tmp.name
    shutil.copy2(_tmp, METRICS_OUT)
    os.unlink(_tmp)
    print(f"\nMetrics saved → {METRICS_OUT}")
except Exception as _e:
    print(f"\n[WARN] Could not save anomaly metrics: {_e}")

# ─────────────────────────────────────────────
# SAVE ANOMALIES
# ─────────────────────────────────────────────
anomalies = df[df["is_anomaly"]].copy()
anomalies = anomalies.sort_values(["ensemble_proba", "anomaly_score"],
                                  ascending=False, na_position="last")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
anomalies[[
    "date", "manufacturer", "distributor", "retailer",
    "retailer_state", "quantity", "z_quantity",
    "flag_volume", "flag_frequency", "flag_surge", "flag_concentration",
    "flag_iforest", "iforest_score", "anomaly_score", "ensemble_proba"
]].to_csv(OUTPUT, index=False)

print(f"\nTop anomalies:")
print(anomalies[["manufacturer", "retailer", "quantity", "anomaly_score"]].head(10).to_string(index=False))
print(f"\nSaved → {OUTPUT}")
