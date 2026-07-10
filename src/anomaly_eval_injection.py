"""
Stage 3b — Anomaly Detection Benchmark (Synthetic Injection)
==============================================================
Replaces the self-referential "pseudo-ground-truth" evaluation in
anomaly_detection.py (which scored z-score-derived signals against a
stricter threshold of themselves — precision was meaningless because
the label wasn't independent of the detectors being tested).

Methodology (standard practice for unlabeled anomaly detection, e.g.
Emmott et al. 2013 "Systematic Construction of Anomaly Detection
Benchmarks"): inject a known quantity of synthetic anomalies into
otherwise-clean data, run the SAME detectors used in production, and
score against the injection label — which is independent of any
detector output.

Three injection types, matching the three real-world anomaly patterns
the pipeline is designed to catch:
  A. Volume spike        — quantity inflated 8-20x on random transactions
  B. Frequency burst     — a manufacturer-retailer pair gets a sudden
                            cluster of extra transactions in a short window
  C. Concentration shift — a retailer's supply is forcibly funneled
                            through a single distributor

Outputs:
    output/anomaly_injection_metrics.txt
    output/anomaly_injection_results.csv     (per-method P/R/F1, machine-readable)
    output/figures/fig_anomaly_injection_eval.png
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT  = os.path.join(ROOT, "data", "processed", "clean_chain.csv")
CONFIG = os.path.join(ROOT, "config.yaml")
METRICS_OUT = os.path.join(ROOT, "output", "anomaly_injection_metrics.txt")
CSV_OUT     = os.path.join(ROOT, "output", "anomaly_injection_results.csv")
FIG_OUT     = os.path.join(ROOT, "output", "figures", "fig_anomaly_injection_eval.png")

RNG = np.random.default_rng(42)

# ── Load the same production thresholds ────────────────────────
_thresholds = {}
if HAS_YAML and os.path.exists(CONFIG):
    with open(CONFIG) as f:
        _cfg = yaml.safe_load(f)
    _thresholds = _cfg.get("thresholds", {})

VOL_Z    = _thresholds.get("volume_zscore",         3.0)
FREQ_Z   = _thresholds.get("frequency_zscore",      2.5)
SURGE_Z  = _thresholds.get("temporal_surge_zscore", 2.5)
CONC_PCT = _thresholds.get("concentration_pct",     0.90)

CONTAMINATION_RATE = 0.02  # ~2% of rows become synthetic anomalies


def inject_anomalies(df):
    """Return an augmented dataframe + boolean ground-truth column `is_injected`."""
    df = df.copy().reset_index(drop=True)
    df["is_injected"] = False

    n_total = len(df)
    n_budget = int(n_total * CONTAMINATION_RATE)
    n_volume = int(n_budget * 0.4)
    n_freq_pairs = max(3, int(n_budget * 0.2 / 15))   # each pair gets ~15 burst rows
    n_conc_retailers = max(3, int(n_budget * 0.2 / 20))  # each retailer gets ~20 reassigned rows

    injected_rows = []

    # A. Volume spike — inflate quantity on random existing transactions
    vol_idx = RNG.choice(df.index, size=n_volume, replace=False)
    factors = RNG.uniform(8, 20, size=n_volume)
    df.loc[vol_idx, "quantity"] = (df.loc[vol_idx, "quantity"].values * factors).astype(int)
    df.loc[vol_idx, "is_injected"] = True

    # B. Frequency burst — pick pairs, append clustered extra transactions
    sample_rows = df.sample(n=n_freq_pairs, random_state=42)
    burst_frames = []
    for _, row in sample_rows.iterrows():
        n_burst = RNG.integers(12, 25)
        base_date = row["date"]
        burst = pd.DataFrame({
            "date": [pd.Timestamp(base_date) + pd.Timedelta(days=int(d))
                     for d in RNG.integers(0, 5, size=n_burst)],
            "manufacturer": row["manufacturer"],
            "distributor": row["distributor"],
            "retailer": row["retailer"],
            "retailer_state": row["retailer_state"],
            "quantity": RNG.integers(
                max(1, int(row["quantity"] * 0.5)), int(row["quantity"] * 1.5) + 2, size=n_burst
            ),
            "is_injected": True,
        })
        burst_frames.append(burst)
    if burst_frames:
        burst_df = pd.concat(burst_frames, ignore_index=True)
        for col in df.columns:
            if col not in burst_df.columns:
                burst_df[col] = np.nan
        df = pd.concat([df, burst_df[df.columns]], ignore_index=True)

    # C. Concentration shift — force one dominant distributor per chosen retailer
    retailers = df["retailer"].dropna().unique()
    chosen_retailers = RNG.choice(retailers, size=min(n_conc_retailers, len(retailers)), replace=False)
    all_distributors = df["distributor"].dropna().unique()
    for ret in chosen_retailers:
        mask = df["retailer"] == ret
        ret_idx = df.index[mask]
        if len(ret_idx) < 5:
            continue
        dominant = RNG.choice(all_distributors)
        n_reassign = max(5, int(len(ret_idx) * 0.9))
        reassign_idx = RNG.choice(ret_idx, size=min(n_reassign, len(ret_idx)), replace=False)
        df.loc[reassign_idx, "distributor"] = dominant
        df.loc[reassign_idx, "is_injected"] = True

    df["is_injected"] = df["is_injected"].fillna(False).astype(bool)
    return df


def run_detectors(df):
    """Re-run the production 5-signal detection logic on the (augmented) dataframe."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # [1] Volume
    df["z_quantity"] = zscore(df["quantity"])
    df["flag_volume"] = df["z_quantity"].abs() > VOL_Z

    # [2] Frequency
    pair_freq = df.groupby(["manufacturer", "retailer"]).size().reset_index(name="pair_freq")
    df = df.merge(pair_freq, on=["manufacturer", "retailer"], how="left")
    df["z_freq"] = zscore(df["pair_freq"]) if df["pair_freq"].std() > 0 else 0.0
    df["flag_frequency"] = df["z_freq"] > FREQ_Z

    # [3] Temporal surge
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby(["manufacturer", "year_month"])["quantity"].sum().reset_index(name="monthly_total")

    def safe_zscore(x):
        if len(x) < 2 or x.std() == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return pd.Series(zscore(x), index=x.index)

    monthly["z_surge"] = monthly.groupby("manufacturer")["monthly_total"].transform(safe_zscore)
    df = df.merge(monthly[["manufacturer", "year_month", "z_surge"]], on=["manufacturer", "year_month"], how="left")
    df["z_surge"] = df["z_surge"].fillna(0)
    df["flag_surge"] = df["z_surge"].abs() > SURGE_Z

    # [4] Concentration
    retailer_total = df.groupby("retailer")["quantity"].sum().reset_index(name="retailer_total")
    dist_retailer = df.groupby(["distributor", "retailer"])["quantity"].sum().reset_index(name="dr_total")
    dist_retailer = dist_retailer.merge(retailer_total, on="retailer")
    dist_retailer["concentration"] = dist_retailer["dr_total"] / dist_retailer["retailer_total"]
    df = df.merge(dist_retailer[["distributor", "retailer", "concentration"]], on=["distributor", "retailer"], how="left")
    df["concentration"] = df["concentration"].fillna(0)
    df["flag_concentration"] = df["concentration"] > CONC_PCT

    # [5] Isolation Forest
    IF_FEATURES = ["z_quantity", "z_freq", "z_surge", "concentration"]
    X_if = df[IF_FEATURES].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X_if)
    iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    preds = iforest.fit_predict(X_scaled)
    df["flag_iforest"] = (preds == -1)

    zscore_cols = ["flag_volume", "flag_frequency", "flag_surge", "flag_concentration"]
    df["zscore_count"] = df[zscore_cols].sum(axis=1)
    df["anomaly_score"] = df[zscore_cols + ["flag_iforest"]].sum(axis=1)
    df["is_anomaly"] = df["anomaly_score"] >= 2

    return df


def prf(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f


def main():
    print("=" * 55)
    print("  STAGE 3b — ANOMALY DETECTION BENCHMARK (INJECTION)")
    print("=" * 55)

    if not os.path.exists(INPUT):
        print(f"[ERROR] {INPUT} not found. Run preprocess.py first.")
        raise SystemExit(1)

    df = pd.read_csv(INPUT)
    n_before = len(df)
    print(f"Loaded {n_before:,} clean transactions.")

    injected = inject_anomalies(df)
    n_injected = int(injected["is_injected"].sum())
    n_after = len(injected)
    print(f"Injected {n_injected:,} synthetic anomalies "
          f"({n_injected/n_after*100:.2f}% of {n_after:,} rows after burst additions)")

    scored = run_detectors(injected)
    y_true = scored["is_injected"].astype(int)

    methods = {
        "Z-score only  (2+ signals)": (scored["zscore_count"] >= 2).astype(int),
        "Isolation Forest alone":     scored["flag_iforest"].astype(int),
        "Ensemble (2+ incl. IF)":     scored["is_anomaly"].astype(int),
    }

    print(f"\n{'Method':<30} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    results = []
    for name, y_pred in methods.items():
        p, r, f = prf(y_true, y_pred)
        print(f"{name:<30} {p:>10.3f} {r:>8.3f} {f:>8.3f}")
        results.append({"Method": name, "Precision": p, "Recall": r, "F1": f})

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    results_df.to_csv(CSV_OUT, index=False)

    # ── Report ──
    lines = [
        "STAGE 3b — ANOMALY DETECTION BENCHMARK (SYNTHETIC INJECTION)",
        "=" * 60,
        "",
        "Methodology: independent ground truth via synthetic anomaly",
        "injection (contamination rate ~2%), following standard practice",
        "for evaluating unlabeled anomaly detectors (e.g. Emmott et al. 2013).",
        "This replaces the earlier self-referential pseudo-GT evaluation,",
        "which scored z-score-derived signals against a stricter threshold",
        "of themselves and produced a circular, near-zero-precision result.",
        "",
        f"Base transactions        : {n_before:,}",
        f"Synthetic anomalies added: {n_injected:,} ({n_injected/n_after*100:.2f}% of {n_after:,} rows)",
        "  A. Volume spike          — quantity inflated 8-20x",
        "  B. Frequency burst       — clustered extra transactions on a pair",
        "  C. Concentration shift   — retailer supply forced through one distributor",
        "",
        f"{'Method':<30} {'Precision':>10} {'Recall':>8} {'F1':>8}",
        "-" * 58,
    ]
    for r in results:
        lines.append(f"{r['Method']:<30} {r['Precision']:>10.3f} {r['Recall']:>8.3f} {r['F1']:>8.3f}")
    lines.append("")

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMetrics saved -> {METRICS_OUT}")
    print(f"Results csv   -> {CSV_OUT}")

    # ── Figure (mirrors the grouped bar chart style already used) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results_df))
    width = 0.25
    ax.bar(x - width, results_df["Precision"], width, label="Precision", color="#0d2b52")
    ax.bar(x,          results_df["Recall"],    width, label="Recall",    color="#4a90d9")
    ax.bar(x + width,  results_df["F1"],        width, label="F1",        color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("(")[0].strip() for m in results_df["Method"]])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Anomaly Detection — Synthetic Injection Benchmark")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"Figure saved  -> {FIG_OUT}")

    print("\nStage 3b complete.")


if __name__ == "__main__":
    main()
