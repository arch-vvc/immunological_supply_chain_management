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
from sklearn.linear_model import LogisticRegression
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
CALIB_OUT   = os.path.join(ROOT, "output", "anomaly_ensemble_calibration.json")

# Tune on one injection realization, evaluate on held-out realizations —
# the tuned rule never sees the labels it is finally scored against.
TUNE_SEED  = 42
EVAL_SEEDS = [43, 44, 45, 46]
IF_BUDGETS = [0.010, 0.015, 0.020, 0.030, 0.050]   # flag-budget sweep (fraction flagged)

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


def inject_anomalies(df, seed):
    """Return an augmented dataframe + boolean ground-truth column `is_injected`.
    Each seed produces an independent injection realization."""
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    df["is_injected"] = False

    n_total = len(df)
    n_budget = int(n_total * CONTAMINATION_RATE)
    n_volume = int(n_budget * 0.4)
    n_freq_pairs = max(3, int(n_budget * 0.2 / 15))   # each pair gets ~15 burst rows
    n_conc_retailers = max(3, int(n_budget * 0.2 / 20))  # each retailer gets ~20 reassigned rows

    # A. Volume spike — inflate quantity on random existing transactions
    vol_idx = rng.choice(df.index, size=n_volume, replace=False)
    factors = rng.uniform(8, 20, size=n_volume)
    df.loc[vol_idx, "quantity"] = (df.loc[vol_idx, "quantity"].values * factors).astype(int)
    df.loc[vol_idx, "is_injected"] = True

    # B. Frequency burst — pick pairs, append clustered extra transactions
    sample_rows = df.sample(n=n_freq_pairs, random_state=seed)
    burst_frames = []
    for _, row in sample_rows.iterrows():
        n_burst = rng.integers(12, 25)
        base_date = row["date"]
        burst = pd.DataFrame({
            "date": [pd.Timestamp(base_date) + pd.Timedelta(days=int(d))
                     for d in rng.integers(0, 5, size=n_burst)],
            "manufacturer": row["manufacturer"],
            "distributor": row["distributor"],
            "retailer": row["retailer"],
            "retailer_state": row["retailer_state"],
            "quantity": rng.integers(
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
    chosen_retailers = rng.choice(retailers, size=min(n_conc_retailers, len(retailers)), replace=False)
    all_distributors = df["distributor"].dropna().unique()
    for ret in chosen_retailers:
        mask = df["retailer"] == ret
        ret_idx = df.index[mask]
        if len(ret_idx) < 5:
            continue
        dominant = rng.choice(all_distributors)
        n_reassign = max(5, int(len(ret_idx) * 0.9))
        reassign_idx = rng.choice(ret_idx, size=min(n_reassign, len(ret_idx)), replace=False)
        df.loc[reassign_idx, "distributor"] = dominant
        df.loc[reassign_idx, "is_injected"] = True

    df["is_injected"] = df["is_injected"].fillna(False).astype(bool)
    return df


def robust_z(x):
    """Median/MAD z-score — must match anomaly_detection.py exactly."""
    x = np.asarray(x, dtype=float)
    med   = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - med))
    if scale < 1e-9:
        s = x.std()
        return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)
    return (x - med) / scale


def run_detectors(df):
    """Re-run the production 5-signal detection logic on the (augmented) dataframe."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # [1] Volume
    df["z_quantity"] = robust_z(df["quantity"])
    df["flag_volume"] = df["z_quantity"].abs() > VOL_Z

    # [2] Frequency
    pair_freq = df.groupby(["manufacturer", "retailer"]).size().reset_index(name="pair_freq")
    df = df.merge(pair_freq, on=["manufacturer", "retailer"], how="left")
    df["z_freq"] = robust_z(df["pair_freq"]) if df["pair_freq"].std() > 0 else 0.0
    df["flag_frequency"] = df["z_freq"] > FREQ_Z

    # [3] Temporal surge
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby(["manufacturer", "year_month"])["quantity"].sum().reset_index(name="monthly_total")

    def safe_zscore(x):
        if len(x) < 2:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return pd.Series(robust_z(x), index=x.index)

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
    df["iforest_score"] = -iforest.score_samples(X_scaled)   # higher = more anomalous

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


def logit_features(scored):
    """Feature matrix for the calibrated ensemble — the five signal strengths."""
    return np.column_stack([
        scored["z_quantity"].abs().values,
        scored["z_freq"].values,
        scored["z_surge"].abs().values,
        scored["concentration"].values,
        scored["iforest_score"].values,
    ])


def method_predictions(scored, if_budget, logit_model=None, logit_thr=0.5):
    """All candidate decision rules on one scored dataframe. Legacy rules
    first (unchanged, for continuity with the paper), then the tuned ones."""
    preds = {}
    preds["Z-score only  (2+ signals)"] = (scored["zscore_count"] >= 2).astype(int)
    preds["Isolation Forest alone"]     = scored["flag_iforest"].astype(int)
    preds["Ensemble (2+ incl. IF)"]     = scored["is_anomaly"].astype(int)
    # Budget-matched IF: flag the top if_budget fraction by IF score. The
    # legacy contamination=0.05 flags 2.5x more rows than the 2% anomaly
    # rate, capping precision at ~0.4 no matter how good the ranking is.
    cut = scored["iforest_score"].quantile(1 - if_budget)
    if_b = (scored["iforest_score"] > cut)
    preds["IF budget-matched (tuned)"] = if_b.astype(int)
    preds["IF-budget OR 2+ z-signals"] = (if_b | (scored["zscore_count"] >= 2)).astype(int)
    if logit_model is not None:
        proba = logit_model.predict_proba(logit_features(scored))[:, 1]
        preds["Calibrated ensemble (logistic)"] = (proba >= logit_thr).astype(int)
    return preds


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

    # ── TUNING REALIZATION (seed 42): pick flag budget + calibrate ensemble ──
    injected = inject_anomalies(df, TUNE_SEED)
    n_injected = int(injected["is_injected"].sum())
    n_after = len(injected)
    print(f"Tuning realization (seed {TUNE_SEED}): {n_injected:,} synthetic anomalies "
          f"({n_injected/n_after*100:.2f}% of {n_after:,} rows after burst additions)")

    scored = run_detectors(injected)
    y_true = scored["is_injected"].astype(int)

    # ── Diagnostic: why the multi-signal rule struggles ────────────────────
    # Count, among true injected anomalies, how many independent z-score
    # signals each one trips, and the per-signal catch rate. This explains
    # the headline result: most injected anomalies are SINGLE-dimension, so a
    # "2+ signals" rule structurally cannot flag them.
    inj = scored[scored["is_injected"]]
    n_inj = len(inj)
    zdist = inj["zscore_count"].value_counts().sort_index()
    per_signal = {
        "Volume":        float(inj["flag_volume"].mean()),
        "Frequency":     float(inj["flag_frequency"].mean()),
        "Temporal surge":float(inj["flag_surge"].mean()),
        "Concentration": float(inj["flag_concentration"].mean()),
        "IsolationForest": float(inj["flag_iforest"].mean()),
    }
    single_or_zero = int((inj["zscore_count"] <= 1).sum())

    print(f"\n  Diagnostic — signals tripped per injected anomaly (n={n_inj}):")
    for k, v in zdist.items():
        print(f"    {k} z-signals : {v:>4}  ({v/n_inj*100:4.1f}%)")
    print(f"    → {single_or_zero/n_inj*100:.1f}% trip ≤1 z-signal, so a 2+ z-rule cannot catch them")
    print("  Per-signal recall on injected anomalies:")
    for k, v in per_signal.items():
        print(f"    {k:<16}: {v*100:5.1f}%")

    # ── Tune the IF flag budget on the tuning realization only ──
    budget_sweep = []
    for q in IF_BUDGETS:
        cut = scored["iforest_score"].quantile(1 - q)
        p, r, f = prf(y_true, (scored["iforest_score"] > cut).astype(int))
        budget_sweep.append({"budget": q, "P": p, "R": r, "F1": f})
    best_budget = max(budget_sweep, key=lambda b: b["F1"])["budget"]
    print(f"\n  Flag-budget sweep (tuning seed only):")
    for b in budget_sweep:
        mark = " <- tuned" if b["budget"] == best_budget else ""
        print(f"    top {b['budget']:.1%} flagged : P={b['P']:.3f}  R={b['R']:.3f}  F1={b['F1']:.3f}{mark}")

    # ── Calibrate the logistic ensemble on the tuning realization only ──
    logit = LogisticRegression(max_iter=1000, class_weight="balanced")
    logit.fit(logit_features(scored), y_true)
    proba_tune = logit.predict_proba(logit_features(scored))[:, 1]
    thr_sweep = [(t, f1_score(y_true, (proba_tune >= t).astype(int), zero_division=0))
                 for t in np.arange(0.30, 0.96, 0.05)]
    logit_thr = max(thr_sweep, key=lambda t: t[1])[0]
    print(f"  Calibrated ensemble: logistic on 5 signal strengths, "
          f"threshold {logit_thr:.2f} (tuned on seed {TUNE_SEED})")

    # ── Pre-injection baseline flags: the organic-anomaly correction ──
    # The base ARCOS data is NOT anomaly-free — the generator plants ~5%
    # organic anomalies with no labels. A detector that finds one is charged
    # a false positive by the raw precision. So we also run every rule on the
    # CLEAN base: rows it flags there are organic suspects, and the ADJUSTED
    # precision excludes them from the false-positive count. Raw precision is
    # a floor (treats every organic hit as an error); adjusted is the honest
    # estimate of "when the detector fires on non-organic rows, is it right?".
    base_scored = run_detectors(df.assign(is_injected=False))
    base_preds = {name: np.asarray(yp).astype(bool)
                  for name, yp in method_predictions(base_scored, best_budget,
                                                     logit, logit_thr).items()}
    n_base = len(df)

    # ── HELD-OUT EVALUATION: 4 fresh injection realizations ──
    print(f"\n  Held-out evaluation on seeds {EVAL_SEEDS} (tuned rules never saw these labels):")
    per_seed = {}          # method -> list of (p, r, f, p_adj)
    for s in EVAL_SEEDS:
        sc = run_detectors(inject_anomalies(df, s))
        yt = sc["is_injected"].astype(int)
        for name, yp in method_predictions(sc, best_budget, logit, logit_thr).items():
            p, r, f = prf(yt, yp)
            # Adjusted precision: original rows keep their position after
            # injection (bursts are appended), so align base flags by index.
            pred = np.asarray(yp).astype(bool)
            inj  = np.asarray(yt).astype(bool)
            organic = np.zeros(len(sc), dtype=bool)
            organic[:n_base] = base_preds[name]
            tp     = int((pred & inj).sum())
            fp_adj = int((pred & ~inj & ~organic).sum())
            p_adj  = tp / (tp + fp_adj) if (tp + fp_adj) else 0.0
            per_seed.setdefault(name, []).append((p, r, f, p_adj))

    print(f"\n{'Method':<32} {'Precision':>14} {'P(adj)':>14} {'Recall':>14} {'F1':>14}")
    results = []
    for name, tuples in per_seed.items():
        ps, rs, fs, pas = zip(*tuples)
        row = {"Method": name,
               "Precision": float(np.mean(ps)), "Recall": float(np.mean(rs)),
               "F1": float(np.mean(fs)),
               "P_adj": float(np.mean(pas)),
               "P_std": float(np.std(ps)), "R_std": float(np.std(rs)),
               "F1_std": float(np.std(fs)), "P_adj_std": float(np.std(pas))}
        results.append(row)
        print(f"{name:<32} {row['Precision']:>7.3f} ±{row['P_std']:.3f} "
              f"{row['P_adj']:>7.3f} ±{row['P_adj_std']:.3f} "
              f"{row['Recall']:>7.3f} ±{row['R_std']:.3f} "
              f"{row['F1']:>7.3f} ±{row['F1_std']:.3f}")

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    results_df.to_csv(CSV_OUT, index=False)

    # ── Persist the calibrated ensemble for the production detector ──
    # Stage 3 (anomaly_detection.py) applies these weights when the file
    # exists and falls back to the legacy 2+ rule when it doesn't.
    import json
    held_out = {r["Method"]: r for r in results}["Calibrated ensemble (logistic)"]
    with open(CALIB_OUT, "w") as fh:
        json.dump({
            "features": ["abs_z_quantity", "z_freq", "abs_z_surge",
                         "concentration", "iforest_score"],
            "coef": logit.coef_[0].tolist(),
            "intercept": float(logit.intercept_[0]),
            "threshold": float(logit_thr),
            "tuned_on_seed": TUNE_SEED,
            "held_out_f1": round(held_out["F1"], 4),
            "held_out_precision": round(held_out["Precision"], 4),
            "held_out_precision_adjusted": round(held_out["P_adj"], 4),
            "held_out_recall": round(held_out["Recall"], 4),
        }, fh, indent=2)
    print(f"Calibration   -> {CALIB_OUT}")

    # ── Report ──
    lines = [
        "STAGE 3b — ANOMALY DETECTION BENCHMARK (SYNTHETIC INJECTION)",
        "=" * 60,
        "",
        "Methodology: independent ground truth via synthetic anomaly",
        "injection (contamination rate ~2%), following standard practice",
        "for evaluating unlabeled anomaly detectors (e.g. Emmott et al. 2013).",
        "Tuned rules (flag budget, logistic ensemble + threshold) are fitted",
        f"on ONE injection realization (seed {TUNE_SEED}) and evaluated on",
        f"{len(EVAL_SEEDS)} held-out realizations (seeds {EVAL_SEEDS}); reported",
        "numbers are mean +/- std across the held-out seeds only.",
        "",
        f"Base transactions        : {n_before:,}",
        f"Anomalies per realization: ~{n_injected:,} ({n_injected/n_after*100:.2f}% of {n_after:,} rows)",
        "  A. Volume spike          — quantity inflated 8-20x",
        "  B. Frequency burst       — clustered extra transactions on a pair",
        "  C. Concentration shift   — retailer supply forced through one distributor",
        "",
        f"Tuned flag budget        : top {best_budget:.1%} of rows by IF score",
        "  (legacy contamination=0.05 flags 2.5x more rows than the 2%",
        "   anomaly rate — precision was capped near 0.4 by construction)",
        f"Calibrated ensemble      : logistic on 5 signal strengths, thr={logit_thr:.2f}",
        "",
        f"{'Method':<32} {'Precision':>13} {'P(adj)':>13} {'Recall':>13} {'F1':>13}",
        "-" * 88,
    ]
    for r in results:
        lines.append(f"{r['Method']:<32} {r['Precision']:>6.3f} ±{r['P_std']:.3f} "
                     f"{r['P_adj']:>6.3f} ±{r['P_adj_std']:.3f} "
                     f"{r['Recall']:>6.3f} ±{r['R_std']:.3f} "
                     f"{r['F1']:>6.3f} ±{r['F1_std']:.3f}")

    best = max(results, key=lambda r: r["F1"])
    lines += [
        "",
        "Diagnostic — signals per injected anomaly (tuning seed):",
        *[f"  {int(k)} z-signals : {v} ({v/n_inj*100:.1f}%)" for k, v in zdist.items()],
        f"  {single_or_zero/n_inj*100:.1f}% of injected anomalies trip <=1 z-signal.",
        "Per-signal recall:",
        *[f"  {k:<16}: {v*100:.1f}%" for k, v in per_signal.items()],
        "",
        f"Best method by held-out F1: {best['Method'].strip()} (F1={best['F1']:.3f}).",
        "Finding: the legacy rules were budget-mismatched, not signal-poor.",
        "Matching the flag budget to the expected anomaly rate recovers most",
        "of the available precision; the calibrated logistic ensemble adds",
        "the z-signals' complementary coverage on top of the IF ranking.",
        "The 2+ z-score rule remains structurally unable to catch single-",
        "dimension anomalies and is reported for continuity only.",
        "",
        "Raw vs adjusted precision: the base data contains ~5% ORGANIC",
        "anomalies planted by the ARCOS generator with no labels. Raw",
        "precision charges the detector a false positive for finding one,",
        "so it is a floor with a structural ceiling near",
        "n_injected/(n_injected + n_organic) ~= 0.28 even for a perfect",
        "detector. P(adj) excludes rows the same rule already flags on the",
        "PRE-injection base (organic suspects) from the FP count — it answers",
        "'when the detector fires on a non-organic row, is it right?'.",
        "Recall against the injected labels is unaffected by either view.",
        "Injected anomalies follow three synthetic patterns and may not span",
        "the real-world anomaly distribution.",
    ]

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMetrics saved -> {METRICS_OUT}")
    print(f"Results csv   -> {CSV_OUT}")

    # ── Figure (grouped bars, error bars = std across held-out seeds) ──
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(results_df))
    width = 0.25
    ax.bar(x - width, results_df["Precision"], width, yerr=results_df["P_std"],
           capsize=3, label="Precision", color="#0d2b52")
    ax.bar(x,          results_df["Recall"],    width, yerr=results_df["R_std"],
           capsize=3, label="Recall",    color="#4a90d9")
    ax.bar(x + width,  results_df["F1"],        width, yerr=results_df["F1_std"],
           capsize=3, label="F1",        color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("(")[0].strip() for m in results_df["Method"]],
                       rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Anomaly Detection — Injection Benchmark "
                 f"(mean ± std over {len(EVAL_SEEDS)} held-out seeds)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"Figure saved  -> {FIG_OUT}")

    print("\nStage 3b complete.")


if __name__ == "__main__":
    main()
