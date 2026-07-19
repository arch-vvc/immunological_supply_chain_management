"""
Stage 8 — Recovery Time Predictor  &  Outcome-Ranked Strategy Recommender
========================================================================
Maps to: Adaptive Immunity — learning from past disruptions to predict
         how long future ones take to recover, and which response gets
         there fastest.

DESIGN NOTE — why this is a regressor, not a classifier
-------------------------------------------------------
An earlier version trained a classifier to predict which response_type a
human historically chose. That target turned out to be near-unlearnable:
mutual information with every observable feature is ~0 except
has_backup_supplier (0.23), and within the no-backup subset the four
fallback strategies are chosen almost uniformly at random with respect to
any recorded feature. No model can recover a pattern the data doesn't hold,
so per-class precision floored out (Customer Delay P=0.32/R=0.10, etc.).

The *outcome* (full_recovery_days), by contrast, carries real signal AND
varies by strategy at fixed severity (e.g. at severity 3, no-backup:
Alternative Supplier ~98d vs Customer Delay ~120d). So we reframe:

    Instead of imitating the historical CHOICE, predict the OUTCOME under
    each candidate strategy and rank them. The recommendation is the
    strategy with the shortest predicted recovery.

`response_type` becomes an INPUT feature to the regressor. At decision time
we hold the disruption's features fixed and sweep the strategy, reading off
predicted recovery for each. When no backup supplier exists, "Alternative
Supplier" is removed from the candidate set — you can't use a backup you
don't have.

IMPORTANT (honest framing for the report): these are OBSERVATIONAL
associations, not proven causal effects. "Cases like this historically
recovered fastest under strategy X" — not "doing X causes faster recovery."
The recommender is decision *support*, not automated command. Individual
recovery predictions carry ~MAE-wide error; the strategy *ranking* is driven
by systematic mean differences across thousands of samples, so it is stable
even where a single prediction is noisy.

Outputs:
    models/recovery_regressor.pkl     — trained regression model (8 features)
    output/recovery_metrics.txt       — MAE / R2 vs baselines + strategy-effect table
    output/recovery_predictions.csv   — per-anomaly recovery under each strategy + pick
    output/figures/fig6_recovery.png  — feature importance + per-strategy recovery
"""

import os
import sys
import pickle
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT, "data", "supplementary", "disruption_processed.csv")
ANOMALY_IN  = os.path.join(ROOT, "output",  "anomalies.csv")
REG_OUT     = os.path.join(ROOT, "models",  "recovery_regressor.pkl")
METRICS_OUT = os.path.join(ROOT, "output",  "recovery_metrics.txt")
PRED_OUT    = os.path.join(ROOT, "output",  "recovery_predictions.csv")
FIG_OUT     = os.path.join(ROOT, "output",  "figures", "fig6_recovery.png")

os.makedirs(os.path.join(ROOT, "models"),           exist_ok=True)
os.makedirs(os.path.join(ROOT, "output", "figures"), exist_ok=True)

print("=" * 55)
print("  STAGE 8 — RECOVERY PREDICTOR + STRATEGY RANKER")
print("=" * 55)

# ── Check dependencies ────────────────────────────────────────
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from xgboost import XGBRegressor
except ImportError:
    print("[ERROR] scikit-learn / xgboost not found. Install with:")
    print("  pip3 install scikit-learn xgboost")
    sys.exit(1)

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] disruption_processed.csv not found at:\n  {DATA_PATH}")
    sys.exit(1)

# ── Load data ─────────────────────────────────────────────────
print(f"  Loading disruption data...")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

# ── Feature engineering ───────────────────────────────────────
# response_type_enc is the TREATMENT variable (the strategy chosen); it is a
# legitimate input to a recovery-time model, not leakage — the outcome we
# predict (full_recovery_days) is realised after the strategy is applied.
df["has_backup_supplier"] = df["has_backup_supplier"].map(
    {True: 1, False: 0, "True": 1, "False": 0}
).fillna(0).astype(int)

df["sev_x_backup"]    = df["disruption_severity"]   * df["has_backup_supplier"]
df["impact_x_backup"] = df["production_impact_pct"] * df["has_backup_supplier"]

# ORDER MATTERS — every X built downstream must use exactly this column order.
FEATURE_COLS = [
    "disruption_type_enc",
    "supplier_size_enc",
    "disruption_severity",
    "production_impact_pct",
    "has_backup_supplier",
    "sev_x_backup",
    "impact_x_backup",
    "response_type_enc",     # ← treatment: swept at decision time to rank strategies
]
FEATURE_LABELS = [
    "Disruption Type", "Supplier Size", "Severity", "Production Impact %",
    "Has Backup Supplier", "Severity x Backup", "Impact % x Backup",
    "Response Strategy",
]
RESP_IDX   = FEATURE_COLS.index("response_type_enc")
TARGET_REG = "full_recovery_days"

needed = FEATURE_COLS + [TARGET_REG]
df = df.dropna(subset=needed)
print(f"  Clean rows for training: {len(df):,}")

X     = df[FEATURE_COLS].values.astype(float)
y_reg = df[TARGET_REG].values.astype(float)

# Strategy label maps
response_map = dict(zip(df["response_type_enc"].astype(int), df["response_type"]))
name_to_enc  = {v: k for k, v in response_map.items()}
strategies   = sorted(response_map.keys())
ALT_ENC      = name_to_enc.get("Alternative Supplier")   # excluded when no backup

# ── Train / test split ────────────────────────────────────────
X_train, X_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

# ── Train regressor ───────────────────────────────────────────
print("\n  Training XGBoost Regressor (recovery days, strategy-aware)...")
regressor = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
regressor.fit(X_train, yr_train)

# ── Evaluate against the naive floor ──────────────────────────
yr_pred  = regressor.predict(X_test)
mae      = mean_absolute_error(yr_test, yr_pred)
r2       = r2_score(yr_test, yr_pred)
base_mae = mean_absolute_error(yr_test, np.full(len(yr_test), yr_train.mean()))

print(f"\n  ── Regression (Recovery Days) ──")
print(f"  MAE : {mae:.1f} days   (predict-the-mean baseline: {base_mae:.1f} days)")
print(f"  R²  : {r2:.3f}")

# ── Strategy-effect table: does the model actually distinguish strategies? ──
# Counterfactual sweep on the held-out test set: hold every disruption's
# features fixed, override the strategy, average the predicted recovery.
# Compared against the empirical mean recovery per strategy in the data.
print(f"\n  ── Predicted recovery by strategy (test-set counterfactual) ──")
print(f"  {'Strategy':<22}{'model pred':>11}{'empirical':>11}")
strat_pred_mean, strat_emp_mean = {}, {}
for s in strategies:
    Xc = X_test.copy()
    Xc[:, RESP_IDX] = s
    strat_pred_mean[s] = float(regressor.predict(Xc).mean())
    strat_emp_mean[s]  = float(df.loc[df["response_type_enc"] == s, TARGET_REG].mean())
    print(f"  {response_map[s]:<22}{strat_pred_mean[s]:>9.1f}d{strat_emp_mean[s]:>10.1f}d")
spread = max(strat_pred_mean.values()) - min(strat_pred_mean.values())
print(f"  → model spread across strategies: {spread:.1f} days "
      f"(vs regressor MAE {mae:.1f}d — ranking is a mean effect, not a per-case guarantee)")

# ── Save model ────────────────────────────────────────────────
with open(REG_OUT, "wb") as f:
    pickle.dump({"model": regressor, "feature_cols": FEATURE_COLS,
                 "response_map": response_map}, f)
print(f"\n  Model saved → {REG_OUT}")

# ── Save metrics ──────────────────────────────────────────────
_metrics = "\n".join([
    "RECOVERY PREDICTOR + STRATEGY RANKER — METRICS",
    "=" * 50, "",
    f"Training samples : {len(X_train):,}",
    f"Test samples     : {len(X_test):,}",
    "",
    "── Regression (full_recovery_days) ──",
    f"MAE  : {mae:.2f} days   (predict-the-mean baseline: {base_mae:.2f} days)",
    f"R²   : {r2:.4f}",
    "",
    "── Predicted recovery by strategy (test-set counterfactual sweep) ──",
    f"{'Strategy':<22}{'model pred':>12}{'empirical':>12}",
    *[f"{response_map[s]:<22}{strat_pred_mean[s]:>10.1f}d{strat_emp_mean[s]:>11.1f}d"
      for s in strategies],
    f"model spread across strategies: {spread:.1f} days",
    "",
    "The classifier that predicted which strategy a human chose was REMOVED:",
    "its target was near-unlearnable (MI ~0 for all features except",
    "has_backup_supplier; the 4 no-backup strategies are ~uniformly assigned",
    "w.r.t. every recorded feature). We instead predict recovery days under",
    "each candidate strategy and recommend the fastest.",
    "",
    "CAVEAT: these are observational associations, not controlled causal",
    "effects. Read as 'cases like this historically recovered fastest under X'.",
    "The per-case recovery number carries ~MAE-wide error; the strategy",
    "ranking is a stable mean effect over thousands of samples.",
])
try:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(_metrics)
        _tmp_path = tmp.name
    shutil.copy2(_tmp_path, METRICS_OUT)
    os.unlink(_tmp_path)
    print(f"  Metrics saved → {METRICS_OUT}")
except Exception as _e:
    print(f"  [WARN] Could not save metrics file: {_e}")

# ── Apply to current anomalies: rank strategies by predicted recovery ──
# Per-anomaly features come from its own detection signals (see mapping).
# disruption_type is unobservable from transaction data, so predictions are
# marginalised over the historical type distribution.
if os.path.exists(ANOMALY_IN):
    adf = pd.read_csv(ANOMALY_IN)
    n_anom = len(adf)
    print(f"\n  Ranking strategies for {n_anom} detected anomalies...")

    severity   = adf["anomaly_score"].clip(1, 5).astype(float).values
    impact     = (adf["z_quantity"].abs() / 10.0 * 100).clip(0, 100).values
    has_backup = (~adf["flag_concentration"].astype(bool)).astype(float).values

    # Supplier size proxy: distributor out-volume tercile → Large/Medium/Small enc
    enc_of = {str(s).lower(): float(e)
              for s, e in zip(df["supplier_size"], df["supplier_size_enc"])}
    size_default = float(df["supplier_size_enc"].mode()[0])
    sizes = np.full(n_anom, size_default)
    GRAPH_RISK = os.path.join(ROOT, "output", "graph_risk_scores.csv")
    if os.path.exists(GRAPH_RISK) and enc_of:
        gdf  = pd.read_csv(GRAPH_RISK)
        dvol = gdf[gdf["type"] == "distributor"].set_index("entity")["out_volume"]
        if len(dvol) >= 3:
            t1, t2 = dvol.quantile(1 / 3), dvol.quantile(2 / 3)
            vols = adf["distributor"].map(dvol)
            sizes = np.where(vols >= t2, enc_of.get("large", size_default),
                    np.where(vols >= t1, enc_of.get("medium", size_default),
                                         enc_of.get("small", size_default)))
            sizes = np.where(vols.isna(), size_default, sizes)

    type_dist = df["disruption_type_enc"].value_counts(normalize=True)

    # days_matrix[i, j] = predicted recovery for anomaly i under strategy j,
    # marginalised over disruption type.
    days_matrix = np.zeros((n_anom, len(strategies)))
    for j, s in enumerate(strategies):
        acc = np.zeros(n_anom)
        for t_enc, w in type_dist.items():
            X_ts = np.column_stack([
                np.full(n_anom, float(t_enc)),   # disruption_type_enc
                sizes,                            # supplier_size_enc
                severity,                         # disruption_severity
                impact,                           # production_impact_pct
                has_backup,                       # has_backup_supplier
                severity * has_backup,            # sev_x_backup
                impact * has_backup,              # impact_x_backup
                np.full(n_anom, float(s)),        # response_type_enc
            ])
            acc += w * regressor.predict(X_ts)
        days_matrix[:, j] = acc

    # Backup-aware candidate set: without a backup, Alternative Supplier is
    # not an available option, so it can never be recommended there.
    if ALT_ENC is not None and ALT_ENC in strategies:
        alt_j = strategies.index(ALT_ENC)
        days_matrix[has_backup == 0, alt_j] = np.inf

    best_j          = days_matrix.argmin(axis=1)
    recommended_enc = np.array(strategies)[best_j]
    predicted_days  = days_matrix[np.arange(n_anom), best_j]

    # Per-strategy transparency columns (inf → NaN so unavailable options are blank)
    vis = np.where(np.isinf(days_matrix), np.nan, days_matrix).round(1)
    for j, s in enumerate(strategies):
        adf[f"days_{response_map[s].replace(' ', '_')}"] = vis[:, j]

    adf["predicted_recovery_days"]     = predicted_days.round(1)
    adf["recommended_strategy"]        = [response_map[int(e)] for e in recommended_enc]
    adf["predicted_response_strategy"] = adf["recommended_strategy"]   # back-compat
    adf.to_csv(PRED_OUT, index=False)

    print(f"  Predictions saved → {PRED_OUT}")
    print(f"  Recommended recovery — min {predicted_days.min():.1f}  "
          f"mean {predicted_days.mean():.1f}  max {predicted_days.max():.1f} days")
    print(f"  Recommended-strategy mix: "
          f"{adf['recommended_strategy'].value_counts().to_dict()}")
    n_nobackup = int((has_backup == 0).sum())
    print(f"  ({n_nobackup} anomalies had no backup supplier → "
          f"Alternative Supplier excluded from their ranking)")

# ── Visualisation ─────────────────────────────────────────────
print("\n  Generating Fig 6: Feature Importance + Recovery by Strategy...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0f0f1a")
for ax in [ax1, ax2]:
    ax.set_facecolor("#0f0f1a")

# Left: feature importances
importances = regressor.feature_importances_
sorted_idx  = np.argsort(importances)
colors      = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(FEATURE_LABELS)))
ax1.barh(range(len(FEATURE_LABELS)), importances[sorted_idx],
         color=colors, edgecolor="white", linewidth=0.3)
ax1.set_yticks(range(len(FEATURE_LABELS)))
ax1.set_yticklabels([FEATURE_LABELS[i] for i in sorted_idx], color="white", fontsize=9)
ax1.set_xlabel("Feature Importance", color="white")
ax1.set_title("What Drives Recovery Time?\n(XGBoost gradient-boosted trees)",
              color="white", fontsize=10, pad=10)
ax1.tick_params(colors="white")
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax1.spines[spine].set_color("#4a4a6a")

# Right: predicted vs empirical recovery by strategy (the Option-A story)
order   = sorted(strategies, key=lambda s: strat_pred_mean[s])
labels  = [response_map[s] for s in order]
ypos    = np.arange(len(order))
ax2.barh(ypos - 0.2, [strat_pred_mean[s] for s in order], height=0.4,
         color="#3498db", label="Model predicted", edgecolor="white", linewidth=0.3)
ax2.barh(ypos + 0.2, [strat_emp_mean[s] for s in order], height=0.4,
         color="#f39c12", label="Empirical mean", edgecolor="white", linewidth=0.3)
ax2.set_yticks(ypos)
ax2.set_yticklabels(labels, color="white", fontsize=9)
ax2.set_xlabel("Mean Recovery (days)", color="white")
ax2.set_title("Recovery by Strategy — historical association\n"
              "(fastest = recommended; not a causal claim)",
              color="white", fontsize=10, pad=10)
ax2.tick_params(colors="white")
ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, loc="lower right")
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax2.spines[spine].set_color("#4a4a6a")

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {FIG_OUT}")
print("\n  Stage 8 complete.")
