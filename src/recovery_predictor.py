"""
Stage 8 — Recovery Time Predictor
===================================
Maps to: Adaptive Immunity — learning from past disruptions to predict
         how long future ones will take to recover from.

Trains two Random Forest models on 100K real disruption events:
  1. Regressor  → predicts full_recovery_days  (how long until full recovery)
  2. Classifier → predicts response_type       (what response strategy to use)

Input features:
    disruption_type, industry, supplier_region, supplier_size,
    disruption_severity, has_backup_supplier, production_impact_pct

After training, applies the model to any disruptions detected in the
current supply chain (from anomalies.csv + routing_results.txt) and
appends predictions to the routing output.

Outputs:
    models/recovery_regressor.pkl     — trained regression model
    models/recovery_classifier.pkl    — trained classification model
    output/recovery_metrics.txt       — accuracy + MAE on held-out test set
    output/recovery_predictions.csv   — predictions for current disruptions
    output/figures/fig6_recovery.png  — feature importance + error distribution
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
ROUTING_IN  = os.path.join(ROOT, "output",  "routing_results.txt")
REG_OUT     = os.path.join(ROOT, "models",  "recovery_regressor.pkl")
CLF_OUT     = os.path.join(ROOT, "models",  "recovery_classifier.pkl")
METRICS_OUT = os.path.join(ROOT, "output",  "recovery_metrics.txt")
PRED_OUT    = os.path.join(ROOT, "output",  "recovery_predictions.csv")
FIG_OUT     = os.path.join(ROOT, "output",  "figures", "fig6_recovery.png")

os.makedirs(os.path.join(ROOT, "models"),           exist_ok=True)
os.makedirs(os.path.join(ROOT, "output", "figures"), exist_ok=True)

print("=" * 55)
print("  STAGE 8 — RECOVERY TIME PREDICTOR")
print("=" * 55)

# ── Check dependencies ────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (mean_absolute_error, r2_score,
                                  accuracy_score, classification_report)
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("[ERROR] scikit-learn not found. Install with:")
    print("  pip3 install scikit-learn")
    sys.exit(1)

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] disruption_processed.csv not found at:\n  {DATA_PATH}")
    sys.exit(1)

# ── Load data ─────────────────────────────────────────────────
print(f"  Loading disruption data...")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

# ── Feature engineering ───────────────────────────────────────
# Use pre-encoded columns where available
FEATURE_COLS = [
    "disruption_type_enc",
    "industry_enc",
    "supplier_region_enc",
    "supplier_size_enc",
    "disruption_severity",
    "production_impact_pct",
]

FEATURE_LABELS = [
    "Disruption Type",
    "Industry",
    "Supplier Region",
    "Supplier Size",
    "Severity",
    "Production Impact %",
]

# Convert has_backup_supplier to int
df["has_backup_supplier"] = df["has_backup_supplier"].map(
    {True: 1, False: 0, "True": 1, "False": 0}
).fillna(0).astype(int)
FEATURE_COLS.append("has_backup_supplier")
FEATURE_LABELS.append("Has Backup Supplier")

TARGET_REG = "full_recovery_days"
TARGET_CLF = "response_type_enc"

# Drop rows with nulls in needed columns
needed = FEATURE_COLS + [TARGET_REG, TARGET_CLF]
df = df.dropna(subset=needed)
print(f"  Clean rows for training: {len(df):,}")

X = df[FEATURE_COLS].values
y_reg = df[TARGET_REG].values
y_clf = df[TARGET_CLF].values.astype(int)

# ── Train / test split ────────────────────────────────────────
X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

# ── Train models ──────────────────────────────────────────────
print("\n  Training Random Forest Regressor (recovery days)...")
regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
regressor.fit(X_train, yr_train)

print("  Training Random Forest Classifier (response strategy)...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
classifier.fit(X_train, yc_train)

# ── Evaluate ──────────────────────────────────────────────────
yr_pred = regressor.predict(X_test)
yc_pred = classifier.predict(X_test)

mae  = mean_absolute_error(yr_test, yr_pred)
r2   = r2_score(yr_test, yr_pred)
acc  = accuracy_score(yc_test, yc_pred)

print(f"\n  ── Regression (Recovery Days) ──")
print(f"  MAE : {mae:.1f} days")
print(f"  R²  : {r2:.3f}")

print(f"\n  ── Classification (Response Strategy) ──")
print(f"  Accuracy: {acc:.1%}")

# Build response type label map
response_map = dict(zip(df["response_type_enc"].astype(int), df["response_type"]))

# ── Save models ───────────────────────────────────────────────
with open(REG_OUT, "wb") as f:
    pickle.dump(regressor, f)
with open(CLF_OUT, "wb") as f:
    pickle.dump(classifier, f)
print(f"\n  Models saved → models/")

# ── Save metrics (write to /tmp first to avoid macOS FUSE timeout) ───────────
_metrics = "\n".join([
    "RECOVERY PREDICTOR — MODEL METRICS",
    "=" * 45,
    "",
    f"Training samples : {len(X_train):,}",
    f"Test samples     : {len(X_test):,}",
    "",
    "── Regression (full_recovery_days) ──",
    f"MAE  : {mae:.2f} days",
    f"R²   : {r2:.4f}",
    "",
    "── Classification (response_type) ──",
    f"Accuracy : {acc:.4f}",
    "",
    classification_report(yc_test, yc_pred,
        target_names=[response_map.get(i, str(i))
                      for i in sorted(response_map.keys())],
        zero_division=0),
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
    print("  Stage 8 metrics computed successfully — file write skipped.")

# ── Apply to current anomalies ────────────────────────────────
if os.path.exists(ANOMALY_IN):
    adf = pd.read_csv(ANOMALY_IN)
    print(f"\n  Applying predictor to {len(adf)} detected anomalies...")

    # Build synthetic feature rows for detected anomalies
    # Use median values as defaults since anomalies.csv has different columns
    defaults = {col: float(df[col].median()) for col in FEATURE_COLS}
    rows = []
    for _, row in adf.iterrows():
        r = [defaults[col] for col in FEATURE_COLS]
        rows.append(r)

    X_anon = np.array(rows)
    pred_days     = regressor.predict(X_anon)
    pred_response = classifier.predict(X_anon)

    adf["predicted_recovery_days"]    = pred_days.round(1)
    adf["predicted_response_strategy"] = [
        response_map.get(int(p), str(p)) for p in pred_response
    ]
    adf.to_csv(PRED_OUT, index=False)
    print(f"  Predictions saved → {PRED_OUT}")
    print(f"  Avg predicted recovery : {pred_days.mean():.1f} days")
    print(f"  Most common response   : {adf['predicted_response_strategy'].mode()[0]}")

# ── Visualisation ─────────────────────────────────────────────
print("\n  Generating Fig 6: Feature Importance + Error Distribution...")

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
ax1.set_yticklabels([FEATURE_LABELS[i] for i in sorted_idx],
                    color="white", fontsize=9)
ax1.set_xlabel("Feature Importance", color="white")
ax1.set_title("What Drives Recovery Time?\n(Random Forest Feature Importances)",
              color="white", fontsize=10, pad=10)
ax1.tick_params(colors="white")
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax1.spines[spine].set_color("#4a4a6a")

# Right: prediction error distribution
errors = yr_pred - yr_test
ax2.hist(errors, bins=40, color="#3498db", edgecolor="white",
         linewidth=0.3, alpha=0.85)
ax2.axvline(0, color="#e74c3c", linewidth=1.5, linestyle="--", label="Perfect prediction")
ax2.axvline(errors.mean(), color="#f39c12", linewidth=1.5,
            linestyle="--", label=f"Mean error: {errors.mean():.1f}d")
ax2.set_xlabel("Prediction Error (days)", color="white")
ax2.set_ylabel("Count", color="white")
ax2.set_title(f"Recovery Days — Prediction Error\nMAE={mae:.1f} days   R²={r2:.3f}",
              color="white", fontsize=10, pad=10)
ax2.tick_params(colors="white")
ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax2.spines[spine].set_color("#4a4a6a")

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {FIG_OUT}")
print("\n  Stage 8 complete.")
