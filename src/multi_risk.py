"""
Stage 12 — Multi-Domain Risk Modelling
Trains XGBoost classifiers on disruption data and compares weighted F1 by industry.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


INPUT_PATH = os.path.join("data", "supplementary", "disruption_processed.csv")
OUT_CSV = os.path.join("output", "multi_domain_f1.csv")
OUT_FIG = os.path.join("output", "figures", "fig10_multi_domain_risk.png")

FEATURES = [
    "disruption_type_enc",
    "industry_enc",
    "supplier_tier",
    "supplier_region_enc",
    "supplier_size_enc",
    "has_backup_supplier",
    "domain_enc",
]
TARGET = "disruption_severity"


def prepare_xy(frame):
    x = frame[FEATURES].copy()
    for col in FEATURES:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.fillna(0.0)

    y_raw = frame[TARGET]
    y_codes, _ = pd.factorize(y_raw, sort=True)
    y = pd.Series(y_codes, index=frame.index)
    return x, y


def train_and_score(frame):
    """Train on an 80% split and score on the held-out 20% (never on training rows)."""
    x, y = prepare_xy(frame)
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # a class has too few members to stratify on this subset — fall back to a plain split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return f1_score(y_test, preds, average="weighted")


def main():
    start = time.time()

    print("=" * 55)
    print("  STAGE 12 — MULTI-DOMAIN RISK MODELLING")
    print("=" * 55)

    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input file not found: {INPUT_PATH}")
        raise SystemExit(1)

    df = pd.read_csv(INPUT_PATH)

    required = FEATURES + [TARGET, "industry"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        raise SystemExit(1)

    print(f"Loaded {len(df):,} rows from {INPUT_PATH}")

    full_f1 = train_and_score(df)
    print(f"Weighted F1 (full dataset, held-out test split): {full_f1:.4f}")

    rows = []
    industries = sorted([v for v in df["industry"].dropna().unique()])
    for ind in industries:
        sub = df[df["industry"] == ind]
        n = len(sub)
        if n < 100:
            continue
        f1 = train_and_score(sub)
        rows.append({"Industry": ind, "F1 Score": float(f1), "Sample Size": int(n)})

    if not rows:
        print("No industry subsets with at least 100 rows; nothing to save/plot.")
        print("Stage 12 complete")
        elapsed = time.time() - start
        print(f"  ✓ Stage 12 completed in {elapsed:.1f}s")
        return

    out_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)

    print("\nIndustry | F1 Score | Sample Size")
    print("-" * 55)
    for _, r in out_df.iterrows():
        print(f"{str(r['Industry']):<25} | {r['F1 Score']:.4f} | {int(r['Sample Size']):>11}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved table -> {OUT_CSV}")

    plot_df = out_df.sort_values("F1 Score", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["Industry"], plot_df["F1 Score"])
    plt.xlabel("Weighted F1 Score")
    plt.ylabel("Industry")
    plt.title("Multi-Domain Risk Model — F1 by Industry")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()
    print(f"Saved figure -> {OUT_FIG}")

    print("Stage 12 complete")
    elapsed = time.time() - start
    print(f"  ✓ Stage 12 completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()