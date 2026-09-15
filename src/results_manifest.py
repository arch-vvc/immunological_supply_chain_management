"""
RESULTS MANIFEST — single source of truth for the project's headline numbers
===========================================================================
Reads the artifacts already written under output/ (through the same parsers
project_metrics.py uses) and writes output/RESULTS_MANIFEST.json with:

  provenance : git commit, timestamp, sha256 of every input dataset
  seeds      : the fixed seeds each stage uses
  headline   : every number the paper / README / dashboard quotes

`--check` rebuilds the manifest in memory and diffs it against the committed
file (relative tolerance 2%, absolute 1e-3); any drifted key is printed and
the exit code is 1. tests/test_results_manifest.py does the same under CI, so
"did a rerun silently move a number" becomes a failing test rather than a
stale table in the paper.

    python3 src/results_manifest.py          # write
    python3 src/results_manifest.py --check  # compare
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from project_metrics import collect

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
MANIFEST = OUT / "RESULTS_MANIFEST.json"

DATASETS = [
    "data/raw/arcos_sampled_50k.csv",
    "data/raw/SCMS_Delivery_History_Dataset.csv",
    "data/supplementary/disruption_processed.csv",
    "data/supplementary/Supply_Chain_and_Freight_Indicators.csv",
]

# Training stages run from scratch unless ISC_CONTINUAL=1 (see README); the
# manifest records from-scratch numbers only.
# Seeds as set in the stage modules (kept here so they are visible in one place;
# the modules remain the source — this is documentation that travels with the numbers).
SEEDS = {
    "generate_arcos": 18,
    "isolation_forest": 42,
    "anomaly_benchmark_tune_seed": 42,
    "anomaly_benchmark_holdout_seeds": [43, 44, 45, 46],
    "gnn_encoder": 42,
    "lstm_forecaster": 42,
    "recovery_xgboost": 42,
    "multi_domain_xgboost": 42,
    "immunological_memory_split": 42,
    "ppo_routing_agent": 42,
    "scms_counterfactual_bootstrap": 42,
    "ir_stages_betweenness": 42,
}

REL_TOL = 0.02
ABS_TOL = 1e-3


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _json(name: str) -> dict | None:
    p = OUT / name
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _macro_events() -> dict:
    p = OUT / "macro_event_validation.csv"
    if not p.exists():
        return {}
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        r["event"]: {
            "delta": float(r["delta"]),
            "placebo_percentile": float(r["placebo_percentile"]),
            "verdict": r["verdict"],
            "top_driver": r["top_driver"],
        } for r in rows
    }


def build() -> dict:
    m = collect()
    cal = _json("anomaly_ensemble_calibration.json") or {}
    ppo = _json("ppo_routing_stats.json") or {}
    cf = _json("scms_counterfactual_stats.json") or {}
    anom = m["anomaly"]

    def _method(name: str, key: str):
        return anom.get(name, {}).get(key)

    headline = {
        "anomaly_detection": {
            "n_high_confidence": None,  # filled below from anomalies.csv
            "calibrated_ensemble": {
                "f1": cal.get("held_out_f1", _method("Calibrated ensemble (logistic)", "f1")),
                "precision": cal.get("held_out_precision"),
                "precision_adjusted": cal.get("held_out_precision_adjusted"),
                "recall": cal.get("held_out_recall"),
                "threshold": cal.get("threshold"),
            },
            "isolation_forest_f1": _method("Isolation Forest alone", "f1"),
            "zscore_2plus_f1": _method("Z-score only  (2+ signals)", "f1"),
        },
        "recovery_predictor": {
            "mae_days": m["recovery"].get("mae"),
            "r2": m["recovery"].get("r2"),
            "baseline_mae_days": m["recovery"].get("baseline_mae"),
        },
        "lstm_stress_forecaster": {
            "mae": m["lstm"].get("mae"),
            "persistence_mae": m["lstm"].get("baseline_mae"),
            "beats_persistence": (m["lstm"].get("mae") is not None
                                  and m["lstm"].get("baseline_mae") is not None
                                  and m["lstm"]["mae"] < m["lstm"]["baseline_mae"]),
        },
        "immunological_memory": {
            "knn_mae_days": m["memory"].get("knn_mae"),
            "baseline_mae_days": m["memory"].get("base_mae"),
            "knn_response_accuracy_pct": m["memory"].get("knn_acc"),
            "majority_accuracy_pct": m["memory"].get("base_acc"),
        },
        "ppo_cascade": {
            "n_eval_episodes": ppo.get("n_eval_episodes"),
            "avg_total_reward": ppo.get("avg_total_reward"),
            "sd_total_reward": ppo.get("sd_total_reward"),
            "unserved": ppo.get("unserved"),
            "ppo_safer_than_dijkstra_pct": ppo.get("ppo_safer_than_dijkstra_pct",
                                                    m["ppo"].get("safer_pct")),
            "paired_delta_reward": ppo.get("paired_delta_reward"),
        },
        "multi_domain_risk": dict(m["multi"]),
        "macro_crisis_validation": _macro_events(),
        "scms_counterfactual": {
            k: cf.get(k) for k in (
                "n_comparable", "wins", "ties", "losses", "win_rate", "win_rate_ci95",
                "a_late_rate", "b_late_rate", "r_late_rate", "late_rate_gap",
                "late_rate_gap_ci95", "lateness_saved_mean", "lateness_saved_ci95",
                "sign_test_p")
        },
        "response_planner": {
            "anomalies_planned": m["response_plan"].get("planned"),
            "proven_supplier_pct": m["response_plan"].get("resolved_pct"),
        },
    }
    p = OUT / "anomalies.csv"
    if p.exists():
        with p.open(newline="", encoding="utf-8") as f:
            headline["anomaly_detection"]["n_high_confidence"] = sum(1 for _ in f) - 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "dataset_sha256": {d: _sha256(ROOT / d) for d in DATASETS},
        "seeds": SEEDS,
        "headline": headline,
    }


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            for i, x in enumerate(v):
                out[f"{key}[{i}]"] = x
        else:
            out[key] = v
    return out


def compare(current: dict, committed: dict, rel_tol=REL_TOL, abs_tol=ABS_TOL) -> list[str]:
    """Return human-readable drift lines; empty list means everything matches."""
    drift = []
    for d, h in committed.get("dataset_sha256", {}).items():
        if current["dataset_sha256"].get(d) != h:
            drift.append(f"dataset changed: {d}")
    cur, old = _flatten(current["headline"]), _flatten(committed.get("headline", {}))
    for key, ov in old.items():
        cv = cur.get(key)
        if ov is None:
            continue
        if cv is None:
            drift.append(f"missing now: {key} (was {ov})")
        elif isinstance(ov, bool) or isinstance(ov, str):
            if cv != ov:
                drift.append(f"{key}: {ov!r} -> {cv!r}")
        elif isinstance(ov, (int, float)):
            tol = max(abs_tol, rel_tol * abs(ov))
            if abs(float(cv) - float(ov)) > tol:
                drift.append(f"{key}: {ov} -> {cv} (tol ±{tol:.4g})")
    return drift


def main() -> None:
    ap = argparse.ArgumentParser(description="Write or check the results manifest")
    ap.add_argument("--check", action="store_true",
                    help="compare current outputs against the committed manifest; exit 1 on drift")
    args = ap.parse_args()

    current = build()
    if args.check:
        if not MANIFEST.exists():
            print(f"[ERROR] No committed manifest at {MANIFEST}. Run without --check first.")
            sys.exit(2)
        with MANIFEST.open() as f:
            committed = json.load(f)
        drift = compare(current, committed)
        if drift:
            print(f"RESULTS DRIFT vs {MANIFEST.name} ({len(drift)} item(s)):")
            for line in drift:
                print(f"  - {line}")
            sys.exit(1)
        print(f"Results match {MANIFEST.name} (commit {str(committed.get('git_commit'))[:10]}).")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w") as f:
        json.dump(current, f, indent=2)
    n = len(_flatten(current["headline"]))
    print(f"Wrote {MANIFEST} ({n} headline values, commit {str(current['git_commit'])[:10]})")


if __name__ == "__main__":
    main()
