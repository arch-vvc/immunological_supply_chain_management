"""The headline numbers currently in output/ must match the committed manifest.

Guards against silent drift: a rerun that moves a number by more than the
tolerance fails here and names the key, instead of leaving the paper stale.
Datasets are compared by hash so a changed input is also caught.
"""
import json
import os

import results_manifest as rm


def test_committed_manifest_exists():
    assert rm.MANIFEST.exists(), "run `python3 src/results_manifest.py` and commit output/RESULTS_MANIFEST.json"


def test_current_outputs_match_committed_manifest():
    if not rm.MANIFEST.exists():
        return
    with open(rm.MANIFEST) as f:
        committed = json.load(f)
    drift = rm.compare(rm.build(), committed)
    assert not drift, "results drift vs RESULTS_MANIFEST.json:\n  " + "\n  ".join(drift)


def test_compare_flags_numeric_drift_and_dataset_change():
    base = {"dataset_sha256": {"x": "aaa"}, "headline": {"a": {"f1": 0.50, "name": "ok"}}}
    same = {"dataset_sha256": {"x": "aaa"}, "headline": {"a": {"f1": 0.505, "name": "ok"}}}
    assert rm.compare(same, base) == []                       # within 2%
    moved = {"dataset_sha256": {"x": "bbb"}, "headline": {"a": {"f1": 0.60, "name": "changed"}}}
    drift = rm.compare(moved, base)
    assert any("dataset changed" in d for d in drift)
    assert any("a.f1" in d for d in drift)
    assert any("a.name" in d for d in drift)


def test_macro_thresholds_actually_apply_to_transactions():
    """Guard for the calendar mismatch: if the transaction calendar and the
    macro stress calendar stop overlapping, every row silently falls back to
    lambda = 1 and the per-week adaptation is dead code again."""
    import re
    path = rm.OUT / "anomaly_metrics.txt"
    if not path.exists():
        return
    m = re.search(r"Macro coverage share\s*:\s*([0-9.]+)", path.read_text(encoding="utf-8"))
    assert m, "anomaly_metrics.txt no longer reports the macro coverage share"
    assert float(m.group(1)) >= 0.5, f"only {float(m.group(1)):.0%} of transactions get a per-week stress multiplier"
