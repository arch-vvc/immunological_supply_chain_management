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


def test_macro_threshold_fallback_is_explicit():
    """The synthetic ARCOS dates (2006-2014) predate the freight indicators
    (2017 onward), and they are deliberately NOT shifted onto that calendar:
    pairing synthetic transactions with real stress from years later would be
    artificial. So no row gets a per-week multiplier and every row uses the
    logged fallback (paper Section 6.3); the per-week mechanism is exercised by
    the streaming consumer. Guard that the coverage and the fallback lambda are
    reported, and that uncovered rows all use that fallback."""
    import re
    path = rm.OUT / "anomaly_metrics.txt"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    cov = re.search(r"Macro coverage share\s*:\s*([0-9.]+)", text)
    assert cov, "anomaly_metrics.txt no longer reports the macro coverage share"
    line = re.search(r"Macro stress\s*:(.*)", text)
    assert line and "fallback λ=" in line.group(1), "the fallback lambda is no longer logged"
    if float(cov.group(1)) == 0.0:
        fallback = re.search(r"fallback λ=([0-9.]+)", line.group(1)).group(1)
        buckets = re.findall(r"λ=([0-9.]+):", line.group(1))
        assert buckets == [fallback], (
            f"0% coverage but rows use λ buckets {buckets}, not only the fallback λ={fallback}")
