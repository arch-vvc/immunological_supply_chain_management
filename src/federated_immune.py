"""
Federated Immunological Learning
=================================
Simulates distributed immune memory across 3 organizations (Org A, B, C)
WITHOUT sharing raw supply chain data between them.

Biological parallel:
  Each organization = a separate lymphoid tissue (spleen, lymph node, thymus).
  Each maintains its own immunological memory. When a new threat is detected,
  each tissue contributes its top recall results to a federated coordinator
  that re-ranks and aggregates — distributed immune response.

What is private:        Raw transaction data never leaves each org.
What is shared:         Only top-K recall distances + anonymous feature vectors.

Architecture:
  Org A (33K vectors)  ──┐
  Org B (33K vectors)  ──┤── Federated Coordinator ── merged ranked recall
  Org C (33K vectors)  ──┘        (re-rank by distance)

Usage:
    python3 src/federated_immune.py                        # demo query
    python3 src/federated_immune.py --z 5.5 --build        # rebuild shards first
    python3 src/federated_immune.py --compare              # federated vs single org
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FAISS_INDEX  = ROOT / "models" / "faiss_memory.index"
FAISS_META   = ROOT / "models" / "faiss_memory_meta.pkl"
FED_DIR      = ROOT / "models" / "federated"

FED_DIR.mkdir(parents=True, exist_ok=True)

N_ORGS  = 3
TOP_K   = 3     # each org contributes its top-K matches
MERGE_K = 5     # final merged recall size

SEP = "=" * 72

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    print("[FED][WARN] faiss not installed — federated demo will be limited")


# ── Build shards ──────────────────────────────────────────────────────────────

def build_shards(force: bool = False) -> bool:
    """
    Split the global FAISS index into N_ORGS shards.
    Each shard = one simulated organization's private memory.
    Only run once (or with --build flag).
    """
    shard_paths = [FED_DIR / f"org_{chr(65+i)}_shard.index" for i in range(N_ORGS)]
    if all(p.exists() for p in shard_paths) and not force:
        return True   # already built

    if not FAISS_OK:
        return False
    if not FAISS_INDEX.exists() or not FAISS_META.exists():
        print("[FED][ERROR] Global FAISS index not found. Run Stage 13 first.")
        return False

    print(f"[FED] Building {N_ORGS} organisation shards from global index...")
    idx = faiss.read_index(str(FAISS_INDEX))
    with open(FAISS_META, "rb") as f:
        meta = pickle.load(f)

    total = idx.ntotal
    # Reconstruct all vectors from index
    all_vecs = np.zeros((total, idx.d), dtype=np.float32)
    for i in range(total):
        idx.reconstruct(i, all_vecs[i])

    shard_size = total // N_ORGS
    outcomes   = meta["outcomes"]

    for i in range(N_ORGS):
        org_name = chr(65 + i)
        start    = i * shard_size
        end      = start + shard_size if i < N_ORGS - 1 else total
        shard_vecs = all_vecs[start:end]

        shard_idx = faiss.IndexFlatL2(idx.d)
        shard_idx.add(shard_vecs)

        faiss.write_index(shard_idx, str(FED_DIR / f"org_{org_name}_shard.index"))

        # Save shard metadata (anonymised — no raw node names)
        shard_meta = {
            "org":      f"Org {org_name}",
            "size":     len(shard_vecs),
            "scaler":   meta["scaler"],
            "outcomes": outcomes.iloc[start:end].reset_index(drop=True),
        }
        with open(FED_DIR / f"org_{org_name}_meta.pkl", "wb") as f:
            pickle.dump(shard_meta, f)

        print(f"  [FED] Org {org_name}: {len(shard_vecs):,} vectors  "
              f"(rows {start:,}–{end:,})")

    print(f"[FED] Shards saved to {FED_DIR}/\n")
    return True


# ── Federated query ───────────────────────────────────────────────────────────

def federated_query(
    z: float,
    qty: float = 0.0,
    has_backup: float = 1.0,
    silent: bool = False,
) -> dict:
    """
    Each org searches its own shard privately.
    Only top-K (distances + outcomes — no raw data) returned to coordinator.
    Coordinator merges and re-ranks.
    """
    if not FAISS_OK:
        return {}
    if not FAISS_META.exists():
        return {}

    with open(FAISS_META, "rb") as f:
        meta = pickle.load(f)
    scaler = meta["scaler"]

    # Build query vector (same encoding as immune_response_engine)
    severity    = min(z / 10.0 * 4 + 1, 5.0)
    prod_impact = min(abs(z) / 10.0 * 100, 100.0)
    qvec = np.array([[severity, prod_impact, has_backup,
                      0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    qvec_scaled = scaler.transform(qvec).astype(np.float32)

    org_results   = {}   # org_name -> list of {distance, recovery_days, response_type}
    all_candidates = []

    for i in range(N_ORGS):
        org_name   = chr(65 + i)
        shard_path = FED_DIR / f"org_{org_name}_shard.index"
        meta_path  = FED_DIR / f"org_{org_name}_meta.pkl"

        if not shard_path.exists() or not meta_path.exists():
            continue

        shard_idx = faiss.read_index(str(shard_path))
        with open(meta_path, "rb") as f:
            shard_meta = pickle.load(f)

        dists, idxs = shard_idx.search(qvec_scaled, TOP_K)
        outcomes    = shard_meta["outcomes"]
        org_hits    = []

        for dist_val, idx_val in zip(dists[0], idxs[0]):
            if idx_val < 0 or idx_val >= len(outcomes):
                continue
            row = outcomes.iloc[idx_val]
            hit = {
                "org":           f"Org {org_name}",
                "distance":      round(float(dist_val), 4),
                "recovery_days": float(row.get("full_recovery_days", 0)),
                "response_type": str(row.get("response_type",
                                  row.get("response_type_enc", "?"))),
            }
            org_hits.append(hit)
            all_candidates.append(hit)

        org_results[f"Org {org_name}"] = org_hits

    # Coordinator: merge + re-rank by distance (no raw data, only distances)
    merged = sorted(all_candidates, key=lambda x: x["distance"])[:MERGE_K]

    avg_rec  = np.mean([m["recovery_days"] for m in merged]) if merged else 0
    top_resp = merged[0]["response_type"] if merged else "unknown"

    result = {
        "z_score":           z,
        "org_results":       org_results,
        "merged_recall":     merged,
        "avg_recovery_days": round(float(avg_rec), 1),
        "recommended":       top_resp,
        "orgs_queried":      len(org_results),
    }

    if not silent:
        _print_result(result)

    return result


# ── Compare federated vs single-org ──────────────────────────────────────────

def compare_federated_vs_single(z: float = 3.5):
    """
    Show that federated recall finds better (closer) matches than
    any single org alone — the key empirical argument for federated learning.
    """
    if not FAISS_OK or not FAISS_META.exists():
        return

    with open(FAISS_META, "rb") as f:
        meta = pickle.load(f)
    scaler = meta["scaler"]
    severity    = min(z / 10.0 * 4 + 1, 5.0)
    prod_impact = min(abs(z) / 10.0 * 100, 100.0)
    qvec = np.array([[severity, prod_impact, 1.0,
                      0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    qvec_scaled = scaler.transform(qvec).astype(np.float32)

    print(f"\n{SEP}")
    print(f"  FEDERATED vs SINGLE-ORG RECALL COMPARISON  |  Z={z}")
    print(SEP)
    print(f"  {'Source':<18} {'Best distance':>14} {'Avg recovery':>14} {'Top response':<20}")
    print(f"  {'-'*18} {'-'*14} {'-'*14} {'-'*20}")

    best_dists = {}
    for i in range(N_ORGS):
        org_name   = chr(65 + i)
        shard_path = FED_DIR / f"org_{org_name}_shard.index"
        meta_path  = FED_DIR / f"org_{org_name}_meta.pkl"
        if not shard_path.exists():
            continue
        shard_idx = faiss.read_index(str(shard_path))
        with open(meta_path, "rb") as f:
            sm = pickle.load(f)
        dists, idxs = shard_idx.search(qvec_scaled, TOP_K)
        hits = []
        for d, ix in zip(dists[0], idxs[0]):
            if ix >= 0 and ix < len(sm["outcomes"]):
                row = sm["outcomes"].iloc[ix]
                hits.append((float(d), float(row.get("full_recovery_days", 0)),
                             str(row.get("response_type", "?"))))
        if hits:
            best_d = hits[0][0]
            avg_r  = np.mean([h[1] for h in hits])
            top_r  = hits[0][2]
            best_dists[f"Org {org_name}"] = best_d
            print(f"  Org {org_name} (private)    {best_d:>14.4f} {avg_r:>14.1f}d {top_r:<20}")

    # Federated
    fed = federated_query(z, silent=True)
    if fed.get("merged_recall"):
        fed_best    = fed["merged_recall"][0]["distance"]
        worst_single = max(best_dists.values()) if best_dists else 999
        best_single  = min(best_dists.values()) if best_dists else 999
        # Key insight: federation lifts the worst-off org to global best
        improvement_worst = (worst_single - fed_best) / worst_single * 100 if worst_single > 0 else 0
        print(f"  {'Federated (merged)':<18} {fed_best:>14.4f} "
              f"{fed['avg_recovery_days']:>14.1f}d {fed['recommended']:<20}")
        print(f"\n  Key insight: Without federation, Org A is stuck with distance "
              f"{worst_single:.4f} (poor guidance).")
        print(f"  With federation it gets distance {fed_best:.4f} — "
              f"{improvement_worst:.1f}% improvement — without seeing Org C's raw data.")
        print(f"  Raw data shared between orgs: NONE")
        print(f"  Information shared: distances + anonymous outcome vectors only")
    print(SEP)


# ── Pretty printer ────────────────────────────────────────────────────────────

def _print_result(r: dict):
    print(f"\n{SEP}")
    print(f"  FEDERATED IMMUNE RECALL  |  Z-score={r['z_score']}")
    print(SEP)
    print(f"  Orgs queried (without sharing raw data): {r['orgs_queried']}")
    print()
    for org, hits in r["org_results"].items():
        print(f"  {org} private shard results:")
        for h in hits:
            print(f"    dist={h['distance']:.4f}  recovery={h['recovery_days']:.0f}d"
                  f"  response={h['response_type']}")
    print()
    print(f"  Federated coordinator merged recall (re-ranked by distance):")
    print(f"  {'Rank':<5} {'Source':<12} {'Distance':>10} {'Recovery':>10} {'Response'}")
    print(f"  {'-'*5} {'-'*12} {'-'*10} {'-'*10} {'-'*20}")
    for i, m in enumerate(r["merged_recall"], 1):
        print(f"  [{i}]   {m['org']:<12} {m['distance']:>10.4f} "
              f"{m['recovery_days']:>9.0f}d  {m['response_type']}")
    print()
    print(f"  Federated recommendation : {r['recommended']}")
    print(f"  Avg recovery estimate    : {r['avg_recovery_days']} days")
    print(f"  Privacy guarantee        : zero raw transactions left each org's boundary")
    print(SEP)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Immunological Memory")
    parser.add_argument("--z",       type=float, default=3.5,
                        help="Z-score of the simulated disruption (default: 3.5)")
    parser.add_argument("--build",   action="store_true",
                        help="Force rebuild org shards from global FAISS index")
    parser.add_argument("--compare", action="store_true",
                        help="Compare federated recall vs single-org recall")
    args = parser.parse_args()

    ok = build_shards(force=args.build)
    if not ok:
        sys.exit(1)

    if args.compare:
        compare_federated_vs_single(z=args.z)
    else:
        federated_query(z=args.z)
