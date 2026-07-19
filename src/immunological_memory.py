"""
Stage 13 — Immunological Memory (FAISS)
Maps to: Adaptive Immunity — memory B-cells that store past disruption signatures
         and enable faster, more targeted responses on re-exposure.

How it works:
  EVALUATE — Before building the production index, memory quality is measured
             on a held-out split: 80% of disruption history is indexed, the
             remaining 20% query it, and the k-NN estimate of recovery days /
             response type is scored against the true outcome — with naive
             baselines (predict-the-mean, majority-class) for context.

  BUILD    — Index historical disruption records from disruption_processed.csv
             into a FAISS vector database.

  QUERY    — For each anomaly detected in the current pipeline run, map it into
             the same feature space and retrieve the top-K most similar past
             disruptions (expected recovery days + recommended response type).

Methodology notes:
  The similarity index uses ONLY features that are observable at query time:
      disruption_severity, production_impact_pct, has_backup_supplier
  The previous 8-dim index also included five categorical features that are
  unknown for a live anomaly and were filled with dataset means — after
  standardisation those dimensions contributed ~0 to every query distance,
  so they were dead weight that diluted the similarity metric. Features the
  query cannot observe do not belong in the distance computation.

Outputs:
  models/faiss_memory.index       — FAISS flat L2 index (exact nearest-neighbour)
  models/faiss_memory_meta.pkl    — scaler + outcome labels + feature config
  output/memory_retrieval.csv     — top-K matches per current anomaly
  output/memory_report.txt        — held-out evaluation + retrieval summary
"""

import os
import pickle
import shutil
import tempfile
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import StandardScaler

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISRUPT_PATH = os.path.join(ROOT, "data",   "supplementary", "disruption_processed.csv")
ANOMALY_PATH = os.path.join(ROOT, "output", "anomalies.csv")
INDEX_PATH   = os.path.join(ROOT, "models", "faiss_memory.index")
META_PATH    = os.path.join(ROOT, "models", "faiss_memory_meta.pkl")
RETRIEVAL_OUT= os.path.join(ROOT, "output", "memory_retrieval.csv")
REPORT_OUT   = os.path.join(ROOT, "output", "memory_report.txt")

# Only features observable at query time may enter the similarity metric.
INDEX_FEATURES = [
    "disruption_severity",      # numeric 1–5
    "production_impact_pct",    # numeric 0–100
    "has_backup_supplier",      # binary 0/1
]
N_FEATURES = len(INDEX_FEATURES)
TOP_K      = 3     # nearest neighbours to retrieve per query
RNG_SEED   = 42

print("=" * 55)
print("  STAGE 13 — IMMUNOLOGICAL MEMORY (FAISS)")
print("=" * 55)

if not os.path.exists(DISRUPT_PATH):
    print(f"[ERROR] {DISRUPT_PATH} not found.")
    exit(1)

print(f"\n[LOAD] Loading historical disruption data...")
ddf = pd.read_csv(DISRUPT_PATH)
print(f"  Loaded {len(ddf):,} disruption records  ({len(ddf.columns)} columns)")

missing = [c for c in INDEX_FEATURES if c not in ddf.columns]
if missing:
    print(f"[ERROR] Missing required columns: {missing}")
    exit(1)

ddf["has_backup_supplier"] = (
    ddf["has_backup_supplier"].map({True: 1, False: 0, "True": 1, "False": 0})
    .fillna(0).astype(int)
)

# ─────────────────────────────────────────────────────────────
# STEP 1: HELD-OUT EVALUATION — does memory recall actually
# predict outcomes better than naive baselines?
# ─────────────────────────────────────────────────────────────
print(f"\n[EVAL] Held-out evaluation (80% indexed, 20% querying)...")

rng      = np.random.default_rng(RNG_SEED)
perm     = rng.permutation(len(ddf))
n_train  = int(len(ddf) * 0.8)
tr_idx, te_idx = perm[:n_train], perm[n_train:]
train, test    = ddf.iloc[tr_idx], ddf.iloc[te_idx]

ev_scaler = StandardScaler()
Xtr = ev_scaler.fit_transform(train[INDEX_FEATURES].fillna(0).values).astype(np.float32)
Xte = ev_scaler.transform(test[INDEX_FEATURES].fillna(0).values).astype(np.float32)

ev_index = faiss.IndexFlatL2(N_FEATURES)
ev_index.add(Xtr)
_, nbrs = ev_index.search(Xte, TOP_K)

tr_recovery = train["full_recovery_days"].values
tr_response = train["response_type_enc"].values.astype(int)

knn_recovery = tr_recovery[nbrs].mean(axis=1)
knn_response = np.array([np.bincount(tr_response[row]).argmax() for row in nbrs])

te_recovery = test["full_recovery_days"].values
te_response = test["response_type_enc"].values.astype(int)

knn_mae   = float(np.mean(np.abs(knn_recovery - te_recovery)))
base_mae  = float(np.mean(np.abs(tr_recovery.mean() - te_recovery)))
knn_acc   = float((knn_response == te_response).mean())
maj_class = int(np.bincount(tr_response).argmax())
base_acc  = float((te_response == maj_class).mean())

print(f"  Recovery days  — k-NN MAE : {knn_mae:.1f}   predict-the-mean baseline : {base_mae:.1f}")
print(f"  Response type  — k-NN acc : {knn_acc:.1%}   majority-class baseline   : {base_acc:.1%}")
if knn_mae >= base_mae and knn_acc <= base_acc:
    print("  [NOTE] Memory recall does not beat naive baselines on the observable")
    print("         features — report this honestly rather than as a capability.")

# ─────────────────────────────────────────────────────────────
# STEP 2: BUILD PRODUCTION INDEX on the full history
# ─────────────────────────────────────────────────────────────
X_raw = ddf[INDEX_FEATURES].fillna(0).values.astype(np.float32)

outcomes = ddf[["full_recovery_days", "response_type_enc"]].copy()
outcomes["response_type"] = (
    ddf["response_type"].values
    if "response_type" in ddf.columns
    else outcomes["response_type_enc"].astype(str)
)
outcomes["disruption_type"] = (
    ddf["disruption_type"].values
    if "disruption_type" in ddf.columns
    else ddf["disruption_type_enc"].astype(str)
)
outcomes["industry"] = (
    ddf["industry"].values
    if "industry" in ddf.columns
    else ddf["industry_enc"].astype(str)
)
outcomes = outcomes.reset_index(drop=True)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

index = faiss.IndexFlatL2(N_FEATURES)
index.add(X_scaled)

print(f"\n[BUILD] FAISS index built: {index.ntotal:,} vectors  ({N_FEATURES} observable dims)")
print(f"  Index type: IndexFlatL2 (exact search)")

os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
faiss.write_index(index, INDEX_PATH)

meta = {
    "scaler":         scaler,
    "outcomes":       outcomes,
    "index_features": INDEX_FEATURES,
    "n_features":     N_FEATURES,
    "eval": {
        "knn_mae": knn_mae, "baseline_mae": base_mae,
        "knn_acc": knn_acc, "baseline_acc": base_acc,
        "top_k": TOP_K, "split": "80/20 held-out, seed 42",
    },
}
with open(META_PATH, "wb") as f:
    pickle.dump(meta, f)

print(f"  Index saved → {INDEX_PATH}")
print(f"  Metadata saved → {META_PATH}")

# ─────────────────────────────────────────────────────────────
# STEP 3: QUERY with current anomalies from anomalies.csv
# Every query dimension is derived from the anomaly's own signals:
#   severity   <- number of detection signals fired (2–5)
#   impact %   <- |volume z-score|, saturating at z=10
#   has_backup <- 0 when the concentration flag fired (single-supplier
#                 dependency = no backup available)
# ─────────────────────────────────────────────────────────────
print(f"\n[QUERY] Loading current anomalies...")

if not os.path.exists(ANOMALY_PATH):
    print(f"[WARN] {ANOMALY_PATH} not found. Skipping retrieval.")
    exit(0)

adf = pd.read_csv(ANOMALY_PATH)
print(f"  {len(adf)} current anomalies to query")

if len(adf) == 0:
    print("  No anomalies to query. Memory retrieval skipped.")
    exit(0)

severity   = adf["anomaly_score"].clip(1, 5).astype(float).values
impact     = (adf["z_quantity"].abs() / 10.0 * 100).clip(0, 100).values
has_backup = (~adf["flag_concentration"].astype(bool)).astype(float).values

Q_raw    = np.column_stack([severity, impact, has_backup]).astype(np.float32)
Q_scaled = scaler.transform(Q_raw).astype(np.float32)

distances, indices = index.search(Q_scaled, TOP_K)

# ─────────────────────────────────────────────────────────────
# STEP 4: BUILD RETRIEVAL REPORT
# ─────────────────────────────────────────────────────────────
print(f"\n[RETRIEVAL RESULTS]")
print(f"  Showing top-{TOP_K} memory matches per anomaly\n")

retrieval_rows = []
for i, (_, arow) in enumerate(adf.iterrows()):
    nbr_indices = indices[i]
    nbr_dists   = distances[i]

    rec_days_list  = []
    resp_type_list = []
    for j, (idx, dist) in enumerate(zip(nbr_indices, nbr_dists)):
        if idx < 0:
            continue
        nb = outcomes.iloc[idx]
        rec_days_list.append(float(nb["full_recovery_days"]))
        resp_type_list.append(str(nb["response_type"]))
        retrieval_rows.append({
            "anomaly_idx":        i,
            "manufacturer":       arow.get("manufacturer", ""),
            "retailer":           arow.get("retailer", ""),
            "anomaly_score":      arow.get("anomaly_score", ""),
            "match_rank":         j + 1,
            "match_distance":     round(float(dist), 4),
            "match_recovery_days":float(nb["full_recovery_days"]),
            "match_response_type":str(nb["response_type"]),
            "match_disruption_type": str(nb["disruption_type"]),
            "match_industry":     str(nb["industry"]),
        })

    if rec_days_list and i < 5:
        avg_rec  = round(np.mean(rec_days_list), 1)
        top_resp = max(set(resp_type_list), key=resp_type_list.count)
        mfr = str(arow.get("manufacturer", ""))[:40]
        print(f"  Anomaly {i+1}: {mfr}")
        print(f"    Score: {arow.get('anomaly_score','')}  |  "
              f"Expected recovery: {avg_rec} days  |  "
              f"Recommended response: {top_resp}")
        print(f"    Closest match distance: {nbr_dists[0]:.4f}")
    elif i == 5:
        print(f"  ... ({len(adf) - 5} more anomalies processed, see CSV for full results)")

retrieval_df = pd.DataFrame(retrieval_rows)
os.makedirs(os.path.dirname(RETRIEVAL_OUT), exist_ok=True)
retrieval_df.to_csv(RETRIEVAL_OUT, index=False)
print(f"\n  Retrieval results saved → {RETRIEVAL_OUT}")

# ─────────────────────────────────────────────────────────────
# STEP 5: MEMORY REPORT
# ─────────────────────────────────────────────────────────────
if len(retrieval_df) > 0:
    avg_recovery_all  = retrieval_df["match_recovery_days"].mean()
    std_recovery      = retrieval_df.groupby("anomaly_idx")["match_recovery_days"].mean().std()
    top_response_all  = retrieval_df["match_response_type"].mode()[0]
    top_disruption    = retrieval_df["match_disruption_type"].mode()[0]
    avg_distance      = retrieval_df["match_distance"].mean()

    print(f"\n──────────────────────────────────────────────────")
    print(f"  MEMORY SUMMARY")
    print(f"──────────────────────────────────────────────────")
    print(f"  Anomalies queried       : {len(adf)}")
    print(f"  Index size              : {index.ntotal:,} historical disruptions")
    print(f"  Avg retrieval distance  : {avg_distance:.4f}  (lower = closer match)")
    print(f"  Expected recovery       : {avg_recovery_all:.1f} days avg "
          f"(per-anomaly spread std={std_recovery:.1f})")
    print(f"  Most recommended response: {top_response_all}")

    _rpt = "\n".join([
        "STAGE 13 — IMMUNOLOGICAL MEMORY REPORT",
        "=" * 55,
        "",
        "Held-out evaluation (80% indexed, 20% querying, seed 42):",
        f"  Recovery days  — k-NN(k={TOP_K}) MAE : {knn_mae:.1f} days",
        f"                   predict-the-mean   : {base_mae:.1f} days",
        f"  Response type  — k-NN(k={TOP_K}) acc : {knn_acc:.1%}",
        f"                   majority-class     : {base_acc:.1%}",
        "  (Memory recall is only a capability to the extent it beats",
        "   these baselines on features observable at query time.)",
        "",
        f"Index size              : {index.ntotal:,} historical disruptions",
        f"Dimensions              : {N_FEATURES} observable features "
        f"({', '.join(INDEX_FEATURES)})",
        f"Index type              : FAISS IndexFlatL2 (exact search)",
        f"Anomalies queried       : {len(adf)}",
        f"Top-K retrieved per query: {TOP_K}",
        "",
        "Aggregate retrieval results:",
        f"  Avg match distance    : {avg_distance:.4f}",
        f"  Avg recovery days     : {avg_recovery_all:.1f} "
        f"(per-anomaly spread std={std_recovery:.1f})",
        f"  Top response type     : {top_response_all}",
        f"  Top disruption type   : {top_disruption}",
        "",
        "Design note: only query-time-observable features enter the distance",
        "metric. The earlier 8-dim index padded five unobservable categorical",
        "dims with dataset means, which contributed ~0 distance for every",
        "query and diluted the similarity signal.",
        "",
        "Immune analogy:",
        "  Memory B-cells (FAISS vectors) recognise similar antigens (disruption",
        "  signatures) and activate a targeted response faster than the innate",
        "  immune system could derive from scratch.",
    ])
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write(_rpt)
            _tmp = tmp.name
        shutil.copy2(_tmp, REPORT_OUT)
        os.unlink(_tmp)
        print(f"  Report saved → {REPORT_OUT}")
    except Exception as _e:
        print(f"  [WARN] Could not save report: {_e}")
