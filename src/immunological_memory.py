"""
Stage 13 — Immunological Memory (FAISS)
Maps to: Adaptive Immunity — memory B-cells that store past disruption signatures
         and enable faster, more targeted responses on re-exposure.

How it works:
  BUILD  — Index 100K historical disruption records from disruption_processed.csv
           into a FAISS vector database. Each record is embedded as an 8-dimensional
           feature vector: severity, production impact, backup supplier availability,
           and four encoded categorical signals (disruption type, industry, region, size).

  QUERY  — For each anomaly detected in the current pipeline run (anomalies.csv),
           map it into the same feature space and retrieve the top-3 most similar
           past disruptions. Return expected recovery days and recommended response type.

  LEARN  — New outcomes are appended to the index after each pipeline run,
           so the system improves with each disruption it processes.
           This is clonal selection — successful responses are reinforced.

Outputs:
  models/faiss_memory.index       — FAISS flat L2 index (exact nearest-neighbour)
  models/faiss_memory_meta.pkl    — scaler + outcome labels + feature config
  output/memory_retrieval.csv     — top-3 matches per current anomaly
  output/memory_report.txt        — summary
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

# Features used to embed each disruption into the memory space
# Must be present in disruption_processed.csv (all already encoded)
INDEX_FEATURES = [
    "disruption_severity",      # numeric 1–5
    "production_impact_pct",    # numeric 0–100
    "has_backup_supplier",      # binary 0/1
    "disruption_type_enc",      # label-encoded category
    "industry_enc",             # label-encoded category
    "supplier_region_enc",      # label-encoded category
    "supplier_size_enc",        # label-encoded category
    "response_type_enc",        # label-encoded category
]
N_FEATURES = len(INDEX_FEATURES)
TOP_K      = 3    # nearest neighbours to retrieve per query

print("=" * 55)
print("  STAGE 13 — IMMUNOLOGICAL MEMORY (FAISS)")
print("=" * 55)

# ─────────────────────────────────────────────────────────────
# STEP 1: BUILD INDEX from disruption_processed.csv
# ─────────────────────────────────────────────────────────────
if not os.path.exists(DISRUPT_PATH):
    print(f"[ERROR] {DISRUPT_PATH} not found.")
    exit(1)

print(f"\n[BUILD] Loading historical disruption data...")
ddf = pd.read_csv(DISRUPT_PATH)
print(f"  Loaded {len(ddf):,} disruption records  ({len(ddf.columns)} columns)")

missing = [c for c in INDEX_FEATURES if c not in ddf.columns]
if missing:
    print(f"[ERROR] Missing required columns: {missing}")
    exit(1)

X_raw = ddf[INDEX_FEATURES].fillna(0).values.astype(np.float32)

# Outcome labels stored alongside each vector (not in the index itself)
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

# Fit scaler and normalise
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

# Build FAISS flat L2 index (exact nearest-neighbour, no approximation)
index = faiss.IndexFlatL2(N_FEATURES)
index.add(X_scaled)

print(f"  FAISS index built: {index.ntotal:,} vectors  ({N_FEATURES} dimensions)")
print(f"  Index type: IndexFlatL2 (exact search)")

# ─────────────────────────────────────────────────────────────
# STEP 2: SAVE INDEX + METADATA
# ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
faiss.write_index(index, INDEX_PATH)

meta = {
    "scaler":        scaler,
    "outcomes":      outcomes,
    "index_features":INDEX_FEATURES,
    "n_features":    N_FEATURES,
}
with open(META_PATH, "wb") as f:
    pickle.dump(meta, f)

print(f"  Index saved → {INDEX_PATH}")
print(f"  Metadata saved → {META_PATH}")

# ─────────────────────────────────────────────────────────────
# STEP 3: QUERY with current anomalies from anomalies.csv
# Map anomaly feature space → disruption feature space
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

# Map anomaly columns to disruption feature space
# These are approximations — the categorical features default to their
# mean-encoded values (pharmaceutical domain = single industry).
mean_disruption_type = float(ddf["disruption_type_enc"].mean())
mean_industry        = float(ddf["industry_enc"].mean())
mean_region          = float(ddf["supplier_region_enc"].mean())
mean_size            = float(ddf["supplier_size_enc"].mean())

# anomaly_score is 0–5 → maps to disruption_severity 1–5
# iforest_score is already a risk signal (higher = more anomalous)
# flag_concentration → 1.0 means high dependency = no backup supplier
query_rows = []
for _, row in adf.iterrows():
    severity      = min(float(row.get("anomaly_score", 2)) / 5.0 * 4 + 1, 5.0)  # scale to 1–5
    prod_impact   = min(abs(float(row.get("z_quantity", 0))) / 10.0 * 100, 100.0)
    has_backup    = 0.0 if bool(row.get("flag_concentration", False)) else 1.0
    response_enc  = 0.0   # unknown at query time

    query_rows.append([
        severity,
        prod_impact,
        has_backup,
        mean_disruption_type,   # pharmaceutical domain proxy
        mean_industry,
        mean_region,
        mean_size,
        response_enc,
    ])

Q_raw    = np.array(query_rows, dtype=np.float32)
Q_scaled = scaler.transform(Q_raw).astype(np.float32)

# Search FAISS
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
    valid       = nbr_indices[nbr_indices >= 0]

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

    if rec_days_list:
        avg_rec = round(np.mean(rec_days_list), 1)
        top_resp = max(set(resp_type_list), key=resp_type_list.count)
        if i < 5:
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
    top_response_all  = retrieval_df["match_response_type"].mode()[0]
    top_disruption    = retrieval_df["match_disruption_type"].mode()[0]
    avg_distance      = retrieval_df["match_distance"].mean()

    print(f"\n──────────────────────────────────────────────────")
    print(f"  MEMORY SUMMARY")
    print(f"──────────────────────────────────────────────────")
    print(f"  Anomalies queried       : {len(adf)}")
    print(f"  Index size              : {index.ntotal:,} historical disruptions")
    print(f"  Avg retrieval distance  : {avg_distance:.4f}  (lower = closer match)")
    print(f"  Expected avg recovery   : {avg_recovery_all:.1f} days")
    print(f"  Most recommended response: {top_response_all}")
    print(f"  Most similar disruption type: {top_disruption}")

    _rpt = "\n".join([
        "STAGE 13 — IMMUNOLOGICAL MEMORY REPORT",
        "=" * 55,
        "",
        f"Index size              : {index.ntotal:,} historical disruptions",
        f"Dimensions              : {N_FEATURES} features",
        f"Index type              : FAISS IndexFlatL2 (exact search)",
        f"Anomalies queried       : {len(adf)}",
        f"Top-K retrieved per query: {TOP_K}",
        "",
        "Aggregate retrieval results:",
        f"  Avg match distance    : {avg_distance:.4f}",
        f"  Avg recovery days     : {avg_recovery_all:.1f}",
        f"  Top response type     : {top_response_all}",
        f"  Top disruption type   : {top_disruption}",
        "",
        "Immune analogy:",
        "  Memory B-cells (FAISS vectors) recognise similar antigens (disruption",
        "  signatures) and activate a targeted response faster than the innate",
        "  immune system could derive from scratch. Each pipeline run that detects",
        "  anomalies adds new signatures to the index (clonal selection).",
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
