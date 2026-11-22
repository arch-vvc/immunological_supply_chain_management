"""
Sensitivity of the heuristic composite weights (paper Section 8.1.7).

Each weight vector is perturbed by an independent uniform factor in
[1 - 0.20, 1 + 0.20] per weight, renormalised to sum to 1, and the ranking or
verdicts are recomputed. 200 draws, seed 42. Three weight vectors are tested:

  1. Graph risk (Eq. 1, Stage 4): 0.5 betweenness + 0.3 in-degree + 0.2 PageRank.
     Kendall tau between perturbed and default scores over the default top 20.
  2. Supplier agent composite (Stage 14): 0.5 safety + 0.3 volume + 0.2 efficiency,
     on the pool of all manufacturer and distributor nodes. Kendall tau over the
     default top 20 of the pool, and over the full pool.
  3. Macro stress composite (Eq. 3, Stage 9): five indicator weights. The crisis
     event study of Stage 27 is rerun under each draw; verdict flips are counted.

Run after main.py (needs output/risk_scores.csv, output/graph_risk_scores.csv
and the freight indicator data):

    python3 src/weight_sensitivity.py

Writes output/weight_sensitivity.json and output/weight_sensitivity.txt.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isc_common import SUPPLIER_WEIGHTS          # noqa: E402
from macro_event_validation import run_event_study  # noqa: E402
from macro_risk import build_stress               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "output", "weight_sensitivity.json")
OUT_TXT = os.path.join(ROOT, "output", "weight_sensitivity.txt")

N_DRAWS = 200
SPREAD = 0.20
SEED = 42
TOP_K = 20
MIN_INDICATORS = 3   # same availability rule as macro_risk.build_stress

rng = np.random.default_rng(SEED)


def perturb(w):
    w = np.asarray(w, dtype=float) * rng.uniform(1 - SPREAD, 1 + SPREAD, len(w))
    return w / w.sum()


def tau_summary(taus):
    taus = np.asarray(taus)
    return {"mean": round(float(taus.mean()), 4), "sd": round(float(taus.std()), 4),
            "min": round(float(taus.min()), 4), "max": round(float(taus.max()), 4)}


# ── 1. Graph risk ranking (Eq. 1) ────────────────────────────────────────────
risk = pd.read_csv(os.path.join(ROOT, "output", "risk_scores.csv"))
cols = ["betweenness", "in_degree", "pagerank"]
norm = np.column_stack([risk[c] / risk[c].max() if risk[c].max() > 0 else risk[c] * 0
                        for c in cols])
w_graph = np.array([0.5, 0.3, 0.2])
default = norm @ w_graph
top = np.argsort(-default)[:TOP_K]
graph_taus = [kendalltau(default[top], (norm @ perturb(w_graph))[top])[0]
              for _ in range(N_DRAWS)]

# ── 2. Supplier agent composite ──────────────────────────────────────────────
g = pd.read_csv(os.path.join(ROOT, "output", "graph_risk_scores.csv"))
pool = g[g["type"].isin(["manufacturer", "distributor"])].copy()
pool["efficiency"] = pool["out_volume"] / pool["out_degree"].clip(lower=1)


def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


sub = np.column_stack([1.0 - pool["composite_risk"], minmax(pool["out_volume"]),
                       minmax(pool["efficiency"])])
w_sup = np.array([SUPPLIER_WEIGHTS["safety"], SUPPLIER_WEIGHTS["volume"],
                  SUPPLIER_WEIGHTS["efficiency"]])
sup_default = sub @ w_sup
sup_top = np.argsort(-sup_default)[:TOP_K]
sup_taus, sup_taus_all = [], []
for _ in range(N_DRAWS):
    s = sub @ perturb(w_sup)
    sup_taus.append(kendalltau(sup_default[sup_top], s[sup_top])[0])
    sup_taus_all.append(kendalltau(sup_default, s)[0])

# ── 3. Crisis event-study verdicts under the macro composite weights (Eq. 3) ─
stress = build_stress(verbose=False).sort_values("date").reset_index(drop=True)
ind_cols = [c for c in stress.columns
            if c not in ("date", "stress_score", "stress_level", "n_indicators")]
w_macro = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
assert len(ind_cols) == len(w_macro), f"expected 5 indicator columns, got {ind_cols}"
mat = stress[ind_cols].to_numpy(dtype=float)
avail = ~np.isnan(mat)


def verdicts(w):
    w_sum = (avail * w).sum(axis=1)
    comp = np.where(w_sum > 0, np.nansum(mat * w, axis=1) / np.where(w_sum > 0, w_sum, 1), np.nan)
    comp[avail.sum(axis=1) < MIN_INDICATORS] = np.nan
    # Keep the indicator columns: the event study also names each crisis's top driver.
    df = stress.drop(columns=["stress_score"]).assign(stress_score=comp)
    df = df.dropna(subset=["stress_score"])
    df["stress_score"] = df["stress_score"].rolling(4, min_periods=1).mean()
    rows, _ = run_event_study(df.reset_index(drop=True))
    return {r["event"]: r["verdict"] for r in rows}


base = verdicts(w_macro)
reproduced = pd.read_csv(os.path.join(ROOT, "output", "macro_event_validation.csv"))
assert base == dict(zip(reproduced["event"], reproduced["verdict"])), \
    "default weights do not reproduce Stage 27's verdicts"
flips = {e: 0 for e in base}
all_same = 0
for _ in range(N_DRAWS):
    v = verdicts(perturb(w_macro))
    changed = [e for e in base if v[e] != base[e]]
    all_same += not changed
    for e in changed:
        flips[e] += 1

# ── Report ───────────────────────────────────────────────────────────────────
result = {
    "draws": N_DRAWS, "spread": SPREAD, "seed": SEED,
    "graph_risk_top20_tau": tau_summary(graph_taus),
    "graph_risk_draws_tau_above_0.8": int(sum(t > 0.8 for t in graph_taus)),
    "supplier_pool_size": int(len(pool)),
    "supplier_top20_tau": tau_summary(sup_taus),
    "supplier_full_pool_tau": tau_summary(sup_taus_all),
    "crisis_default_verdicts": base,
    "crisis_all_verdicts_unchanged": all_same,
    "crisis_flips": flips,
}
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

gt, st, sa = (result["graph_risk_top20_tau"], result["supplier_top20_tau"],
              result["supplier_full_pool_tau"])
lines = [
    "WEIGHT SENSITIVITY (±20% per weight, renormalised, 200 draws, seed 42)",
    "=" * 70,
    f"Graph risk, default top {TOP_K}: Kendall tau mean {gt['mean']} (sd {gt['sd']}, "
    f"range {gt['min']} to {gt['max']}); {result['graph_risk_draws_tau_above_0.8']}/{N_DRAWS} draws above 0.8",
    f"Supplier agent, default top {TOP_K} of {len(pool)}: Kendall tau mean {st['mean']} "
    f"(sd {st['sd']}, range {st['min']} to {st['max']})",
    f"Supplier agent, full pool of {len(pool)}: Kendall tau mean {sa['mean']} "
    f"(sd {sa['sd']}, range {sa['min']} to {sa['max']})",
    f"Crisis verdicts: all five unchanged in {all_same}/{N_DRAWS} draws",
] + [f"  {e:<28} default {base[e]:<9} flips {n}/{N_DRAWS}" for e, n in flips.items()]
with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nSaved -> {OUT_JSON}")
