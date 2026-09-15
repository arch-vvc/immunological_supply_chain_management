"""
Test harness for src/scms_vendor_scorecard.py
=============================================
Run:  ./myenv/bin/python src/test_vendor_scorecard.py   (pandas enables T2)
      /usr/bin/python3   src/test_vendor_scorecard.py   (T2 skipped)

Five groups:
  T1  Output invariants — bounds, ordering, incumbent exclusion, T5 lanes.
  T2  Independent recomputation — vendor OTIF/n recomputed from the raw CSV
      via pandas (different parser, different aggregation path).
  T3  Unit behaviour on synthetic data — shrinkage formula exact, walk-forward
      `before=` filtering, deterministic ranking, exclusion.
  T4  Cross-module math — King's SS recomputed from the safety-stock CSV's own
      stored columns must reproduce its stored safety_stock_packs_95.
  T5  Protocol equivalence + validation finding — replicates the canonical
      revealed-preference backtest exactly (same lanes, pools, hits), then
      runs the SCORECARD ranking through the same walk-forward protocol and
      reports its hit-rates vs the planner rule and random. The comparison is
      a FINDING (reported, not asserted); protocol equality IS asserted.

Exit code 0 = all assertions passed.
"""

import csv
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime

import scms_vendor_scorecard as sc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "output")

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  -> {detail}"))
    if not cond:
        failures.append(name)


print("=" * 60)
print("  TESTS — SCMS VENDOR SCORECARD")
print("=" * 60)

# Shared fixtures
recs = sc.load_shipments()
S    = sc.build_stats(recs)
lane_vendors = defaultdict(set)
for x in recs:
    lane_vendors[(x["mol"], x["cty"])].add(x["ven"])

sc_csv = os.path.join(OUT, "scms_vendor_scorecard.csv")
if not os.path.exists(sc_csv):
    sc.main()
rows = list(csv.DictReader(open(sc_csv)))

# ─────────────────────────────────────────────────────────────
print("\nT1 — output invariants")
# ─────────────────────────────────────────────────────────────
ok_bounds = ok_order = ok_incumbent = ok_distinct = ok_t5 = True
bad = ""
for r in rows:
    lane = (r["molecule"], r["country"])
    vens = [r[f"r{i}_vendor"] for i in (1, 2, 3) if r[f"r{i}_vendor"]]
    scores = [float(r[f"r{i}_score"]) for i in (1, 2, 3) if r[f"r{i}_score"]]
    if any(not (0.0 <= s_ <= 1.0001) for s_ in scores):
        ok_bounds, bad = False, f"{lane} scores {scores}"
    if r["r1_otif"] and not (0.0 <= float(r["r1_otif"]) <= 1.0):
        ok_bounds, bad = False, f"{lane} otif {r['r1_otif']}"
    if scores != sorted(scores, reverse=True):
        ok_order, bad = False, f"{lane} not descending {scores}"
    if len(set(vens)) != len(vens):
        ok_distinct, bad = False, f"{lane} duplicate vendors {vens}"
    if any(v in lane_vendors[lane] for v in vens):
        ok_incumbent, bad = False, f"{lane} incumbent leaked into shortlist"
    if r["tier"] == "T5_SINGLE_SOURCED" and int(r["n_candidates"]) != 0:
        ok_t5, bad = False, f"{lane} T5 lane has candidates"
check("scores/otif within bounds", ok_bounds, bad)
check("rank scores descending", ok_order, bad)
check("top-3 vendors distinct", ok_distinct, bad)
check("no incumbent in any shortlist", ok_incumbent, bad)
check("T5 lanes have zero candidates", ok_t5, bad)
check("all 92 exposed lanes present", len(rows) == 92, f"got {len(rows)}")

# ─────────────────────────────────────────────────────────────
print("\nT2 — independent recomputation (pandas)")
# ─────────────────────────────────────────────────────────────
try:
    import pandas as pd

    df = pd.read_csv(sc.SCMS_IN, encoding="utf-8", encoding_errors="replace")
    for col in ("Vendor", "Molecule/Test Type", "Country"):
        df[col] = df[col].astype(str).str.split().str.join(" ")
    dd = pd.to_datetime(df["Delivered to Client Date"], format="mixed", errors="coerce")
    ds = pd.to_datetime(df["Scheduled Delivery Date"], format="mixed", errors="coerce")
    m = (dd.notna() & ds.notna()
         & (df["Vendor"] != "") & (df["Molecule/Test Type"] != "") & (df["Country"] != ""))
    sub = df[m].copy()
    sub["ontime"] = ((dd - ds).dt.days <= 0)[m]

    agg = sub.groupby("Vendor")["ontime"].agg(["count", "mean"])
    directory = {r["vendor"]: r for r in csv.DictReader(
        open(os.path.join(OUT, "scms_vendor_directory.csv")))}

    picks = list(agg.sort_values("count", ascending=False).head(5).index) \
          + list(agg.sort_values("count").head(3).index)
    ok_n = ok_rate = True
    bad = ""
    for v in picks:
        if v not in directory:
            ok_n, bad = False, f"{v} missing from directory"
            continue
        n_mod = int(directory[v]["n_shipments"])
        r_mod = float(directory[v]["otif_raw"])
        n_pd, r_pd = int(agg.loc[v, "count"]), float(agg.loc[v, "mean"])
        if abs(n_mod - n_pd) > 2:
            ok_n, bad = False, f"{v}: n {n_mod} vs pandas {n_pd}"
        if abs(r_mod - r_pd) > 0.02:
            ok_rate, bad = False, f"{v}: otif {r_mod} vs pandas {r_pd:.3f}"
    check(f"shipment counts match pandas path ({len(picks)} vendors)", ok_n, bad)
    check("on-time rates match pandas path (±0.02)", ok_rate, bad)
    check("total parsed rows agree (±5)", abs(len(sub) - len(recs)) <= 5,
          f"pandas {len(sub)} vs module {len(recs)}")
except ImportError:
    print("  SKIP  pandas not available in this interpreter")

# ─────────────────────────────────────────────────────────────
print("\nT3 — unit behaviour on synthetic data")
# ─────────────────────────────────────────────────────────────
def ship(y, m, d, ven, mol="M", cty="X", delay=0, qty=10.0, lead=None):
    return {"date": datetime(y, m, d), "delay": delay, "ven": ven, "mol": mol,
            "cty": cty, "qty": qty, "val": qty * 2.0, "lead": lead}

synth = [ship(2010, 1, 10, "A", delay=5)]                       # A: 1 shipment, late
synth += [ship(2010, 1 + i % 12, 1 + i % 27, "B", delay=0) for i in range(20)]  # B: 20 on time
synth += [ship(2012, 6, 1, "C", delay=0)]                       # C: after cutoff
S3 = sc.build_stats(synth)
p0 = 21 / 22   # A late; B's 20 + C's 1 on time
relA = (0 + sc.SHRINK_K * p0) / (1 + sc.SHRINK_K)
relB = (20 + sc.SHRINK_K * p0) / (20 + sc.SHRINK_K)
check("shrinkage exact for 1-shipment vendor",
      abs(S3["vendor"]["A"]["reliability"] - relA) < 1e-12,
      f"{S3['vendor']['A']['reliability']} vs {relA}")
check("shrinkage exact for 20-shipment vendor",
      abs(S3["vendor"]["B"]["reliability"] - relB) < 1e-12,
      f"{S3['vendor']['B']['reliability']} vs {relB}")
S3b = sc.build_stats(synth, before=datetime(2012, 1, 1))
check("walk-forward `before=` excludes later shipments",
      "C" not in S3b["vendor"] and S3b["vendor"]["B"]["n"] == 20,
      f"vendors={sorted(S3b['vendor'])}")
r_all = sc.rank_candidates("M", set(), S3)
r_exc = sc.rank_candidates("M", {"B"}, S3)
check("ranking excludes requested vendors",
      all(c["vendor"] != "B" for c in r_exc) and len(r_exc) == len(r_all) - 1)
check("ranking deterministic",
      [c["vendor"] for c in sc.rank_candidates("M", set(), S3)] ==
      [c["vendor"] for c in r_all])

# ─────────────────────────────────────────────────────────────
print("\nT4 — King's formula consistent with safety-stock stage")
# ─────────────────────────────────────────────────────────────
ok_king, bad, n_king = True, "", 0
with open(os.path.join(OUT, "scms_safety_stock.csv")) as f:
    for r in csv.DictReader(f):
        ss_re = sc.king_ss(1.65, float(r["lead_mean_days"]),
                           float(r["daily_demand_std"]),
                           float(r["daily_demand_packs"]),
                           float(r["lead_std_days"]))
        stored = float(r["safety_stock_packs_95"])
        n_king += 1
        if abs(ss_re - stored) > max(2.0, 0.01 * stored):
            ok_king, bad = False, (f"{r['molecule'][:20]}|{r['country']}: "
                                   f"recomputed {ss_re:.0f} vs stored {stored:.0f}")
check(f"SS recomputed from stored columns matches ({n_king} lanes)", ok_king, bad)

# ─────────────────────────────────────────────────────────────
print("\nT5 — backtest protocol equivalence + scorecard validation")
# ─────────────────────────────────────────────────────────────
canon = {(r["molecule"], r["country"]): r
         for r in csv.DictReader(open(os.path.join(OUT, "scms_backtest.csv")))}

lanes_hist = defaultdict(list)
mol_hist   = defaultdict(list)
for x in recs:
    lanes_hist[(x["mol"], x["cty"])].append(x)
    mol_hist[x["mol"]].append(x)

n_eval = 0
mismatch_pool = mismatch_hits = 0
sum_h1_plan = sum_h3_plan = 0
sum_h1_sc = sum_h3_sc = 0
sum_h1_sc_c = sum_h3_sc_c = n_cond = 0
bad = ""

for L, hist in lanes_hist.items():
    lates = [x for x in hist if x["delay"] > 0]
    if not lates:
        continue
    t0 = lates[0]["date"]
    first_on_lane = {}
    for x in hist:
        first_on_lane.setdefault(x["ven"], x["date"])
    incumbents = {v for v, fs in first_on_lane.items() if fs <= t0}
    entrants   = {v for v, fs in first_on_lane.items() if fs > t0}
    if not entrants:
        continue
    n_eval += 1

    # exact replication of scms_backtest.py's planner ranking
    pool_qty = defaultdict(list)
    for x in mol_hist[L[0]]:
        if x["date"] < t0 and x["ven"] not in incumbents:
            pool_qty[x["ven"]].append(x["qty"])
    if pool_qty:
        cap_hi = max(statistics.median(q) for q in pool_qty.values()) or 1.0
        est_hi = max(len(q) for q in pool_qty.values()) or 1.0
        scored = {v: 0.6 * statistics.median(q) / cap_hi + 0.4 * len(q) / est_hi
                  for v, q in pool_qty.items()}
        plan_rank = [v for v, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))]
    else:
        plan_rank = []
    h1p = int(any(v in entrants for v in plan_rank[:1]))
    h3p = int(any(v in entrants for v in plan_rank[:3]))
    sum_h1_plan += h1p
    sum_h3_plan += h3p

    c = canon.get(L)
    if c is None or int(c["pool_size"]) != len(pool_qty):
        mismatch_pool += 1
        bad = f"{L}: pool {len(pool_qty)} vs canon {c['pool_size'] if c else '?'}"
    elif int(c["hit1_ours"]) != h1p or int(c["hit3_ours"]) != h3p:
        mismatch_hits += 1
        bad = f"{L}: hits ({h1p},{h3p}) vs canon ({c['hit1_ours']},{c['hit3_ours']})"

    # scorecard ranking under the same walk-forward protocol
    S_t0 = sc.build_stats(recs, before=t0)
    sc_rank = [c_["vendor"] for c_ in sc.rank_candidates(L[0], incumbents, S_t0)]
    h1s = int(any(v in entrants for v in sc_rank[:1]))
    h3s = int(any(v in entrants for v in sc_rank[:3]))
    sum_h1_sc += h1s
    sum_h3_sc += h3s
    if entrants & set(pool_qty):
        n_cond += 1
        sum_h1_sc_c += h1s
        sum_h3_sc_c += h3s

check("evaluable lanes == canonical (85)", n_eval == 85 == len(canon),
      f"replicated {n_eval}, canonical {len(canon)}")
check("per-lane pools identical to canonical", mismatch_pool == 0, bad)
check("per-lane planner hits identical to canonical", mismatch_hits == 0, bad)
check("aggregate planner hits match canonical",
      sum_h1_plan == sum(int(r["hit1_ours"]) for r in canon.values())
      and sum_h3_plan == sum(int(r["hit3_ours"]) for r in canon.values()),
      f"h1 {sum_h1_plan} h3 {sum_h3_plan}")

r1 = 100 * statistics.mean(float(r["p_hit1_rand"]) for r in canon.values())
r3 = 100 * statistics.mean(float(r["p_hit3_rand"]) for r in canon.values())
cond_rows = [r for r in canon.values() if int(r["nameable_at_t0"]) > 0]
r1c = 100 * statistics.mean(float(r["p_hit1_rand"]) for r in cond_rows)
r3c = 100 * statistics.mean(float(r["p_hit3_rand"]) for r in cond_rows)
p1c = 100 * statistics.mean(int(r["hit1_ours"]) for r in cond_rows)
p3c = 100 * statistics.mean(int(r["hit3_ours"]) for r in cond_rows)

print(f"""
  FINDING — scorecard ranking under the walk-forward protocol (n={n_eval}):
                              hit@1    hit@3
    STRICT      scorecard   {100*sum_h1_sc/n_eval:6.1f}%  {100*sum_h3_sc/n_eval:6.1f}%
                planner     {100*sum_h1_plan/n_eval:6.1f}%  {100*sum_h3_plan/n_eval:6.1f}%
                random      {r1:6.1f}%  {r3:6.1f}%
    CONDITIONAL scorecard   {100*sum_h1_sc_c/n_cond:6.1f}%  {100*sum_h3_sc_c/n_cond:6.1f}%   (n={n_cond})
                planner     {p1c:6.1f}%  {p3c:6.1f}%
                random      {r1c:6.1f}%  {r3c:6.1f}%""")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if failures:
    print(f"  {len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("  ALL ASSERTIONS PASSED")
