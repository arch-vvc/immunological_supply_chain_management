"""
SCMS Revealed-Preference Backtest — does the planner's shortlist match
what real procurement actually did next?
======================================================================
THE QUESTION
------------
When a lane (molecule x country) suffered its FIRST late shipment, and real
procurement LATER brought a brand-new vendor onto that lane, would our
planner — using only information available at the time — have named that
vendor on its qualification shortlist?

This is the strongest validation an observational dataset allows:
  * ground truth = the revealed preference of real procurement officers
    (a vendor genuinely entered the lane afterwards), and
  * the planner is evaluated WALK-FORWARD: its candidate pool and scores are
    built strictly from shipments BEFORE the disruption date. No lookahead.

WHAT WE CLAIM (and what we don't)
---------------------------------
HIT  = the vendor reality eventually used was on our top-k shortlist.
We claim CONSISTENCY and FEASIBILITY: "the planner's recommendations are
confirmed by subsequent real procurement behaviour." We do NOT claim the
entry was CAUSED by the disruption, nor that procurement followed advice —
the entrant may have been onboarded for any reason. That is exactly why the
result is meaningful: an independent process arrived at the same vendor.

PROTOCOL (pre-registered choices, kept deliberately simple)
-----------------------------------------------------------
  t0        = lane's first late delivery (delay > 0). One eval per lane.
  incumbents= vendors already on the lane on/before t0 (not "new" by defn).
  pool(t0)  = vendors with >=1 shipment of this molecule ANYWHERE strictly
              before t0, minus incumbents.  (T2 onboarding candidates.)
  entrants  = vendors whose first-ever shipment on this lane is after t0.
  ranking   = planner rule from scms_spine.py: 0.6*capacity + 0.4*established
              (both computed on pre-t0 data only, normalised within pool).
  baselines = analytic random-from-pool  P(hit@k) = 1 - C(N-m,k)/C(N,k),
              popularity-only ranking, capacity-only ranking.
  metrics   = hit@1 / hit@3, STRICT (all entrant-lanes; unnameable entrants
              count as misses) and CONDITIONAL (entrant was in pool at t0).

Pure stdlib. Input: data/raw/SCMS_Delivery_History_Dataset.csv
Outputs: output/scms_backtest.csv, output/scms_backtest_report.txt
"""

import csv
import math
import os
import statistics
from collections import defaultdict
from isc_common import parse_date, clean, to_float   # shared SCMS parsing helpers (one definition)
from datetime import datetime

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCMS_IN = os.path.join(ROOT, "data", "raw", "SCMS_Delivery_History_Dataset.csv")
CSV_OUT = os.path.join(ROOT, "output", "scms_backtest.csv")
RPT_OUT = os.path.join(ROOT, "output", "scms_backtest_report.txt")

from isc_common import W_CAP, W_ESTAB     # planner rule — one definition for Stages 19/20/28

print("=" * 60)
print("  SCMS REVEALED-PREFERENCE BACKTEST (walk-forward)")
print("=" * 60)

if not os.path.exists(SCMS_IN):
    print(f"[ERROR] Missing input: {SCMS_IN}")
    raise SystemExit(1)


# ─────────────────────────────────────────────────────────────
# 1. Load shipments in delivery-time order
# ─────────────────────────────────────────────────────────────
recs = []
with open(SCMS_IN, encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        d = parse_date(r.get("Delivered to Client Date"))
        s = parse_date(r.get("Scheduled Delivery Date"))
        ven = clean(r.get("Vendor"))
        mol = clean(r.get("Molecule/Test Type"))
        cty = clean(r.get("Country"))
        if not (d and s and ven and mol and cty):
            continue
        qty = to_float(r.get("Line Item Quantity"))
        if qty is None:                   # malformed quantity: skip, never zero-fill
            continue
        recs.append({"date": d, "delay": (d - s).days,
                     "ven": ven, "mol": mol, "cty": cty, "qty": qty})
recs.sort(key=lambda x: x["date"])
print(f"\n  Shipments loaded: {len(recs):,}")

lanes    = defaultdict(list)     # (mol,cty) -> shipments in time order
mol_hist = defaultdict(list)     # mol -> shipments in time order (any country)
for x in recs:
    lanes[(x["mol"], x["cty"])].append(x)
    mol_hist[x["mol"]].append(x)


# ─────────────────────────────────────────────────────────────
# 2. Walk-forward evaluation per lane
# ─────────────────────────────────────────────────────────────
def rank(pool_stats, key):
    """Deterministic ranking: score desc, then vendor name for ties."""
    return [v for v, _ in sorted(pool_stats.items(), key=lambda kv: (-kv[1][key], kv[0]))]


def p_hit_random(N, m, k):
    """P(random k-subset of pool size N contains >=1 of m entrants)."""
    if N <= 0 or m <= 0:
        return 0.0
    k = min(k, N)
    return 1.0 - (math.comb(N - m, k) / math.comb(N, k))


rows = []
n_late_lanes = n_no_entrant = 0

for L, hist in lanes.items():
    mol, cty = L
    lates = [x for x in hist if x["delay"] > 0]
    if not lates:
        continue
    n_late_lanes += 1
    t0 = lates[0]["date"]

    first_on_lane = {}
    for x in hist:
        first_on_lane.setdefault(x["ven"], x["date"])
    incumbents = {v for v, fs in first_on_lane.items() if fs <= t0}
    entrants   = {v for v, fs in first_on_lane.items() if fs > t0}
    if not entrants:
        n_no_entrant += 1
        continue

    # Candidate pool + per-candidate stats from STRICTLY pre-t0 data only
    pre = [x for x in mol_hist[mol] if x["date"] < t0]
    pool_qty = defaultdict(list)
    for x in pre:
        if x["ven"] not in incumbents:
            pool_qty[x["ven"]].append(x["qty"])
    if pool_qty:
        stats = {}
        cap_hi = max(statistics.median(q) for q in pool_qty.values()) or 1.0
        est_hi = max(len(q) for q in pool_qty.values()) or 1.0
        for v, q in pool_qty.items():
            cap, est = statistics.median(q), len(q)
            stats[v] = {"cap": cap, "est": est,
                        "score": W_CAP * cap / cap_hi + W_ESTAB * est / est_hi}
        top_ours = rank(stats, "score")
        top_pop  = rank(stats, "est")
        top_cap  = rank(stats, "cap")
    else:
        stats, top_ours, top_pop, top_cap = {}, [], [], []

    nameable = entrants & set(stats)
    N, m = len(stats), len(nameable)

    def hits(ranking):
        return (int(any(v in entrants for v in ranking[:1])),
                int(any(v in entrants for v in ranking[:3])))

    h1o, h3o = hits(top_ours)
    h1p, h3p = hits(top_pop)
    h1c, h3c = hits(top_cap)

    rows.append({
        "molecule": mol, "country": cty, "t0": t0.date().isoformat(),
        "pool_size": N, "n_entrants": len(entrants),
        "entrants": " | ".join(sorted(entrants)),
        "nameable_at_t0": m,
        "top3_ours": " | ".join(top_ours[:3]),
        "hit1_ours": h1o, "hit3_ours": h3o,
        "hit1_pop": h1p, "hit3_pop": h3p,
        "hit1_cap": h1c, "hit3_cap": h3c,
        "p_hit1_rand": round(p_hit_random(N, m, 1), 4),
        "p_hit3_rand": round(p_hit_random(N, m, 3), 4),
    })

if not rows:
    print("[ERROR] No evaluable lanes found.")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────
# 3. Aggregate + write outputs
# ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

cond = [r for r in rows if r["nameable_at_t0"] > 0]

def agg(rs, key):
    return 100.0 * sum(r[key] for r in rs) / len(rs) if rs else 0.0

def agg_rand(rs, key):
    return 100.0 * sum(r[key] for r in rs) / len(rs) if rs else 0.0

pool_sizes = [r["pool_size"] for r in rows]

lines = [
    "SCMS REVEALED-PREFERENCE BACKTEST — REPORT",
    "=" * 55,
    "",
    "Walk-forward: at each lane's FIRST late delivery (t0), the planner",
    "builds its shortlist from data strictly before t0. Ground truth = a",
    "brand-new vendor that really entered the lane afterwards.",
    "",
    f"Lanes with >=1 late shipment          : {n_late_lanes}",
    f"  no new vendor ever entered          : {n_no_entrant}  (not evaluable)",
    f"  EVALUABLE (new vendor entered)      : {len(rows)}",
    f"  entrant nameable at t0 (in pool)    : {len(cond)}  "
    f"({100*len(cond)/len(rows):.0f}% — pool-containment rate)",
    f"  candidate pool size                 : median {statistics.median(pool_sizes):.0f}, "
    f"mean {statistics.mean(pool_sizes):.1f}",
    "",
    "── Hit rates: was the actual future vendor on the shortlist? ──",
    "",
    f"{'':<26}{'hit@1':>8}{'hit@3':>8}",
    "STRICT (all evaluable lanes; unnameable = miss)",
    f"{'  planner (0.6cap+0.4est)':<26}{agg(rows,'hit1_ours'):>7.1f}%{agg(rows,'hit3_ours'):>7.1f}%",
    f"{'  popularity-only':<26}{agg(rows,'hit1_pop'):>7.1f}%{agg(rows,'hit3_pop'):>7.1f}%",
    f"{'  capacity-only':<26}{agg(rows,'hit1_cap'):>7.1f}%{agg(rows,'hit3_cap'):>7.1f}%",
    f"{'  random-from-pool':<26}{agg_rand(rows,'p_hit1_rand'):>7.1f}%{agg_rand(rows,'p_hit3_rand'):>7.1f}%",
    "CONDITIONAL (entrant was in pool at t0)",
    f"{'  planner (0.6cap+0.4est)':<26}{agg(cond,'hit1_ours'):>7.1f}%{agg(cond,'hit3_ours'):>7.1f}%",
    f"{'  popularity-only':<26}{agg(cond,'hit1_pop'):>7.1f}%{agg(cond,'hit3_pop'):>7.1f}%",
    f"{'  capacity-only':<26}{agg(cond,'hit1_cap'):>7.1f}%{agg(cond,'hit3_cap'):>7.1f}%",
    f"{'  random-from-pool':<26}{agg_rand(cond,'p_hit1_rand'):>7.1f}%{agg_rand(cond,'p_hit3_rand'):>7.1f}%",
    "",
    "HONEST NOTES",
    "  * Claim is consistency/feasibility with real subsequent procurement,",
    "    NOT causation and NOT 'procurement followed the tool'.",
    "  * t0 = first late delivery (single pre-registered choice, no tuning).",
    "  * Scoring rule is the planner's existing 0.6/0.4 — not fitted here.",
    "  * Entrants outside the pool at t0 (new-to-molecule vendors) are",
    "    unnameable in principle; STRICT counts them as misses anyway.",
]
with open(RPT_OUT, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\n  Per-lane results -> {CSV_OUT}")
print(f"  Report           -> {RPT_OUT}")
print("\n  Backtest complete.")
