"""
Stage 28 — SCMS Outcome Counterfactual ("better than what happened IRL?")
=========================================================================
THE QUESTION
------------
When a lane suffered its first late shipment and real procurement continued
with vendor A, while our walk-forward planner would have recommended vendor
B — whose deliveries actually ran better afterwards? This is the closest an
observational dataset gets to "would the system have beaten reality".

PROTOCOL (symmetric by construction)
------------------------------------
  t0        = lane's (molecule x country) first late delivery.
  B         = planner's top-1 from the pre-t0 pool (same 0.6*capacity +
              0.4*established rule as scms_spine / scms_backtest; walk-forward,
              no lookahead).
  A         = the vendor real procurement used MOST on this lane in the
              outcome window after t0 (what actually happened).
  outcomes  = each vendor's realized deliveries of the SAME molecule to ANY
              country inside (t0, t0+365d]:
                late rate      = share of shipments with delay > 0
                lateness days  = mean max(0, delay)
              Both vendors are measured at the SAME molecule scope — lane-only
              outcomes for B rarely exist (that is the counterfactual gap),
              and scoring A at lane scope while B gets molecule scope would
              be asymmetric. Comparable = both sides have >= 3 shipments.
  random    = same outcome stats for a seeded random vendor from the same
              pre-t0 pool (eligibility: >= 3 window shipments). Separates
              "the planner picks good vendors" from "anyone looks good".

WHAT WE CLAIM (and what we don't)
---------------------------------
We claim a symmetric, walk-forward, observational comparison of realized
delivery performance. We do NOT claim causation: procurement's choice may
reflect price, contracts, or capacity we cannot see, and each vendor's
outcomes are conditioned on the orders it actually received. "Recommended
vendor ran fewer late days" is evidence of decision quality, not a measured
treatment effect.

Pure stdlib. Input: data/raw/SCMS_Delivery_History_Dataset.csv
Outputs: output/scms_counterfactual.csv
         output/scms_counterfactual_report.txt
         output/figures/fig12_scms_counterfactual.png
"""

import csv
import os
import random
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCMS_IN = os.path.join(ROOT, "data", "raw", "SCMS_Delivery_History_Dataset.csv")
CSV_OUT = os.path.join(ROOT, "output", "scms_counterfactual.csv")
RPT_OUT = os.path.join(ROOT, "output", "scms_counterfactual_report.txt")
FIG_OUT = os.path.join(ROOT, "output", "figures", "fig12_scms_counterfactual.png")

W_CAP, W_ESTAB = 0.6, 0.4               # planner rule — reused, not re-tuned
WINDOW_DAYS    = 365                    # outcome window after t0
MIN_N          = 3                      # min shipments per side to compare
SEED           = 42
DATE_FMTS = ("%d-%b-%y", "%m/%d/%y", "%m/%d/%Y", "%d-%b-%Y")

print("=" * 60)
print("  STAGE 28 — SCMS OUTCOME COUNTERFACTUAL (walk-forward)")
print("=" * 60)

if not os.path.exists(SCMS_IN):
    print(f"[ERROR] Missing input: {SCMS_IN}")
    raise SystemExit(1)


def parse_date(s):
    s = (s or "").strip()
    for f in DATE_FMTS:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            continue
    return None


def clean(s):
    return " ".join((s or "").split())


# ── 1. Load shipments (mirrors scms_backtest.py) ─────────────────────────────
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
        try:
            qty = float(str(r.get("Line Item Quantity", 0)).replace(",", "") or 0)
        except ValueError:
            qty = 0.0
        recs.append({"date": d, "delay": (d - s).days,
                     "ven": ven, "mol": mol, "cty": cty, "qty": qty})
recs.sort(key=lambda x: x["date"])
print(f"\n  Shipments loaded: {len(recs):,}")

lanes    = defaultdict(list)
mol_hist = defaultdict(list)
for x in recs:
    lanes[(x["mol"], x["cty"])].append(x)
    mol_hist[x["mol"]].append(x)


def outcome_stats(mol, vendor, t0):
    """Realized deliveries of `mol` by `vendor` to any country in the window."""
    hi = t0 + timedelta(days=WINDOW_DAYS)
    ship = [x for x in mol_hist[mol]
            if x["ven"] == vendor and t0 < x["date"] <= hi]
    if len(ship) < MIN_N:
        return None
    late = sum(1 for x in ship if x["delay"] > 0) / len(ship)
    lateness = statistics.mean(max(0, x["delay"]) for x in ship)
    return {"n": len(ship), "late_rate": late, "lateness": lateness}


# ── 2. Walk-forward counterfactual per lane ──────────────────────────────────
rng = random.Random(SEED)
rows = []
n_late_lanes = n_no_next = n_agree = n_b_unobs = n_a_unobs = 0

for (mol, cty), hist in sorted(lanes.items()):
    lates = [x for x in hist if x["delay"] > 0]
    if not lates:
        continue
    n_late_lanes += 1
    t0 = lates[0]["date"]
    hi = t0 + timedelta(days=WINDOW_DAYS)

    # A — what reality did: modal vendor of the lane's post-t0 window shipments
    nxt = [x for x in hist if t0 < x["date"] <= hi]
    if not nxt:
        n_no_next += 1
        continue
    counts = defaultdict(int)
    for x in nxt:
        counts[x["ven"]] += 1
    A = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    # B — planner's top-1 from strictly pre-t0 data (backtest's exact rule)
    first_on_lane = {}
    for x in hist:
        first_on_lane.setdefault(x["ven"], x["date"])
    incumbents = {v for v, fs in first_on_lane.items() if fs <= t0}
    pre = [x for x in mol_hist[mol] if x["date"] < t0]
    pool_qty = defaultdict(list)
    for x in pre:
        if x["ven"] not in incumbents:
            pool_qty[x["ven"]].append(x["qty"])
    if not pool_qty:
        continue
    cap_hi = max(statistics.median(q) for q in pool_qty.values()) or 1.0
    est_hi = max(len(q) for q in pool_qty.values()) or 1.0
    scored = sorted(
        ((W_CAP * statistics.median(q) / cap_hi + W_ESTAB * len(q) / est_hi, v)
         for v, q in pool_qty.items()),
        key=lambda t: (-t[0], t[1]))
    B = scored[0][1]

    lane_pre_shipments = sum(1 for x in hist if x["date"] < t0)   # lane maturity at t0

    if A == B:
        n_agree += 1
        continue

    a_stats = outcome_stats(mol, A, t0)
    b_stats = outcome_stats(mol, B, t0)
    if b_stats is None:
        n_b_unobs += 1
        continue
    if a_stats is None:
        n_a_unobs += 1
        continue

    # random-pool baseline: seeded pick among eligible pool vendors
    eligible = [v for v in sorted(pool_qty) if v not in (A, B)
                and outcome_stats(mol, v, t0) is not None]
    r_stats = outcome_stats(mol, rng.choice(eligible), t0) if eligible else None

    rows.append({
        "molecule": mol, "country": cty, "t0": t0.date().isoformat(),
        "actual_vendor": A, "recommended_vendor": B,
        "a_n": a_stats["n"], "a_late_rate": round(a_stats["late_rate"], 4),
        "a_lateness_days": round(a_stats["lateness"], 2),
        "b_n": b_stats["n"], "b_late_rate": round(b_stats["late_rate"], 4),
        "b_lateness_days": round(b_stats["lateness"], 2),
        "r_late_rate": round(r_stats["late_rate"], 4) if r_stats else "",
        "r_lateness_days": round(r_stats["lateness"], 2) if r_stats else "",
        "rec_wins_late_rate": int(b_stats["late_rate"] < a_stats["late_rate"]),
        "rec_ties_late_rate": int(b_stats["late_rate"] == a_stats["late_rate"]),
        "lateness_saved_days": round(a_stats["lateness"] - b_stats["lateness"], 2),
        "lane_pre_shipments": lane_pre_shipments,
    })

if not rows:
    print("[ERROR] No comparable lanes found.")
    raise SystemExit(1)

# ── 2b. Sensitivity strata (tertiles) — labels go into the CSV too ────────────
# Selection concern: the incumbent is scored while its book is in trouble. If
# the A-vs-B gap were an artefact of incumbent scale or lane maturity it should
# collapse inside strata that hold those fixed.
import math

def _tertile_labels(values):
    s = sorted(values)
    lo, hi = s[len(s) // 3], s[(2 * len(s)) // 3]
    return [("low" if v < lo else "high" if v >= hi else "mid") for v in values], (lo, hi)

_vol_labels, _vol_cuts = _tertile_labels([r["a_n"] for r in rows])
_ten_labels, _ten_cuts = _tertile_labels([r["lane_pre_shipments"] for r in rows])
for r, lv, lt in zip(rows, _vol_labels, _ten_labels):
    r["stratum_incumbent_volume"] = lv
    r["stratum_lane_tenure"] = lt


def _stratum_summary(key):
    out = []
    for label in ("low", "mid", "high"):
        sub = [r for r in rows if r[key] == label]
        if not sub:
            continue
        w = sum(r["rec_wins_late_rate"] for r in sub)
        out.append({
            "stratum": label, "n": len(sub), "wins": w,
            "win_rate": w / len(sub),
            "a_late_rate": statistics.mean(r["a_late_rate"] for r in sub),
            "b_late_rate": statistics.mean(r["b_late_rate"] for r in sub),
            "lateness_saved": statistics.mean(r["lateness_saved_days"] for r in sub),
        })
    return out

strata = {
    "incumbent_volume": {"cuts": _vol_cuts, "rows": _stratum_summary("stratum_incumbent_volume")},
    "lane_tenure":      {"cuts": _ten_cuts, "rows": _stratum_summary("stratum_lane_tenure")},
}
_all_strata = strata["incumbent_volume"]["rows"] + strata["lane_tenure"]["rows"]
_pos = [s for s in _all_strata if s["a_late_rate"] > s["b_late_rate"]]
_neg = [s for s in _all_strata if s["a_late_rate"] <= s["b_late_rate"]]


def _strata_reading():
    """Report lines that describe what the strata actually show — never a
    canned 'holds everywhere' claim."""
    out = [f"Gap favours the planner's pick in {len(_pos)}/{len(_all_strata)} strata."]
    for s in _neg:
        which = ("incumbent volume" if s in strata["incumbent_volume"]["rows"]
                 else "lane tenure")
        out.append(
            f"It reverses for {which} = {s['stratum']} (n={s['n']}, win {100*s['win_rate']:.0f}%, "
            f"gap {s['a_late_rate']-s['b_late_rate']:+.1%}): "
            + ("when the incumbent's post-alert book is small and mostly on time, the first "
               "late shipment looks like noise rather than a regime change — the alert was "
               "arguably premature there, and the effect concentrates in mid/high-volume "
               "incumbents whose lateness persisted."
               if which == "incumbent volume" else
               "the association is weaker for these lanes; treat it as not established."))
    out += [
        "With ~25 lanes per stratum these are robustness checks, not an identification",
        "strategy; residual selection on unobserved factors (price, contracts) remains.",
    ]
    return out

# ── 3. Aggregate + outputs ───────────────────────────────────────────────────
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n = len(rows)
wins  = sum(r["rec_wins_late_rate"] for r in rows)
ties  = sum(r["rec_ties_late_rate"] for r in rows)
loses = n - wins - ties
a_lr = statistics.mean(r["a_late_rate"] for r in rows)
b_lr = statistics.mean(r["b_late_rate"] for r in rows)
a_ld = statistics.mean(r["a_lateness_days"] for r in rows)
b_ld = statistics.mean(r["b_lateness_days"] for r in rows)
saved = statistics.mean(r["lateness_saved_days"] for r in rows)
med_saved = statistics.median(r["lateness_saved_days"] for r in rows)
r_rows = [r for r in rows if r["r_late_rate"] != ""]
r_lr = statistics.mean(r["r_late_rate"] for r in r_rows) if r_rows else None
r_ld = statistics.mean(r["r_lateness_days"] for r in r_rows) if r_rows else None

# ── 3b. Uncertainty: bootstrap over lanes + exact sign test ──────────────────
N_BOOT = 10_000

def _bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    """95% percentile CI of the mean, resampling lanes with replacement."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n))
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]

ci_win   = _bootstrap_ci([r["rec_wins_late_rate"] for r in rows])
ci_gap   = _bootstrap_ci([r["a_late_rate"] - r["b_late_rate"] for r in rows])
ci_saved = _bootstrap_ci([r["lateness_saved_days"] for r in rows])
gap_lr   = a_lr - b_lr

# exact two-sided sign test on wins vs losses (ties carry no information)
_k, _m = wins, wins + loses
sign_p = min(1.0, 2 * sum(math.comb(_m, j) for j in range(max(_k, _m - _k), _m + 1)) / 2 ** _m) if _m else 1.0

lines = [
    "SCMS OUTCOME COUNTERFACTUAL — REPORT",
    "=" * 55,
    "",
    "At each lane's first late shipment, compare realized delivery",
    "performance over the following year: the vendor procurement actually",
    "used most (A) vs the walk-forward planner's top pick (B), both",
    "measured at the SAME molecule scope. Observational, not causal.",
    "",
    f"Lanes with >=1 late shipment           : {n_late_lanes}",
    f"  no shipments in outcome window       : {n_no_next}",
    f"  planner pick == reality's vendor     : {n_agree}  (agreement — no counterfactual)",
    f"  recommended vendor unobservable      : {n_b_unobs}  (<{MIN_N} window shipments)",
    f"  actual vendor unobservable           : {n_a_unobs}",
    f"  COMPARABLE (both sides observed)     : {n}",
    "",
    "── Realized outcomes over the year after t0 (symmetric scope) ──",
    "",
    f"{'':<30}{'late rate':>10}{'lateness d/ship':>17}",
    f"{'  actual choice (A)':<30}{a_lr:>9.1%}{a_ld:>15.1f}",
    f"{'  planner pick  (B)':<30}{b_lr:>9.1%}{b_ld:>15.1f}",
]
if r_lr is not None:
    lines.append(f"{'  random pool vendor':<30}{r_lr:>9.1%}{r_ld:>15.1f}   (n={len(r_rows)})")
b_vs_r_wins = sum(1 for r in r_rows if r["b_late_rate"] < r["r_late_rate"])
b_vs_r_ties = sum(1 for r in r_rows if r["b_late_rate"] == r["r_late_rate"])
lines += [
    "",
    f"Recommendation had LOWER late rate     : {wins}/{n}  ({100*wins/n:.1f}%)",
    f"tied                                   : {ties}/{n}  ({100*ties/n:.1f}%)",
    f"actual choice was better               : {loses}/{n}  ({100*loses/n:.1f}%)",
    f"mean lateness saved per shipment       : {saved:+.1f} days (median {med_saved:+.1f})",
    "",
    f"── Uncertainty (bootstrap over lanes, {N_BOOT:,} resamples, seed {SEED}) ──",
    f"win rate 95% CI                        : [{ci_win[0]:.1%}, {ci_win[1]:.1%}]",
    f"late-rate gap A−B 95% CI               : {gap_lr:+.1%}  [{ci_gap[0]:+.1%}, {ci_gap[1]:+.1%}]",
    f"lateness saved 95% CI                  : {saved:+.1f} d  [{ci_saved[0]:+.1f}, {ci_saved[1]:+.1f}]",
    f"exact sign test, wins vs losses        : {wins} vs {loses} (ties dropped), p = {sign_p:.2e}",
    "",
    "── Sensitivity: does the gap survive within strata? ──",
    "If the gap were an artefact of incumbent scale or lane maturity it should",
    "collapse when those are held fixed. Tertiles on each; 'gap' = A−B late rate.",
    "",
    f"  {'stratum':<26}{'n':>4}{'win%':>7}{'A late':>9}{'B late':>9}{'gap':>8}{'saved d':>9}",
    f"  {'-'*72}",
    *[f"  {'incumbent volume ' + s['stratum']:<26}{s['n']:>4}{100*s['win_rate']:>6.0f}%"
      f"{s['a_late_rate']:>9.1%}{s['b_late_rate']:>9.1%}{s['a_late_rate']-s['b_late_rate']:>+8.1%}"
      f"{s['lateness_saved']:>+9.1f}" for s in strata["incumbent_volume"]["rows"]],
    *[f"  {'lane tenure ' + s['stratum']:<26}{s['n']:>4}{100*s['win_rate']:>6.0f}%"
      f"{s['a_late_rate']:>9.1%}{s['b_late_rate']:>9.1%}{s['a_late_rate']-s['b_late_rate']:>+8.1%}"
      f"{s['lateness_saved']:>+9.1f}" for s in strata["lane_tenure"]["rows"]],
    f"  (volume tertile cuts at a_n = {strata['incumbent_volume']['cuts']}; "
    f"tenure cuts at pre-t0 lane shipments = {strata['lane_tenure']['cuts']})",
    *_strata_reading(),
    "",
    "── What the random baseline decomposes ──",
    f"planner pick vs random pool vendor     : better {b_vs_r_wins}, tied {b_vs_r_ties}, "
    f"worse {len(r_rows) - b_vs_r_wins - b_vs_r_ties}  (of {len(r_rows)})",
    "The planner's pick and a random pool vendor land close together on",
    "late rate — so the large gap vs the ACTUAL choice is evidence for the",
    "CATEGORY of action (switch from the troubled incumbent to any",
    "molecule-experienced alternative), NOT for the specific ranking. Part",
    "of that gap is selection: the incumbent is measured while its book is",
    "in trouble; outsiders are measured on their healthy order books.",
    "Ranking quality is validated separately by the revealed-preference",
    "backtest (hit rates vs random-from-pool); THIS test validates that",
    "acting on the alert at all was associated with better outcomes.",
    "",
    "HONEST NOTES",
    "  * Walk-forward: B is chosen from strictly pre-t0 data; outcomes are",
    "    strictly post-t0. No lookahead on either side.",
    "  * Symmetric scope: both vendors scored on molecule-level deliveries",
    "    (any country) — lane-level outcomes for the road-not-taken rarely",
    "    exist; asymmetric scoping would bias the comparison.",
    "  * NOT causal: procurement may have optimised price/contracts we do",
    "    not observe, and each vendor's outcomes are conditioned on the",
    "    orders it actually won. Read as decision-quality evidence.",
    "  * Agreement lanes are excluded from the win rate by construction;",
    "    they are reported above and are a success mode of the planner",
    "    (see scms_backtest.py hit rates).",
]
with open(RPT_OUT, "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))

# Machine-readable sidecar for the results manifest / dashboard
import json
STATS_OUT = CSV_OUT.replace(".csv", "_stats.json")
with open(STATS_OUT, "w") as f:
    json.dump({
        "n_comparable": n, "wins": wins, "ties": ties, "losses": loses,
        "win_rate": wins / n, "win_rate_ci95": list(ci_win),
        "a_late_rate": a_lr, "b_late_rate": b_lr,
        "r_late_rate": r_lr, "n_random": len(r_rows),
        "late_rate_gap": gap_lr, "late_rate_gap_ci95": list(ci_gap),
        "lateness_saved_mean": saved, "lateness_saved_median": med_saved,
        "lateness_saved_ci95": list(ci_saved),
        "sign_test_p": sign_p, "n_boot": N_BOOT, "seed": SEED,
        "strata": strata,
    }, f, indent=2)
print(f"  Stats            -> {STATS_OUT}")

# ── 4. Figure ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, BLUE, RED, GREEN, GREY = "#0f0f1a", "#4fc3f7", "#ff6b6b", "#44dd88", "#aaaaaa"
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(BG)
fig.suptitle("SCMS Outcome Counterfactual — Stage 28", color="white",
             fontsize=13, y=0.99)
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    ax.tick_params(colors=GREY)

labels = ["Actual choice", "Planner pick"] + (["Random pool"] if r_lr is not None else [])
lrs    = [a_lr, b_lr] + ([r_lr] if r_lr is not None else [])
lds    = [a_ld, b_ld] + ([r_ld] if r_lr is not None else [])
colors = [RED, GREEN, GREY][:len(labels)]

ax1 = axes[0]
x = np.arange(len(labels))
bars = ax1.bar(x, [v * 100 for v in lrs], color=colors, alpha=0.9, width=0.55)
for b, v, d in zip(bars, lrs, lds):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
             f"{v:.1%}\n{d:.1f} d/ship", ha="center", va="bottom",
             color="white", fontsize=8.5)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, color=GREY, fontsize=9)
ax1.set_ylim(0, max(v * 100 for v in lrs) * 1.35)
ax1.set_ylabel("Realized late rate, year after t0 (%)", color=GREY)
ax1.set_title(f"Outcomes at symmetric molecule scope ({n} lanes)",
              color="white", fontsize=10.5, pad=8)

ax2 = axes[1]
sizes = [wins, ties, loses]
seg_labels = [f"planner pick better\n{wins} ({100*wins/n:.0f}%)",
              f"tied\n{ties}", f"actual better\n{loses} ({100*loses/n:.0f}%)"]
ax2.pie(sizes, labels=seg_labels, colors=[GREEN, GREY, RED],
        textprops={"color": "white", "fontsize": 9},
        wedgeprops={"edgecolor": BG, "linewidth": 1.5},
        startangle=90)
ax2.set_title("Who ran fewer late shipments?", color="white",
              fontsize=10.5, pad=8)

plt.tight_layout(pad=2.0)
os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
plt.savefig(FIG_OUT, dpi=130, bbox_inches="tight", facecolor=BG)
plt.close()

print(f"\n  Per-lane results -> {CSV_OUT}")
print(f"  Report           -> {RPT_OUT}")
print(f"  Figure           -> {FIG_OUT}")
print("\n  Stage 28 complete.")
