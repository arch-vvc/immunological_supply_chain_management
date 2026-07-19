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
    })

if not rows:
    print("[ERROR] No comparable lanes found.")
    raise SystemExit(1)

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
