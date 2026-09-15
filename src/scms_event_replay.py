"""
SCMS Real-Event Replay — does blind detection find a documented disaster?
=========================================================================
THE TEST
--------
The January 2010 Haiti earthquake is a real, dated, externally documented
supply-chain disruption. SCMS shipped health commodities into Haiti before
and after it. If our detection philosophy is sound, a detector that walks
the delivery data FORWARD IN TIME — knowing nothing about earthquakes —
should raise an alert on Haiti's lanes in early 2010 on service data alone.

DETECTOR (walk-forward, deliberately simple and untuned)
--------------------------------------------------------
Per country, per 3-month trailing window (stepped monthly):
    p0    = country's own historical late-rate BEFORE the window
            (Laplace-smoothed; strictly past data only — no lookahead)
    k, n  = late deliveries / total deliveries inside the window
    p-val = one-sided binomial tail  P(X >= k | n, p0)
    ALERT if p-val < 1e-3 and n >= MIN_WINDOW_N
Secondary signal (reported, not used for alerting): robust z of the window's
mean delay vs the country's own past monthly mean delays (median/MAD).

This mirrors the project's detection philosophy (robust deviation from an
entity's OWN history) applied to lane service series. Multiple-comparison
context is reported honestly: we state how many windows were tested and how
many alerts fired across ALL countries, so Haiti's alert can be judged
against the detector's overall alert budget — not cherry-picked.

Pure stdlib. Input: data/raw/SCMS_Delivery_History_Dataset.csv
Outputs: output/scms_event_replay.csv, output/scms_event_replay_report.txt
"""

import csv
import math
import os
import statistics
from collections import defaultdict
from isc_common import parse_date   # shared SCMS parsing helpers (one definition)
from datetime import datetime

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCMS_IN = os.path.join(ROOT, "data", "raw", "SCMS_Delivery_History_Dataset.csv")
CSV_OUT = os.path.join(ROOT, "output", "scms_event_replay.csv")
RPT_OUT = os.path.join(ROOT, "output", "scms_event_replay_report.txt")

WINDOW_MONTHS   = 3
ALPHA           = 1e-3
MIN_WINDOW_N    = 10      # shipments inside the window
MIN_PRIOR_N     = 30      # shipments before the window (baseline stability)
MIN_PRIOR_MONTHS = 12

print("=" * 60)
print("  SCMS REAL-EVENT REPLAY (blind walk-forward detection)")
print("=" * 60)

if not os.path.exists(SCMS_IN):
    print(f"[ERROR] Missing input: {SCMS_IN}")
    raise SystemExit(1)


def month_index(d):
    return d.year * 12 + (d.month - 1)


def month_label(mi):
    return f"{mi // 12:04d}-{mi % 12 + 1:02d}"


def binom_tail(k, n, p):
    """One-sided P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    q = 1.0 - p
    return sum(math.comb(n, i) * (p ** i) * (q ** (n - i)) for i in range(k, n + 1))


# ─────────────────────────────────────────────────────────────
# 1. Monthly service buckets per country
# ─────────────────────────────────────────────────────────────
buckets = defaultdict(lambda: defaultdict(lambda: {"n": 0, "late": 0, "delays": []}))
with open(SCMS_IN, encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        d = parse_date(r.get("Delivered to Client Date"))
        s = parse_date(r.get("Scheduled Delivery Date"))
        cty = " ".join((r.get("Country") or "").split())
        if not (d and s and cty):
            continue
        b = buckets[cty][month_index(d)]
        b["n"] += 1
        b["late"] += 1 if (d - s).days > 0 else 0
        b["delays"].append((d - s).days)

print(f"\n  Countries: {len(buckets)}")

# ─────────────────────────────────────────────────────────────
# 2. Walk forward month by month
# ─────────────────────────────────────────────────────────────
rows = []
n_tested = 0
for cty, months in buckets.items():
    idxs = sorted(months)
    first = idxs[0]
    for mi in range(first + MIN_PRIOR_MONTHS, idxs[-1] + 1):
        win = range(mi - WINDOW_MONTHS + 1, mi + 1)
        n_win    = sum(months[j]["n"]    for j in win if j in months)
        late_win = sum(months[j]["late"] for j in win if j in months)
        if n_win < MIN_WINDOW_N:
            continue
        prior = [j for j in idxs if j < mi - WINDOW_MONTHS + 1]
        n_pri    = sum(months[j]["n"]    for j in prior)
        late_pri = sum(months[j]["late"] for j in prior)
        if n_pri < MIN_PRIOR_N:
            continue
        p0   = (late_pri + 1) / (n_pri + 2)            # Laplace-smoothed
        pval = binom_tail(late_win, n_win, p0)
        n_tested += 1

        # secondary: robust z of window mean delay vs past monthly means
        past_means = [statistics.mean(months[j]["delays"]) for j in prior if months[j]["n"] > 0]
        win_delays = [x for j in win if j in months for x in months[j]["delays"]]
        z = ""
        if len(past_means) >= 6 and win_delays:
            med = statistics.median(past_means)
            mad = statistics.median([abs(m - med) for m in past_means]) or 1.0
            z = round((statistics.mean(win_delays) - med) / (1.4826 * mad), 2)

        rows.append({
            "country": cty,
            "window_end": month_label(mi),
            "n_window": n_win, "late_window": late_win,
            "late_rate": round(late_win / n_win, 3),
            "baseline_rate": round(p0, 4),
            "p_value": f"{pval:.2e}",
            "alert": int(pval < ALPHA),
            "z_delay": z,
        })

alerts = [r for r in rows if r["alert"]]

# group consecutive alert windows per country into episodes
episodes = []
by_cty = defaultdict(list)
for r in alerts:
    by_cty[r["country"]].append(r)
for cty, rs in by_cty.items():
    rs.sort(key=lambda r: r["window_end"])
    cur = [rs[0]]
    def _mi(lbl):
        y, m = lbl.split("-")
        return int(y) * 12 + int(m) - 1
    for r in rs[1:]:
        if _mi(r["window_end"]) - _mi(cur[-1]["window_end"]) <= 2:
            cur.append(r)
        else:
            episodes.append(cur)
            cur = [r]
    episodes.append(cur)

def peak(ep):
    return min(ep, key=lambda r: float(r["p_value"]))

episodes.sort(key=lambda ep: float(peak(ep)["p_value"]))

# ─────────────────────────────────────────────────────────────
# 3. Outputs
# ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

haiti_eps = [ep for ep in episodes if peak(ep)["country"] == "Haiti"]
haiti_2010 = [ep for ep in haiti_eps
              if any(r["window_end"].startswith("2010") for r in ep)]

lines = [
    "SCMS REAL-EVENT REPLAY — BLIND WALK-FORWARD DETECTION",
    "=" * 55,
    "",
    f"Windows tested (all countries, 2006-2015) : {n_tested:,}",
    f"Alerts fired (p < {ALPHA:g}, n >= {MIN_WINDOW_N})          : {len(alerts)}"
    f"  ->  {len(episodes)} distinct episodes",
    "",
    "Episodes ranked by peak significance:",
]
for i, ep in enumerate(episodes[:10], 1):
    pk = peak(ep)
    span = f"{ep[0]['window_end']}..{ep[-1]['window_end']}" if len(ep) > 1 else ep[0]["window_end"]
    mark = "  <-- HAITI EARTHQUAKE (Jan 12, 2010)" if (
        pk["country"] == "Haiti" and any(r["window_end"].startswith("2010") for r in ep)) else ""
    lines.append(f"  {i}. {pk['country']:<22} {span:<18} peak p={pk['p_value']}"
                 f"  late {pk['late_window']}/{pk['n_window']}"
                 f" (baseline {float(pk['baseline_rate'])*100:.0f}%){mark}")

lines += [
    "",
    "── The pre-registered question: is Haiti flagged in 2010? ──",
]
if haiti_2010:
    ep = haiti_2010[0]
    pk = peak(ep)
    rank_pos = episodes.index(ep) + 1
    lines += [
        f"YES — first alert window ends {ep[0]['window_end']}, "
        f"peak p={pk['p_value']} ({pk['late_window']}/{pk['n_window']} late vs "
        f"{float(pk['baseline_rate'])*100:.1f}% baseline).",
        f"Episode rank among all {len(episodes)} episodes: #{rank_pos}.",
        "The detector had NO knowledge of the earthquake — it saw only",
        "scheduled-vs-delivered dates, walking forward in time.",
    ]
else:
    lines += [
        "NO — Haiti 2010 was NOT flagged under the pre-registered settings.",
        "This is reported as a negative result; settings were not re-tuned",
        "to force a hit.",
    ]

lines += [
    "",
    "HONEST NOTES",
    f"  * ~{n_tested:,} windows tested at alpha={ALPHA:g} -> expect roughly "
    f"{n_tested * ALPHA:.0f} false alerts by chance;",
    "    judge the Haiti alert against the full episode list above.",
    "  * Detector settings (3-month window, alpha, minimum counts) were fixed",
    "    before looking at results and match the project's robust-deviation",
    "    philosophy; no per-event tuning.",
    "  * Other episodes are not necessarily false alarms — several may be",
    "    real but undocumented logistics problems; we simply cannot verify",
    "    them externally the way we can verify Haiti.",
]
with open(RPT_OUT, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\n  Windows -> {CSV_OUT}")
print(f"  Report  -> {RPT_OUT}")
print("\n  Event replay complete.")
