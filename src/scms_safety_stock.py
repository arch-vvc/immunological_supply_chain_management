"""
SCMS Safety-Stock Sizing — the priced answer for exposed lanes
==============================================================
WHY THIS EXISTS
---------------
The SCMS spine flags EXPOSED lanes: a real disruption hit and the network
contains no proven in-country alternate vendor (tier T2 = a vendor ships the
molecule elsewhere but would need onboarding; T5 = no vendor anywhere). For
those lanes "find a backup" is not an answer available today — the textbook
response is a sized inventory buffer. This module sizes and prices it from
the SAME real dataset. No synthetic inputs.

METHOD (standard, stated plainly)
---------------------------------
King's safety-stock formula under demand AND lead-time uncertainty:

    SS = z * sqrt( Lbar * sigma_d^2  +  Dbar^2 * sigma_L^2 )

  Dbar, sigma_d : lane's daily demand mean/std — from its own quarterly
                  shipment series (active span, zero quarters included),
                  disaggregated to daily under an iid assumption.
  Lbar, sigma_L : vendor lead time (PO Sent -> Delivered), real dates.
                  Source hierarchy: lane (n>=3) -> country (n>=5) -> global.
  z             : 1.65 (95% service level). Report also totals at 99%.

Priced at the lane's own median pack price (LineItemValue / Quantity, which
matches SCMS's Pack Price column), holding cost 25%/yr of buffer value.

HONEST CAVEATS (in the report too)
----------------------------------
* Demand on these lanes is lumpy/intermittent; King's formula assumes
  smoother demand, so treat sizes as planning magnitudes, not SKU truth.
* The buffers are large because demand is LUMPY and lead times are LONG
  (global mean 115d): the L*sigma_d^2 term dominates on nearly all lanes.
  That is not a modelling artifact — it IS the measured carrying cost of
  slow single-sourced international supply. Buffer size = price of no backup.
* Lanes with too little history to estimate variance are listed, not sized.

Inputs : data/raw/SCMS_Delivery_History_Dataset.csv
         output/scms_response_plan.csv        (exposed lanes)
Outputs: output/scms_safety_stock.csv
         output/scms_safety_stock_report.txt
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
PLAN_IN = os.path.join(ROOT, "output", "scms_response_plan.csv")
CSV_OUT = os.path.join(ROOT, "output", "scms_safety_stock.csv")
RPT_OUT = os.path.join(ROOT, "output", "scms_safety_stock_report.txt")

from isc_common import kings_safety_stock, Z_95, Z_99   # one implementation (Stages 22/23/26)
HOLDING_RATE   = 0.25          # annual holding cost as share of buffer value
DAYS_PER_QTR   = 91.31
MIN_SHIPMENTS  = 3             # to size a lane at all
MIN_QUARTERS   = 4             # active-span quarters incl. zeros

print("=" * 60)
print("  SCMS SAFETY-STOCK SIZING (exposed lanes, real inputs)")
print("=" * 60)

for p in (SCMS_IN, PLAN_IN):
    if not os.path.exists(p):
        print(f"[ERROR] Missing input: {p} (run src/scms_spine.py first)")
        raise SystemExit(1)


# ─────────────────────────────────────────────────────────────
# 1. Exposed lanes from the spine's response plan
# ─────────────────────────────────────────────────────────────
exposed = set()
with open(PLAN_IN, newline="") as f:
    for r in csv.DictReader(f):
        if r.get("action") in ("T2_ALT_VENDOR_OTHER_COUNTRY", "T5_SINGLE_SOURCED"):
            exposed.add((clean(r.get("molecule")), clean(r.get("country"))))
print(f"\n  Exposed lanes (T2/T5, from spine): {len(exposed)}")

# ─────────────────────────────────────────────────────────────
# 2. One pass over SCMS: demand, prices, lead times
# ─────────────────────────────────────────────────────────────
lane_ship   = defaultdict(list)   # lane -> [(date, qty, value)]
lane_lt     = defaultdict(list)   # lane -> [lead days]
cty_lt      = defaultdict(list)   # country -> [lead days]
global_lt   = []

with open(SCMS_IN, encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        d   = parse_date(r.get("Delivered to Client Date"))
        mol = clean(r.get("Molecule/Test Type"))
        cty = clean(r.get("Country"))
        if not (d and mol and cty):
            continue
        qty = to_float(r.get("Line Item Quantity"))
        val = to_float(r.get("Line Item Value"))     # None = unpriced shipment (kept for demand, excluded from price)
        if qty is None:                              # malformed quantity: skip, never zero-fill
            continue
        lane_ship[(mol, cty)].append((d, qty, val))
        po = parse_date(r.get("PO Sent to Vendor Date"))
        if po:
            lt = (d - po).days
            if 0 < lt < 1000:
                lane_lt[(mol, cty)].append(lt)
                cty_lt[cty].append(lt)
                global_lt.append(lt)

G_LT_MEAN, G_LT_STD = statistics.mean(global_lt), statistics.stdev(global_lt)
print(f"  Global lead time: mean {G_LT_MEAN:.0f}d  std {G_LT_STD:.0f}d  (n={len(global_lt):,})")


def lead_stats(lane):
    mol, cty = lane
    if len(lane_lt[lane]) >= 3:
        s = lane_lt[lane]
        return statistics.mean(s), (statistics.stdev(s) if len(s) >= 2 else G_LT_STD), "lane"
    if len(cty_lt[cty]) >= 5:
        s = cty_lt[cty]
        return statistics.mean(s), statistics.stdev(s), "country"
    return G_LT_MEAN, G_LT_STD, "global"


def qtr_index(d):
    return d.year * 4 + (d.month - 1) // 3


# ─────────────────────────────────────────────────────────────
# 3. Size every exposed lane
# ─────────────────────────────────────────────────────────────
rows, skipped = [], []
for lane in sorted(exposed):
    mol, cty = lane
    ships = sorted(lane_ship.get(lane, []))
    if len(ships) < MIN_SHIPMENTS:
        skipped.append((mol, cty, f"only {len(ships)} shipment(s)"))
        continue

    q0, q1 = qtr_index(ships[0][0]), qtr_index(ships[-1][0])
    n_qtrs = q1 - q0 + 1
    if n_qtrs < MIN_QUARTERS:
        skipped.append((mol, cty, f"history spans only {n_qtrs} quarter(s)"))
        continue

    per_qtr = defaultdict(float)
    for d, qty, _ in ships:
        per_qtr[qtr_index(d)] += qty
    series  = [per_qtr.get(q, 0.0) for q in range(q0, q1 + 1)]   # zeros included
    d_mean  = statistics.mean(series) / DAYS_PER_QTR             # packs/day
    d_std   = statistics.stdev(series) / math.sqrt(DAYS_PER_QTR)
    if d_mean <= 0:
        skipped.append((mol, cty, "zero demand rate"))
        continue

    lt_mean, lt_std, lt_src = lead_stats(lane)

    ss95 = kings_safety_stock(lt_mean, lt_std, d_mean, d_std, Z_95)
    ss99 = kings_safety_stock(lt_mean, lt_std, d_mean, d_std, Z_99)

    # pack price = per-shipment value/qty, lane median (matches Pack Price col)
    prices = [v / q for _, q, v in ships if q > 0 and v is not None and v > 0]
    price  = statistics.median(prices) if prices else 0.0

    span_years = max((ships[-1][0] - ships[0][0]).days, DAYS_PER_QTR) / 365.25
    annual_flow_value = sum(v for _, _, v in ships) / span_years

    rows.append({
        "molecule": mol, "country": cty,
        "n_shipments": len(ships), "history_quarters": n_qtrs,
        "daily_demand_packs": round(d_mean, 3),
        "daily_demand_std": round(d_std, 3),
        "lead_mean_days": round(lt_mean, 1),
        "lead_std_days": round(lt_std, 1),
        "lead_source": lt_src,
        "safety_stock_packs_95": round(ss95, 0),
        "days_of_cover_95": round(ss95 / d_mean, 0),
        "pack_price_usd": round(price, 2),
        "buffer_value_usd_95": round(ss95 * price, 0),
        "holding_cost_usd_per_yr": round(ss95 * price * HOLDING_RATE, 0),
        "annual_flow_value_usd": round(annual_flow_value, 0),
        "sizeable": True,
    })

# ─────────────────────────────────────────────────────────────
# 4. Outputs
# ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

tot_buf95  = sum(r["buffer_value_usd_95"] for r in rows)
tot_buf99  = tot_buf95 / Z_95 * Z_99          # value scales linearly with z
tot_hold   = sum(r["holding_cost_usd_per_yr"] for r in rows)
tot_flow   = sum(r["annual_flow_value_usd"] for r in rows)
med_cover  = statistics.median(r["days_of_cover_95"] for r in rows)
lt_dominated = sum(
    1 for r in rows
    if (r["daily_demand_packs"] ** 2) * (r["lead_std_days"] ** 2)
       > r["lead_mean_days"] * (r["daily_demand_std"] ** 2)
)
by_buf = sorted(rows, key=lambda r: -r["buffer_value_usd_95"])

lines = [
    "SCMS SAFETY-STOCK SIZING — EXPOSED LANES, REAL INPUTS",
    "=" * 55,
    "",
    "King's formula  SS = z*sqrt(L*sd^2 + D^2*sL^2)  with the lane's own",
    "demand history and real vendor lead times (PO->Delivered).",
    f"Service level 95% (z={Z_95}); holding rate {HOLDING_RATE*100:.0f}%/yr.",
    "",
    f"Exposed lanes (spine T2/T5)        : {len(exposed)}",
    f"  sized                            : {len(rows)}",
    f"  insufficient history (listed)    : {len(skipped)}",
    "",
    f"TOTAL buffer investment @95%       : ${tot_buf95:,.0f}",
    f"TOTAL buffer investment @99%       : ${tot_buf99:,.0f}",
    f"Annual holding cost @95%           : ${tot_hold:,.0f} /yr",
    f"Annual supply flow protected       : ${tot_flow:,.0f} /yr",
    f"  buffer as share of annual flow   : {tot_buf95 / tot_flow * 100 if tot_flow else 0:.1f}%",
    f"Median days of cover               : {med_cover:.0f} days",
    f"Lanes where LEAD-TIME variance dominates demand variance: "
    f"{lt_dominated}/{len(rows)}",
    "",
    "Top 10 buffers by investment:",
    f"  {'molecule':<32}{'country':<16}{'SS packs':>9}{'cover':>7}{'buffer $':>12}{'LT src':>8}",
]
for r in by_buf[:10]:
    lines.append(f"  {r['molecule'][:31]:<32}{r['country'][:15]:<16}"
                 f"{r['safety_stock_packs_95']:>9,.0f}{r['days_of_cover_95']:>6.0f}d"
                 f"{r['buffer_value_usd_95']:>12,.0f}{r['lead_source']:>8}")

if skipped:
    lines += ["", "Not sized (insufficient history):"]
    lines += [f"  {m[:40]} | {c}  — {why}" for m, c, why in skipped[:12]]
    if len(skipped) > 12:
        lines.append(f"  ... and {len(skipped) - 12} more")

lines += [
    "",
    "READ THIS BEFORE QUOTING",
    f"  * What drives the buffers: demand-variance x lead-time (L*sd^2) term",
    f"    dominates on {len(rows) - lt_dominated}/{len(rows)} lanes; pure lead-time variance (sigma_L="
    f"{G_LT_STD:.0f}d) on {lt_dominated}.",
    "    i.e. the cost comes from LUMPY demand flowing through LONG (mean",
    f"    {G_LT_MEAN:.0f}d) single-sourced pipelines — not a formula artifact.",
    "    Dual-sourcing (qualification shortlist) shortens and de-risks the",
    "    pipeline, attacking both terms; that business case is now in dollars.",
    "  * Demand here is lumpy; treat sizes as planning magnitudes.",
    "  * Buffer economics assume the lane keeps flowing at its historical",
    "    rate and value; prices are the lane's own median pack price.",
]
with open(RPT_OUT, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\n  Lane sizing -> {CSV_OUT}")
print(f"  Report      -> {RPT_OUT}")
print("\n  Safety-stock sizing complete.")
