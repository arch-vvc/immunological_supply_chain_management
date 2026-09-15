"""
SCMS SPINE — Real, Unified No-Backup / Recovery Analysis
========================================================
Why this file exists
--------------------
The ARCOS pipeline has a real transaction NETWORK but no disruptions, so
recovery times had to be borrowed from a SEPARATE synthetic table
(disruption_processed.csv). Those two datasets share no entities, so any
"days saved" number crossed an illegitimate seam.

The SCMS Delivery History dataset fixes this: network structure AND real
disruptions live in the SAME rows.
  • network      : Vendor  →  (Molecule/Test Type)  →  destination Country
  • disruption   : a LATE shipment  (Delivered − Scheduled > 0)  — a real
                   event with a real duration, not a generated label
  • no-backup    : for a disrupted (molecule, country) lane, does ANOTHER
                   vendor already supply that molecule?  (a real, entity-
                   linked de-facto backup)
  • capacity     : vendor shipment volumes  (Line Item Quantity)

So every claim below rests on one coherent, real dataset.

HONEST FRAMING (read before quoting any number)
-----------------------------------------------
The delay of a late shipment is the disruption's COST. It is NOT "recovery
time after switching suppliers" — an alternate vendor helps the NEXT order,
not the one already in transit. So we do not claim "backup shortens this
shipment's delay." We report only what the data proves:
  1. whether a proven alternate vendor exists for the disrupted lane, and
  2. whether that alternate has the CAPACITY to absorb the disrupted volume.
Single-sourced lanes (no alternate vendor anywhere) are flagged as
structural exposure — the correct answer there is a buffer, not a reroute.

This mirrors the REVISED ARCOS methodology in app_copy.py, on real unified
data — so the two dashboards are directly comparable.

Pure stdlib on purpose (csv / collections / datetime): runs anywhere, and
makes the point that this layer is a lookup over a network we trust, not
another model.

Input:
    data/raw/SCMS_Delivery_History_Dataset.csv
Outputs:
    output/scms_response_plan.csv   one row per real disruption + its plan
    output/scms_spine_report.txt    summary (comparable to Stage-18 report)
"""

import csv
import os
import statistics
from collections import defaultdict
from isc_common import parse_date, to_float   # shared SCMS parsing helpers (one definition)
N_MALFORMED_QTY = [0]                            # rows skipped for an unparseable quantity
from datetime import datetime

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCMS_IN  = os.path.join(ROOT, "data", "raw", "SCMS_Delivery_History_Dataset.csv")
PLAN_OUT = os.path.join(ROOT, "output", "scms_response_plan.csv")
RPT_OUT  = os.path.join(ROOT, "output", "scms_spine_report.txt")

# candidate score for choosing among alternate vendors: capacity-first,
# then how established the vendor is on this molecule (shipment count).
from isc_common import W_CAP, W_ESTAB     # planner rule — one definition for Stages 19/20/28

print("=" * 55)
print("  SCMS SPINE — REAL UNIFIED NO-BACKUP / RECOVERY")
print("=" * 55)

if not os.path.exists(SCMS_IN):
    print(f"[ERROR] Missing input: {SCMS_IN}")
    raise SystemExit(1)


def severity(delay_days):
    """Real disruption magnitude bucketed 1-5 from the actual delay."""
    d = delay_days
    return 1 if d <= 7 else 2 if d <= 21 else 3 if d <= 45 else 4 if d <= 90 else 5


# ─────────────────────────────────────────────────────────────
# 1. Load rows, build the network maps (all real, same dataset)
# ─────────────────────────────────────────────────────────────
print("\n  Loading SCMS delivery history ...")
rows = []
with open(SCMS_IN, encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        rows.append(r)
print(f"  Rows: {len(rows):,}")

# lane = (molecule, country). Who supplies it, and with what typical volume?
lane_vendors      = defaultdict(set)                       # (mol,country) -> {vendors}
mol_vendors_any   = defaultdict(set)                       # molecule      -> {vendors anywhere}
vendor_mol_qty    = defaultdict(list)                      # (vendor,mol)  -> [shipment qtys]
lane_qty          = defaultdict(list)                      # (mol,country) -> [shipment qtys]

for r in rows:
    mol = r.get("Molecule/Test Type", "").strip()
    cty = r.get("Country", "").strip()
    ven = r.get("Vendor", "").strip()
    qty = to_float(r.get("Line Item Quantity"))
    if not (mol and cty and ven):
        continue
    if qty is None:                       # malformed quantity: skip, never zero-fill
        N_MALFORMED_QTY[0] += 1
        continue
    lane_vendors[(mol, cty)].add(ven)
    mol_vendors_any[mol].add(ven)
    vendor_mol_qty[(ven, mol)].append(qty)
    lane_qty[(mol, cty)].append(qty)

n_lanes  = len(lane_vendors)
n_single = sum(1 for v in lane_vendors.values() if len(v) == 1)
print(f"  Lanes (molecule x country): {n_lanes:,}   "
      f"single-vendor: {n_single:,} ({n_single / n_lanes * 100:.0f}%)")


def alt_vendor_capacity(vendor, mol):
    """Median shipment quantity this vendor ships of this molecule."""
    q = vendor_mol_qty.get((vendor, mol), [])
    return statistics.median(q) if q else 0.0


def pick_alternate(candidates, mol, disrupted_qty):
    """Choose the best alternate vendor: capacity to cover, then how
    established it is on the molecule. Returns (vendor, coverage)."""
    caps    = {v: alt_vendor_capacity(v, mol) for v in candidates}
    estab   = {v: len(vendor_mol_qty.get((v, mol), [])) for v in candidates}
    cap_hi  = max(caps.values()) or 1.0
    est_hi  = max(estab.values()) or 1.0
    best = max(candidates,
               key=lambda v: W_CAP * (caps[v] / cap_hi) + W_ESTAB * (estab[v] / est_hi))
    cov = min(1.0, caps[best] / disrupted_qty) if disrupted_qty > 0 else None
    return best, cov


# ─────────────────────────────────────────────────────────────
# 2. Walk every REAL disruption (late shipment) through the ladder
# ─────────────────────────────────────────────────────────────
print("  Planning responses for real disruptions (late shipments) ...\n")

plans          = []
tier_counts    = defaultdict(int)
total_disr_vol = 0.0

for r in rows:
    sched = parse_date(r.get("Scheduled Delivery Date"))
    deliv = parse_date(r.get("Delivered to Client Date"))
    if not (sched and deliv):
        continue
    delay = (deliv - sched).days
    if delay <= 0:
        continue                       # not a disruption — on time or early

    mol = r.get("Molecule/Test Type", "").strip()
    cty = r.get("Country", "").strip()
    ven = r.get("Vendor", "").strip()
    qty = to_float(r.get("Line Item Quantity"))
    if not (mol and cty and ven):
        continue
    if qty is None:                       # malformed quantity: skip, never zero-fill
        N_MALFORMED_QTY[0] += 1
        continue

    sev  = severity(delay)
    total_disr_vol += qty

    # Ladder over REAL alternate vendors:
    same_lane = lane_vendors[(mol, cty)] - {ven}     # already ship mol to THIS country
    any_where = mol_vendors_any[mol] - {ven}         # ship mol to some country

    if same_lane:
        tier = "T1_ALT_VENDOR_SAME_LANE"             # proven backup, same destination
        alt, cov = pick_alternate(same_lane, mol, qty)
    elif any_where:
        tier = "T2_ALT_VENDOR_OTHER_COUNTRY"         # proven product, new destination
        alt, cov = pick_alternate(any_where, mol, qty)
    else:
        tier = "T5_SINGLE_SOURCED"                   # structural exposure, needs buffer
        alt, cov = "", None

    tier_counts[tier] += 1
    plans.append({
        "vendor_disrupted":  ven,
        "molecule":          mol,
        "country":           cty,
        "shipment_qty":      round(qty, 1),
        "delay_days":        delay,             # REAL disruption duration
        "severity":          sev,
        "lane_vendor_count": len(lane_vendors[(mol, cty)]),
        "has_alternate":     tier != "T5_SINGLE_SOURCED",
        "action":            tier,
        "alternate_vendor":  alt,
        "volume_coverage":   round(cov, 3) if cov is not None else "",
        "route":             (f"{alt} -> {cty}" if alt else "(single-sourced: hold & buffer)"),
    })

if not plans:
    print("[ERROR] No dated late shipments found — check the SCMS date columns.")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────
# 3. Write plan + summary report (shape matches Stage-18 report)
# ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(PLAN_OUT), exist_ok=True)
with open(PLAN_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(plans[0].keys()))
    w.writeheader()
    w.writerows(plans)

n          = len(plans)
# Honest categories, avoiding the "100% resolved" overclaim:
#   RESOLVED  = T1 only  — a proven in-country alternate vendor (drop-in).
#   EXPOSED   = T2 + T5  — NO in-country alternate. T2 has a global vendor for
#               the molecule (onboarding candidate, not drop-in); T5 has none
#               anywhere. Both mean this country currently has no working
#               fallback → the correct answer is a buffer / qualification plan.
resolved   = [p for p in plans if p["action"] == "T1_ALT_VENDOR_SAME_LANE"]
onboarding = [p for p in plans if p["action"] == "T2_ALT_VENDOR_OTHER_COUNTRY"]
none_any   = [p for p in plans if p["action"] == "T5_SINGLE_SOURCED"]
exposed    = onboarding + none_any
covs       = [p["volume_coverage"] for p in resolved if p["volume_coverage"] != ""]
delays     = [p["delay_days"] for p in plans]
avg_cov    = statistics.mean(covs) if covs else 0.0
full_cov   = sum(1 for c in covs if c >= 0.999)
exposed_vol = sum(float(p["shipment_qty"]) for p in exposed)

lines = [
    "SCMS SPINE — REAL UNIFIED NO-BACKUP / RECOVERY ANALYSIS",
    "=" * 55,
    "",
    "One real dataset: network + disruptions in the same rows. No synthetic",
    "recovery table, no cross-dataset subtraction.",
    "",
    f"Real disruptions analysed (late shipments) : {n:,}",
    f"  delay days  median {statistics.median(delays):.0f}  "
    f"mean {statistics.mean(delays):.1f}  p90 {sorted(delays)[int(n * 0.9)]}",
    "",
    "Action mix (ladder over REAL alternate vendors):",
    "  T1  proven IN-COUNTRY alternate vendor (drop-in backup)",
    "  T2  vendor supplies molecule elsewhere, NOT this country (onboarding)",
    "  T5  no vendor supplies this molecule anywhere (true single-source)",
]
for tier in ("T1_ALT_VENDOR_SAME_LANE", "T2_ALT_VENDOR_OTHER_COUNTRY", "T5_SINGLE_SOURCED"):
    c = tier_counts.get(tier, 0)
    lines.append(f"    {tier:<28} {c:>6}  ({c / n * 100:5.1f}%)")

lines += [
    "",
    "── What the data proves (and what it doesn't) ──────────────────",
    f"RESOLVED  — proven in-country alternate      : {len(resolved):,}  "
    f"({len(resolved) / n * 100:.0f}%)",
    f"  avg volume coverage of alternate           : {avg_cov * 100:.0f}%  "
    f"({full_cov}/{len(covs)} fully covered)",
    f"EXPOSED   — no in-country alternate           : {len(exposed):,}  "
    f"({len(exposed) / n * 100:.0f}%)  -> buffer / qualify a vendor",
    f"    of which onboarding-candidate (T2)        : {len(onboarding):,}",
    f"    of which no vendor anywhere  (T5)         : {len(none_any):,}",
    f"  volume on exposed (no in-country) lanes     : "
    f"{(exposed_vol / total_disr_vol * 100) if total_disr_vol else 0:.0f}% "
    "of disrupted units",
    "",
    "HONEST CAVEAT: a shipment's delay is the disruption COST, not recovery-",
    "after-switching. An alternate vendor helps the NEXT order, not the one",
    "in transit. We therefore claim only alternate EXISTENCE and CAPACITY",
    "coverage — both directly observable — never a delay reduction for the",
    "disrupted shipment itself.",
]
with open(RPT_OUT, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\n  Plan saved   -> {PLAN_OUT}")
print(f"  Report saved -> {RPT_OUT}")
print("\n  SCMS spine complete.")
print(f"  Rows skipped for an unparseable quantity: {N_MALFORMED_QTY[0]} (never zero-filled)")
