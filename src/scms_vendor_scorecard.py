"""
SCMS Vendor Scorecard + Qualification Shortlist — the counter-offer
===================================================================
WHY THIS EXISTS
---------------
The safety-stock module priced the cost of staying single-sourced on the
exposed lanes (~$18M/yr holding at 95% service). The business alternative is
to QUALIFY a second vendor. This module builds the procurement-standard
scorecard for that decision: for every exposed lane, rank the vendors that
already ship this molecule elsewhere (the T2 onboarding candidates) by
evidence in the SAME real dataset:

  RELIABILITY  on-time rate across ALL the vendor's shipments, shrunk toward
               the global mean for small samples (Laplace/Bayesian shrinkage,
               k=5) so a 1-shipment vendor can't post a perfect score.
  CAPACITY     the vendor's median shipment size of THIS molecule.
  EXPERIENCE   how many times it has shipped THIS molecule (log-scaled).

      score = 0.35*reliability + 0.35*capacity_norm + 0.30*experience_norm

Weights are a stated design choice, NOT fitted. They are validated by the
companion test (src/test_vendor_scorecard.py), which replays this ranking
through the walk-forward revealed-preference backtest and reports hit-rates
against the planner rule and random — a finding, not a tuned number.

THE $ LINK (what-if, labelled as such)
--------------------------------------
For each lane's top candidate we recompute King's safety stock with the
candidate's OWN observed lead-time profile (vendor→country if n>=3, else
vendor-global if n>=3). The delta prices what dual-sourcing could release
from the buffer. Lead times mix vendor and destination effects, so this is
a comparative indicator, not a promise — and negative deltas (candidate's
pipeline is slower) are shown, not hidden.

Inputs : data/raw/SCMS_Delivery_History_Dataset.csv
         output/scms_response_plan.csv     (exposed lanes)
         output/scms_safety_stock.csv      (current buffers, demand params)
Outputs: output/scms_vendor_scorecard.csv  (per-lane shortlists)
         output/scms_vendor_directory.csv  (per-vendor stats)
         output/scms_vendor_scorecard_report.txt
"""

import csv
import math
import os
import statistics
from collections import defaultdict
from isc_common import parse_date, clean, to_float   # shared SCMS parsing helpers (one definition)
from datetime import datetime

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCMS_IN  = os.path.join(ROOT, "data", "raw", "SCMS_Delivery_History_Dataset.csv")
PLAN_IN  = os.path.join(ROOT, "output", "scms_response_plan.csv")
SS_IN    = os.path.join(ROOT, "output", "scms_safety_stock.csv")
CSV_OUT  = os.path.join(ROOT, "output", "scms_vendor_scorecard.csv")
DIR_OUT  = os.path.join(ROOT, "output", "scms_vendor_directory.csv")
RPT_OUT  = os.path.join(ROOT, "output", "scms_vendor_scorecard_report.txt")

W_REL, W_CAP, W_EXP = 0.35, 0.35, 0.30
SHRINK_K = 5.0
Z_95     = 1.65


def load_shipments(path=SCMS_IN):
    """All dated shipments, time-sorted. Each: date, delay, ven, mol, cty,
    qty, val, lead (PO->Delivered days or None)."""
    recs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            d = parse_date(r.get("Delivered to Client Date"))
            s = parse_date(r.get("Scheduled Delivery Date"))
            ven, mol, cty = (clean(r.get("Vendor")), clean(r.get("Molecule/Test Type")),
                             clean(r.get("Country")))
            if not (d and s and ven and mol and cty):
                continue
            po = parse_date(r.get("PO Sent to Vendor Date"))
            lead = (d - po).days if po and 0 < (d - po).days < 1000 else None
            qty = to_float(r.get("Line Item Quantity"))
            if qty is None:                          # malformed quantity: skip, never zero-fill
                continue
            recs.append({"date": d, "delay": (d - s).days, "ven": ven, "mol": mol,
                         "cty": cty, "qty": qty,
                         "val": to_float(r.get("Line Item Value")), "lead": lead})
    recs.sort(key=lambda x: x["date"])
    return recs


def build_stats(recs, before=None):
    """Vendor-level and (vendor, molecule)-level stats. If `before` is set,
    only shipments STRICTLY before it count (walk-forward evaluation)."""
    v_raw   = defaultdict(lambda: {"n": 0, "ontime": 0, "late_delays": [],
                                   "leads": [], "mols": set(), "value": 0.0})
    vm_raw  = defaultdict(lambda: {"n": 0, "qtys": []})
    vc_lead = defaultdict(list)
    mol_vendors = defaultdict(set)
    tot_n = tot_on = 0
    for x in recs:
        if before is not None and x["date"] >= before:
            continue
        ven, mol = x["ven"], x["mol"]
        st = v_raw[ven]
        st["n"] += 1
        st["value"] += x["val"] or 0.0        # unpriced shipment adds no value
        st["mols"].add(mol)
        tot_n += 1
        if x["delay"] <= 0:
            st["ontime"] += 1
            tot_on += 1
        else:
            st["late_delays"].append(x["delay"])
        if x["lead"] is not None:
            st["leads"].append(x["lead"])
            vc_lead[(ven, x["cty"])].append(x["lead"])
        vm = vm_raw[(ven, mol)]
        vm["n"] += 1
        vm["qtys"].append(x["qty"])
        mol_vendors[mol].add(ven)

    p0 = (tot_on / tot_n) if tot_n else 0.0
    vendor = {}
    for ven, st in v_raw.items():
        vendor[ven] = {
            "n": st["n"],
            "otif_raw": st["ontime"] / st["n"],
            "reliability": (st["ontime"] + SHRINK_K * p0) / (st["n"] + SHRINK_K),
            "med_delay_late": (statistics.median(st["late_delays"])
                               if st["late_delays"] else 0.0),
            "lead_n": len(st["leads"]),
            "lead_mean": statistics.mean(st["leads"]) if st["leads"] else None,
            "lead_std": (statistics.stdev(st["leads"])
                         if len(st["leads"]) >= 3 else None),
            "n_molecules": len(st["mols"]),
            "total_value": st["value"],
        }
    vm = {k: {"n": s["n"], "med_qty": statistics.median(s["qtys"])}
          for k, s in vm_raw.items()}
    return {"vendor": vendor, "vm": vm, "vc_lead": dict(vc_lead),
            "mol_vendors": mol_vendors, "p0": p0}


def rank_candidates(mol, exclude, S):
    """Score every vendor shipping `mol` (minus `exclude`), best first."""
    pool = [v for v in S["mol_vendors"].get(mol, set()) if v not in exclude]
    if not pool:
        return []
    cap_hi = max(S["vm"][(v, mol)]["med_qty"] for v in pool) or 1.0
    exp_hi = max(math.log1p(S["vm"][(v, mol)]["n"]) for v in pool) or 1.0
    out = []
    for v in pool:
        vs, vm = S["vendor"][v], S["vm"][(v, mol)]
        score = (W_REL * vs["reliability"]
                 + W_CAP * vm["med_qty"] / cap_hi
                 + W_EXP * math.log1p(vm["n"]) / exp_hi)
        out.append({"vendor": v, "score": score, "otif_raw": vs["otif_raw"],
                    "reliability": vs["reliability"], "ship_n_mol": vm["n"],
                    "med_qty_mol": vm["med_qty"], "lead_mean": vs["lead_mean"],
                    "lead_std": vs["lead_std"], "lead_n": vs["lead_n"]})
    out.sort(key=lambda c: (-c["score"], c["vendor"]))
    return out


def candidate_lead(cand, cty, S):
    """Best available lead-time profile for a candidate serving `cty`:
    vendor→country (n>=3) beats vendor-global (n>=3). Returns
    (mean, std, source) or (None, None, '')."""
    vc = S["vc_lead"].get((cand["vendor"], cty), [])
    if len(vc) >= 3:
        return statistics.mean(vc), statistics.stdev(vc), "vendor→country"
    if cand["lead_n"] >= 3 and cand["lead_std"] is not None:
        return cand["lead_mean"], cand["lead_std"], "vendor-global"
    return None, None, ""


def king_ss(z, lt_mean, d_std, d_mean, lt_std):
    """Thin wrapper kept for this module's call sites/tests; the formula
    lives in isc_common.kings_safety_stock."""
    from isc_common import kings_safety_stock
    return kings_safety_stock(lt_mean, lt_std, d_mean, d_std, z)


def main():
    print("=" * 60)
    print("  SCMS VENDOR SCORECARD + QUALIFICATION SHORTLIST")
    print("=" * 60)
    for p in (SCMS_IN, PLAN_IN, SS_IN):
        if not os.path.exists(p):
            print(f"[ERROR] Missing input: {p} (run stages 19 and 22 first)")
            raise SystemExit(1)

    recs = load_shipments()
    S = build_stats(recs)
    print(f"\n  Shipments: {len(recs):,}   Vendors: {len(S['vendor'])}   "
          f"Global on-time rate p0: {S['p0']*100:.1f}%")

    # exposed lanes + their incumbent vendors
    exposed = {}
    with open(PLAN_IN, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("action") in ("T2_ALT_VENDOR_OTHER_COUNTRY", "T5_SINGLE_SOURCED"):
                lane = (clean(r.get("molecule")), clean(r.get("country")))
                exposed[lane] = r.get("action")
    lane_vendors = defaultdict(set)
    for x in recs:
        lane_vendors[(x["mol"], x["cty"])].add(x["ven"])

    # current buffers + demand params from the safety-stock stage
    ss_info = {}
    with open(SS_IN, newline="") as f:
        for r in csv.DictReader(f):
            ss_info[(r["molecule"], r["country"])] = {
                "d_mean": float(r["daily_demand_packs"]),
                "d_std": float(r["daily_demand_std"]),
                "price": float(r["pack_price_usd"]),
                "ss95": float(r["safety_stock_packs_95"]),
                "buffer": float(r["buffer_value_usd_95"]),
            }

    rows = []
    for (mol, cty), tier in sorted(exposed.items()):
        cands = rank_candidates(mol, lane_vendors[(mol, cty)], S)
        info = ss_info.get((mol, cty))
        row = {"molecule": mol, "country": cty, "tier": tier,
               "n_candidates": len(cands),
               "buffer_now_usd": round(info["buffer"]) if info else "",
               "note": "" if cands else "no vendor ships this molecule anywhere — buffer-only lane"}
        for i, c in enumerate(cands[:3], 1):
            row[f"r{i}_vendor"] = c["vendor"]
            row[f"r{i}_score"] = round(c["score"], 4)
            row[f"r{i}_otif"] = round(c["otif_raw"], 3)
            if i == 1:
                row["r1_reliability"] = round(c["reliability"], 3)
                row["r1_ship_n_mol"] = c["ship_n_mol"]
                row["r1_med_qty_mol"] = round(c["med_qty_mol"], 1)
                lm, ls, src = candidate_lead(c, cty, S)
                row["r1_lead_mean_days"] = round(lm, 1) if lm is not None else ""
                row["r1_lead_src"] = src
                if info and lm is not None:
                    ss_alt = king_ss(Z_95, lm, info["d_std"], info["d_mean"], ls)
                    row["r1_buffer_whatif_usd"] = round(ss_alt * info["price"])
                    row["r1_buffer_delta_usd"] = round(info["buffer"] - ss_alt * info["price"])
                else:
                    row["r1_buffer_whatif_usd"] = ""
                    row["r1_buffer_delta_usd"] = ""
        for i in range(len(cands) + 1, 4):          # blank unused rank columns
            row.setdefault(f"r{i}_vendor", "")
            row.setdefault(f"r{i}_score", "")
            row.setdefault(f"r{i}_otif", "")
        for k in ("r1_reliability", "r1_ship_n_mol", "r1_med_qty_mol",
                  "r1_lead_mean_days", "r1_lead_src",
                  "r1_buffer_whatif_usd", "r1_buffer_delta_usd"):
            row.setdefault(k, "")
        rows.append(row)

    FIELDS = ["molecule", "country", "tier", "n_candidates", "buffer_now_usd",
              "r1_vendor", "r1_score", "r1_otif", "r1_reliability",
              "r1_ship_n_mol", "r1_med_qty_mol", "r1_lead_mean_days",
              "r1_lead_src", "r1_buffer_whatif_usd", "r1_buffer_delta_usd",
              "r2_vendor", "r2_score", "r2_otif",
              "r3_vendor", "r3_score", "r3_otif", "note"]
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    with open(DIR_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vendor", "n_shipments", "otif_raw", "reliability_shrunk",
                    "med_delay_when_late_days", "n_molecules",
                    "lead_n", "lead_mean_days", "lead_std_days", "total_value_usd"])
        for ven, s in sorted(S["vendor"].items(), key=lambda kv: -kv[1]["n"]):
            w.writerow([ven, s["n"], round(s["otif_raw"], 3),
                        round(s["reliability"], 3), round(s["med_delay_late"], 1),
                        s["n_molecules"], s["lead_n"],
                        round(s["lead_mean"], 1) if s["lead_mean"] is not None else "",
                        round(s["lead_std"], 1) if s["lead_std"] is not None else "",
                        round(s["total_value"])])

    with_cand = [r for r in rows if r["n_candidates"] > 0]
    deltas = [r["r1_buffer_delta_usd"] for r in rows
              if isinstance(r["r1_buffer_delta_usd"], (int, float))]
    pos_release = sum(d for d in deltas if d > 0)
    neg_lanes = sum(1 for d in deltas if d <= 0)
    by_delta = sorted((r for r in rows if isinstance(r["r1_buffer_delta_usd"], (int, float))),
                      key=lambda r: -r["r1_buffer_delta_usd"])

    lines = [
        "SCMS VENDOR SCORECARD + QUALIFICATION SHORTLIST",
        "=" * 55,
        "",
        f"score = {W_REL}*reliability(shrunk k={SHRINK_K:.0f}) + {W_CAP}*capacity "
        f"+ {W_EXP}*experience(log)",
        "Weights are a stated design choice — validated (not fitted) via the",
        "walk-forward backtest in src/test_vendor_scorecard.py.",
        "",
        f"Exposed lanes                 : {len(rows)}",
        f"  with >=1 qualification candidate : {len(with_cand)}",
        f"  buffer-only (no vendor anywhere) : {len(rows) - len(with_cand)}",
        f"  median candidates per lane       : "
        f"{statistics.median([r['n_candidates'] for r in with_cand]) if with_cand else 0:.0f}",
        "",
        "── The counter-offer (what-if, comparative indicator) ──",
        f"Lanes with computable buffer what-if : {len(deltas)}",
        f"  potential buffer release (positive deltas): ${pos_release:,.0f}",
        f"  lanes where candidate pipeline is SLOWER  : {neg_lanes} (shown, not hidden)",
        "",
        "Top 10 lanes by potential buffer release:",
        f"  {'molecule':<30}{'country':<15}{'top candidate':<28}{'OTIF':>6}{'release $':>12}",
    ]
    for r in by_delta[:10]:
        lines.append(f"  {r['molecule'][:29]:<30}{r['country'][:14]:<15}"
                     f"{str(r['r1_vendor'])[:27]:<28}{r['r1_otif']:>6}"
                     f"{r['r1_buffer_delta_usd']:>12,.0f}")
    lines += [
        "",
        "HONEST NOTES",
        "  * Reliability is shrunk toward the global mean (k=5): a vendor with",
        "    1 perfect shipment cannot outrank one with 200 at 95%.",
        "  * The buffer what-if swaps the lane's lead-time profile for the",
        "    candidate's OWN observed profile (vendor→country preferred over",
        "    vendor-global). Lead times mix vendor and destination effects;",
        "    read as a comparative indicator, not a promise.",
        "  * Qualification cost/time is NOT modelled — this ranks WHO to",
        "    qualify, it does not claim qualification is free.",
    ]
    with open(RPT_OUT, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n  Shortlists -> {CSV_OUT}")
    print(f"  Directory  -> {DIR_OUT}")
    print(f"  Report     -> {RPT_OUT}")
    print("\n  Vendor scorecard complete.")


if __name__ == "__main__":
    main()
