"""
Stage 18 — Network-Grounded Response Planner (business layer)
=============================================================
Maps to: Adaptive Immunity — the immune system doesn't *classify* a threat
         and stop; it routes around it using the tissue it already has.

WHY THIS EXISTS
---------------
The disruption dataset's "no backup supplier" flag means no *contractually
listed* backup — it does not mean no alternative exists. This project builds
a full supply network graph, and that graph usually contains a working
alternative: a distributor that already serves the affected retailer, or one
that already carries the manufacturer's product, or one with spare capacity
shipping into the same state. Choosing among those is a business decision
over known infrastructure, not an ML prediction problem.

So instead of guessing an abstract strategy label (shown to be near-random
in the data), this planner walks a preference ladder over the network,
most-proven option first:

  T1 REROUTE_EXISTING      distributor linked to BOTH this manufacturer and
                           this retailer — zero new relationships needed
  T2 DEFACTO_ALT_SUPPLIER  distributor already delivering to this retailer —
                           proven last mile, one new upstream link
  T3 NEW_LASTMILE          distributor already carrying this manufacturer,
                           with a real shipping footprint in the retailer's
                           state — proximity evidenced by its own behaviour
  T4 INVENTORY_TRANSFER    low-risk distributor with spare capacity and
                           in-state footprint pre-positions stock
  T5 CUSTOMER_DELAY        the network genuinely has nothing — say so

Every recommendation is a named entity with a route and the graph evidence
for it (risk score, capacity, in-state volume share) — explainable line by
line, no model in the decision loop.

EXPECTED-RECOVERY ESTIMATES (and the headline metric)
-----------------------------------------------------
Each concrete action maps to the historical strategy vocabulary and gets an
expected recovery time from an actuarial lookup over the 100K disruption
events: mean full_recovery_days by (strategy, has_backup, severity).
When the ladder finds a T1/T2 supplier for a case with NO listed backup,
the case moves from "best no-backup strategy" to "Alternative Supplier with
backup" — the day difference is the measured value of grounding decisions
in the network. These are historical associations, not causal guarantees.

Pure stdlib on purpose (csv / collections). Two reasons:
  1. It runs even where the scientific Python stack is unavailable.
  2. It makes the point of the layer: this part is a lookup over a network
     we trust, not another model.

Inputs:
    data/processed/clean_chain.csv                (network + states + volumes)
    output/anomalies.csv                          (detected incidents)
    output/graph_risk_scores.csv                  (composite risk per entity)
    data/supplementary/disruption_processed.csv   (empirical recovery table)
Outputs:
    output/response_plan.csv          one concrete plan per anomaly
    output/response_plan_report.txt   summary + network-resolution metric
"""

import csv
import os
from collections import defaultdict

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAIN_IN  = os.path.join(ROOT, "data", "processed", "clean_chain.csv")
ANOM_IN   = os.path.join(ROOT, "output", "anomalies.csv")
RISK_IN   = os.path.join(ROOT, "output", "graph_risk_scores.csv")
DISR_IN   = os.path.join(ROOT, "data", "supplementary", "disruption_processed.csv")
PLAN_OUT  = os.path.join(ROOT, "output", "response_plan.csv")
RPT_OUT   = os.path.join(ROOT, "output", "response_plan_report.txt")

# Candidate score = proximity-weighted, risk-discounted, capacity-aware.
W_PROX, W_SAFE, W_CAP = 0.40, 0.35, 0.25

print("=" * 55)
print("  STAGE 18 — NETWORK-GROUNDED RESPONSE PLANNER")
print("=" * 55)

for p in (CHAIN_IN, ANOM_IN, RISK_IN, DISR_IN):
    if not os.path.exists(p):
        print(f"[ERROR] Missing input: {p}")
        raise SystemExit(1)

# ─────────────────────────────────────────────────────────────
# 1. Rebuild the network maps straight from the transaction log
#    (equivalent to the pickled graph, without heavy imports)
# ─────────────────────────────────────────────────────────────
print("\n  Building network maps from clean_chain.csv ...")

ret_suppliers  = defaultdict(set)    # retailer  -> {distributors that already deliver to it}
mfr_dists      = defaultdict(set)    # manufacturer -> {distributors that already carry it}
dist_retailers = defaultdict(set)    # distributor -> {retailers it serves}
dist_state_vol = defaultdict(lambda: defaultdict(float))  # distributor -> state -> shipped volume
dist_out_vol   = defaultdict(float)  # distributor -> total volume shipped to retailers
retailer_state = {}                  # retailer -> state

with open(CHAIN_IN, newline="") as f:
    for row in csv.DictReader(f):
        m, d, r = row["manufacturer"], row["distributor"], row["retailer"]
        s       = row.get("retailer_state", "")
        try:
            q = float(row.get("quantity", 0) or 0)
        except ValueError:
            q = 0.0
        ret_suppliers[r].add(d)
        mfr_dists[m].add(d)
        dist_retailers[d].add(r)
        dist_state_vol[d][s] += q
        dist_out_vol[d]      += q
        retailer_state.setdefault(r, s)

n_dists = len(dist_out_vol)
print(f"  Distributors: {n_dists}   Retailers: {len(ret_suppliers)}   "
      f"Manufacturers: {len(mfr_dists)}")

# ─────────────────────────────────────────────────────────────
# 2. Risk + spare-capacity per distributor (from Stage 2 output)
# ─────────────────────────────────────────────────────────────
risk_of  = {}
cap_raw  = {}
with open(RISK_IN, newline="") as f:
    for row in csv.DictReader(f):
        ent = row["entity"]
        try:
            risk_of[ent] = float(row.get("composite_risk", 0.5) or 0.5)
        except ValueError:
            risk_of[ent] = 0.5
        if row.get("type") == "distributor":
            try:
                ov = float(row.get("out_volume", 0) or 0)
                od = max(float(row.get("out_degree", 1) or 1), 1.0)
                cap_raw[ent] = ov / od          # volume per served retailer
            except ValueError:
                pass

cap_vals = list(cap_raw.values()) or [0.0]
cap_lo, cap_hi = min(cap_vals), max(cap_vals)
cap_span = (cap_hi - cap_lo) or 1.0
cap_norm = {d: (v - cap_lo) / cap_span for d, v in cap_raw.items()}
cap_median = sorted(cap_vals)[len(cap_vals) // 2]

def proximity(d, state):
    """Share of d's shipped volume that already goes into `state` —
    proximity evidenced by the network's own behaviour, not geocoding."""
    tot = dist_out_vol.get(d, 0.0)
    return (dist_state_vol[d].get(state, 0.0) / tot) if tot > 0 else 0.0

def score(d, state):
    return (W_PROX * proximity(d, state)
            + W_SAFE * (1.0 - risk_of.get(d, 0.5))
            + W_CAP  * cap_norm.get(d, 0.0))

# ─────────────────────────────────────────────────────────────
# 3. Actuarial recovery table from 100K historical disruptions:
#    mean full_recovery_days by (strategy, has_backup, severity)
# ─────────────────────────────────────────────────────────────
print("  Building empirical recovery table from disruption history ...")
_sum = defaultdict(float)
_cnt = defaultdict(int)
with open(DISR_IN, newline="") as f:
    for row in csv.DictReader(f):
        resp = row.get("response_type", "")
        bk   = 1 if str(row.get("has_backup_supplier", "")).strip().lower() in ("1", "true") else 0
        try:
            sev  = int(float(row.get("disruption_severity", 0)))
            days = float(row.get("full_recovery_days", ""))
        except ValueError:
            continue
        for key in ((resp, bk, sev), (resp, bk), (resp,), ()):
            _sum[key] += days
            _cnt[key] += 1

def mean_days(resp, backup, sev):
    """(strategy, backup, severity) → mean days, falling back to coarser keys."""
    for key in ((resp, backup, sev), (resp, backup), (resp,), ()):
        if _cnt.get(key):
            return _sum[key] / _cnt[key]
    return 0.0

NO_BACKUP_STRATS = ["Production Reroute", "Inventory Buffer",
                    "Customer Delay", "Combined Strategy"]

def baseline_days(sev):
    """ML-only world with no listed backup: best abstract strategy's mean.
    Alternative Supplier is excluded — without the graph there is no backup
    to invoke, which is exactly the dead end this planner removes."""
    return min(mean_days(s, 0, sev) for s in NO_BACKUP_STRATS)

# Action tier → (strategy vocabulary, backup status once network resolves it)
TIER_TO_STRAT = {
    "T1_REROUTE_EXISTING":     ("Alternative Supplier", 1),
    "T2_DEFACTO_ALT_SUPPLIER": ("Alternative Supplier", 1),
    "T3_NEW_LASTMILE":         ("Production Reroute",   0),
    "T4_INVENTORY_TRANSFER":   ("Inventory Buffer",     0),
    "T5_CUSTOMER_DELAY":       ("Customer Delay",       0),
}

# Two different clocks, reported separately rather than conflated into one
# number:
#   est_recovery_days   — full downstream BUSINESS recovery (inventory
#                          rebuilt, contracts normalised, demand backlog
#                          cleared). Looked up from 100K historical
#                          disruptions — genuinely long because severity-4/5
#                          events in that data are things like natural
#                          disasters and labor strikes, not paperwork.
#   activation_days      — how fast shipments can physically start moving
#                          through the found route. Derived directly from
#                          the tier definition, NOT the historical lookup:
#                          T1 needs zero new relationships (the hub already
#                          ships both the manufacturer's product and to this
#                          retailer), so it activates in days, not months.
#                          Each step up the ladder needs one more new
#                          relationship to establish, hence +2 days per tier.
# This is a labelled heuristic, not a fitted model — it exists so the
# network's actual advantage (speed) doesn't get hidden inside a historical
# average that was never measuring "time to find a name" in the first place.
TIER_ACTIVATION_DAYS = {
    "T1_REROUTE_EXISTING":     1.5,   # redirect existing shipments only
    "T2_DEFACTO_ALT_SUPPLIER": 3.0,   # one new upstream link to establish
    "T3_NEW_LASTMILE":         5.0,   # new last-mile relationship
    "T4_INVENTORY_TRANSFER":   4.0,   # inter-distributor transfer, no new retailer link
    "T5_CUSTOMER_DELAY":       None,  # no route found — nothing to activate
}

# ─────────────────────────────────────────────────────────────
# 4. Walk the ladder for every detected anomaly
# ─────────────────────────────────────────────────────────────
print("  Planning responses for detected anomalies ...\n")

plans = []
tier_counts   = defaultdict(int)
nb_total      = 0          # anomalies with NO listed backup (concentration-flagged)
nb_resolved   = 0          # ... of those, resolved to a proven supplier (T1/T2)
nb_days_saved = []

with open(ANOM_IN, newline="") as f:
    for row in csv.DictReader(f):
        M  = row["manufacturer"]
        D0 = row["distributor"]
        R  = row["retailer"]
        S  = row.get("retailer_state") or retailer_state.get(R, "")
        no_listed_backup = str(row.get("flag_concentration", "")).strip().lower() == "true"
        try:
            sev = max(1, min(5, int(float(row.get("anomaly_score", 2)))))
        except ValueError:
            sev = 2

        # Tier 1: already linked to BOTH the manufacturer and the retailer
        t1 = [d for d in (mfr_dists[M] & ret_suppliers[R]) if d != D0]
        # Tier 2: already delivers to this retailer
        t2 = [d for d in ret_suppliers[R] if d != D0]
        # Tier 3: already carries this manufacturer + real footprint in state S
        t3 = [d for d in mfr_dists[M] if d != D0 and proximity(d, S) > 0]
        # Tier 4: not serving R yet, but in-state footprint + above-median capacity
        t4 = [d for d in dist_out_vol
              if d != D0 and R not in dist_retailers[d]
              and proximity(d, S) > 0 and cap_raw.get(d, 0.0) >= cap_median]

        for tier, cands in (("T1_REROUTE_EXISTING", t1),
                            ("T2_DEFACTO_ALT_SUPPLIER", t2),
                            ("T3_NEW_LASTMILE", t3),
                            ("T4_INVENTORY_TRANSFER", t4)):
            if cands:
                best   = max(cands, key=lambda d: score(d, S))
                action = tier
                break
        else:
            best, action = None, "T5_CUSTOMER_DELAY"

        strat, backup = TIER_TO_STRAT[action]
        grounded = mean_days(strat, backup, sev)
        basel    = baseline_days(sev)
        # Days saved is only claimed where the network genuinely upgraded the
        # case: no listed backup, but a proven supplier found (T1/T2).
        saved = round(basel - grounded, 1) if (no_listed_backup and backup == 1) else 0.0

        tier_counts[action] += 1
        if no_listed_backup:
            nb_total += 1
            if backup == 1:
                nb_resolved += 1
                nb_days_saved.append(basel - grounded)

        plans.append({
            "manufacturer":       M,
            "disrupted_node":     D0,
            "retailer":           R,
            "retailer_state":     S,
            "severity":           sev,
            "no_listed_backup":   no_listed_backup,
            "action":             action,
            "entity":             best or "",
            "route":              f"{M} -> {best} -> {R}" if best else "(hold & communicate)",
            "entity_risk":        round(risk_of.get(best, 0.5), 4) if best else "",
            "entity_capacity":    round(cap_norm.get(best, 0.0), 4) if best else "",
            "entity_state_share": round(proximity(best, S), 4) if best else "",
            "plan_score":         round(score(best, S), 4) if best else "",
            "activation_days":    TIER_ACTIVATION_DAYS[action],
            "est_recovery_days":  round(grounded, 1),
            "baseline_days_no_graph": round(basel, 1),
            "days_saved_vs_no_graph": saved,
        })

if not plans:
    # Seen in practice when anomalies.csv is an iCloud dataless placeholder:
    # open() succeeds but yields zero rows. Fail loudly instead of writing
    # an empty plan.
    print("[ERROR] anomalies.csv yielded 0 rows — file may be empty or an")
    print("        un-materialised iCloud placeholder. Re-run Stage 3, or read")
    print("        the file once (e.g. `wc -l output/anomalies.csv`) and retry.")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────
# 5. Write plan + summary report
# ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(PLAN_OUT), exist_ok=True)
with open(PLAN_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(plans[0].keys()))
    w.writeheader()
    w.writerows(plans)

n = len(plans)
avg_saved = (sum(nb_days_saved) / len(nb_days_saved)) if nb_days_saved else 0.0
resolved_pct = (nb_resolved / nb_total * 100) if nb_total else 0.0

lines = [
    "STAGE 18 — NETWORK-GROUNDED RESPONSE PLANNER",
    "=" * 55,
    "",
    f"Anomalies planned            : {n:,}",
    "",
    "Action mix (preference ladder, most-proven first):",
]
for tier in ("T1_REROUTE_EXISTING", "T2_DEFACTO_ALT_SUPPLIER", "T3_NEW_LASTMILE",
             "T4_INVENTORY_TRANSFER", "T5_CUSTOMER_DELAY"):
    c = tier_counts.get(tier, 0)
    lines.append(f"  {tier:<26} {c:>6}  ({c/n*100:5.1f}%)")

resolved_plans = [p for p in plans if p["no_listed_backup"] and p["days_saved_vs_no_graph"]]
avg_activation = (sum(p["activation_days"] for p in resolved_plans) / len(resolved_plans)
                  if resolved_plans else 0.0)

lines += [
    "",
    "── Headline: two different clocks, not one ──────────────────────",
    "'Recovery days' historically means full downstream business recovery",
    "(inventory rebuilt, contracts normalised) — genuinely long because the",
    "training data's severe cases are things like labor strikes and natural",
    "disasters. That number barely moves just because a name was found.",
    "'Activation days' is a different, much smaller number: how fast",
    "shipments can physically start moving through the found route. THAT is",
    "where the network's advantage actually shows up.",
    "",
    f"Anomalies with NO listed backup supplier : {nb_total:,}",
    f"  resolved to a PROVEN supplier (T1/T2)  : {nb_resolved:,}  ({resolved_pct:.1f}%)",
    f"  avg reroute activation time            : {avg_activation:.1f} days"
    if resolved_plans else "  avg reroute activation time            : n/a",
    f"  avg full recovery: graph-grounded      : "
    f"{(sum(p['est_recovery_days'] for p in resolved_plans)/len(resolved_plans)):.1f} days"
    if resolved_plans else "  avg full recovery: graph-grounded      : n/a",
    f"  avg full recovery: ML-only baseline    : "
    f"{(sum(p['baseline_days_no_graph'] for p in resolved_plans)/len(resolved_plans)):.1f} days"
    if resolved_plans else "  avg full recovery: ML-only baseline    : n/a",
    f"  mean full-recovery days saved per case : {avg_saved:.1f} days",
    "",
    "Method: 'no backup supplier' in the disruption data means no LISTED",
    "backup. The supply graph usually contains a de-facto one — a distributor",
    "already serving the same retailer or carrying the same manufacturer.",
    "The planner walks a preference ladder (reroute via existing hub → " ,
    "de-facto alternate supplier → new last-mile → inventory transfer →",
    "customer delay), scoring candidates by in-state volume share (proximity",
    "evidenced by the network's own behaviour), composite risk, and spare",
    "capacity. Recovery estimates come from an actuarial lookup over 100K",
    "historical disruptions (strategy x backup x severity means) — they are",
    "historical associations, not causal guarantees.",
    "",
    "Note: on the synthetic ARCOS graph (20 well-connected distributors),",
    "T1 resolves nearly everything — the ladder's lower tiers matter on",
    "sparser real networks (e.g. DataCo regions).",
]
with open(RPT_OUT, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines[:20]))
print(f"\n  Plan saved   → {PLAN_OUT}")
print(f"  Report saved → {RPT_OUT}")
print("\n  Stage 18 complete.")
