"""
SCMS Event-Reaction Harness — watch the model respond to a shock
================================================================
Two ways to stress the system against reality, both on the SAME real SCMS
data and reusing the SAME response logic as the rest of the pipeline
(imported from scms_vendor_scorecard):

  REPLAY   — a documented historical event (country + date window). We replay
             the ACTUAL shipments, fire the detector, and — using only data
             from BEFORE the event (walk-forward, no hindsight) — run the
             response ladder for every disrupted lane. Then compare to what
             really happened.

  KNOCKOUT — a counterfactual: remove a vendor from the network TODAY and
             watch which lanes flip from covered to exposed, how much volume
             and value is suddenly at risk, and what the system recommends.

Both emit a step-by-step TRACE:  event -> detection -> per-lane reaction
-> recommendation -> dollars.  Presets are precomputed to JSON for the
dashboard; any scenario can also be run from the CLI:

    python3 src/scms_event_harness.py replay  --country Haiti --start 2010-01 --end 2010-12
    python3 src/scms_event_harness.py knockout --vendor "Aurobindo Pharma Limited"
    python3 src/scms_event_harness.py                       # rebuild all presets

Outputs: output/event_harness.json        (all presets, for the app)
         output/event_harness_report.txt  (human-readable traces)
"""

import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scms_vendor_scorecard as sc          # reuse the real response logic

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_OUT = os.path.join(ROOT, "output", "event_harness.json")
RPT_OUT  = os.path.join(ROOT, "output", "event_harness_report.txt")

Z_95, HOLDING, DAYS_Q = 1.65, 0.25, 91.31

# Documented events → detector already confirms these fire (Stage 21).
REPLAY_PRESETS = [
    {"label": "Haiti earthquake (Jan 2010)", "country": "Haiti",
     "start": "2010-01", "end": "2010-12",
     "documented": "M7.0 quake destroyed Port-au-Prince port; aid surged through a broken gateway."},
    {"label": "Cote d'Ivoire post-election crisis (2010-11)", "country": "Côte d'Ivoire",
     "start": "2010-12", "end": "2011-08",
     "documented": "Disputed election -> sanctions, Jan-2011 export ban, civil conflict to Apr-2011."},
    {"label": "South Africa strike wave (2010)", "country": "South Africa",
     "start": "2010-05", "end": "2011-02",
     "documented": "Transnet ports/rail strike (May) + 1M-worker public-sector strike shut clinics (Aug)."},
]


# ─────────────────────────────────────────────────────────────
# Shared prep: network maps + lead-time stats from the real data
# ─────────────────────────────────────────────────────────────
def build_context():
    recs = sc.load_shipments()
    lane_vendors = defaultdict(set)      # (mol,cty) -> {vendors ever}
    mol_vendors  = defaultdict(set)      # mol       -> {vendors ever}
    lane_ships   = defaultdict(list)     # (mol,cty) -> [(date,qty,val)]
    cty_lead     = defaultdict(list)
    all_lead     = []
    for x in recs:
        lane = (x["mol"], x["cty"])
        lane_vendors[lane].add(x["ven"])
        mol_vendors[x["mol"]].add(x["ven"])
        lane_ships[lane].append((x["date"], x["qty"], x["val"]))
        if x["lead"] is not None:
            cty_lead[x["cty"]].append(x["lead"])
            all_lead.append(x["lead"])
    return {"recs": recs, "lane_vendors": lane_vendors, "mol_vendors": mol_vendors,
            "lane_ships": lane_ships, "cty_lead": cty_lead,
            "g_lead_mean": statistics.mean(all_lead),
            "g_lead_std": statistics.stdev(all_lead)}


def month_key(d):
    return d.year * 12 + d.month - 1


def parse_month(s):
    y, m = s.split("-")
    return int(y) * 12 + int(m) - 1


def binom_tail(k, n, p):
    if k <= 0:
        return 1.0
    q = 1.0 - p
    return sum(math.comb(n, i) * p ** i * q ** (n - i) for i in range(k, n + 1))


def lane_demand(ships):
    """(daily mean, daily std, median pack price) from a lane's shipments,
    or None if too little history to estimate variance."""
    ships = sorted(ships)
    if len(ships) < 3:
        return None
    q = lambda d: d.year * 4 + (d.month - 1) // 3
    q0, q1 = q(ships[0][0]), q(ships[-1][0])
    if q1 - q0 + 1 < 4:
        return None
    per_q = defaultdict(float)
    for d, qty, _ in ships:
        per_q[q(d)] += qty
    series = [per_q.get(k, 0.0) for k in range(q0, q1 + 1)]
    d_mean = statistics.mean(series) / DAYS_Q
    d_std  = statistics.stdev(series) / math.sqrt(DAYS_Q)
    prices = [v / qty for _, qty, v in ships if qty > 0 and v > 0]
    if d_mean <= 0 or not prices:
        return None
    return d_mean, d_std, statistics.median(prices)


def size_buffer(ctx, lane):
    dem = lane_demand(ctx["lane_ships"].get(lane, []))
    if dem is None:
        return None
    d_mean, d_std, price = dem
    _, cty = lane
    lead = ctx["cty_lead"].get(cty, [])
    lt_mean = statistics.mean(lead) if len(lead) >= 5 else ctx["g_lead_mean"]
    lt_std  = statistics.stdev(lead) if len(lead) >= 5 else ctx["g_lead_std"]
    ss = sc.king_ss(Z_95, lt_mean, d_std, d_mean, lt_std)
    return {"safety_stock_packs": round(ss),
            "days_of_cover": round(ss / d_mean) if d_mean else 0,
            "buffer_usd": round(ss * price),
            "holding_usd_per_yr": round(ss * price * HOLDING)}


def best_incountry(alts, mol, S):
    """Pick the strongest already-on-lane alternate (reliability x capacity)."""
    def key(v):
        rel = S["vendor"].get(v, {}).get("reliability", 0.0)
        cap = S["vm"].get((v, mol), {}).get("med_qty", 0.0)
        return rel * cap
    return max(alts, key=key)


# ─────────────────────────────────────────────────────────────
# REPLAY — a documented event, walk-forward
# ─────────────────────────────────────────────────────────────
def replay(ctx, country, start, end, label="", documented=""):
    recs = ctx["recs"]
    m0, m1 = parse_month(start), parse_month(end)
    before = datetime(m0 // 12, m0 % 12 + 1, 1)          # first day of window

    in_win  = [x for x in recs if x["cty"] == country and m0 <= month_key(x["date"]) <= m1]
    pre     = [x for x in recs if x["cty"] == country and month_key(x["date"]) < m0]

    # detection: window late-rate vs the country's own pre-window baseline
    n_win  = len(in_win)
    late_w = sum(1 for x in in_win if x["delay"] > 0)
    n_pre  = len(pre)
    late_p = sum(1 for x in pre if x["delay"] > 0)
    p0     = (late_p + 1) / (n_pre + 2)
    pval   = binom_tail(late_w, n_win, p0) if n_win else 1.0
    detection = {
        "n_window": n_win, "late_window": late_w,
        "late_rate": round(late_w / n_win, 3) if n_win else 0.0,
        "baseline_rate": round(p0, 3),
        "p_value": f"{pval:.2e}",
        "fired": bool(n_win >= 10 and pval < 1e-3),
    }

    # response, using ONLY pre-window knowledge (walk-forward)
    S = sc.build_stats(recs, before=before)
    lane_vendors_before = defaultdict(set)
    for x in pre:
        lane_vendors_before[(x["mol"], x["cty"])].add(x["ven"])

    disrupted = defaultdict(set)                           # lane -> {vendors late in window}
    for x in in_win:
        if x["delay"] > 0:
            disrupted[(x["mol"], x["cty"])].add(x["ven"])

    lanes, n_res, n_exp, buf_total = [], 0, 0, 0
    for lane, bad_vendors in sorted(disrupted.items()):
        mol, cty = lane
        in_country = lane_vendors_before[lane] - bad_vendors     # proven, on-lane, not disrupted
        entry = {"molecule": mol, "disrupted_vendor": " / ".join(sorted(bad_vendors))}
        if in_country:
            alt = best_incountry(in_country, mol, S)
            lane_qtys = [q for _, q, _ in ctx["lane_ships"][lane]]
            typ = statistics.median(lane_qtys) if lane_qtys else 0.0
            cap = S["vm"].get((alt, mol), {}).get("med_qty", 0.0)
            entry.update(tier="T1_IN_COUNTRY_ALTERNATE", action="reroute",
                         recommendation=f"Reroute via {alt}",
                         alternate=alt,
                         coverage=round(min(1.0, cap / typ), 2) if typ else None)
            n_res += 1
        else:
            cands = sc.rank_candidates(mol, lane_vendors_before[lane] | bad_vendors, S)
            buf = size_buffer(ctx, lane)
            if buf:
                buf_total += buf["buffer_usd"]
            if cands:
                entry.update(tier="T2_QUALIFY_VENDOR", action="buffer+qualify",
                             recommendation=f"Buffer now, qualify {cands[0]['vendor']}",
                             shortlist=[c["vendor"] for c in cands[:3]], buffer=buf)
            else:
                entry.update(tier="T5_SINGLE_SOURCED", action="buffer-only",
                             recommendation="No alternate anywhere — buffer + delay comms",
                             shortlist=[], buffer=buf)
            n_exp += 1
        lanes.append(entry)

    return {
        "type": "replay", "label": label, "country": country,
        "window": f"{start}..{end}", "documented": documented,
        "detection": detection,
        "lanes": lanes,
        "summary": {"disrupted_lanes": len(lanes), "resolved": n_res,
                    "exposed": n_exp, "buffer_usd_total": buf_total},
    }


# ─────────────────────────────────────────────────────────────
# KNOCKOUT — remove a vendor, see what breaks
# ─────────────────────────────────────────────────────────────
def knockout(ctx, vendor, label=""):
    recs = ctx["recs"]
    S = sc.build_stats(recs)
    served = {lane for lane, vs in ctx["lane_vendors"].items() if vendor in vs}

    # annualised volume/value the vendor moves, per lane
    span_days = (max(x["date"] for x in recs) - min(x["date"] for x in recs)).days or 1
    yrs = span_days / 365.25
    v_qty = defaultdict(float)
    v_val = defaultdict(float)
    for x in recs:
        if x["ven"] == vendor:
            v_qty[(x["mol"], x["cty"])] += x["qty"]
            v_val[(x["mol"], x["cty"])] += x["val"]

    lanes, exposed_val, buf_total = [], 0.0, 0
    n_exposed = n_resilient = 0
    for lane in sorted(served):
        mol, cty = lane
        others_in_country = ctx["lane_vendors"][lane] - {vendor}
        if others_in_country:
            n_resilient += 1
            continue                                     # lane still covered
        n_exposed += 1
        val_yr = v_val[lane] / yrs
        exposed_val += val_yr
        cands = sc.rank_candidates(mol, ctx["lane_vendors"][lane] | {vendor}, S)
        buf = size_buffer(ctx, lane)
        if buf:
            buf_total += buf["buffer_usd"]
        lanes.append({
            "molecule": mol, "country": cty,
            "value_at_risk_usd_per_yr": round(val_yr),
            "tier": "T2_QUALIFY_VENDOR" if cands else "T5_SINGLE_SOURCED",
            "recommendation": (f"Qualify {cands[0]['vendor']}" if cands
                               else "No alternate anywhere — buffer only"),
            "shortlist": [c["vendor"] for c in cands[:3]],
            "buffer": buf,
        })
    lanes.sort(key=lambda e: -e["value_at_risk_usd_per_yr"])

    return {
        "type": "knockout", "label": label or f"Knockout: {vendor}",
        "vendor": vendor,
        "summary": {"lanes_served": len(served), "resilient": n_resilient,
                    "newly_exposed": n_exposed,
                    "value_at_risk_usd_per_yr": round(exposed_val),
                    "buffer_usd_total": buf_total},
        "lanes": lanes,
    }


# ─────────────────────────────────────────────────────────────
# Human-readable trace
# ─────────────────────────────────────────────────────────────
def render(sc_res):
    L = []
    if sc_res["type"] == "replay":
        d = sc_res["detection"]
        L += [f"REPLAY — {sc_res['label']}",
              f"  Real event : {sc_res['documented']}",
              f"  Window     : {sc_res['country']}  {sc_res['window']}",
              "",
              "  STEP 1 — DETECTION (blind, walk-forward)",
              f"    late-rate {d['late_rate']*100:.0f}% ({d['late_window']}/{d['n_window']}) "
              f"vs baseline {d['baseline_rate']*100:.0f}%  ->  p={d['p_value']}  "
              f"{'*** ALERT FIRED ***' if d['fired'] else '(no alert)'}",
              "",
              f"  STEP 2 — RESPONSE  ({sc_res['summary']['disrupted_lanes']} disrupted lanes: "
              f"{sc_res['summary']['resolved']} reroutable, {sc_res['summary']['exposed']} exposed)"]
        for e in sc_res["lanes"][:12]:
            tag = e["tier"].split("_")[0]
            extra = (f"coverage {e['coverage']*100:.0f}%" if e.get("coverage") is not None
                     else (f"buffer ${e['buffer']['buffer_usd']:,}" if e.get("buffer") else ""))
            L.append(f"    [{tag}] {e['molecule'][:34]:<34} -> {e['recommendation'][:44]:<44} {extra}")
        if len(sc_res["lanes"]) > 12:
            L.append(f"    ... and {len(sc_res['lanes'])-12} more lanes")
        L += ["",
              f"  STEP 3 — TOTALS: buffer exposure ${sc_res['summary']['buffer_usd_total']:,}"]
    else:
        s = sc_res["summary"]
        L += [f"KNOCKOUT — {sc_res['vendor']}",
              "",
              f"  STEP 1 — IMPACT: served {s['lanes_served']} lanes -> "
              f"{s['resilient']} still covered, {s['newly_exposed']} NEWLY EXPOSED",
              f"  STEP 2 — VALUE AT RISK: ${s['value_at_risk_usd_per_yr']:,}/yr on newly-exposed lanes",
              "",
              "  STEP 3 — RESPONSE (top exposed lanes by value):"]
        for e in sc_res["lanes"][:12]:
            L.append(f"    {e['molecule'][:30]:<30} {e['country'][:14]:<14} "
                     f"${e['value_at_risk_usd_per_yr']:>10,}/yr -> {e['recommendation'][:38]}")
        if len(sc_res["lanes"]) > 12:
            L.append(f"    ... and {len(sc_res['lanes'])-12} more exposed lanes")
        L += ["",
              f"  STEP 4 — MITIGATION: buffer exposure ${s['buffer_usd_total']:,} "
              "(the cost of absorbing this shock with stock alone)"]
    return "\n".join(L)


def is_distribution_node(vendor):
    """'SCMS from RDC' etc. are the program's own regional distribution /
    buffer mechanism, not a manufacturer. Knocking one out is not a
    'supplier failure' — exclude from vendor-shock scenarios (its
    dependency is reported separately)."""
    v = vendor.lower()
    return "from rdc" in v or v.startswith("scms")


def top_knockout_vendors(ctx, k=3):
    """Manufacturers whose removal exposes the most lanes (they're the sole
    in-country supplier) — the most instructive genuine supplier shocks."""
    sole = defaultdict(int)
    for lane, vs in ctx["lane_vendors"].items():
        if len(vs) == 1:
            v = next(iter(vs))
            if not is_distribution_node(v):
                sole[v] += 1
    return [v for v, _ in sorted(sole.items(), key=lambda kv: -kv[1])[:k]]


def main(argv):
    ctx = build_context()

    if len(argv) >= 2 and argv[1] == "replay":
        a = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1, 2)}
        res = replay(ctx, a["--country"], a["--start"], a["--end"])
        print(render(res))
        return
    if len(argv) >= 2 and argv[1] == "knockout":
        a = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1, 2)}
        res = knockout(ctx, a["--vendor"])
        print(render(res))
        return

    # no args → rebuild all presets for the dashboard
    print("=" * 60)
    print("  SCMS EVENT-REACTION HARNESS — building presets")
    print("=" * 60)
    scenarios = []
    for p in REPLAY_PRESETS:
        scenarios.append(replay(ctx, p["country"], p["start"], p["end"],
                                p["label"], p["documented"]))
    for v in top_knockout_vendors(ctx, 3):
        scenarios.append(knockout(ctx, v))

    with open(JSON_OUT, "w") as f:
        json.dump(scenarios, f, indent=2, default=str)
    report = ("SCMS EVENT-REACTION HARNESS — TRACES\n" + "=" * 55 + "\n\n"
              + "\n\n".join(render(s) for s in scenarios))
    with open(RPT_OUT, "w") as f:
        f.write(report)

    print(f"\n  Scenarios built: {len(scenarios)}")
    print(f"  JSON   -> {JSON_OUT}")
    print(f"  Report -> {RPT_OUT}\n")
    print(report)


if __name__ == "__main__":
    main(sys.argv)
