"""
TEST HARNESS — src/ir_stages.py (gated IR analytics)
====================================================
Follows the test_vendor_scorecard.py pattern: numbered tests, independent
recomputation, PASS/FAIL summary, and — most importantly — CROSS-VALIDATION
against the project's canonical outputs, so the IR-native stages are anchored
to numbers already verified elsewhere:

  T1  metric math (_prf) against hand-computed confusion counts
  T2  anomaly_detection: determinism (same seed), label conservation
      (tp+fn == injected), and beats the random-flagging baseline
  T3  capability gate enforcement: ARCOS runs anomaly + both graph stages /
      skips the 4 flow stages it lacks; SCMS (3-tier adapter) runs all 7;
      DataCo runs the 5 flow stages and skips both graph stages citing
      HAS_TOPOLOGY — and every skipped stage names the missing capability
  T8  graph stages cross-validated: ARCOS IR graph == canonical 2,131 nodes /
      23,264 edges; risk concentration beats the random-decile baseline;
      cascade PPO reports totals/risks/unserved per method, learns at least
      the random floor, and stays risk-aware vs Dijkstra
  T4  SCMS value_at_risk == canonical feasibility probe ($259M late /
      $1.63B total) and late count == spine's 1,186 real disruptions
  T5  event_replay independently re-finds documented episodes that the
      canonical scms_event_replay.py verified (Tanzania 2014-02,
      Nigeria 2011-07, Congo DRC 2014-02)
  T6  disruption_detection: temporal holdout is leak-free by construction;
      SCMS precision lift over base rate >= 2x; baselines present in output
  T7  safety_stock determinism + all-lanes SCMS total within sane range of
      the canonical Stage-22 exposed-lane figure ($72.2M)

Pure stdlib. Run:
    python3 src/test_ir_stages.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ir_stages
from ir_stages import (_prf, anomaly_detection, disruption_detection,
                       event_replay, load_flows, run, safety_stock,
                       value_at_risk)

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


print("=" * 64)
print("  TESTS — gated IR analytics (ir_stages.py)")
print("=" * 64)

# ── T1: metric math ──────────────────────────────────────────────────────────
print("\nT1  metric math")
p, r, f1 = _prf(tp=8, fp=2, fn=8)
check("precision 8/(8+2) = 0.8", p == 0.8)
check("recall    8/(8+8) = 0.5", r == 0.5)
check("f1 = 2*0.8*0.5/1.3 ≈ 0.6154", abs(f1 - 0.6154) < 1e-4)
check("zero-division safe", _prf(0, 0, 0) == (0.0, 0.0, 0.0))

# ── load flows once ─────────────────────────────────────────────────────────
scms_flows   = load_flows("scms")
arcos_flows  = load_flows("arcos")
dataco_flows = load_flows("dataco")

# ── T2: anomaly_detection invariants ────────────────────────────────────────
print("\nT2  anomaly_detection (synthetic injection)")
a1 = anomaly_detection(scms_flows, seed=42)
a2 = anomaly_detection(scms_flows, seed=42)
a3 = anomaly_detection(scms_flows, seed=7)
check("deterministic under same seed", a1 == a2)
check("seed actually matters", a1 != a3, "different seeds gave identical output")
# tp+fn must equal number injected: recall*n_injected must be an integer (tp)
# up to the 4-decimal rounding of the reported recall (error <= 5e-5 * n).
x = a1["recall"] * a1["n_injected"]
check("recall denominator == injected count (tp is integral)",
      abs(x - round(x)) <= 5e-5 * a1["n_injected"] + 1e-9,
      f'recall*injected = {x}')
for name, res in (("arcos", anomaly_detection(arcos_flows)),
                  ("scms", a1),
                  ("dataco", anomaly_detection(dataco_flows))):
    check(f"{name}: f1 > random-flagging baseline",
          res["f1"] > res["baseline_random_f1"],
          f'f1={res["f1"]} vs random={res["baseline_random_f1"]}')

# ── T3: gate enforcement ────────────────────────────────────────────────────
print("\nT3  capability gate enforcement")
res_arcos = run("arcos")
ran     = sorted(s for s, o in res_arcos.items() if o["status"] == "ran")
skipped = {s: o["missing"] for s, o in res_arcos.items() if o["status"] == "skipped"}
check("ARCOS runs anomaly + both graph stages",
      ran == ["anomaly_detection", "graph_risk_routing", "ppo_recovery_routing"],
      str(ran))
check("ARCOS skips the 4 flow stages it lacks", len(skipped) == 4, str(list(skipped)))
check("every skip names its missing capability",
      all(len(m) > 0 for m in skipped.values()))
check("disruption_detection skip cites HAS_DISRUPTIONS",
      skipped.get("disruption_detection") == ["HAS_DISRUPTIONS"])
res_scms = run("scms")
check("SCMS (3-tier adapter) runs all 7 stages",
      all(o["status"] == "ran" for o in res_scms.values()),
      str({s: o["status"] for s, o in res_scms.items()}))
res_dataco = run("dataco")
dataco_skipped = {s: o["missing"] for s, o in res_dataco.items()
                  if o["status"] == "skipped"}
check("DataCo runs the 5 flow stages",
      sum(1 for o in res_dataco.values() if o["status"] == "ran") == 5,
      str({s: o["status"] for s, o in res_dataco.items()}))
check("DataCo skips both graph stages citing HAS_TOPOLOGY",
      dataco_skipped.get("graph_risk_routing") == ["HAS_TOPOLOGY"]
      and dataco_skipped.get("ppo_recovery_routing") == ["HAS_TOPOLOGY"])

# ── T4: cross-validate value_at_risk vs canonical numbers ───────────────────
print("\nT4  value_at_risk vs canonical SCMS probes")
var = value_at_risk(scms_flows)
check("late flow count == spine's 1,186 real disruptions",
      var["n_late_flows"] == 1186, str(var["n_late_flows"]))
check("value-at-risk ≈ $259M (canonical feasibility probe)",
      abs(var["value_at_risk_usd"] - 259_000_000) < 2_000_000,
      f'${var["value_at_risk_usd"]:,.0f}')
check("total value ≈ $1.63B (canonical)",
      abs(var["total_value_usd"] - 1_630_000_000) < 10_000_000,
      f'${var["total_value_usd"]:,.0f}')

# ── T5: event_replay re-finds documented episodes ───────────────────────────
print("\nT5  event_replay vs canonical documented episodes")
er = event_replay(scms_flows)
check("alerts fired", er["n_alerts"] > 0)
# Canonical scms_event_replay.py episodes (web/machine-verified in case_table):
# Tanzania 2014-02..04, Nigeria 2011-07..09, Congo DRC 2014-02..04.
top_text = " | ".join(er["top_alerts"])
found = sum(1 for marker in ("Tanzania 2014-02", "Nigeria 2011-07", "Congo, DRC 2014-02")
            if marker in top_text)
check("top alerts re-find >=2 canonical episodes independently",
      found >= 2, top_text)

# ── T6: disruption_detection honesty ────────────────────────────────────────
print("\nT6  disruption_detection (real labels, temporal holdout)")
dd_scms = disruption_detection(scms_flows)
dd_dc   = disruption_detection(dataco_flows)
check("baselines present in output",
      "baseline_always_late_f1" in dd_scms and "precision_lift_over_base" in dd_scms)
check("SCMS precision lift over base rate >= 2x",
      dd_scms["precision_lift_over_base"] >= 2.0, str(dd_scms))
check("SCMS test-set base rate ≈ known 11-16% late band",
      0.08 <= dd_scms["base_late_rate"] <= 0.20, str(dd_scms["base_late_rate"]))
check("DataCo evaluated on large test set (>50K)", dd_dc["n_test"] > 50_000)

# ── T7: safety_stock determinism + sanity vs canonical ──────────────────────
print("\nT7  safety_stock")
s1, s2 = safety_stock(scms_flows), safety_stock(scms_flows)
check("deterministic (no randomness)", s1 == s2)
check("SCMS all-lanes total within 0.5x-1.5x of canonical $72.2M exposed-lane figure",
      0.5 * 72_200_000 <= s1["total_buffer_usd"] <= 1.5 * 72_200_000,
      f'${s1["total_buffer_usd"]:,.0f}')
check("holding cost == 25% of buffer",
      abs(s1["holding_usd_per_yr"] - 0.25 * s1["total_buffer_usd"]) < 1.0)

# ── T8: graph stages — cross-validation + well-formedness ───────────────────
print("\nT8  graph stages (topology-gated)")
gr_arcos = res_arcos["graph_risk_routing"]
check("ARCOS IR graph == canonical 2,131 nodes",
      gr_arcos["n_nodes"] == 2131, str(gr_arcos["n_nodes"]))
check("ARCOS IR graph == canonical 23,264 edges",
      gr_arcos["n_edges"] == 23264, str(gr_arcos["n_edges"]))
check("ARCOS top-decile volume share beats random-decile baseline",
      gr_arcos["top_decile_volume_share"] > gr_arcos["random_decile_volume_share"],
      f'{gr_arcos["top_decile_volume_share"]} vs {gr_arcos["random_decile_volume_share"]}')
gr_scms = res_scms["graph_risk_routing"]
check("SCMS graph risk ran on the 3-tier adapter output",
      gr_scms.get("n_nodes", 0) > 100, str(gr_scms.get("n_nodes")))
check("SCMS late-rate lift computed from REAL labels (present)",
      gr_scms.get("late_rate_lift_top_decile") is not None, str(gr_scms))
ppo_scms = res_scms["ppo_recovery_routing"]
check("SCMS cascade PPO trained and evaluated (n_test >= 100)",
      ppo_scms.get("n_test", 0) >= 100, str(ppo_scms))
check("cascade reports total reward for all four methods",
      all(k in ppo_scms.get("avg_total_reward", {})
          for k in ("ppo", "dijkstra", "risk_greedy", "random")))
check("PPO reported against Dijkstra + risk-greedy + random baselines",
      all(k in ppo_scms.get("avg_route_risk", {})
          for k in ("ppo", "dijkstra", "risk_greedy", "random")))
check("SCMS cascade PPO total reward >= random baseline (policy learned)",
      ppo_scms["avg_total_reward"]["ppo"] >= ppo_scms["avg_total_reward"]["random"],
      str(ppo_scms.get("avg_total_reward")))
check("SCMS cascade PPO avg route risk <= Dijkstra baseline (risk-aware policy)",
      ppo_scms["avg_route_risk"]["ppo"] <= ppo_scms["avg_route_risk"]["dijkstra"],
      str(ppo_scms.get("avg_route_risk")))
check("unserved-demand counts reported per method",
      all(k in ppo_scms.get("unserved", {})
          for k in ("ppo", "dijkstra", "risk_greedy", "random")))
ppo_arcos = res_arcos["ppo_recovery_routing"]
check("ARCOS cascade trains despite compressed risk scale (adaptive spread filter)",
      ppo_arcos.get("n_test", 0) >= 100, str(ppo_arcos))

# ── summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print(f"  {PASS} passed, {FAIL} failed")
print("=" * 64)
sys.exit(1 if FAIL else 0)
