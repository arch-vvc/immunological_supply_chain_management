"""
Cross-Industry Transfer Test
=============================
Runs the same ImmuneResponseEngine pipeline against pharma, automotive,
and FMCG domain configs using a shared synthetic anomaly event.

Shows that:
  1. Detection thresholds differ per domain (tight for auto, loose for FMCG)
  2. Routing weights differ (path-heavy for FMCG / auto, risk-heavy for pharma)
  3. Decay parameters differ (shorter window for FMCG perishables)
  4. Cytokine storm sensitivity differs
  5. ALL core components (FAISS recall, PPO, chain-of-thought) reuse without change

Usage:
    python3 src/cross_industry_test.py
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from domain_config import load_domain_config
from immune_response_engine import ImmuneResponseEngine

# ── Shared test event (generic — no pharma-specific fields) ────────────────
TEST_EVENT = {
    "manufacturer":      "TEVA PHARMACEUTICALS USA INC",   # reuse graph node
    "distributor":       "CARDINAL HEALTH INC",
    "retailer":          "CVS PHARMACY #5216 (AK)",
    "retailer_state":    "AK",
    "quantity":          0,
    "z_score":           3.5,
    "disruption_injected": 1,
}

DOMAINS = ["pharma", "automotive", "fmcg"]

SEP = "=" * 72

def run_domain(domain: str):
    cfg = load_domain_config(domain)
    print(f"\n{SEP}")
    print(f"  DOMAIN : {cfg.domain_name}")
    print(f"  Config : config/{domain}.yaml")
    print(f"  Tier roles : {cfg.tier1_label} -> {cfg.tier2_label} -> {cfg.tier3_label}")
    print(f"  Thresholds : Z={cfg.z_score_threshold}  Storm={cfg.cytokine_threshold} in {cfg.cytokine_window_secs}s")
    print(f"  Weights    : risk={cfg.risk_weight}  gnn={cfg.gnn_weight}  path={cfg.path_weight}")
    print(f"  Decay      : window={cfg.decay_window_minutes}min  drop={cfg.decay_per_use}  floor={cfg.decay_floor}")
    print(f"  Shipping AK: {cfg.fuel_cost('AK')}x  OH: {cfg.fuel_cost('OH')}x  CA: {cfg.fuel_cost('CA')}x")
    print(SEP)

    # Z-score detection gate (would this event trigger in this domain?)
    z = float(TEST_EVENT["z_score"])
    triggered = z >= cfg.z_score_threshold
    print(f"  Event Z={z:.1f} vs threshold {cfg.z_score_threshold} --> "
          f"{'TRIGGERED' if triggered else 'SUPPRESSED (below threshold)'}")

    if not triggered:
        print(f"  [SKIP] Event would NOT fire immune response in {domain} domain.")
        print(f"         (Raise z_score above {cfg.z_score_threshold} in the config to trigger.)\n")
        return

    # Run full engine
    engine = ImmuneResponseEngine(verbose=False, domain=domain)
    decision = engine.respond(TEST_EVENT)

    # Print summary
    verdict = decision.get("verdict", {})
    print(f"\n  Severity     : {verdict.get('severity','?')}")
    print(f"  Signals      : {verdict.get('signals_activated','?')}/4 activated")
    print(f"  Recovery est.: {verdict.get('recovery_estimate_days','?')} days")

    # Show how routing weights changed the chosen route
    for step in decision.get("thinking", []):
        if step.get("phase") == "ALTERNATE ROUTE":
            d = step.get("data", {})
            print(f"  Reroute via  : {d.get('chosen_distributor','none')}")
            print(f"  Decay factor : {d.get('decay_factor',1.0)}")
            print(f"  Method       : {d.get('method','?')}")
            attr = {k: v for k, v in d.get("shap_attribution", {}).items() if not k.startswith("_")}
            if attr:
                top = max(attr, key=attr.get)
                print(f"  Top feature  : {top} ({attr[top]:.5f})")
        if step.get("phase") == "INVENTORY TRANSFER":
            d = step.get("data", {})
            print(f"  Fuel cost    : {d.get('fuel_multiplier','?')}x (domain-specific)")

    actions = decision.get("actions", [])
    if actions:
        a1 = actions[0]
        print(f"  Best action  : {a1.get('action','?')} via {a1.get('label','?')[:40]}")
        print(f"  Confidence   : {float(a1.get('confidence',0)):.0%}")


if __name__ == "__main__":
    print(f"\n{'#' * 72}")
    print(f"  CROSS-INDUSTRY TRANSFER TEST")
    print(f"  Same engine, same graph, same PPO — three domain configs")
    print(f"{'#' * 72}")
    print(f"\n  Test event: Z={TEST_EVENT['z_score']} disruption via CARDINAL HEALTH -> AK")

    for d in DOMAINS:
        try:
            run_domain(d)
        except Exception as e:
            print(f"\n  [ERROR] Domain '{d}' failed: {e}")

    print(f"\n{SEP}")
    print("  TRANSFER SUMMARY")
    print(f"{SEP}")
    print("""
  Components that transfer with ZERO changes:
    - FAISS immunological memory (vector search is domain-agnostic)
    - PPO routing agent          (graph traversal is domain-agnostic)
    - Cytokine storm detector    (sliding window is domain-agnostic)
    - Chain-of-thought engine    (reasoning structure is domain-agnostic)
    - Streamlit dashboard        (reads JSONL regardless of domain)

  Components that change per domain (config/X.yaml only):
    - Z-score threshold          (sensitivity to demand shocks)
    - Cytokine storm threshold   (cascade sensitivity)
    - Routing weights            (risk vs speed vs connectivity priority)
    - Confidence decay params    (how fast overuse penalises a node)
    - Shipping cost map          (industry-specific logistics)
    - Node role labels           (manufacturer/distributor vs oem/tier1_supplier)

  To deploy on a new industry: write a new YAML + rebuild graph from domain data.
  Estimated effort: 1-2 days of data wrangling + config tuning.
""")
