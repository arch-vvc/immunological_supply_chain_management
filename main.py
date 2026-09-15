"""
IMMUNOLOGICAL SUPPLY CHAIN — Unified Pipeline Runner
=====================================================
Runs all stages in sequence.

Usage:
    python3 main.py                              # full pipeline (Stages 1-6)
    python3 main.py --from 3                    # start from stage 3
    python3 main.py --only 6                    # run only visualization
    python3 main.py --only 0                    # run Stage 0 (sample raw ARCOS)
    python3 main.py --onboard /path/to/data.csv # auto-detect columns + run pipeline
    python3 main.py --metrics                   # run pipeline and print project metrics
    python3 main.py --stages 1-6,19-25          # run an explicit set / ranges of stages
    python3 main.py --skip-training             # skip 7/10/11 when their model + output already exist
    python3 main.py --check                     # verify every stage's expected artifacts exist (no run)

Stages:
    0 — Dataset Sampling           (optional: needs data/raw/datasetuc.csv)
    1 — Preprocessing              (reads config.yaml or ARCOS defaults)
    2 — Supply Chain Graph Construction
    3 — Multi-Dimensional Anomaly Detection
    4 — Risk & Vulnerability Analysis
    5 — Disruption Injection & Recovery Routing
    6 — Visualization
    7 — GNN Node Encoder           (requires: pip install torch)
    8 — Recovery Time Predictor    (requires: pip install scikit-learn)
    9  — Macro Freight Risk Scorer
    10 — LSTM Stress Forecaster       (requires Stage 9 output)
    11 — PPO Recovery Routing Agent
    12 — Multi-Domain Risk Modelling
    13 — Immunological Memory (FAISS) — builds vector index + queries current anomalies
    14 — Supplier Agent (backup supplier scoring)
    15 — Inventory Agent (emergency stock transfers)
    16 — Immune Response Engine (real-time test)
    17 — Anomaly Detection Benchmark (independent injection ground truth)
    18 — Network-Grounded Response Planner (preference ladder over the graph)
    19 — SCMS Spine (real unified network + disruption analysis)
    20 — SCMS Revealed-Preference Backtest (walk-forward validation)
    21 — SCMS Real-Event Replay (blind detection of Haiti 2010)
    22 — SCMS Safety-Stock Sizing (priced buffers for exposed lanes)
    23 — SCMS Vendor Scorecard (qualification shortlists; tests in src/test_vendor_scorecard.py)
    24 — Case Table (documented real-world events vs this system)
    25 — Event-Reaction Harness (replay documented events + vendor knockout)
    26 — Dataset-Independent IR Pipeline (adapters -> canonical IR -> capability-gated stages)
    27 — Macro-Stress Crisis Validation (event study vs documented crises + placebo inference)
    28 — SCMS Outcome Counterfactual (planner's pick vs procurement's real choice, realized outcomes)
"""

import os
import sys
import subprocess
import argparse
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

ALL_STAGES = [
    (0, "Dataset Sampling (raw ARCOS → 50K rows)", "src/sample_dataset.py"),
    (1, "Preprocessing",                            "src/preprocess.py"),
    (2, "Supply Chain Graph Construction",           "src/build_chain.py"),
    (3, "Multi-Dimensional Anomaly Detection",       "src/anomaly_detection.py"),
    (4, "Risk & Vulnerability Analysis",             "src/risk_analysis.py"),
    (5, "Disruption Injection & Recovery Routing",   "src/routing.py"),
    (6, "Visualization",                             "src/visualize.py"),
    (7, "GNN Node Encoder (Adaptive Immunity)",      "src/gnn_encoder.py"),
    (8, "Recovery Time Predictor",                   "src/recovery_predictor.py"),
    (9,  "Macro Freight Risk Scorer",                 "src/macro_risk.py"),
    (10, "LSTM Stress Forecaster",                    "src/lstm_forecaster.py"),
    (11, "PPO Recovery Routing Agent",                "src/ppo_routing_agent.py"),
    (12, "Multi-Domain Risk Modelling",               "src/multi_risk.py"),
    (13, "Immunological Memory (FAISS)",              "src/immunological_memory.py"),
    (14, "Supplier Agent (Digital Antibody #2)",      "src/supplier_agent.py"),
    (15, "Inventory Agent (Digital Antibody #4)",     "src/inventory_agent.py"),
    (16, "Immune Response Engine (Real-Time Test)",   "src/immune_response_engine.py"),
    (17, "Anomaly Detection Benchmark (Injection GT)","src/anomaly_eval_injection.py"),
    (18, "Network-Grounded Response Planner",         "src/response_planner.py"),
    (19, "SCMS Spine (Real Unified Analysis)",        "src/scms_spine.py"),
    (20, "SCMS Revealed-Preference Backtest",         "src/scms_backtest.py"),
    (21, "SCMS Real-Event Replay (Haiti 2010)",       "src/scms_event_replay.py"),
    (22, "SCMS Safety-Stock Sizing (Exposed Lanes)",  "src/scms_safety_stock.py"),
    (23, "SCMS Vendor Scorecard + Shortlists",        "src/scms_vendor_scorecard.py"),
    (24, "Case Table (Documented Real Events)",       "src/case_table.py"),
    (25, "Event-Reaction Harness (Replay + Knockout)", "src/scms_event_harness.py"),
    (26, "Dataset-Independent IR Pipeline (adapters -> IR -> gated stages)", "src/ir_pipeline.py"),
    (27, "Macro-Stress Crisis Validation (documented real events)", "src/macro_event_validation.py"),
    (28, "SCMS Outcome Counterfactual (recommendation vs what happened IRL)", "src/scms_counterfactual.py"),
]

# Default pipeline skips Stage 0 (optional sampling step)
STAGES = [s for s in ALL_STAGES if s[0] >= 1]

# Artifacts each stage is expected to leave behind (relative to ROOT).
# Used by --check and by --skip-training. Stage 16 is a real-time smoke test
# and writes nothing; Stage 0 is optional and its input is not shipped.
EXPECTED_OUTPUTS = {
    1:  ["data/processed/clean_chain.csv"],
    2:  ["models/supplychain_graph.pkl", "output/graph_risk_scores.csv", "output/figures/fig_centrality.png"],
    3:  ["output/anomalies.csv", "output/anomaly_metrics.txt"],
    4:  ["output/risk_scores.csv"],
    5:  ["output/routing_results.txt"],
    6:  ["output/figures/fig1_supply_chain_graph.png", "output/figures/fig2_top_risk_entities.png",
         "output/figures/fig3_anomaly_timeline.png", "output/figures/fig4_anomaly_dimensions.png"],
    7:  ["models/gnn_autoencoder.pth", "models/node_embeddings.pkl", "output/gnn_risk_scores.csv",
         "output/figures/fig5_gnn_embeddings.png"],
    8:  ["models/recovery_regressor.pkl", "output/recovery_predictions.csv", "output/recovery_metrics.txt",
         "output/figures/fig6_recovery.png"],
    9:  ["output/macro_stress_scores.csv", "output/figures/fig7_macro_stress.png"],
    10: ["models/lstm_stress_model.pth", "output/stress_forecast.csv", "output/lstm_metrics.txt",
         "output/figures/fig8_stress_forecast.png"],
    11: ["models/ppo_routing_agent.pth", "output/ppo_routing_results.txt", "output/figures/fig9_ppo_training.png"],
    12: ["output/multi_domain_f1.csv", "output/figures/fig10_multi_domain_risk.png"],
    13: ["models/faiss_memory.index", "models/faiss_memory_meta.pkl", "output/memory_retrieval.csv",
         "output/memory_report.txt"],
    14: ["output/supplier_agent_results.csv", "output/supplier_agent_report.txt"],
    15: ["output/inventory_agent_results.csv", "output/inventory_agent_report.txt"],
    16: [],
    17: ["output/anomaly_injection_results.csv", "output/anomaly_injection_metrics.txt",
         "output/anomaly_ensemble_calibration.json", "output/figures/fig_anomaly_injection_eval.png"],
    18: ["output/response_plan.csv", "output/response_plan_report.txt"],
    19: ["output/scms_response_plan.csv", "output/scms_spine_report.txt"],
    20: ["output/scms_backtest.csv", "output/scms_backtest_report.txt"],
    21: ["output/scms_event_replay.csv", "output/scms_event_replay_report.txt"],
    22: ["output/scms_safety_stock.csv", "output/scms_safety_stock_report.txt"],
    23: ["output/scms_vendor_scorecard.csv", "output/scms_vendor_directory.csv",
         "output/scms_vendor_scorecard_report.txt"],
    24: ["output/case_table.csv", "output/case_table_report.txt"],
    25: ["output/event_harness.json", "output/event_harness_report.txt"],
    26: ["output/ir/arcos/manifest.json", "output/ir/arcos/stage_results.json",
         "output/ir/scms/manifest.json", "output/ir/scms/stage_results.json",
         "output/ir/dataco/manifest.json", "output/ir/dataco/stage_results.json"],
    27: ["output/macro_event_validation.csv", "output/macro_event_validation_report.txt",
         "output/figures/fig11_macro_event_validation.png"],
    28: ["output/scms_counterfactual.csv", "output/scms_counterfactual_report.txt",
         "output/figures/fig12_scms_counterfactual.png"],
}

# Stages that train a model from scratch and dominate wall-clock time.
# --skip-training skips them when every expected artifact is already present.
TRAINING_STAGES = {7, 10, 11}


def missing_outputs(num):
    """Return the expected artifacts of stage `num` that are absent or empty."""
    out = []
    for rel in EXPECTED_OUTPUTS.get(num, []):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            out.append(rel)
    return out


def parse_stage_spec(spec):
    """'1-6,19-25,28' -> sorted list of stage numbers. Raises ValueError on junk."""
    nums = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if lo > hi:
                raise ValueError(f"bad range '{part}'")
            nums.update(range(lo, hi + 1))
        else:
            nums.add(int(part))
    known = {s[0] for s in ALL_STAGES}
    unknown = sorted(nums - known)
    if unknown:
        raise ValueError(f"unknown stage(s): {unknown}")
    return sorted(nums)


def check_outputs(stages=None):
    """Print a per-stage artifact audit. Returns the number of stages with gaps."""
    stages = stages or [s[0] for s in STAGES]
    by_num = {s[0]: s for s in ALL_STAGES}
    gaps = 0
    print("  ARTIFACT CHECK")
    print(f"  {'-' * 56}")
    for num in stages:
        _, name, _ = by_num[num]
        expected = EXPECTED_OUTPUTS.get(num, [])
        if not expected:
            print(f"  {num:>2}  {'n/a ':<8} {name}  (writes nothing)")
            continue
        miss = missing_outputs(num)
        status = "ok" if not miss else "MISSING"
        print(f"  {num:>2}  {status:<8} {name}")
        for rel in miss:
            print(f"        - {rel}")
        gaps += bool(miss)
    print(f"  {'-' * 56}")
    if gaps:
        print(f"  {gaps} stage(s) have missing artifacts. Re-run them with --stages.")
    else:
        print("  All expected artifacts present.")
    return gaps

BANNER = """
╔══════════════════════════════════════════════════════╗
║   IMMUNOLOGICAL SUPPLY CHAIN                         ║
║   Self-Healing Supply Chains with AI Antibodies      ║
║   28-Stage Pipeline · Multi-Dataset · Real-Time      ║
║   PES University — ISA Capstone  PW26_RGP_01         ║
╚══════════════════════════════════════════════════════╝
"""

def run_stage(num, name, script):
    print(f"\n{'━' * 56}")
    print(f"  STAGE {num}: {name}")
    print(f"{'━' * 56}")
    start = time.time()
    result = subprocess.run([sys.executable, script], cwd=ROOT)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[FAILED] Stage {num} exited with error.")
        return False
    print(f"\n  ✓ Stage {num} completed in {elapsed:.1f}s")
    return True


def run_onboard(filepath):
    """Run auto-onboarding for a new company dataset."""
    script = os.path.join(ROOT, "src", "auto_onboard.py")
    result = subprocess.run([sys.executable, script, filepath], cwd=ROOT)
    if result.returncode != 0:
        print("[FAILED] Onboarding failed.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run the Immunological Supply Chain pipeline.")
    parser.add_argument("--from",    type=int,  dest="from_stage",   default=1,    help="Start from stage N (default: 1)")
    parser.add_argument("--only",    type=int,  dest="only_stage",   default=None, help="Run only stage N")
    parser.add_argument("--onboard", type=str,  dest="onboard_file", default=None, help="Path to company CSV — auto-detects columns and runs pipeline")
    parser.add_argument("--metrics", action="store_true", dest="show_metrics", help="Print the project-wide metrics summary after the pipeline finishes")
    parser.add_argument("--stages",  type=str,  dest="stage_spec",   default=None, help="Explicit stages to run, e.g. '1-6,19-25,28' (overrides --from/--only)")
    parser.add_argument("--skip-training", action="store_true", dest="skip_training",
                        help=f"Skip training stages {sorted(TRAINING_STAGES)} when their model and outputs already exist")
    parser.add_argument("--check",   action="store_true", dest="check_only",
                        help="Do not run anything; report which expected artifacts are missing per stage")
    args = parser.parse_args()

    print(BANNER)

    # ── Artifact audit only ────────────────────────────────────
    if args.check_only:
        wanted = parse_stage_spec(args.stage_spec) if args.stage_spec else None
        sys.exit(1 if check_outputs(wanted) else 0)

    # ── Auto-onboard mode ──────────────────────────────────────
    if args.onboard_file:
        filepath = os.path.abspath(args.onboard_file)
        print(f"  Onboarding: {filepath}\n")
        run_onboard(filepath)
        # After onboarding, run full pipeline from Stage 1
        stages_to_run = [s for s in ALL_STAGES if s[0] >= 1]

    elif args.stage_spec:
        try:
            wanted = set(parse_stage_spec(args.stage_spec))
        except ValueError as e:
            print(f"[ERROR] --stages: {e}")
            sys.exit(1)
        stages_to_run = [s for s in ALL_STAGES if s[0] in wanted]

    elif args.only_stage is not None:
        stages_to_run = [s for s in ALL_STAGES if s[0] == args.only_stage]
        if not stages_to_run:
            print(f"[ERROR] Stage {args.only_stage} not found.")
            sys.exit(1)
    else:
        stages_to_run = [s for s in ALL_STAGES if s[0] >= args.from_stage]

    total_start = time.time()
    timings = []
    skipped = []

    for num, name, script in stages_to_run:
        script_path = os.path.join(ROOT, script)
        if not os.path.exists(script_path):
            print(f"[ERROR] Script not found: {script_path}")
            sys.exit(1)
        if args.skip_training and num in TRAINING_STAGES and not missing_outputs(num):
            print(f"\n  ↷ Stage {num} skipped (--skip-training; model + outputs present)")
            skipped.append(num)
            continue
        start = time.time()
        success = run_stage(num, name, script_path)
        timings.append((num, name, time.time() - start))
        if not success:
            print(f"\n Pipeline halted at Stage {num}.")
            sys.exit(1)
        miss = missing_outputs(num)
        if miss:
            print(f"  [WARN] Stage {num} finished but did not produce: {', '.join(miss)}")

    total = time.time() - total_start
    print(f"\n{'═' * 56}")
    print(f"  PIPELINE COMPLETE  ({total:.1f}s total)")
    print(f"{'═' * 56}")
    if timings:
        print("  Stage timings:")
        for num, name, secs in timings:
            print(f"    {num:>2}  {secs:7.1f}s  {name}")
    if skipped:
        print(f"  Skipped (training, already trained): {skipped}")
    print(f"  Outputs  → {os.path.join(ROOT, 'output')}")
    print(f"  Figures  → {os.path.join(ROOT, 'output', 'figures')}")
    print(f"  Models   → {os.path.join(ROOT, 'models')}")
    print()

    if args.show_metrics:
        metrics_script = os.path.join(ROOT, "src", "project_metrics.py")
        print("  PROJECT METRICS")
        print(f"  {'-' * 56}")
        result = subprocess.run([sys.executable, metrics_script], cwd=ROOT)
        if result.returncode != 0:
            print("[WARN] Project metrics summary failed.")


if __name__ == "__main__":
    main()
