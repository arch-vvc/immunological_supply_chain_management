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
]

# Default pipeline skips Stage 0 (optional sampling step)
STAGES = [s for s in ALL_STAGES if s[0] >= 1]

BANNER = """
╔══════════════════════════════════════════════════════╗
║   IMMUNOLOGICAL SUPPLY CHAIN                         ║
║   Self-Healing Supply Chains with AI Antibodies      ║
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
    args = parser.parse_args()

    print(BANNER)

    # ── Auto-onboard mode ──────────────────────────────────────
    if args.onboard_file:
        filepath = os.path.abspath(args.onboard_file)
        print(f"  Onboarding: {filepath}\n")
        run_onboard(filepath)
        # After onboarding, run full pipeline from Stage 1
        stages_to_run = [s for s in ALL_STAGES if s[0] >= 1]

    elif args.only_stage is not None:
        stages_to_run = [s for s in ALL_STAGES if s[0] == args.only_stage]
        if not stages_to_run:
            print(f"[ERROR] Stage {args.only_stage} not found.")
            sys.exit(1)
    else:
        stages_to_run = [s for s in ALL_STAGES if s[0] >= args.from_stage]

    total_start = time.time()

    for num, name, script in stages_to_run:
        script_path = os.path.join(ROOT, script)
        if not os.path.exists(script_path):
            print(f"[ERROR] Script not found: {script_path}")
            sys.exit(1)
        success = run_stage(num, name, script_path)
        if not success:
            print(f"\n Pipeline halted at Stage {num}.")
            sys.exit(1)

    total = time.time() - total_start
    print(f"\n{'═' * 56}")
    print(f"  PIPELINE COMPLETE  ({total:.1f}s total)")
    print(f"{'═' * 56}")
    print(f"  Outputs  → {os.path.join(ROOT, 'output')}")
    print(f"  Figures  → {os.path.join(ROOT, 'output', 'figures')}")
    print(f"  Models   → {os.path.join(ROOT, 'models')}")
    print()


if __name__ == "__main__":
    main()
