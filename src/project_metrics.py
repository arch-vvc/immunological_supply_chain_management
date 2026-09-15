"""Project-wide metrics audit for the immunological supply chain repo.

By default this script reads the current metric artifacts already written to
output/ and summarizes the project in one place. Use --refresh to rerun the
main benchmark-producing stages before collecting the report.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def find_float(text: str, pattern: str) -> float | None:
    """First capture group as float, or None. A miss on a NON-empty text is
    printed: it means a report's wording changed and this parser is stale —
    silently rendering 'n/a' hid exactly that before. (results_manifest.py
    also flags any headline value that goes from a number to None as drift.)"""
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        if text.strip():
            print(f"[project_metrics][WARN] pattern not found in report — parser may be stale: {pattern!r}")
        return None
    return float(match.group(1).replace(",", ""))


def parse_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_anomaly_injection(path: Path) -> dict:
    rows = parse_csv_rows(path)
    out = {}
    for row in rows:
        method = row.get("Method", "").strip()
        if not method:
            continue
        try:
            out[method] = {
                "precision": float(row.get("Precision", 0) or 0),
                "recall": float(row.get("Recall", 0) or 0),
                "f1": float(row.get("F1", 0) or 0),
            }
        except ValueError:
            continue
    return out


def parse_multi_risk(path: Path) -> dict:
    rows = parse_csv_rows(path)
    if not rows:
        return {}
    top = rows[0]
    return {
        "best_industry": top.get("Industry", ""),
        "f1": float(top.get("F1 Score", 0) or 0),
        "baseline": float(top.get("Baseline F1", 0) or 0),
        "lift": float(top.get("Lift", 0) or 0),
    }


def parse_recovery_metrics(path: Path) -> dict:
    text = read_text(path)
    return {
        "mae": find_float(text, r"MAE\s*:\s*([0-9.]+)\s*days"),
        "r2": find_float(text, r"R²\s*:\s*([0-9.]+)"),
        "baseline_mae": find_float(text, r"predict-the-mean baseline:\s*([0-9.]+)\s*days"),
        "strategy_spread": find_float(text, r"model spread across strategies:\s*([0-9.]+)\s*days"),
    }


def parse_lstm_metrics(path: Path) -> dict:
    text = read_text(path)
    return {
        "mae": find_float(text, r"^LSTM[^0-9]*([0-9.]+)\s+[0-9.]+$"),
        "rmse": find_float(text, r"^LSTM[^0-9]*[0-9.]+\s+([0-9.]+)$"),
        "baseline_mae": find_float(text, r"^Persistence \(last week\)\s+([0-9.]+)"),
    }


def parse_memory_report(path: Path) -> dict:
    text = read_text(path)
    return {
        "knn_mae": find_float(text, r"k-NN\(k=\d+\) MAE\s*:\s*([0-9.]+)\s*days"),
        "base_mae": find_float(text, r"predict-the-mean\s*:\s*([0-9.]+)\s*days"),
        "knn_acc": find_float(text, r"k-NN\(k=\d+\) acc\s*:\s*([0-9.]+)%"),
        "base_acc": find_float(text, r"majority-class\s*:\s*([0-9.]+)%"),
    }


def parse_ppo_results(path: Path) -> dict:
    text = read_text(path)
    return {
        "safer_pct": find_float(text, r"PPO strictly safer than Dijkstra\s*:\s*([0-9.]+)%"),
    }


def parse_supplier_report(path: Path) -> dict:
    text = read_text(path)
    return {
        "analysed": find_float(text, r"Disrupted entities analysed\s*:\s*([0-9.]+)"),
        "avg_top1": find_float(text, r"Avg backup score \(rank #1\)\s*:\s*([0-9.]+)"),
    }


def parse_inventory_report(path: Path) -> dict:
    text = read_text(path)
    return {
        "analysed": find_float(text, r"At-risk retailers analysed\s*:\s*([0-9.]+)"),
        "avg_top1": find_float(text, r"Avg transfer score \(rank #1\)\s*:\s*([0-9.]+)"),
        "avg_days": find_float(text, r"Avg estimated delivery days\s*:\s*([0-9.]+)"),
    }


def parse_response_plan(path: Path) -> dict:
    text = read_text(path)
    return {
        "planned": find_float(text, r"Anomalies planned\s*:\s*([0-9,]+)"),
        "resolved_pct": find_float(text, r"resolved to a PROVEN supplier \(T1/T2\)\s*:\s*[0-9,]+\s*\(([0-9.]+)%\)"),
        "days_saved": find_float(text, r"days saved per case\s*:\s*([0-9.]+)"),   # report says "mean full-recovery days saved per case"
    }


def parse_routing_report(path: Path) -> dict:
    text = read_text(path)
    status = "missing"
    if text:
        status = "SUCCESS" if "Status            : SUCCESS" in text else "CHECK"
    return {
        "status": status,
        "recovery_days": find_float(text, r"Recovery days\s*:\s*([0-9.]+)"),
        "baseline_days": find_float(text, r"Baseline days\s*:\s*([0-9.]+)"),
    }


def refresh_benchmarks() -> None:
    scripts = [
        "src/anomaly_eval_injection.py",
        "src/multi_risk.py",
        "src/recovery_predictor.py",
        "src/lstm_forecaster.py",
        "src/immunological_memory.py",
        "src/ppo_routing_agent.py",
        "src/supplier_agent.py",
        "src/inventory_agent.py",
        "src/response_planner.py",
        "src/routing.py",
    ]
    for script in scripts:
        print(f"[REFRESH] {script}")
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(f"{script} failed with exit code {result.returncode}")


def fmt(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def collect() -> dict:
    """Parse every metric artifact under output/ into one dict.

    Shared by the text summary below and by results_manifest.py, so both
    read the same numbers through the same parsers.
    """
    return {
        "anomaly": parse_anomaly_injection(OUT / "anomaly_injection_results.csv"),
        "multi": parse_multi_risk(OUT / "multi_domain_f1.csv"),
        "recovery": parse_recovery_metrics(OUT / "recovery_metrics.txt"),
        "lstm": parse_lstm_metrics(OUT / "lstm_metrics.txt"),
        "memory": parse_memory_report(OUT / "memory_report.txt"),
        "ppo": parse_ppo_results(OUT / "ppo_routing_results.txt"),
        "supplier": parse_supplier_report(OUT / "supplier_agent_report.txt"),
        "inventory": parse_inventory_report(OUT / "inventory_agent_report.txt"),
        "response_plan": parse_response_plan(OUT / "response_plan_report.txt"),
        "routing": parse_routing_report(OUT / "routing_results.txt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise project-wide metrics")
    parser.add_argument("--refresh", action="store_true", help="rerun benchmark-producing stages first")
    args = parser.parse_args()

    if args.refresh:
        refresh_benchmarks()

    data = collect()
    anomaly, multi, recovery, lstm, memory = (data[k] for k in ("anomaly", "multi", "recovery", "lstm", "memory"))
    ppo, supplier, inventory, response_plan, routing = (
        data[k] for k in ("ppo", "supplier", "inventory", "response_plan", "routing"))

    lines = [
        "PROJECT METRICS AUDIT",
        "=" * 60,
        "",
        "Anomaly detection (independent injection benchmark):",
    ]
    if anomaly:
        for method, values in anomaly.items():
            lines.append(
                f"  {method:<24} P={values['precision']:.3f}  R={values['recall']:.3f}  F1={values['f1']:.3f}"
            )
    else:
        lines.append("  n/a")

    lines += [
        "",
        "Multi-domain risk modelling:",
        f"  best industry            {multi.get('best_industry', 'n/a')}",
        f"  weighted F1              {fmt(multi.get('f1'))}",
        f"  baseline F1              {fmt(multi.get('baseline'))}",
        f"  lift                     {fmt(multi.get('lift'))}",
        "",
        "Recovery predictor:",
        f"  MAE                      {fmt(recovery.get('mae'), ' days')}",
        f"  R²                       {fmt(recovery.get('r2'))}",
        f"  baseline MAE             {fmt(recovery.get('baseline_mae'), ' days')}",
        f"  strategy spread          {fmt(recovery.get('strategy_spread'), ' days')}",
        "",
        "LSTM stress forecaster:",
        f"  MAE                      {fmt(lstm.get('mae'))}",
        f"  RMSE                     {fmt(lstm.get('rmse'))}",
        f"  persistence MAE          {fmt(lstm.get('baseline_mae'))}",
        "",
        "Immunological memory:",
        f"  k-NN MAE                 {fmt(memory.get('knn_mae'), ' days')}",
        f"  baseline MAE             {fmt(memory.get('base_mae'), ' days')}",
        f"  k-NN accuracy            {fmt(memory.get('knn_acc'), '%')}",
        f"  baseline accuracy        {fmt(memory.get('base_acc'), '%')}",
        "",
        "PPO routing:",
        f"  safer than Dijkstra      {fmt(ppo.get('safer_pct'), '%')}",
        "",
        "Supplier and inventory support:",
        f"  supplier entities        {fmt(supplier.get('analysed'))}",
        f"  supplier top-1 score     {fmt(supplier.get('avg_top1'))}",
        f"  inventory retailers      {fmt(inventory.get('analysed'))}",
        f"  inventory top-1 score    {fmt(inventory.get('avg_top1'))}",
        f"  delivery days            {fmt(inventory.get('avg_days'), ' days')}",
        "",
        "Response planner and routing:",
        f"  planned anomalies        {fmt(response_plan.get('planned'))}",
        f"  proven supplier rate     {fmt(response_plan.get('resolved_pct'), '%')}",
        f"  days saved per case      {fmt(response_plan.get('days_saved'), ' days')}",
        f"  routing status           {routing.get('status', 'n/a')}",
        f"  routing recovery days    {fmt(routing.get('recovery_days'), ' days')}",
        f"  routing baseline days    {fmt(routing.get('baseline_days'), ' days')}",
    ]

    summary = "\n".join(lines)
    print(summary)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "project_metrics_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"\nSaved summary -> {OUT / 'project_metrics_summary.txt'}")


if __name__ == "__main__":
    main()