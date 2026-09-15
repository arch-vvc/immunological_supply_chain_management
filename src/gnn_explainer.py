"""
GNN Subgraph Explainer
======================
Identifies which neighboring nodes in the supply chain graph are most
responsible for a given node's high GNN risk score.

Approach: a similarity-and-tension HEURISTIC over the k-hop neighbourhood —
no gradients are computed. Each neighbour is scored by
    |cos(target_emb, neighbour_emb)| * neighbour_gnn_score
  + 0.3 * |projection of (target − neighbour) onto target|
The first term rewards neighbours that share the target's embedding pattern
AND are themselves anomalous; the second rewards neighbours whose embedding
pulls away from the target (tension). It is a fast attribution proxy, not
GNNExplainer/SubgraphX, and should be read as "which neighbours look most
implicated", not as a causal or gradient-based attribution.

Usage (standalone):
    python3 src/gnn_explainer.py --node "CARDINAL HEALTH INC"
    python3 src/gnn_explainer.py --node "MIAMI-LUKEN INC" --hops 2
Usage (from code — the dashboard's Risk tab does this):
    from gnn_explainer import explain_node_risk
    result = explain_node_risk("CARDINAL HEALTH INC", silent=True)
    # pass artifacts=(G, embeddings, risk_map) to reuse already-loaded objects
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

GRAPH_PATH   = ROOT / "models" / "supplychain_graph.pkl"
EMBED_PATH   = ROOT / "models" / "node_embeddings.pkl"
GNN_CSV      = ROOT / "output"  / "gnn_risk_scores.csv"

SEP = "=" * 72


# ── Load artefacts ────────────────────────────────────────────────────────────

def _load():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")
    if not EMBED_PATH.exists():
        raise FileNotFoundError(f"Node embeddings not found: {EMBED_PATH}")

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    with open(EMBED_PATH, "rb") as f:
        embeddings = pickle.load(f)   # dict: node_name -> np.ndarray (16-dim)

    risk_map = {}
    if GNN_CSV.exists():
        df = pd.read_csv(GNN_CSV)
        for _, row in df.iterrows():
            risk_map[row["entity"]] = {
                "gnn_score":    float(row.get("gnn_score", 0)),
                "recon_error":  float(row.get("recon_error", 0)),
                "node_type":    str(row.get("node_type", "?")),
                "rank":         int(row.get("rank", 9999)),
            }
    return G, embeddings, risk_map


# ── Core explainability ───────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def explain_node_risk(
    node: str,
    hops: int = 1,
    top_n: int = 5,
    silent: bool = False,
    artifacts: Optional[tuple] = None,
) -> dict:
    """
    Explain why `node` has a high GNN risk score.

    Returns a dict with:
        node            : target node name
        node_risk       : GNN risk score for the target
        recon_error     : reconstruction error (anomaly signal)
        top_drivers     : list of dicts — the most influential neighbours
        risk_subgraph   : all nodes in the explanation subgraph
    """
    # `artifacts` may supply a preloaded (G, embeddings, risk_map) triple so a
    # caller that already holds them (the dashboard) skips the pickle loads.
    G, embeddings, risk_map = artifacts if artifacts is not None else _load()

    if node not in G:
        raise ValueError(f"Node '{node}' not found in graph.")
    if node not in embeddings:
        raise ValueError(f"No embedding for '{node}'. Run Stage 2 first.")

    target_emb  = np.array(embeddings[node], dtype=np.float32)
    target_info = risk_map.get(node, {})

    # ── Collect k-hop neighbourhood, recording each node's true BFS hop ──────
    hop_of: dict = {}
    frontier = {node}
    for h in range(1, hops + 1):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.predecessors(n))
            next_frontier.update(G.successors(n))
        for n in next_frontier:
            if n != node and n not in hop_of:
                hop_of[n] = h
        frontier = next_frontier
    neighbours = sorted(hop_of)            # deterministic order

    # ── Score each neighbour ──────────────────────────────────────────────────
    # influence = |cos sim| * neighbour risk + 0.3 * |embedding tension|
    scored = []
    for nb in neighbours:
        if nb not in embeddings:
            continue
        nb_emb  = np.array(embeddings[nb], dtype=np.float32)
        sim     = _cosine(target_emb, nb_emb)
        nb_info = risk_map.get(nb, {})
        nb_risk = nb_info.get("gnn_score", 0.0)

        # Tension: projection of (target − neighbour) onto the target
        # direction, normalised. Large → the neighbour sits far from the
        # target along the target's own axis in embedding space.
        diff_proj = float(np.dot(target_emb - nb_emb, target_emb)) / (
            np.dot(target_emb, target_emb) + 1e-9
        )
        influence = abs(sim) * nb_risk + abs(diff_proj) * 0.3
        hop_dist = hop_of[nb]                 # true BFS distance (1..hops)

        scored.append({
            "neighbour":    nb,
            "type":         nb_info.get("node_type", "?"),
            "gnn_risk":     round(nb_risk, 4),
            "similarity":   round(sim, 4),
            "influence":    round(influence, 5),
            "hop":          hop_dist,
        })

    scored.sort(key=lambda x: x["influence"], reverse=True)
    top_drivers = scored[:top_n]

    result = {
        "node":         node,
        "node_type":    target_info.get("node_type", "?"),
        "node_risk":    round(target_info.get("gnn_score", 0.0), 4),
        "recon_error":  round(target_info.get("recon_error", 0.0), 4),
        "rank":         target_info.get("rank", "?"),
        "top_drivers":  top_drivers,
        "risk_subgraph":[node] + [d["neighbour"] for d in top_drivers],
    }

    if not silent:
        _print_result(result)

    return result


# ── Pretty printer ────────────────────────────────────────────────────────────

def _print_result(r: dict):
    print(f"\n{SEP}")
    print(f"  GNN RISK EXPLANATION  |  {r['node']}")
    print(SEP)
    print(f"  Node type    : {r['node_type']}")
    print(f"  GNN risk     : {r['node_risk']}  (rank #{r['rank']} in network)")
    print(f"  Recon error  : {r['recon_error']}  (higher = more anomalous neighbourhood)")
    print()
    print(f"  Top {len(r['top_drivers'])} neighbours driving this risk score:")
    print(f"  {'Rank':<5} {'Neighbour':<38} {'Type':<14} {'Risk':>7} {'Similarity':>11} {'Influence':>10}")
    print(f"  {'-'*5} {'-'*38} {'-'*14} {'-'*7} {'-'*11} {'-'*10}")
    for i, d in enumerate(r["top_drivers"], 1):
        print(
            f"  [{i}]   {d['neighbour']:<38} {d['type']:<14} "
            f"{d['gnn_risk']:>7.4f} {d['similarity']:>11.4f} {d['influence']:>10.5f}"
        )
    print()
    print(f"  Risk subgraph: {' -> '.join(r['risk_subgraph'][:4])}" +
          (f" (+{len(r['risk_subgraph'])-4} more)" if len(r['risk_subgraph']) > 4 else ""))
    print(f"  Interpretation: The neighbours above have embedding patterns that")
    print(f"  'pull' {r['node'][:28]} toward an anomalous reconstruction region.")
    print(f"  Disrupting or monitoring these nodes first reduces cascade risk.")
    print(SEP)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Subgraph Risk Explainer")
    parser.add_argument("--node", type=str, default="MIAMI-LUKEN INC",
                        help="Node to explain")
    parser.add_argument("--hops", type=int, default=1,
                        help="Neighbourhood depth (1 or 2)")
    parser.add_argument("--top",  type=int, default=5,
                        help="Number of top driver nodes to show")
    args = parser.parse_args()
    explain_node_risk(args.node, hops=args.hops, top_n=args.top)
