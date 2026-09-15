"""
Stage 7 — GNN Node Encoder
===========================
Maps to: Adaptive Immunity — the system LEARNS structural patterns
         rather than relying on fixed hand-crafted metrics.

Implements a 2-layer Graph Convolutional Network (GCN) in pure PyTorch.
Trained as an autoencoder — learns to compress each node into a 16-dim
embedding and reconstruct its original features from that embedding.

Nodes the model reconstructs poorly are scored as structurally anomalous.
Because that is a heuristic until tested, this stage also runs an
INDEPENDENT NODE-LEVEL INJECTION BENCHMARK: on held-out seeds it plants
structural anomalies into copies of the graph (starved / flooded / rewired
nodes), retrains from scratch on each copy, and measures how well the GNN
score ranks the planted nodes against simple degree/volume baselines.
Those numbers (AUC, precision@k, mean ± std over seeds) are written to
output/gnn_injection_metrics.txt / .json and are the only basis on which
the GNN score should be called an anomaly detector.

Node features used (7 per node):
    in_degree, out_degree, total_in_volume, total_out_volume,
    is_manufacturer, is_distributor, is_retailer

Architecture:
    Encoder: 7 → 32 → 16  (GCN layers with ReLU)
    Decoder: 16 → 32 → 7  (linear layers)
    Loss:    MSE reconstruction

Outputs:
    models/node_embeddings.pkl        — {node: 16-dim numpy array}
    models/gnn_autoencoder.pth        — trained weights
    output/gnn_risk_scores.csv        — enhanced risk (centrality + GNN)
    output/gnn_injection_metrics.txt  — node-level injection benchmark
    output/gnn_injection_eval.json    — same, machine-readable
    output/figures/fig5_gnn_embeddings.png  — PCA plot of embeddings

Run:  python3 src/gnn_encoder.py        (or python3 main.py --only 7)
"""

import json
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_IN   = os.path.join(ROOT, "models",  "supplychain_graph.pkl")
RISK_IN    = os.path.join(ROOT, "output",  "risk_scores.csv")
EMBED_OUT  = os.path.join(ROOT, "models",  "node_embeddings.pkl")
GNN_CKPT   = os.path.join(ROOT, "models",  "gnn_autoencoder.pth")
RISK_OUT   = os.path.join(ROOT, "output",  "gnn_risk_scores.csv")
FIG_OUT    = os.path.join(ROOT, "output",  "figures", "fig5_gnn_embeddings.png")
EVAL_TXT   = os.path.join(ROOT, "output",  "gnn_injection_metrics.txt")
EVAL_JSON  = os.path.join(ROOT, "output",  "gnn_injection_eval.json")

EPOCHS = 300
LR     = 0.01
SEED   = 42

# Injection benchmark: held-out seeds (the model is never tuned on them),
# how many nodes to plant per seed, and the perturbations used.
EVAL_SEEDS    = (43, 44, 45, 46)
N_INJECT      = 40
STARVE_FACTOR = 0.05      # incident edge weights x0.05  (a node that went quiet)
FLOOD_FACTOR  = 20.0      # incident edge weights x20    (a node that surged)
REWIRE_EDGES  = 6         # new edges to random partners (a node that changed shape)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("[ERROR] PyTorch not found. Install it with:\n  pip install torch")
    sys.exit(1)

import networkx as nx


def set_seed(seed: int) -> None:
    """Seed every RNG the training loop touches (Python, NumPy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Features + normalised adjacency ────────────────────────────────────────

def build_inputs(G):
    """Return (nodes, X, A_hat, type_map) for graph G — deterministic in G's
    node order."""
    nodes    = list(G.nodes())
    n_nodes  = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_vol  = {n: 0.0 for n in nodes}
    out_vol = {n: 0.0 for n in nodes}
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        out_vol[u] = out_vol.get(u, 0.0) + w
        in_vol[v]  = in_vol.get(v, 0.0) + w
    type_map = {n: G.nodes[n].get("type", "unknown") for n in nodes}

    X = np.zeros((n_nodes, 7), dtype=np.float32)
    for i, n in enumerate(nodes):
        X[i, 0] = in_deg.get(n, 0)
        X[i, 1] = out_deg.get(n, 0)
        X[i, 2] = np.log1p(in_vol.get(n, 0))
        X[i, 3] = np.log1p(out_vol.get(n, 0))
        X[i, 4] = 1.0 if type_map[n] == "manufacturer" else 0.0
        X[i, 5] = 1.0 if type_map[n] == "distributor"  else 0.0
        X[i, 6] = 1.0 if type_map[n] == "retailer"     else 0.0
    for col in range(4):                       # continuous features → 0-1
        col_max = X[:, col].max()
        if col_max > 0:
            X[:, col] /= col_max

    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        A[i, j] += 1.0
        A[j, i] += 1.0                          # undirected message passing
    A += np.eye(n_nodes, dtype=np.float32)      # self-loops
    D_inv = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-8)))
    A_hat = D_inv @ A @ D_inv
    return nodes, X, A_hat, type_map


# ── Model ──────────────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim,     hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, embed_dim,  bias=False)

    def forward(self, A, X):
        H = F.relu(self.W1(A @ X))          # H = ReLU(A_hat X W1)
        return F.relu(self.W2(A @ H))       # Z = ReLU(A_hat H W2)


class GCNAutoencoder(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=32, embed_dim=16):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_dim))

    def forward(self, A, X):
        Z = self.encoder(A, X)
        return Z, self.decoder(Z)


def train_autoencoder(X, A_hat, seed=SEED, epochs=EPOCHS, warm_start_path=None, verbose=True):
    """Train from scratch (seeded). Warm start only if a path is given — the
    caller decides (ISC_CONTINUAL=1 in main); the benchmark never warm-starts."""
    set_seed(seed)
    model = GCNAutoencoder()
    if warm_start_path and os.path.exists(warm_start_path):
        try:
            model.load_state_dict(torch.load(warm_start_path, weights_only=True))
            if verbose:
                print("  [CONTINUAL] ISC_CONTINUAL=1 — loaded previous GNN weights, fine-tuning")
        except Exception:
            if verbose:
                print("  [CONTINUAL] Previous weights incompatible — training from scratch")
    A_t = torch.tensor(A_hat, dtype=torch.float32)
    X_t = torch.tensor(X,     dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        _, X_hat = model(A_t, X_t)
        loss = crit(X_hat, X_t)
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:>3}/{epochs}  loss={loss.item():.6f}")
    return model


def score_nodes(model, X, A_hat):
    """Return (embeddings, recon_errors, gnn_score in 0-1)."""
    model.eval()
    with torch.no_grad():
        Z, X_hat = model(torch.tensor(A_hat, dtype=torch.float32),
                         torch.tensor(X, dtype=torch.float32))
    emb = Z.numpy()
    rec = np.mean((X - X_hat.numpy()) ** 2, axis=1)
    z = (rec - rec.mean()) / (rec.std() + 1e-8)
    return emb, rec, (z - z.min()) / (z.max() - z.min() + 1e-8)


# ── Independent node-level injection benchmark ─────────────────────────────

def inject_structural_anomalies(G, seed, n_inject=N_INJECT):
    """Plant n_inject structural anomalies into a copy of G. Each planted node
    is starved, flooded, or rewired (an equal mix, chosen by the seed) — the
    three ways a node's neighbourhood pattern breaks in real disruptions.
    Returns (G_perturbed, set_of_planted_nodes, {node: kind})."""
    rng = random.Random(seed)
    H = G.copy()
    candidates = [n for n, d in H.nodes(data=True)
                  if d.get("type") in ("distributor", "retailer") and H.degree(n) >= 2]
    planted = rng.sample(sorted(candidates), n_inject)
    kinds = {}
    by_type = {}
    for n, d in H.nodes(data=True):
        by_type.setdefault(d.get("type", "unknown"), []).append(n)
    for i, n in enumerate(planted):
        kind = ("starve", "flood", "rewire")[i % 3]
        kinds[n] = kind
        if kind in ("starve", "flood"):
            f = STARVE_FACTOR if kind == "starve" else FLOOD_FACTOR
            for u, v, d in list(H.in_edges(n, data=True)) + list(H.out_edges(n, data=True)):
                d["weight"] = d.get("weight", 1.0) * f
        else:
            # new edges to random partners of the tier this node usually connects to
            ntype = H.nodes[n].get("type")
            partner_type = "retailer" if ntype == "distributor" else "distributor"
            partners = [p for p in by_type.get(partner_type, []) if p != n]
            for p in rng.sample(partners, min(REWIRE_EDGES, len(partners))):
                if ntype == "distributor":
                    H.add_edge(n, p, weight=1.0)
                else:
                    H.add_edge(p, n, weight=1.0)
    return H, set(planted), kinds


def _auc(scores, labels):
    """Rank-based AUC (Mann-Whitney), no sklearn needed."""
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels, dtype=int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    for v in np.unique(scores):
        idx = np.where(scores == v)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _prec_at_k(scores, labels, k):
    top = np.argsort(scores)[::-1][:k]
    return float(np.asarray(labels)[top].mean())


def injection_benchmark(G, seeds=EVAL_SEEDS, n_inject=N_INJECT):
    """For each held-out seed: perturb, retrain from scratch, score every
    node with the GNN and with two baselines, and measure how well each ranks
    the planted nodes. Returns a dict with per-seed rows and mean ± std."""
    rows = []
    for seed in seeds:
        H, planted, kinds = inject_structural_anomalies(G, seed, n_inject)
        nodes, X, A_hat, _ = build_inputs(H)
        model = train_autoencoder(X, A_hat, seed=seed, verbose=False)
        _, rec, gnn = score_nodes(model, X, A_hat)
        labels = np.array([1 if n in planted else 0 for n in nodes])
        # Baselines on the SAME perturbed graph:
        #   volume_z  — |z| of log in+out volume (what "flood/starve" changes)
        #   degree_z  — |z| of in+out degree     (what "rewire" changes)
        vol = X[:, 2] + X[:, 3]; deg = X[:, 0] + X[:, 1]
        vol_z = np.abs((vol - vol.mean()) / (vol.std() + 1e-8))
        deg_z = np.abs((deg - deg.mean()) / (deg.std() + 1e-8))
        naive = np.maximum(vol_z, deg_z)
        row = {"seed": seed, "n_nodes": len(nodes), "n_injected": int(labels.sum())}
        for name, s in (("gnn", gnn), ("volume_z", vol_z), ("degree_z", deg_z), ("max_z", naive)):
            row[f"{name}_auc"] = round(_auc(s, labels), 4)
            row[f"{name}_p_at_k"] = round(_prec_at_k(s, labels, n_inject), 4)
        # per-kind recall@k for the GNN
        top = set(np.asarray(nodes)[np.argsort(gnn)[::-1][:n_inject]].tolist())
        for kind in ("starve", "flood", "rewire"):
            ks = [n for n, k in kinds.items() if k == kind]
            row[f"gnn_recall_{kind}"] = round(sum(1 for n in ks if n in top) / max(1, len(ks)), 4)
        rows.append(row)
    keys = [k for k in rows[0] if k not in ("seed", "n_nodes", "n_injected")]
    summary = {k: {"mean": round(float(np.mean([r[k] for r in rows])), 4),
                   "std":  round(float(np.std([r[k] for r in rows], ddof=1)), 4)} for k in keys}
    return {"seeds": list(seeds), "n_injected_per_seed": n_inject,
            "perturbations": {"starve_factor": STARVE_FACTOR, "flood_factor": FLOOD_FACTOR,
                              "rewire_edges": REWIRE_EDGES},
            "per_seed": rows, "summary": summary}


def _benchmark_text(b):
    s = b["summary"]
    def ms(k): return f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}"
    lines = [
        "STAGE 7 — GNN SCORE: NODE-LEVEL INJECTION BENCHMARK",
        "=" * 58,
        "",
        f"Held-out seeds {b['seeds']}; {b['n_injected_per_seed']} structural anomalies planted",
        "per seed into a copy of the real graph (equal mix of starved x"
        f"{b['perturbations']['starve_factor']}, flooded x{b['perturbations']['flood_factor']}, "
        f"rewired +{b['perturbations']['rewire_edges']} edges); the GCN autoencoder is retrained",
        "from scratch on each copy and its score is asked to rank the planted",
        "nodes. Baselines are |z| of log volume, |z| of degree, and their max —",
        "the obvious non-learned detectors for the same three perturbations.",
        "",
        f"{'detector':<12}{'AUC':>16}{'precision@k':>18}",
        f"{'-'*46}",
        f"{'GNN score':<12}{ms('gnn_auc'):>16}{ms('gnn_p_at_k'):>18}",
        f"{'|z| volume':<12}{ms('volume_z_auc'):>16}{ms('volume_z_p_at_k'):>18}",
        f"{'|z| degree':<12}{ms('degree_z_auc'):>16}{ms('degree_z_p_at_k'):>18}",
        f"{'max of both':<12}{ms('max_z_auc'):>16}{ms('max_z_p_at_k'):>18}",
        "",
        f"GNN recall@k by perturbation: starve {ms('gnn_recall_starve')}, "
        f"flood {ms('gnn_recall_flood')}, rewire {ms('gnn_recall_rewire')}",
        "",
    ]
    g, m = s["gnn_auc"]["mean"], s["max_z_auc"]["mean"]
    if g > m + 0.02:
        lines.append("Reading: the learned score ranks planted structural anomalies better than")
        lines.append("the degree/volume z-score baselines; the GNN adds signal beyond simple stats.")
    elif g >= m - 0.02:
        lines.append("Reading: the learned score is on par with the degree/volume z-score baselines;")
        lines.append("it does not add detection power beyond simple stats on these perturbations.")
    else:
        lines.append("Reading: the learned score ranks planted anomalies WORSE than simple degree/")
        lines.append("volume z-scores. The GNN score should not be presented as an anomaly detector.")
    lines.append("This is the only ground truth behind the 'structurally anomalous' label;")
    lines.append("the 60/40 blend into enhanced_risk should be read with it in mind.")
    return "\n".join(lines)


# ── Stage entry point ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STAGE 7 — GNN NODE ENCODER (Adaptive Immunity)")
    print("=" * 60)
    if not os.path.exists(MODEL_IN):
        print(f"[ERROR] {MODEL_IN} not found. Run build_chain.py first.")
        sys.exit(1)
    with open(MODEL_IN, "rb") as f:
        G = pickle.load(f)
    print(f"  Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("  Building node features + adjacency...")
    nodes, X, A_hat, type_map = build_inputs(G)

    # ── Warm start is OPT-IN (ISC_CONTINUAL=1). Default trains from scratch so
    #    every number is reproducible from a fresh clone and the results manifest
    #    means something; fine-tuning from a prior checkpoint makes the result
    #    depend on how many times the stage was run before on that machine.
    warm = GNN_CKPT if os.environ.get("ISC_CONTINUAL") == "1" else None
    if not warm:
        print("  Training from scratch (seed 42). Set ISC_CONTINUAL=1 to fine-tune a prior checkpoint.")
    print("  Training GCN autoencoder...")
    model = train_autoencoder(X, A_hat, seed=SEED, warm_start_path=warm)
    os.makedirs(os.path.dirname(GNN_CKPT), exist_ok=True)
    torch.save(model.state_dict(), GNN_CKPT)

    embeddings, recon_errors, gnn_scores_norm = score_nodes(model, X, A_hat)
    print(f"\n  Embeddings shape : {embeddings.shape}")
    print(f"  Recon error      : mean={recon_errors.mean():.6f}  max={recon_errors.max():.6f}")

    with open(EMBED_OUT, "wb") as f:
        pickle.dump({nodes[i]: embeddings[i] for i in range(len(nodes))}, f)
    print(f"\n  Embeddings saved → {EMBED_OUT}")

    gnn_df = pd.DataFrame({"entity": nodes, "node_type": [type_map[n] for n in nodes],
                           "recon_error": recon_errors, "gnn_score": gnn_scores_norm})
    merged = None
    if os.path.exists(RISK_IN):
        risk_df = pd.read_csv(RISK_IN)
        merged = gnn_df.merge(risk_df[["entity", "risk_score"]], on="entity", how="left")
        merged["risk_score"] = merged["risk_score"].fillna(0)
        merged["enhanced_risk"] = 0.60 * merged["risk_score"] + 0.40 * merged["gnn_score"]  # 60% centrality + 40% GNN
        merged = merged.sort_values("enhanced_risk", ascending=False).reset_index(drop=True)
        merged["rank"] = merged.index + 1
        merged.to_csv(RISK_OUT, index=False)
        print(f"  Enhanced risk scores saved → {RISK_OUT}")
        print(f"\n  Top 10 Nodes (Enhanced Risk):")
        print(f"  {'Rank':<5} {'Entity':<40} {'Type':<15} {'Enhanced':<10} {'GNN':<8}")
        print("  " + "─" * 78)
        for _, row in merged.head(10).iterrows():
            print(f"  {int(row['rank']):<5} {str(row['entity'])[:39]:<40} "
                  f"{row['node_type']:<15} {row['enhanced_risk']:.4f}    {row['gnn_score']:.4f}")
    else:
        gnn_df.to_csv(RISK_OUT, index=False)
        print(f"  GNN risk scores saved → {RISK_OUT}")

    # ── Injection benchmark (held-out seeds, from-scratch retrains) ─────────
    print(f"\n  Node-level injection benchmark on seeds {list(EVAL_SEEDS)} ...")
    bench = injection_benchmark(G)
    txt = _benchmark_text(bench)
    with open(EVAL_TXT, "w") as f:
        f.write(txt + "\n")
    with open(EVAL_JSON, "w") as f:
        json.dump(bench, f, indent=2)
    print("\n" + "\n".join("  " + l for l in txt.splitlines()[9:]))
    print(f"\n  Benchmark saved → {EVAL_TXT}")

    # ── PCA figure ───────────────────────────────────────────────────────────
    print("\n  Generating Fig 5: GNN Embedding Space (PCA)...")
    E = embeddings - embeddings.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(E.T))
    proj = E @ vecs[:, [-1, -2]]
    color_map = {"manufacturer": "#e74c3c", "distributor": "#f39c12",
                 "retailer": "#2ecc71", "unknown": "#95a5a6"}
    node_colors = [color_map.get(type_map[n], "#95a5a6") for n in nodes]
    sizes = 40 + 300 * gnn_scores_norm
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0f0f1a"); ax.set_facecolor("#0f0f1a")
    ax.scatter(proj[:, 0], proj[:, 1], c=node_colors, s=sizes, alpha=0.75,
               edgecolors="white", linewidths=0.3)
    if merged is not None:
        top_nodes = set(merged.head(10)["entity"].tolist())
    else:
        top_nodes = {nodes[i] for i in np.argsort(gnn_scores_norm)[-10:]}
    for i, n in enumerate(nodes):
        if n in top_nodes:
            ax.annotate(str(n)[:25], (proj[i, 0], proj[i, 1]), fontsize=6, color="white",
                        alpha=0.85, xytext=(5, 5), textcoords="offset points")
    patches = [mpatches.Patch(color=v, label=k.capitalize())
               for k, v in color_map.items() if k != "unknown"]
    ax.legend(handles=patches, loc="upper left", facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.set_title("GNN Embedding Space — PCA Projection\n"
                 "(Node size = reconstruction error / anomaly score, Color = entity type)",
                 color="white", fontsize=11, pad=12)
    ax.set_xlabel("Principal Component 1", color="white")
    ax.set_ylabel("Principal Component 2", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#4a4a6a")
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {FIG_OUT}")
    print("\n  Stage 7 complete.")


if __name__ == "__main__":
    main()
