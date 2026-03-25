"""
Stage 7 — GNN Node Encoder
===========================
Maps to: Adaptive Immunity — the system LEARNS structural patterns
         rather than relying on fixed hand-crafted metrics.

Implements a 2-layer Graph Convolutional Network (GCN) in pure PyTorch.
Trained as an autoencoder — learns to compress each node into a 16-dim
embedding and reconstruct its original features from that embedding.

Nodes the model struggles to reconstruct are STRUCTURALLY ANOMALOUS —
they behave differently from their neighbours. This is a learned signal
on top of the rule-based centrality scores from Stage 4.

Node features used (7 per node):
    in_degree, out_degree, total_in_volume, total_out_volume,
    is_manufacturer, is_distributor, is_retailer

Architecture:
    Encoder: 7 → 32 → 16  (GCN layers with ReLU)
    Decoder: 16 → 32 → 7  (linear layers)
    Loss:    MSE reconstruction

Outputs:
    models/node_embeddings.pkl        — {node: 16-dim numpy array}
    output/gnn_risk_scores.csv        — enhanced risk (centrality + GNN)
    output/figures/fig5_gnn_embeddings.png  — PCA plot of embeddings
"""

import os
import sys
import pickle
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
RISK_OUT   = os.path.join(ROOT, "output",  "gnn_risk_scores.csv")
FIG_OUT    = os.path.join(ROOT, "output",  "figures", "fig5_gnn_embeddings.png")

os.makedirs(os.path.join(ROOT, "models"),          exist_ok=True)
os.makedirs(os.path.join(ROOT, "output", "figures"), exist_ok=True)

print("=" * 55)
print("  STAGE 7 — GNN NODE ENCODER")
print("=" * 55)

# ── Check dependencies ────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("[ERROR] PyTorch not found. Install it with:")
    print("  pip install torch")
    sys.exit(1)

if not os.path.exists(MODEL_IN):
    print(f"[ERROR] {MODEL_IN} not found. Run build_chain.py first.")
    sys.exit(1)

# ── Load graph ────────────────────────────────────────────────
import networkx as nx
with open(MODEL_IN, "rb") as f:
    G = pickle.load(f)

nodes     = list(G.nodes())
n_nodes   = len(nodes)
node_idx  = {n: i for i, n in enumerate(nodes)}

print(f"  Graph loaded: {n_nodes} nodes, {G.number_of_edges()} edges")

# ── Build node feature matrix (N x 7) ────────────────────────
print("  Building node features...")

in_deg   = dict(G.in_degree())
out_deg  = dict(G.out_degree())

in_vol   = {n: 0.0 for n in nodes}
out_vol  = {n: 0.0 for n in nodes}
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

# Normalise continuous features to 0-1
for col in range(4):
    col_max = X[:, col].max()
    if col_max > 0:
        X[:, col] /= col_max

# ── Build normalised adjacency matrix A_hat ───────────────────
# A_hat = D^{-1/2} (A + I) D^{-1/2}
print("  Building adjacency matrix...")

A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
for u, v in G.edges():
    i, j     = node_idx[u], node_idx[v]
    A[i, j] += 1.0
    A[j, i] += 1.0   # treat as undirected for message passing

A += np.eye(n_nodes, dtype=np.float32)  # self-loops
D     = np.diag(A.sum(axis=1))
D_inv = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-8)))
A_hat = D_inv @ A @ D_inv

A_hat_t = torch.tensor(A_hat, dtype=torch.float32)
X_t     = torch.tensor(X,     dtype=torch.float32)

# ── GCN Model ─────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim,     hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, embed_dim,  bias=False)

    def forward(self, A, X):
        # Layer 1: H = ReLU(A_hat @ X @ W1)
        H = F.relu(self.W1(A @ X))
        # Layer 2: Z = ReLU(A_hat @ H @ W2)
        Z = F.relu(self.W2(A @ H))
        return Z


class GCNAutoencoder(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=32, embed_dim=16):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, A, X):
        Z    = self.encoder(A, X)
        X_hat = self.decoder(Z)
        return Z, X_hat


# ── Training ──────────────────────────────────────────────────
print("  Training GCN autoencoder...")

EPOCHS = 300
LR     = 0.01

model     = GCNAutoencoder(in_dim=7, hidden_dim=32, embed_dim=16)

# ── Continual learning: load previous weights if they exist ───
GNN_CKPT = os.path.join(ROOT, "models", "gnn_autoencoder.pth")
if os.path.exists(GNN_CKPT):
    try:
        model.load_state_dict(torch.load(GNN_CKPT, weights_only=True))
        print("  [CONTINUAL] Loaded previous GNN weights — fine-tuning on new data")
    except Exception:
        print("  [CONTINUAL] Previous weights incompatible — training from scratch")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    Z, X_hat = model(A_hat_t, X_t)
    loss     = criterion(X_hat, X_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"    Epoch {epoch+1:>3}/{EPOCHS}  loss={loss.item():.6f}")

# ── Save weights for continual learning ───────────────────────
torch.save(model.state_dict(), GNN_CKPT)

# ── Extract embeddings & reconstruction error ─────────────────
model.eval()
with torch.no_grad():
    Z, X_hat = model(A_hat_t, X_t)

embeddings   = Z.numpy()         # (N, 16)
X_hat_np     = X_hat.numpy()
recon_errors = np.mean((X - X_hat_np) ** 2, axis=1)   # per-node MSE

# Z-score the reconstruction errors → GNN anomaly score
recon_mean = recon_errors.mean()
recon_std  = recon_errors.std()
gnn_scores = (recon_errors - recon_mean) / (recon_std + 1e-8)
gnn_scores_norm = (gnn_scores - gnn_scores.min()) / (gnn_scores.max() - gnn_scores.min() + 1e-8)

print(f"\n  Embeddings shape : {embeddings.shape}")
print(f"  Recon error      : mean={recon_mean:.6f}  max={recon_errors.max():.6f}")

# ── Save embeddings ───────────────────────────────────────────
embed_dict = {nodes[i]: embeddings[i] for i in range(n_nodes)}
with open(EMBED_OUT, "wb") as f:
    pickle.dump(embed_dict, f)
print(f"\n  Embeddings saved → {EMBED_OUT}")

# ── Enhanced risk scores (centrality + GNN) ───────────────────
gnn_df = pd.DataFrame({
    "entity"       : nodes,
    "node_type"    : [type_map[n] for n in nodes],
    "recon_error"  : recon_errors,
    "gnn_score"    : gnn_scores_norm,
})

if os.path.exists(RISK_IN):
    risk_df = pd.read_csv(RISK_IN)
    merged  = gnn_df.merge(risk_df[["entity", "risk_score"]], on="entity", how="left")
    merged["risk_score"] = merged["risk_score"].fillna(0)
    # Enhanced = 60% centrality + 40% GNN
    merged["enhanced_risk"] = (0.60 * merged["risk_score"] + 0.40 * merged["gnn_score"])
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

# ── PCA Visualisation ─────────────────────────────────────────
print("\n  Generating Fig 5: GNN Embedding Space (PCA)...")

# Manual 2-component PCA (no sklearn needed)
E  = embeddings - embeddings.mean(axis=0)
cov = np.cov(E.T)
vals, vecs = np.linalg.eigh(cov)
top2 = vecs[:, [-1, -2]]   # top 2 eigenvectors
proj = E @ top2             # (N, 2)

color_map  = {"manufacturer": "#e74c3c", "distributor": "#f39c12",
              "retailer": "#2ecc71",     "unknown": "#95a5a6"}
node_colors = [color_map.get(type_map[n], "#95a5a6") for n in nodes]

# Size by GNN anomaly score
sizes = 40 + 300 * gnn_scores_norm

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")

scatter = ax.scatter(proj[:, 0], proj[:, 1],
                     c=node_colors, s=sizes, alpha=0.75, edgecolors="white", linewidths=0.3)

# Label top 10 highest GNN anomaly nodes
if os.path.exists(RISK_IN):
    top_nodes = set(merged.head(10)["entity"].tolist())
else:
    top_idx   = np.argsort(gnn_scores_norm)[-10:]
    top_nodes = {nodes[i] for i in top_idx}

for i, n in enumerate(nodes):
    if n in top_nodes:
        ax.annotate(str(n)[:25], (proj[i, 0], proj[i, 1]),
                    fontsize=6, color="white", alpha=0.85,
                    xytext=(5, 5), textcoords="offset points")

patches = [mpatches.Patch(color=v, label=k.capitalize())
           for k, v in color_map.items() if k != "unknown"]
ax.legend(handles=patches, loc="upper left", facecolor="#1a1a2e",
          labelcolor="white", fontsize=9)

ax.set_title("GNN Embedding Space — PCA Projection\n"
             "(Node size = reconstruction error / anomaly score, Color = entity type)",
             color="white", fontsize=11, pad=12)
ax.set_xlabel("Principal Component 1", color="white")
ax.set_ylabel("Principal Component 2", color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("#4a4a6a")

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {FIG_OUT}")

print("\n  Stage 7 complete.")
print(f"  The GNN learned {embeddings.shape[1]}-dim embeddings for {n_nodes} supply chain nodes.")
print(f"  Nodes with high reconstruction error are structurally anomalous.")
