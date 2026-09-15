"""
Stage 6 — Visualization
Generates 4 figures saved to output/figures/:

  Fig 1 — Supply Chain Graph (subgraph of top 40 nodes by risk, colored by type)
  Fig 2 — Top 15 High-Risk Entities (composite score bar chart)
  Fig 3 — Anomaly Timeline (quantity over time, anomalies highlighted)
  Fig 4 — Anomaly Dimension Breakdown (how many transactions flagged per dimension)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL   = os.path.join(ROOT, "models", "supplychain_graph.pkl")
DATA    = os.path.join(ROOT, "data", "processed", "clean_chain.csv")
RISK    = os.path.join(ROOT, "output", "risk_scores.csv")
ANOMALY = os.path.join(ROOT, "output", "anomalies.csv")
FIGDIR  = os.path.join(ROOT, "output", "figures")

os.makedirs(FIGDIR, exist_ok=True)

print("=" * 55)
print("  STAGE 6 — VISUALIZATION")
print("=" * 55)

# ─────────────────────────────────────────────
# FIG 1 — Supply Chain Graph (top 40 nodes)
# ─────────────────────────────────────────────
print("Generating Figure 1: Supply Chain Network Graph...")

if os.path.exists(MODEL) and os.path.exists(RISK):
    with open(MODEL, "rb") as f:
        G = pickle.load(f)

    risk_df = pd.read_csv(RISK)
    # Build the drawn subgraph in a FIXED node order. G.subgraph(set) iterates
    # the set, whose order changes with Python's per-process hash seed, and
    # spring_layout's seeded initial positions follow node order — so the
    # same seed gave a different picture every run.
    top_list  = risk_df.head(40)["entity"].tolist()
    top_nodes = set(top_list)
    subG = nx.DiGraph()
    subG.add_nodes_from((n, dict(G.nodes[n])) for n in top_list if n in G)
    subG.add_edges_from((u, v, dict(d)) for u, v, d in G.edges(top_list, data=True)
                        if v in top_nodes)

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Node colors by type
    color_map = {"manufacturer": "#e74c3c", "distributor": "#f39c12", "retailer": "#2ecc71", "unknown": "#95a5a6"}
    node_colors = [color_map.get(G.nodes[n].get("type", "unknown"), "#95a5a6") for n in subG.nodes()]

    # Node sizes by risk score
    risk_lookup = dict(zip(risk_df["entity"], risk_df["risk_score"]))
    node_sizes  = [800 + 2000 * risk_lookup.get(n, 0) for n in subG.nodes()]

    pos = nx.spring_layout(subG, seed=42, k=2.5)

    nx.draw_networkx_edges(subG, pos, ax=ax, alpha=0.3, edge_color="#4a4a6a",
                           arrows=True, arrowsize=10, width=0.8)
    nx.draw_networkx_nodes(subG, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(subG, pos, ax=ax, font_size=6, font_color="white")

    # Legend
    patches = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in color_map.items() if k != "unknown"]
    ax.legend(handles=patches, loc="upper left", facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    ax.set_title("Supply Chain Network — Top 40 Risk Nodes\n(Node size = composite risk score, Color = entity type)",
                 color="white", fontsize=12, pad=15)
    ax.axis("off")

    out = os.path.join(FIGDIR, "fig1_supply_chain_graph.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")
else:
    print("  [SKIP] Model or risk scores not found.")

# ─────────────────────────────────────────────
# FIG 2 — Top 15 High-Risk Entities
# ─────────────────────────────────────────────
print("Generating Figure 2: Top 15 High-Risk Entities...")

if os.path.exists(RISK):
    risk_df = pd.read_csv(RISK).head(15)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    type_colors = {"manufacturer": "#e74c3c", "distributor": "#f39c12", "retailer": "#2ecc71", "unknown": "#95a5a6"}
    bar_colors  = [type_colors.get(t, "#95a5a6") for t in risk_df["node_type"]]

    bars = ax.barh(range(len(risk_df)), risk_df["risk_score"], color=bar_colors, edgecolor="white", linewidth=0.3)

    ax.set_yticks(range(len(risk_df)))
    ax.set_yticklabels([e[:40] for e in risk_df["entity"]], color="white", fontsize=8)
    ax.set_xlabel("Composite Risk Score", color="white")
    ax.set_title("Top 15 High-Risk Supply Chain Entities\n(Betweenness 50% + In-degree 30% + PageRank 20%)",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#4a4a6a")

    patches = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in type_colors.items() if k != "unknown"]
    ax.legend(handles=patches, loc="lower right", facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    ax.invert_yaxis()

    out = os.path.join(FIGDIR, "fig2_top_risk_entities.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")
else:
    print("  [SKIP] risk_scores.csv not found.")

# ─────────────────────────────────────────────
# FIG 3 — Anomaly Timeline
# ─────────────────────────────────────────────
print("Generating Figure 3: Anomaly Timeline...")

if os.path.exists(DATA) and os.path.exists(ANOMALY):
    df  = pd.read_csv(DATA)
    adf = pd.read_csv(ANOMALY)
    df["date"]  = pd.to_datetime(df["date"])
    adf["date"] = pd.to_datetime(adf["date"])

    monthly     = df.groupby(df["date"].dt.to_period("M"))["quantity"].sum().reset_index()
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    ax.fill_between(monthly["date"], monthly["quantity"], alpha=0.3, color="#3498db")
    ax.plot(monthly["date"], monthly["quantity"], color="#3498db", linewidth=1.5, label="Monthly Volume")

    # Overlay anomaly events
    for _, row in adf.iterrows():
        ax.axvline(x=pd.to_datetime(row["date"]), color="#e74c3c", alpha=0.6, linewidth=1.2, linestyle="--")

    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Total Transaction Quantity", color="white")
    ax.set_title("Transaction Volume Over Time — Anomaly Events Marked (red dashed)",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#4a4a6a")

    anomaly_line = mpatches.Patch(color="#e74c3c", alpha=0.6, label=f"Anomaly Events ({len(adf)})")
    vol_line     = mpatches.Patch(color="#3498db", alpha=0.5, label="Monthly Volume")
    ax.legend(handles=[vol_line, anomaly_line], facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    out = os.path.join(FIGDIR, "fig3_anomaly_timeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")
else:
    print("  [SKIP] clean_chain.csv or anomalies.csv not found.")

# ─────────────────────────────────────────────
# FIG 4 — Anomaly Dimension Breakdown
# ─────────────────────────────────────────────
print("Generating Figure 4: Anomaly Dimension Breakdown...")

if os.path.exists(ANOMALY):
    adf = pd.read_csv(ANOMALY)

    flag_cols = ["flag_volume", "flag_frequency", "flag_surge", "flag_concentration"]
    labels    = ["Volume\n(Z-score)", "Frequency\n(Pair count)", "Temporal\nSurge", "Concentration\nRisk"]

    counts = []
    for col in flag_cols:
        if col in adf.columns:
            counts.append(adf[col].sum())
        else:
            counts.append(0)

    colors = ["#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5, width=0.55)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(int(count)), ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")

    ax.set_ylabel("Number of Flagged Transactions", color="white")
    ax.set_title("Anomaly Dimension Breakdown\n(Multi-Dimensional Detection — flags per category)",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#4a4a6a")
    ax.set_ylim(0, max(counts) * 1.25 if counts else 5)

    out = os.path.join(FIGDIR, "fig4_anomaly_dimensions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")
else:
    print("  [SKIP] anomalies.csv not found.")

print(f"\nAll figures saved to: {FIGDIR}")
