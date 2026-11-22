"""
Paper figure: per-signal anomaly flag counts vs the calibrated ensemble.

Reads the counts from output/anomaly_metrics.txt (written by Stage 3), so the
figure always matches the current pipeline run. Run after main.py:

    python3 src/fig_anomalycounts.py

Writes output/figures/fig_anomalycounts.pdf (and a .png preview).
"""
import os
import re

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "output", "anomaly_metrics.txt")
OUT_DIR = os.path.join(ROOT, "output", "figures")

SIGNAL_BLUE = "#2a78d6"
ENSEMBLE_ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
AXIS = "#b5b4ae"

text = open(METRICS).read()
n_rows = int(re.search(r"Dataset size\s*:\s*([\d,]+)", text).group(1).replace(",", ""))
signals = [
    (label, int(re.search(rf"\[{i}\][^:]*:\s*(\d+)", text).group(1)))
    for i, label in enumerate(
        ["Volume", "Frequency", "Temporal surge", "Concentration", "Isolation Forest"], 1
    )
]
ensemble = int(re.search(r"High-confidence anomalies\s*:\s*(\d+)", text).group(1))

# Top to bottom: the five input signals, a gap, then the ensemble decision.
rows = signals + [("Calibrated ensemble", ensemble)]
y = [len(rows) - i for i in range(len(signals))] + [0.2]
colors = [SIGNAL_BLUE] * len(signals) + [ENSEMBLE_ORANGE]

plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.2, 2.9))

ax.barh(y, [c for _, c in rows], height=0.62, color=colors, edgecolor="white", linewidth=2)
xmax = max(c for _, c in rows)
for yi, (_, c) in zip(y, rows):
    ax.text(c + xmax * 0.012, yi, f"{c:,} ({100 * c / n_rows:.1f}%)",
            va="center", ha="left", color=INK, fontsize=8.5)

ax.set_yticks(y)
ax.set_yticklabels([label for label, _ in rows], color=INK)
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.axhline(0.85, color=AXIS, lw=0.8, ls=(0, (3, 3)))
ax.text(xmax * 1.22, 0.85 + 0.12, "input signals", ha="right", va="bottom",
        color=INK_2, fontsize=7.5, style="italic")
ax.text(xmax * 1.22, 0.85 - 0.12, "final decision", ha="right", va="top",
        color=INK_2, fontsize=7.5, style="italic")

ax.set_xlim(0, xmax * 1.22)
ax.set_ylim(-0.4, max(y) + 0.5)
ax.set_xlabel(f"Transactions flagged (of {n_rows:,})", color=INK_2)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.tick_params(axis="x", colors=INK_2, labelsize=8)
ax.tick_params(axis="y", length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)

fig.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT_DIR, f"fig_anomalycounts.{ext}"), dpi=200, bbox_inches="tight")
print("Saved figure -> output/figures/fig_anomalycounts.pdf")
