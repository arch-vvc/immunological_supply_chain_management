"""
Stage 10 — LSTM Macro Stress Forecaster
========================================
Trains a single-layer LSTM on weekly macro freight stress scores (Stage 9 output),
predicts 4 weeks ahead, and exports a forecast timeline + figure.

Architecture : LSTM(input=1, hidden=64, layers=1) → Linear(64, 4)
Training     : MSE loss, Adam, 300 epochs, 80/20 split
Output       : models/lstm_stress_model.pth
               output/stress_forecast.csv
               output/figures/fig8_stress_forecast.png
"""

import sys, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(BASE, "output", "macro_stress_scores.csv")
OUT_CSV = os.path.join(BASE, "output", "stress_forecast.csv")
OUT_FIG = os.path.join(BASE, "output", "figures", "fig8_stress_forecast.png")
OUT_MDL = os.path.join(BASE, "models",  "lstm_stress_model.pth")

# ── Hyperparameters ────────────────────────────────────────────────────────
SEQ_LEN    = 12    # weeks of history fed into LSTM per sample
PRED_STEPS = 4     # weeks to forecast ahead
HIDDEN     = 64    # LSTM hidden units
LAYERS     = 1     # LSTM stacked layers
EPOCHS     = 300
LR         = 0.001
TRAIN_FRAC = 0.80

print("=" * 55)
print("  STAGE 10 — LSTM MACRO STRESS FORECASTER")
print("=" * 55)

# ── Load stress scores ─────────────────────────────────────────────────────
if not os.path.exists(IN_CSV):
    print(f"[ERROR] {IN_CSV} not found. Run Stage 9 first.")
    sys.exit(1)

df     = pd.read_csv(IN_CSV, parse_dates=["date"])
df     = df.sort_values("date").reset_index(drop=True)

# Fill any NaN stress scores by interpolation then forward/back fill
nan_before = int(df["stress_score"].isna().sum())
df["stress_score"] = (df["stress_score"]
                      .interpolate(method="linear")
                      .ffill()
                      .bfill())
if nan_before:
    print(f"  ⚠  Filled {nan_before} NaN scores via linear interpolation")

scores = df["stress_score"].values.astype(np.float32)

print(f"  Loaded {len(scores)} weekly stress scores")
print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Score range: {scores.min():.3f} → {scores.max():.3f}")

# ── Min-max normalise ──────────────────────────────────────────────────────
s_min, s_max = float(scores.min()), float(scores.max())
norm = (scores - s_min) / (s_max - s_min + 1e-8)

# ── Build sliding-window sequences ─────────────────────────────────────────
def make_sequences(data, seq_len, pred_steps):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_steps + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_steps])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y   = make_sequences(norm, SEQ_LEN, PRED_STEPS)
split  = int(len(X) * TRAIN_FRAC)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# PyTorch tensors — shape (batch, seq_len, features=1)
Xt = torch.from_numpy(X_tr).unsqueeze(-1)
yt = torch.from_numpy(y_tr)
Xv = torch.from_numpy(X_te).unsqueeze(-1)
yv = torch.from_numpy(y_te)

print(f"\n  Sequences — train: {len(Xt)}   test: {len(Xv)}")
print(f"  Window: {SEQ_LEN} weeks → predict {PRED_STEPS} weeks ahead")

# ── Model definition ───────────────────────────────────────────────────────
class StressLSTM(nn.Module):
    def __init__(self, hidden, layers, pred_steps):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=1, hidden_size=hidden,
                              num_layers=layers, batch_first=True,
                              dropout=0.0)
        self.linear = nn.Linear(hidden, pred_steps)

    def forward(self, x):
        out, _ = self.lstm(x)       # (batch, seq, hidden)
        last   = out[:, -1, :]      # last timestep only
        return self.linear(last)    # (batch, pred_steps)

model     = StressLSTM(HIDDEN, LAYERS, PRED_STEPS)

# ── Continual learning: load previous weights if they exist ───
if os.path.exists(OUT_MDL):
    try:
        model.load_state_dict(torch.load(OUT_MDL, weights_only=True))
        print("  [CONTINUAL] Loaded previous LSTM weights — fine-tuning on new data")
    except Exception:
        print("  [CONTINUAL] Previous weights incompatible — training from scratch")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Training loop ──────────────────────────────────────────────────────────
print(f"\n  Training LSTM ({EPOCHS} epochs, hidden={HIDDEN}) ...")
model.train()
train_losses = []
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    pred  = model(Xt)
    loss  = criterion(pred, yt)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if epoch % 50 == 0:
        print(f"    Epoch {epoch:3d}/{EPOCHS}  loss={loss.item():.6f}")

# ── Evaluation ─────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    val_pred_norm = model(Xv).numpy()
    val_true_norm = yv.numpy()

# Denormalise
def denorm(x):
    return x * (s_max - s_min) + s_min

val_pred = denorm(val_pred_norm)
val_true = denorm(val_true_norm)

mae  = float(np.mean(np.abs(val_pred - val_true)))
rmse = float(np.sqrt(np.mean((val_pred - val_true) ** 2)))

print(f"\n  ── Test Set Metrics ──────────────────────")
print(f"  MAE  : {mae:.4f}  (avg error per predicted week)")
print(f"  RMSE : {rmse:.4f}")

torch.save(model.state_dict(), OUT_MDL)
print(f"  Model saved → {OUT_MDL}")

# ── Forecast: next PRED_STEPS weeks beyond dataset ─────────────────────────
last_seq   = torch.from_numpy(norm[-SEQ_LEN:]).float().unsqueeze(0).unsqueeze(-1)
with torch.no_grad():
    future_norm   = model(last_seq).numpy()[0]

future_scores = np.clip(denorm(future_norm), 0.0, 1.0)
last_date     = df["date"].max()
future_dates  = pd.date_range(
    start   = last_date + pd.Timedelta(weeks=1),
    periods = PRED_STEPS,
    freq    = "W"
)

def stress_label(v):
    if v >= 0.65: return "HIGH"
    if v >= 0.40: return "MEDIUM"
    return "LOW"

forecast_df = pd.DataFrame({
    "date"        : future_dates,
    "stress_score": future_scores,
    "stress_level": [stress_label(v) for v in future_scores],
    "type"        : "forecast",
})

hist_out            = df[["date", "stress_score", "stress_level"]].copy()
hist_out["type"]    = "historical"
combined            = pd.concat([hist_out, forecast_df], ignore_index=True)
combined.to_csv(OUT_CSV, index=False)

print(f"\n  Forecast saved → {OUT_CSV}")
print(f"  Next {PRED_STEPS} weeks:")
for _, row in forecast_df.iterrows():
    lbl   = row["stress_level"]
    badge = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(lbl, "")
    print(f"    {row['date'].date()}  score={row['stress_score']:.3f}  {badge} {lbl}")

# ── Figure ─────────────────────────────────────────────────────────────────
BG      = "#0f0f1a"
BLUE    = "#4fc3f7"
RED     = "#ff6b6b"
ORANGE  = "#ffaa44"
GREEN   = "#44dd88"
GREY    = "#aaaaaa"

fig, axes = plt.subplots(2, 1, figsize=(15, 10))
fig.patch.set_facecolor(BG)
fig.suptitle("LSTM Macro Freight Stress Forecaster — Stage 10",
             color="white", fontsize=14, y=0.98)

for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    ax.tick_params(colors=GREY)
    ax.yaxis.label.set_color(GREY)

# ── Panel 1: Full history ──────────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(hist_out["date"], hist_out["stress_score"],
         color=BLUE, lw=1.1, alpha=0.85, label="Historical stress")
ax1.axhspan(0.65, 1.0,  alpha=0.07, color="red")
ax1.axhspan(0.40, 0.65, alpha=0.07, color="orange")
ax1.axhspan(0.0,  0.40, alpha=0.07, color="green")
ax1.axhline(0.65, color="red",    ls="--", lw=0.7, alpha=0.5, label="HIGH threshold")
ax1.axhline(0.40, color="orange", ls="--", lw=0.7, alpha=0.5, label="MEDIUM threshold")

# Forecast points
ax1.axvspan(hist_out["date"].max(), forecast_df["date"].max(),
            alpha=0.10, color=RED, label="Forecast window")
ax1.plot(forecast_df["date"], forecast_df["stress_score"],
         "o--", color=RED, lw=2, ms=9, zorder=6, label="4-week LSTM forecast")

ax1.set_title("Full Stress History + Forward Forecast", color="white", fontsize=11, pad=8)
ax1.set_ylabel("Stress Score (0-1)", color=GREY)
ax1.set_ylim(0, 1)
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8.5,
           loc="upper left", framealpha=0.8)

# ── Panel 2: Zoom last 52 weeks + annotated forecast ──────────────────────
ax2  = axes[1]
cutoff = hist_out["date"].max() - pd.Timedelta(weeks=52)
recent = hist_out[hist_out["date"] >= cutoff]

ax2.plot(recent["date"], recent["stress_score"],
         color=BLUE, lw=1.6, label="Last 52 weeks")
ax2.axhline(0.65, color="red",    ls="--", lw=0.8, alpha=0.5)
ax2.axhline(0.40, color="orange", ls="--", lw=0.8, alpha=0.5)

ax2.plot(forecast_df["date"], forecast_df["stress_score"],
         "o--", color=RED, lw=2.2, ms=10, zorder=6, label="LSTM Forecast")

colour_map = {"HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#44ff88"}
for _, row in forecast_df.iterrows():
    c = colour_map[row["stress_level"]]
    ax2.annotate(
        f"{row['stress_level']}\n{row['stress_score']:.3f}",
        xy       = (row["date"], row["stress_score"]),
        xytext   = (0, 18),
        textcoords = "offset points",
        ha       = "center",
        fontsize = 8,
        color    = c,
        arrowprops = dict(arrowstyle="-", color=c, lw=0.9),
    )

ax2.set_title("Zoom: Last 52 Weeks + Annotated 4-Week Forecast", color="white",
              fontsize=11, pad=8)
ax2.set_ylabel("Stress Score (0-1)", color=GREY)
ax2.set_ylim(0, 1)
ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8.5, framealpha=0.8)

plt.tight_layout(pad=2.5)
os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
plt.savefig(OUT_FIG, dpi=130, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\n  Figure saved → {OUT_FIG}")

print("\n  Stage 10 complete.")
print(f"  LSTM learned temporal patterns across {len(scores)} weeks of freight data.")
print(f"  Forward forecast feeds into the Streamlit dashboard risk-alert panel.")
