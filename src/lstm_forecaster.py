"""
Stage 10 — LSTM Macro Stress Forecaster
========================================
Trains a single-layer LSTM on weekly macro freight stress (Stage 9 output),
predicts 4 weeks ahead, and exports a forecast timeline + figure.

Design (post-redesign):
  * Multivariate input — the composite plus the 5 raw indicator channels
    Stage 9 now exports (the composite alone is 4-week-smoothed, near random walk).
  * Delta targets — predicts the CHANGE from the last observed week and
    reconstructs levels (ŷ = last + delta), so the model starts at the
    persistence baseline instead of relearning the identity map.
  * Early stopping on a chronological validation slice; per-horizon MAE
    (t+1..t+4) reported against persistence.

HONEST FINDING: even with all of the above, the LSTM does not beat
persistence on this series (see lstm_metrics.txt). Early stopping converges
with validation loss ~= the variance of the deltas — i.e. the best learnable
weekly delta is ~0, which IS persistence. The indicators are interpolated
from monthly data, so weekly innovations simply are not in the data. Keep
quoting persistence as the operative forecast; the LSTM is retained as a
documented negative result, not a capability claim.

Architecture : LSTM(input=6, hidden=64, layers=1) → Linear(64, 4 deltas)
Output       : models/lstm_stress_model.pth
               output/stress_forecast.csv
               output/figures/fig8_stress_forecast.png
"""

import sys, os
import numpy as np
import pandas as pd
import torch
# Reproducibility — seed every RNG the training loop touches (Python, NumPy,
# torch). Without this, two runs of this stage produce different weights and
# different absolute scores (rankings agree, values do not), which the results
# manifest would flag as drift.
import random as _random
SEED = 42
_random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
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
EPOCHS     = 2500   # full-batch training: one epoch == one gradient step,
                    # so 300 was only 300 Adam steps — badly undertrained
LR         = 0.003
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

# Multivariate input: the raw indicator columns (exported by Stage 9) plus the
# composite. The composite is a 4-week-smoothed weighted average, so on its
# own it is nearly a random walk — the un-smoothed components carry the
# leading-indicator signal the composite throws away.
feature_cols = [c for c in df.columns if c not in ("date", "stress_score", "stress_level")]
value_cols   = ["stress_score"] + feature_cols
for c in value_cols:
    nan_before = int(df[c].isna().sum())
    df[c] = df[c].interpolate(method="linear").ffill().bfill()
    if nan_before:
        print(f"  ⚠  Filled {nan_before} NaNs in {c} via linear interpolation")

scores = df["stress_score"].values.astype(np.float32)
feats  = df[value_cols].values.astype(np.float32)     # (weeks, features), all ~0-1
N_FEAT = feats.shape[1]

print(f"  Loaded {len(scores)} weekly stress scores")
print(f"  Input features: composite + {len(feature_cols)} raw indicators "
      f"({N_FEAT} channels)" if feature_cols else
      "  Input features: composite only (rerun Stage 9 to export indicators)")
print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Score range: {scores.min():.3f} → {scores.max():.3f}")

# ── Build sliding-window sequences with DELTA targets ──────────────────────
# The model predicts the CHANGE from the last observed week, not the level:
#     ŷ_level[t+h] = stress[t] + f(window)[h]
# Predicting levels forces the network to first learn the identity map that
# persistence gets for free; residual framing starts AT persistence and
# spends capacity only on the deviation from it.
def make_sequences(mat, target, seq_len, pred_steps):
    X, y, last = [], [], []
    for i in range(len(target) - seq_len - pred_steps + 1):
        X.append(mat[i : i + seq_len])
        anchor = target[i + seq_len - 1]
        y.append(target[i + seq_len : i + seq_len + pred_steps] - anchor)
        last.append(anchor)
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),
            np.array(last, dtype=np.float32))

X, y, anchors = make_sequences(feats, scores, SEQ_LEN, PRED_STEPS)
split  = int(len(X) * TRAIN_FRAC)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
an_tr, an_te = anchors[:split], anchors[split:]

# Scale deltas per horizon (they grow with h) so MSE gradients are well-sized.
d_scale = y_tr.std(axis=0)
d_scale[d_scale < 1e-6] = 1.0

# Carve a chronological validation slice out of the END of the train split
# for early stopping — with ~350 train sequences a fixed epoch count either
# undertrains or memorizes; the val slice decides when to stop.
val_cut = int(len(X_tr) * 0.85)
Xt  = torch.from_numpy(X_tr[:val_cut])
yt  = torch.from_numpy(y_tr[:val_cut] / d_scale)
Xvl = torch.from_numpy(X_tr[val_cut:])
yvl = torch.from_numpy(y_tr[val_cut:] / d_scale)
Xv  = torch.from_numpy(X_te)

print(f"\n  Sequences — train: {len(Xt)}   test: {len(Xv)}")
print(f"  Window: {SEQ_LEN} weeks → predict {PRED_STEPS} weekly deltas ahead")

# ── Model definition ───────────────────────────────────────────────────────
class StressLSTM(nn.Module):
    def __init__(self, n_feat, hidden, layers, pred_steps):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=n_feat, hidden_size=hidden,
                              num_layers=layers, batch_first=True,
                              dropout=0.0)
        self.linear = nn.Linear(hidden, pred_steps)

    def forward(self, x):
        out, _ = self.lstm(x)       # (batch, seq, hidden)
        last   = out[:, -1, :]      # last timestep only
        return self.linear(last)    # (batch, pred_steps) — scaled deltas

model     = StressLSTM(N_FEAT, HIDDEN, LAYERS, PRED_STEPS)

# ── Continual learning: load previous weights if they exist ───
if os.path.exists(OUT_MDL):
    try:
        model.load_state_dict(torch.load(OUT_MDL, weights_only=True))
        print("  [CONTINUAL] Loaded previous LSTM weights — fine-tuning on new data")
    except Exception:
        print("  [CONTINUAL] Previous weights incompatible — training from scratch")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Training loop with early stopping on the validation slice ──────────────
print(f"\n  Training LSTM (max {EPOCHS} epochs, hidden={HIDDEN}, "
      f"early stopping patience=200) ...")
train_losses = []
best_val, best_state, best_epoch, patience = float("inf"), None, 0, 200
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    pred  = model(Xt)
    loss  = criterion(pred, yt)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(Xvl), yvl).item()
    if val_loss < best_val:
        best_val, best_epoch = val_loss, epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if epoch % 500 == 0:
        print(f"    Epoch {epoch:4d}/{EPOCHS}  train={loss.item():.6f}  val={val_loss:.6f}")
    if epoch - best_epoch >= patience:
        print(f"    Early stop at epoch {epoch} (best val at {best_epoch})")
        break

if best_state is not None:
    model.load_state_dict(best_state)
print(f"  Restored best weights (epoch {best_epoch}, val loss {best_val:.6f})")

# ── Evaluation (levels reconstructed from predicted deltas) ────────────────
model.eval()
with torch.no_grad():
    pred_deltas = model(Xv).numpy() * d_scale        # unscale

val_pred = an_te[:, None] + pred_deltas              # anchor + predicted change
val_true = an_te[:, None] + y_te                     # anchor + true change

mae  = float(np.mean(np.abs(val_pred - val_true)))
rmse = float(np.sqrt(np.mean((val_pred - val_true) ** 2)))

# ── Naive baselines on the SAME test split ─────────────────────────────────
# A forecaster earns its keep only by beating these. On a heavily smoothed,
# highly autocorrelated series like macro stress, persistence is a hard
# baseline — reporting it prevents overclaiming the LSTM's value.
#   Persistence  : predicted delta 0 — next PRED_STEPS weeks == last week
#   Window-mean  : next PRED_STEPS weeks == mean of the 12-week input window
persist_pred = np.repeat(an_te[:, None], PRED_STEPS, axis=1)
win_mean     = X_te[:, :, 0].mean(axis=1)            # channel 0 = composite
mean_pred    = np.repeat(win_mean[:, None], PRED_STEPS, axis=1)
persist_mae  = float(np.mean(np.abs(persist_pred - val_true)))
persist_rmse = float(np.sqrt(np.mean((persist_pred - val_true) ** 2)))
mean_mae     = float(np.mean(np.abs(mean_pred - val_true)))

# Per-horizon MAE: persistence decays with horizon; report where (if anywhere)
# the LSTM earns its keep instead of one blended number.
lstm_h    = np.mean(np.abs(val_pred - val_true), axis=0)
persist_h = np.mean(np.abs(persist_pred - val_true), axis=0)

print(f"\n  ── Test Set Metrics ──────────────────────")
print(f"  {'Model':<26}{'MAE':>9}{'RMSE':>9}")
print(f"  {'LSTM (delta, multivar)':<26}{mae:>9.4f}{rmse:>9.4f}")
print(f"  {'Persistence (last week)':<26}{persist_mae:>9.4f}{persist_rmse:>9.4f}")
print(f"  {'Window-mean':<26}{mean_mae:>9.4f}{'':>9}")
print(f"\n  Per-horizon MAE (weeks ahead):")
print(f"  {'Horizon':<12}" + "".join(f"t+{h+1:>6}" for h in range(PRED_STEPS)))
print(f"  {'LSTM':<12}" + "".join(f"{v:>8.4f}" for v in lstm_h))
print(f"  {'Persistence':<12}" + "".join(f"{v:>8.4f}" for v in persist_h))
wins = [h + 1 for h in range(PRED_STEPS) if lstm_h[h] < persist_h[h]]
if mae > persist_mae and not wins:
    print(f"\n  [FINDING] LSTM does NOT beat persistence at any horizon "
          f"({mae:.4f} vs {persist_mae:.4f} MAE overall) — report this honestly")
    print(f"            rather than presenting the LSTM as an improvement.")
elif wins and mae > persist_mae:
    print(f"\n  [FINDING] LSTM beats persistence at horizon(s) {wins} but not overall")
    print(f"            ({mae:.4f} vs {persist_mae:.4f} MAE) — its value is the longer view.")
else:
    print(f"\n  LSTM beats persistence overall by {persist_mae - mae:.4f} MAE "
          f"(wins at horizons {wins}).")

# ── Persist metrics so the report/dashboard can cite the comparison ────────
LSTM_METRICS = os.path.join(BASE, "output", "lstm_metrics.txt")
with open(LSTM_METRICS, "w") as _f:
    _f.write("\n".join([
        "STAGE 10 — LSTM STRESS FORECASTER METRICS",
        "=" * 45, "",
        f"Test sequences : {len(Xv)}",
        f"Window / horizon: {SEQ_LEN} weeks -> {PRED_STEPS} weeks",
        f"Input: {N_FEAT} channels (composite + raw indicators); "
        f"target: per-week deltas from last observation", "",
        f"{'Model':<26}{'MAE':>9}{'RMSE':>9}",
        f"{'LSTM (delta, multivar)':<26}{mae:>9.4f}{rmse:>9.4f}",
        f"{'Persistence (last week)':<26}{persist_mae:>9.4f}{persist_rmse:>9.4f}",
        f"{'Window-mean':<26}{mean_mae:>9.4f}", "",
        "Per-horizon MAE:",
        "  horizon    " + "".join(f"t+{h+1:>6}" for h in range(PRED_STEPS)),
        "  LSTM       " + "".join(f"{v:>8.4f}" for v in lstm_h),
        "  Persistence" + "".join(f"{v:>8.4f}" for v in persist_h), "",
        ("LSTM does not beat persistence at any horizon."
         if mae > persist_mae and not wins else
         (f"LSTM beats persistence at horizon(s) {wins} but not overall."
          if wins and mae > persist_mae else
          f"LSTM beats persistence overall (wins at horizons {wins}).")),
    ]))
print(f"  Metrics saved → {LSTM_METRICS}")

torch.save(model.state_dict(), OUT_MDL)
print(f"  Model saved → {OUT_MDL}")

# ── Forecast: next PRED_STEPS weeks beyond dataset ─────────────────────────
last_seq = torch.from_numpy(feats[-SEQ_LEN:]).float().unsqueeze(0)
with torch.no_grad():
    future_deltas = model(last_seq).numpy()[0] * d_scale

future_scores = np.clip(scores[-1] + future_deltas, 0.0, 1.0)
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
print(f"  Trained on {len(scores)} weeks of freight data; forecast feeds the dashboard.")
print(f"  See lstm_metrics.txt for the persistence-baseline comparison before")
print(f"  citing the LSTM as a forecasting improvement.")
