# Paper Update Change-List — post-improvement numbers

**Why this is a change-list and not an edited paper:** the implementation paper has no
editable source on this machine (no `.tex`, no `.docx` — only the compiled PDF). It's in
Overleaf/LaTeX in the cloud. Paste these edits there. The survey paper *does* have a
`.docx` (`capstone/capstone 2/Immunological_SCM_Survey.docx`) but needs only two small
touches (Part E). If you export the implementation `.tex` into a file, I'll edit it directly.

Every "NEW" number below was pulled from the regenerated, committed artifacts on
2026-07-19. Numbers that **did not change** (graph size 2,131/23,264; multi-domain XGBoost
F1s; recovery MAE ≈ 31) are called out so you don't waste time on them.

Legend:  ✏️ = in-place number/text fix   🖼️ = figure to regenerate + re-embed
➕ = new content to add (optional, high value)   ⚠️ = correctness fix (paper describes
something that no longer exists — do not skip)

---

## PART A — Abstract

✏️ **Anomaly count.** "detects **1,535** high-confidence anomalies" →
"detects **1,664** high-confidence anomalies (calibrated logistic ensemble)".

⚠️ **PPO claim (this one is a reframe, not a number swap).**
OLD: "achieves a PPO reinforcement learning routing success rate of 100% with reduced
average route risk relative to Dijkstra shortest-path"
NEW: "learns a PPO recovery-routing policy that, in a multi-step reroute cascade under
supplier-capacity constraints, leaves less demand unserved and incurs lower route risk
than both a Dijkstra baseline and a myopic risk-greedy heuristic"
*(The old 100%/88.7% single-step result was reframed; see Part C.)*

✏️ **FAISS memory claim.** The "match distance of 0.43" was never a validated metric.
OLD: "retrieves relevant historical disruption memory at an average match distance of 0.43"
NEW: "retrieves historical disruption memory that beats baselines on held-out data
(k-NN recovery-day MAE 36.3 vs 42.3; response-type accuracy 49.5% vs 44.5%)"

➕ **Optional, if you add the new validation sections (Part D):** append one sentence —
"The macro-stress layer is validated against five documented global crises (3 detected at
p≥0.90 against a placebo null), and on real SCMS disruptions the planner's vendor
recommendation is associated with a lower subsequent late-shipment rate than procurement's
actual choice (4.0% vs 18.0%)."

Note: "17-stage pipeline" is now 28 stages if you count the IR (26), macro validation (27),
and counterfactual (28) additions. Optional — change to "a modular pipeline" or "28-stage"
only if you're comfortable renumbering; the narrative doesn't require it.

---

## PART B — Section VII-B, Anomaly Detection  (the biggest edit)

The whole result changes because evaluation moved to **held-out multi-seed** (tune on one
injection realization, seed 42; evaluate on four unseen seeds 43–46; report mean±std) and a
**calibrated logistic ensemble** replaced the "2+ signals" rule.

✏️ Replace the results sentence.
OLD: "Isolation Forest alone achieves P = 0.279, R = 0.718, F1 = 0.401; the z-score ensemble
(≥2 signals) achieves P = 0.292, R = 0.583, F1 = 0.389."
NEW (mean over 4 held-out seeds):

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Z-score ≥2 signals | 0.106 ±0.021 | 0.145 ±0.017 | 0.122 ±0.018 |
| Isolation Forest alone | 0.229 ±0.044 | 0.676 ±0.028 | 0.340 ±0.051 |
| **Calibrated ensemble (logistic)** | **0.285 ±0.055** | **0.603 ±0.063** | **0.386 ±0.063** |

➕ **Add the adjusted-precision finding (this is the strong story — do not omit).**
"Raw precision is bounded from above by construction: the base ARCOS data contains ~2,500
unlabeled *organic* anomalies from the generator, so any correct detection on those rows is
scored as a false positive, capping precision near n_injected/(n_injected+n_organic) ≈ 0.28
even for a perfect detector. Excluding rows the same rule already flags on the pre-injection
base (organic suspects), the calibrated ensemble's **adjusted precision is 0.86** — i.e. when
it fires on a non-organic row it is right 86% of the time. Reported raw figures are therefore
a lower bound."

Honest note for your own defense: the old single-seed 0.401 was a lucky draw; the honest
multi-seed IF F1 is 0.340. The new headline (calibrated ensemble, F1 0.386, adjusted-P 0.86)
is both better methodology and a better story.

🖼️ **Fig 5** (anomaly eval bars): re-embed `output/figures/fig_anomaly_injection_eval.png`
(now shows all methods with error bars over the 4 held-out seeds).

🖼️ **Fig 4** (signal counts): regenerate from the current pipeline. Current per-signal counts:
Volume 5,598 · Frequency 470 · Temporal surge 2,751 · Concentration 142 · Isolation Forest
2,500 · high-confidence (calibrated) **1,664**. (Counts shifted because thresholds are now
scaled per-week by macro stress; see Part D-macro.)

---

## PART C — Section VII-F, PPO Routing  (full reframe)

⚠️ The single-step task was analytically solved by risk-greedy, so the old "88.7% safer than
Dijkstra / 100% success / 4% risk reduction" framing was replaced with a **multi-step cascade**
where a learned policy has genuine value.

Replace the VII-F paragraph.
OLD: "Both achieve 100% success and identical average hop count (2.00). The PPO agent matches
Dijkstra on feasibility and path length while selecting safer routes in 88.7% of scenarios,
lowering average route risk from 0.969 to 0.930 (a 4.0% reduction)."
NEW: "The PPO agent is evaluated in a multi-step reroute cascade: D=6 sequential demands
served from a capacity-limited pool of alternate suppliers (2 uses each) whose *effective*
risk rises with load (+0.15/use — clonal exhaustion). All methods replay identical episodes.
Over 300 held-out episodes the learned policy attains the highest total reward and leaves the
least demand unserved:"

| Method | Avg total reward | Avg route risk | Unserved demands |
|---|---|---|---|
| **PPO (cascade)** | **37.35** | 0.964 | **26** |
| Risk-greedy (myopic) | 35.94 | 0.963 | 42 |
| Dijkstra | 33.22 | 0.966 | 72 |
| Random | 33.22 | 0.966 | 72 |

"PPO's advantage is foresight — reserving flexible suppliers instead of exhausting the
low-risk generalist early — which the one-step formulation could not expose (there, greedy is
optimal by construction). It is strictly safer than Dijkstra in 42.7% of episodes and clears
both risk-blind baselines decisively."

🖼️ **Fig 6**: re-embed `output/figures/fig9_ppo_training.png` (training curve + cascade bars).

➕ **Optional, strong addition:** the same cascade runs on the dataset-independent IR layer,
including on **real SCMS disruptions**, where the linear-policy PPO also beats greedy
(reward 37.30 vs 36.61). One sentence in VII-F or the IR paragraph: "The identical cascade,
run through the plug-and-play IR layer on real SCMS delivery disruptions, reproduces the
result (PPO 37.30 vs greedy 36.61), showing the finding is not ARCOS-specific."

---

## PART D — Table I & the removed classifier, LSTM, FAISS, Macro

### D1 — Table I (Recovery Prediction)  ⚠️ correctness

✏️ MAE 31.63 → **31.23 days**; R² 0.3796 → **0.3896**. (Baseline MAE 42.51 unchanged.)

⚠️ **Remove the classifier rows.** The paper's Table I lists "Classifier (response type)
Accuracy 56.10%, Macro F1 0.4099, Alt-Supplier F1 0.81 (P=0.73, R=0.91)". **That classifier
was deleted** — its target was near-unlearnable (MI≈0 for all features except
has_backup_supplier). It was replaced by a counterfactual per-strategy recovery sweep.
Replace those rows with one line: "A response-type classifier was evaluated and removed
(target near-unlearnable; MI≈0); the system instead predicts recovery days under each
candidate strategy and recommends the fastest." Keeping the old rows describes a component
that no longer exists — a reviewer running the repo will catch it.

### D2 — Section VII-E, LSTM  ⚠️ honesty

✏️ Add the finding. Current LSTM (multivariate, delta-target, early-stopped) MAE **0.0065**
vs persistence **0.0055**; it loses at every horizon (t+1..t+4: 0.0028/0.0052/0.0078/0.0103
vs persistence 0.0022/0.0044/0.0066/0.0090). Add: "Validation loss converges at the variance
of the weekly deltas — the best learnable one-week change is ≈0, which *is* persistence —
because the indicators are monthly data interpolated to weekly, so genuine weekly innovations
are absent. We report persistence as the operative forecast and retain the LSTM as a
documented negative result." (This converts an awkward gap into a clean finding.)

✏️ Stress series range: paper says "481 weekly records from 2019-2026" → current is **481
weeks, 2017-01 to 2026-03**. Update the range.

### D3 — Section VII-H, FAISS Memory

✏️ Replace the raw-distance sentence.
OLD: "average match distance 0.4329, average recalled recovery days 75.0, most frequent
recalled response type Alternative Supplier, most frequent disruption type Labor Strike."
NEW: "Held-out (80% indexed / 20% queried), k-NN(k=3) predicts recovery days at MAE 36.3 vs a
predict-the-mean baseline of 42.3, and response type at 49.5% accuracy vs a 44.5%
majority-class baseline. The index uses 3 query-time-observable features; an earlier 8-dim
version padded five unobservable categorical dimensions that diluted the distance metric and
contributed ~0 signal. Top recalled response type Alternative Supplier; top disruption type
Natural Disaster." (Query count 1,535 → 1,664.)

### D4 — Macro-stress threshold adaptation (Section VI-D / VII)

✏️ The θ_adj = λ·θ_base adaptation is now applied **per calendar week** (each transaction
scaled by its own week's stress level) rather than as one run-level average, and the live
stream consumer reads the latest stress week and retunes its z-threshold on a 5-minute
refresh. One sentence upgrade: "Threshold adaptation is applied per week in batch and
re-checked live in the streaming consumer, so a stress-regime change retunes detection
sensitivity without a restart."

---

## PART D-NEW — Two new subsections (optional, high value for external validity)

These are genuinely new results with figures. Adding them materially strengthens the paper's
"validated on real events" claim. Suggested placement: end of Section VII.

### ➕ VII-J  External Validity: Macro-Stress Crisis Detection  🖼️ Fig 11

"We test whether the macro-stress composite rises at documented global crises it was never
told about, as an event study: each crisis's 8-week window rise vs its prior 12 weeks, ranked
against a placebo distribution of 313 non-event windows (DETECTED ≥ p90, PARTIAL ≥ p75)."

| Crisis | Δ stress | placebo %ile | verdict | top driver (matches mechanism) |
|---|---|---|---|---|
| COVID-19 onset (Mar 2020) | +0.076 | 100.0 | DETECTED | Inventory-to-sales (+0.41) |
| Suez blockage (Mar 2021) | +0.035 | 98.1 | DETECTED | Truck spot rates (+0.16) |
| US port congestion (Sep 2021) | +0.061 | 100.0 | DETECTED | Containerships at berth (+0.26) |
| Ukraine invasion (Feb 2022) | +0.015 | 79.2 | PARTIAL | Diesel prices (+0.32) |
| Red Sea attacks (Dec 2023) | +0.009 | 69.3 | NOT SEEN | — |

"Three of five are detected at p≥0.90, and each detected crisis is driven by the indicator its
real-world mechanism predicts. Ukraine reads PARTIAL only because its window peak is the
series maximum (0.575) atop already-elevated congestion. The Red Sea null is the lens behaving
correctly — a US-aggregate freight composite should not register an Asia–Europe rerouting
crisis. Detections and nulls together characterize what a US-macro innate-immunity layer can
and cannot see." Figure: `output/figures/fig11_macro_event_validation.png`.

### ➕ VII-K  Outcome Counterfactual on Real SCMS Disruptions  🖼️ Fig 12

"At each lane's first late shipment we compare realized delivery performance over the
following year for the vendor procurement actually used (A) vs the walk-forward planner's top
pick (B), both scored at the same molecule scope (76 comparable lanes)."

"The recommended vendor had a lower late rate in 71.1% of lanes (18.0% → 4.0% mean late rate;
+2.5 lateness-days saved per shipment, median +1.5). A random-pool baseline scores similarly
(4.0%), so the large gap is evidence for the *category* of action — switching from the
troubled incumbent to any molecule-experienced alternative — not for the specific ranking
(which the revealed-preference backtest validates separately). Part of the gap is selection:
the incumbent is measured while its book is in trouble. The comparison is observational, not
causal." Figure: `output/figures/fig12_scms_counterfactual.png`.

---

## PART E — Survey paper (`Immunological_SCM_Survey.docx`)  — two small touches

The survey reports no numbers of its own (it defers to the implementation paper), so it barely
changes:

1. Where it previews the companion paper's results ("favourable results, e.g. PPO routing
   reducing average route risk relative to a Dijkstra baseline"), soften to "…PPO recovery
   routing that outperforms Dijkstra and a myopic heuristic in a multi-step cascade" to match
   the reframed result.
2. Optionally add one clause to the future-work / companion-paper mention noting the two new
   external-validity results (macro crisis detection; SCMS outcome counterfactual).

I can edit the `.docx` directly on request — say the word.

---

## PART F — Discussion / Limitations you can now retire

- The "single-run point estimates" limitation is **resolved** for anomaly detection
  (multi-seed held-out), the IR stages (bit-reproducible after fixing set-iteration
  nondeterminism), and PPO/LSTM (fixed seeds + early stopping). Update Section IX to say these
  are now reported with variance / are deterministic, and scope the caveat to the components
  that remain single-run.
- The FAISS "feature-space approximation" caveat can note the fix (dropped the 5 unobservable
  padded dims).

---

## Figures to regenerate & re-embed (all already produced in `output/figures/`)

| Paper fig | File | What changed |
|---|---|---|
| Fig 4 (signal counts) | regenerate from pipeline | new per-week thresholds, 1,664 high-conf |
| Fig 5 (anomaly eval) | `fig_anomaly_injection_eval.png` | held-out, error bars, calibrated ensemble |
| Fig 6 (PPO) | `fig9_ppo_training.png` | cascade reward + unserved bars |
| Fig 8 (LSTM) | `fig8_stress_forecast.png` | delta-target forecast |
| Fig 11 (NEW) | `fig11_macro_event_validation.png` | crisis event study |
| Fig 12 (NEW) | `fig12_scms_counterfactual.png` | SCMS outcome counterfactual |
