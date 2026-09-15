# Immunological Supply Chain

Self-healing supply chains with AI digital antibodies.
PES University ISA Capstone PW26_RGP_01. Team: Arnav Lunia, Anvita Kulkarni, Lohith Anand, Archit Rishabh. Guide: Revathi GP.

The system models a supply chain the way the immune system works. Innate immunity is statistical anomaly detection over transactions. Adaptive immunity is a graph encoder, a recovery-time model and a reinforcement-learning router that learn from the network. Immunological memory is a FAISS index of past disruptions. Digital antibodies are the agents that pick backup suppliers, move inventory and reroute shipments. Everything runs from one command over any transaction-level dataset, and a Streamlit dashboard shows the results.

## Quick start

```bash
python3 -m venv myenv && source myenv/bin/activate
pip install -e ".[test]"          # installs requirements.txt and exposes src/ modules
python3 main.py                   # full 28-stage pipeline (~10 min, trains 3 models)
streamlit run app.py              # dashboard on http://localhost:8501
```

Useful runner flags:

```bash
python3 main.py --check                   # which artifacts are missing, without running anything
python3 main.py --stages 1-6,19-25        # run a subset or ranges
python3 main.py --skip-training           # skip GNN / LSTM / PPO training if models already exist
ISC_CONTINUAL=1 python3 main.py           # fine-tune GNN / LSTM / PPO from existing checkpoints instead of from scratch
python3 main.py --only 27                 # one stage
python3 main.py --onboard path/to.csv     # auto-detect columns of a new dataset, then run
python3 main.py --metrics                 # print the project metrics audit at the end
```

## Layout

```
main.py                 pipeline runner (28 stages, run as subprocesses)
app.py                  Streamlit dashboard, 9 tabs
ai_agent_app.py         optional chat analyst over the immune engine (local Ollama LLM)
config.yaml             column mapping for the primary dataset (auto-written by --onboard)
config/                 per-industry thresholds: pharma, automotive, fmcg
src/                    one module per stage, plus adapters and the IR layer
extras/                 standalone demos not wired into the pipeline (see extras/README.md)
tests/                  pytest suite; tests/run_all.py runs it without pytest
data/raw/               input datasets
data/supplementary/     disruption history and weekly freight indicators
data/processed/         standardised transactions from Stage 1
data/stream/            live-demo feed, results and decision log
output/                 every stage's CSV, report and figure (committed snapshot)
models/                 trained artifacts, gitignored; regenerate with main.py
```

## Pipeline stages

| Stage | What it does | Main output |
|---|---|---|
| 1 | Preprocess any CSV into a standard transaction table | `data/processed/clean_chain.csv` |
| 2 | Build the manufacturer, distributor, retailer graph | `models/supplychain_graph.pkl` |
| 3 | Five-signal anomaly detection, thresholds scaled by macro stress | `output/anomalies.csv` |
| 4 | Centrality-based risk scores | `output/risk_scores.csv` |
| 5 | Disruption injection and Dijkstra recovery routing | `output/routing_results.txt` |
| 6 | Static figures | `output/figures/fig1..4` |
| 7 | GCN autoencoder node embeddings and GNN risk | `output/gnn_risk_scores.csv` |
| 8 | Recovery-time regressor and per-strategy counterfactual sweep | `output/recovery_predictions.csv` |
| 9 | Macro freight stress composite from weekly indicators | `output/macro_stress_scores.csv` |
| 10 | LSTM stress forecast, reported against persistence | `output/stress_forecast.csv` |
| 11 | PPO recovery-routing agent, multi-step cascade benchmark | `output/ppo_routing_results.txt` |
| 12 | Per-industry XGBoost risk models | `output/multi_domain_f1.csv` |
| 13 | FAISS immunological memory, held-out k-NN validation | `models/faiss_memory.index` |
| 14 | Supplier agent: backup supplier scoring | `output/supplier_agent_results.csv` |
| 15 | Inventory agent: emergency stock transfers | `output/inventory_agent_results.csv` |
| 16 | Immune response engine one-shot test | console |
| 17 | Injection benchmark, multi-seed, calibrated logistic ensemble | `output/anomaly_ensemble_calibration.json` |
| 18 | Network-grounded response planner | `output/response_plan.csv` |
| 19 | SCMS spine: real unified network and disruptions | `output/scms_spine_report.txt` |
| 20 | SCMS revealed-preference backtest, walk-forward | `output/scms_backtest.csv` |
| 21 | SCMS blind event replay, Haiti 2010 | `output/scms_event_replay.csv` |
| 22 | SCMS safety-stock sizing | `output/scms_safety_stock.csv` |
| 23 | SCMS vendor scorecard and shortlists | `output/scms_vendor_scorecard.csv` |
| 24 | Case table of documented real-world events | `output/case_table.csv` |
| 25 | Event-reaction harness: replay and vendor knockout | `output/event_harness.json` |
| 26 | Dataset-independent IR: adapters, manifest, capability-gated stages | `output/ir/<dataset>/` |
| 27 | Macro-stress crisis validation against five documented crises | `output/macro_event_validation.csv` |
| 28 | SCMS outcome counterfactual: planner's pick vs procurement's choice | `output/scms_counterfactual.csv` |

Stage 0 samples a raw ARCOS extract down to 50K rows and is only needed if you replace the shipped sample.

## Data

Tracked in git and needed by the pipeline and tests:

- `data/raw/arcos_sampled_50k.csv`, the primary pharmaceutical transaction sample.
- `data/raw/SCMS_Delivery_History_Dataset.csv`, USAID delivery history used by stages 19 to 28.
- `data/supplementary/`, disruption history and the weekly freight indicator series.

Not tracked, because of size:

- `data/raw/DataCoSupplyChainDataset.csv` (91 MB), the Kaggle DataCo supply chain dataset. It is only read by the DataCo adapter in Stage 26. The stage skips the dataset when the file is absent, and the committed IR under `output/ir/dataco/` already covers the dashboard and tests. Drop the file in place to rebuild it.
- `data/raw/superstore_dataset.csv`, unused by the pipeline.

## Dashboard

`streamlit run app.py` opens nine tabs: Overview, Supply Chain Graph, Anomaly Detection, Risk Analysis, Recovery Predictor (with the four SCMS validations and the response planner), Multi-Domain Risk, Immune Memory, Live Response, and Dataset IR.

The Live Response tab starts and stops the stream simulator and the immune-response consumer as background processes from the page, so a demo needs no extra terminals. Trained models under `models/` must exist for the graph, memory and live tabs; run `python3 main.py --check` to see what is missing.

## Tests

```bash
pytest tests/                       # or: python3 tests/run_all.py
ISC_SMOKE=1 pytest tests/test_pipeline_smoke.py -s   # stages 1-6 end to end in a temp copy, ~20 s
```

- `test_runner.py`: stage-spec parsing, artifact map, pyproject module list.
- `test_artifacts.py`: every stage's expected artifacts exist. Fails on a fresh clone until the pipeline has produced `models/`.
- `test_harnesses.py`: wraps the two validation harnesses in `src/` (IR stages, vendor scorecard).
- `test_pipeline_smoke.py`: real end-to-end run of the batch stages, opt-in.
- `test_results_manifest.py`: the headline numbers currently in `output/` match the committed `output/RESULTS_MANIFEST.json` within tolerance, and the input datasets are unchanged.

## Results manifest

`output/RESULTS_MANIFEST.json` is the single source of truth for every headline number: anomaly F1 and adjusted precision, PPO reward and unserved demands with paired CIs, LSTM vs persistence, FAISS k-NN MAE, macro crisis placebo percentiles, the counterfactual late-rate gap with bootstrap CIs, plus the git commit, dataset hashes and seeds that produced them. A full `python3 main.py` run rewrites it. If a rerun moves a number, the manifest test fails and names the key, so drift is caught in CI instead of discovered in the paper.

The three training stages (GNN, LSTM, PPO) are seeded and train from scratch by default, so a fresh clone reproduces every manifest number bit for bit. Warm-starting from the previous checkpoint ("continual learning") is opt-in with `ISC_CONTINUAL=1`; numbers produced that way depend on run history and are not what the manifest records.

```bash
python3 src/results_manifest.py           # rebuild from output/
python3 src/results_manifest.py --check   # diff current outputs against the committed manifest
```

GitHub Actions runs the runner tests, both harnesses and the smoke test on every push.

## Domain configs

`config/pharma.yaml`, `config/automotive.yaml` and `config/fmcg.yaml` set the anomaly z-threshold, cytokine-storm window and rolling window for the live engine. Pick one with `python3 src/stream_consumer.py --domain automotive` or from the Live Response tab.
