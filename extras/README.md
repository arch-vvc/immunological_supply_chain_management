# extras/

Standalone scripts kept out of the primary submission narrative — they work,
they're not wired into `main.py`'s pipeline or `app.py`'s dashboard, and
nothing else in the repo imports them. Run directly with `python3 extras/<file>.py`
if you want to demo one.

- **federated_immune.py** — simulates federated immune memory across three
  organizations by sharding one FAISS index in-process. Real privacy-preserving
  federation would need actual multi-party infrastructure; this is a proof-of-concept
  illustration, not a production federated-learning system. Presented honestly as
  future work, not a load-bearing result.
- **preprocess_dataco.py** — an earlier, DataCo-specific preprocessor. Superseded
  by `src/auto_onboard.py` + `src/preprocess.py`, which handle any dataset
  (including DataCo) via `config.yaml`.
- *(moved)* **gnn_explainer.py** now lives in `src/` and is surfaced in the
  dashboard's Risk Analysis tab ("Why is this node risky?").
- **cross_industry_test.py** — runs the same `ImmuneResponseEngine` against
  three domain configs (pharma/automotive/fmcg) to show the engine transfers
  without code changes, only config. A demo script, not a benchmark.
