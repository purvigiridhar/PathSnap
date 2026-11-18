# PathSnap: Distributed Map-Matching & Route Reconstruction

Team: <team name> | Members: <A>, <B>, <C>, <D>

Pipeline: noisy GPS -> candidate road segments -> classifier scores -> HMM/Viterbi -> reconstructed route.
Served via a FastAPI backend (Laptop 1) and a UI (Laptop 2).

## Repository layout
| Path | Purpose |
|---|---|
| `src/pathsnap/data/` | Raw file loaders, projection, EDA helpers, group splits |
| `src/pathsnap/` `candidates.py`, `features.py` | Candidate generation and feature engineering |
| `src/pathsnap/models/` | Model zoo, training, evaluation, metrics |
| `src/pathsnap/tuning/` | Bayesian hyperparameter optimisation (Optuna) |
| `src/pathsnap/hmm/` | Emission/transition models and Viterbi decoding |
| `src/pathsnap/viz/` | Folium maps and plots |
| `api/` | FastAPI backend, model loading, server-side reference store |
| `ui/` | Evaluator UI |
| `notebooks/` | EDA and experiment notebooks |
| `configs/` | YAML configs (seeds, CRS, candidate params, model params) |
| `experiments/` | Experiment logs and Optuna search records (keep these!) |
| `models/` | Saved model + preprocessing + HMM config artifacts |
| `results/` | CSV result tables and metric summaries |
| `reports/` | Figures and exported Folium HTML maps used in the PDF |
| `docs/` | Interfaces, data-quality log, AI-use disclosure, workflow |
| `tests/` | Basic contract tests |

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .      # makes `pathsnap` importable
```
Place the dataset in `data/raw/` as `train/<trip_id>/<trip_id>.{track,nodes,arcs,route}` and `student_test/...`.
Raw data is git-ignored.

## Running (fill in as you build)
1. EDA: `notebooks/01_eda.ipynb`
2. Build candidates/features: `python -m pathsnap.build_dataset`   (TODO)
3. Train + compare models: `python -m pathsnap.models.train`        (TODO)
4. Bayesian tuning: `python -m pathsnap.tuning.bayes_opt`            (TODO)
5. HMM comparison: `python -m pathsnap.hmm.compare`                  (TODO)
6. API (Laptop 1): `bash scripts/run_api.sh`
7. UI (Laptop 2): `API_URL=http://<laptop1-ip>:8000 bash scripts/run_ui.sh`

## Deployment (two laptops)
Both laptops on the same network. Laptop 1 runs the API on `0.0.0.0:8000` (allow the port through the firewall).
Evaluator reference labels are loaded server-side only from the path in `EVAL_REFERENCE_DIR`.

## Rules we follow
- Evaluator-test data/labels are never used for training, features, thresholds or model/decoder choice.
- All preprocessing is fit on training trips only; CV is `GroupKFold` by trip.
- No hardcoded predictions; all reported results come from logged runs.
- No secrets in the repo.
