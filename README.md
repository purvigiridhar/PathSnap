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
2. Build P1 candidate tables using the commands below.
3. Train + compare models: `python -m pathsnap.models.train`        (TODO)
4. Bayesian tuning: `python -m pathsnap.tuning.bayes_opt`            (TODO)
5. HMM comparison: `python -m pathsnap.hmm.compare`                  (TODO)
6. API (Laptop 1): `bash scripts/run_api.sh`
7. UI (Laptop 2): `API_URL=http://<laptop1-ip>:8000 bash scripts/run_ui.sh`

## Deployment (two laptops)
Both laptops on the same network. Laptop 1 runs the API on `0.0.0.0:8000` (allow the port through the firewall).
Evaluator reference labels are loaded server-side only from the path in `EVAL_REFERENCE_DIR`.

## Build P1 candidate tables

The sample and full builds use the eight closest segments per observation and a
trip-centred azimuthal equidistant projection. Read `coord_crs` when projecting
network nodes for the decoder. Outputs are one Parquet per trip and a
`summary.csv` with coverage and candidate misses.

```powershell
python -m pathsnap.build_dataset data/raw/train data/processed/offline/sample --partition --limit 5 --labeled --local-crs --max-candidates 8
python -m pathsnap.build_dataset data/raw/train data/processed/offline/train --partition --labeled --local-crs --max-candidates 8
python -m pathsnap.build_dataset data/raw/student_test data/processed/offline/student_test --partition --labeled --local-crs --max-candidates 8
python -m pathsnap.build_dataset data/raw/student_test data/processed/prediction/student_test --partition --local-crs --max-candidates 8
```

Fit models using the offline train tables. Use the offline student-test labels
for development evaluation and model comparison. Use the prediction directory
as input to the classifier or decoder. Without `--labeled`, the builder skips
reference routes and writes a label-free schema. It rejects output directories
that already contain Parquets from the other mode. Evaluator references stay
server-side for final scoring. See the [P1 validation run](docs/p1_validation.md).

The single-trip command accepts a trip directory and a Parquet output path.
Its default CRS is EPSG:32632.

For API inference, call `build_candidates_for_trip(trip_dir)` from
`pathsnap.build_dataset`. It uses the candidate settings in
`configs/default.yaml` and the same trip-centred projection as the batch
commands above. It reads the track, nodes, and arcs.

```python
from pathsnap.build_dataset import build_candidates_for_trip
from pathsnap.data.loader import load_track
from pathsnap.data.network import load_projected_network

candidates = build_candidates_for_trip(trip_dir)
track = load_track(trip_dir)
nodes, arcs, crs = load_projected_network(trip_dir)
```

Pass the full track to the decoder so it can return a result for each
observation. To load the network for a table built with an explicit CRS, pass
that table's `coord_crs` to `load_projected_network`.

## Rules we follow
- Evaluator-test data/labels are never used for training, features, thresholds or model/decoder choice.
- All preprocessing is fit on training trips only; CV is `GroupKFold` by trip.
- No hardcoded predictions; all reported results come from logged runs.
- No secrets in the repo.
