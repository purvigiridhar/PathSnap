# Team workflow
## Ownership
- P1 Data and features: `src/pathsnap/data/`, `candidates.py`, `features.py`
- P2 Models and tuning: `src/pathsnap/models/`, `tuning/`
- P3 HMM and metrics: `src/pathsnap/hmm/`, `viz/` (route maps)
- P4 EDA, API, UI: `notebooks/01_eda.ipynb`, `api/`, `ui/`, `scripts/`, README

## Git
- Branch per person: `p1-data`, `p2-models`, `p3-hmm`, `p4-api-ui`. Open PRs into `main`.
- Commit small and often (repository history is reviewed).
- Notebooks: clear outputs or keep them small to avoid merge conflicts; real logic goes in `src/`.
- Never commit raw data, evaluator data, or secrets.

## Reproducibility
- One seed in `configs/default.yaml`, used everywhere.
- Log every experiment to `experiments/logs/` and every Optuna study to `experiments/optuna/`.
