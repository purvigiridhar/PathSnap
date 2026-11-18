# P2 training and tuning

## Reproduce the 19-feature run

After P1's merged 19-feature candidate tables are available, create a deterministic CV sample containing complete GPS-observation candidate sets from all 80 training trips:

```powershell
python scripts/make_p2_cv_sample.py
```

Run the five-model comparison with trip-grouped five-fold CV on that sample and report metrics on the labeled student-test development partition:

```powershell
python -m pathsnap.models.train `
  --train data/processed/offline/p2_cv_sample.parquet `
  --student-test data/processed/offline/student_test `
  --models logistic_regression,random_forest,hist_gradient_boosting,lightgbm,gaussian_nb
```

The comparison selected `hist_gradient_boosting` by grouped-CV Macro F1. The CV table reflects 250,000 candidate rows sampled as complete observation groups, not all 1.67 million rows. The chosen baseline is then refit on the full labeled training partition:

```powershell
python scripts/refit_p2_selected_model.py
```

This writes `models/baseline.joblib` and `results/full_fit_student_test.csv`. The model artifact records the ordered 19 feature names and the independent per-row probability definition expected by `predict_candidate_probabilities()`.

## Tune the selected model

Run Optuna on the same trip-grouped CV sample. Each trial is scored on training folds only; the best settings are refit on all labeled training rows. Student-test is used only for the untuned-versus-tuned development comparison.

```powershell
python -m pathsnap.tuning.hgb_optuna --n-trials 8 --study-name pathsnap-hgb-19features
```

Results are written to `models/baseline_tuned.joblib`, `results/tuned_vs_untuned.csv`, and `experiments/optuna/`. The eight-trial search did not improve grouped-CV or student-test results, so `models/baseline.joblib` remains the P3 handoff baseline.

## Recorded run

- CV sample: 31,250 complete observations / 250,000 candidate rows, seed 42, all 80 trips.
- Full baseline refit: 1,667,336 labeled candidate rows, 80 trips, 19 features.
- Untuned full-refit student-test top-1 accuracy: 85.87%.
- Tuned student-test top-1 accuracy: 85.76%.
- `results/model_comparison.csv` and the matching files in `experiments/logs/` contain model comparison and fold details.
