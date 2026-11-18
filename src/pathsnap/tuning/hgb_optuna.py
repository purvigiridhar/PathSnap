"""Tune the selected HistGradientBoosting model on grouped CV data.

The CV sample contains complete GPS-observation candidate groups from all
training trips. The winning estimator is refit on the full labeled train set.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from pathsnap.models.metrics import candidate_miss_rate, log_loss_safe, top1_accuracy
from pathsnap.models.train import _load_table


def _make_model(params: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", HistGradientBoostingClassifier(**params, random_state=seed)),
    ])


def _metrics(df: pd.DataFrame, probabilities: np.ndarray) -> dict:
    y = df["label"].astype(int).to_numpy()
    prediction = (probabilities >= 0.5).astype(int)
    scored = df[["trip_id", "obs_idx", "label"]].copy()
    scored["score"] = probabilities
    return {
        "candidate_accuracy": float((prediction == y).mean()),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "log_loss": log_loss_safe(y, probabilities),
        "top1_accuracy": top1_accuracy(scored),
        "candidate_miss_rate": candidate_miss_rate(scored),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-data", type=Path, default=Path("data/processed/offline/p2_cv_sample.parquet"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/offline/train"))
    parser.add_argument("--student-test", type=Path, default=Path("data/processed/offline/student_test"))
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--study-name", default="pathsnap-hgb-19features")
    args = parser.parse_args()
    if args.n_trials < 1:
        parser.error("--n-trials must be positive")

    config = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    n_splits = int(config.get("cv", {}).get("n_splits", 5))
    cv = _load_table(args.cv_data)
    train = _load_table(args.train)
    dev = _load_table(args.student_test)
    features = [c for c in train.columns if c.startswith("feat_")]
    if len(features) != 19:
        raise ValueError(f"Expected 19 P1 features in train data, found {len(features)}")
    for name, frame in (("cv", cv), ("student-test", dev)):
        missing = sorted(set(features) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} data missing features: {missing}")
    if set(cv["trip_id"].unique()) != set(train["trip_id"].unique()):
        raise ValueError("CV sample must include the full training trip-group set")

    X_cv = cv[features].reset_index(drop=True)
    y_cv = cv["label"].astype(int).reset_index(drop=True)
    groups = cv["trip_id"].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    folds = list(splitter.split(X_cv, y_cv, groups))
    if len(folds) != n_splits:
        raise ValueError("Could not create all configured group folds")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_iter": trial.suggest_int("max_iter", 30, 100, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [15, 31, 63]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [10, 20, 40, 80]),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            "max_bins": trial.suggest_categorical("max_bins", [63, 127, 255]),
        }
        fold_scores = []
        for train_idx, valid_idx in folds:
            model = _make_model(params, seed)
            model.fit(X_cv.iloc[train_idx], y_cv.iloc[train_idx])
            probabilities = model.predict_proba(X_cv.iloc[valid_idx])[:, 1]
            predicted = (probabilities >= 0.5).astype(int)
            fold_scores.append(float(f1_score(y_cv.iloc[valid_idx], predicted, average="macro", zero_division=0)))
        trial.set_user_attr("fold_macro_f1", fold_scores)
        return float(np.mean(fold_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna_dir = Path("experiments/optuna")
    logs_dir = Path("experiments/logs")
    optuna_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(optuna_dir / 'hist_gradient_boosting_19features.db').as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        load_if_exists=True,
    )
    start = time.perf_counter()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    search_seconds = time.perf_counter() - start

    best = study.best_trial
    baseline = load("models/baseline.joblib")
    baseline_model = baseline["model"]
    baseline_prob = baseline_model.predict_proba(dev[features])[:, 1]
    tuned_model = _make_model(best.params, seed)
    fit_start = time.perf_counter()
    tuned_model.fit(train[features], train["label"].astype(int))
    fit_seconds = time.perf_counter() - fit_start
    tuned_prob = tuned_model.predict_proba(dev[features])[:, 1]
    baseline_metrics = _metrics(dev, baseline_prob)
    tuned_metrics = _metrics(dev, tuned_prob)

    dump({
        "model": tuned_model,
        "model_name": "hist_gradient_boosting_tuned",
        "feature_columns": features,
        "probability_definition": "P(candidate is correct), independent per row; not normalized by obs_idx",
        "positive_label": 1,
        "seed": seed,
        "selection_metric": "mean_group_cv_macro_f1",
        "cv_macro_f1": float(best.value),
        "cv_sample_rows": int(len(cv)),
        "training_trip_count": int(train["trip_id"].nunique()),
        "training_row_count": int(len(train)),
        "fit_seconds": fit_seconds,
        "optuna_study": args.study_name,
        "optuna_best_trial": int(best.number),
        "optuna_params": best.params,
    }, "models/baseline_tuned.joblib")
    pd.DataFrame([
        {"model": "hist_gradient_boosting_untuned", **baseline_metrics},
        {"model": "hist_gradient_boosting_tuned", **tuned_metrics},
    ]).to_csv("results/tuned_vs_untuned.csv", index=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (logs_dir / f"optuna_hgb_{stamp}.json").write_text(json.dumps({
        "created_utc": stamp,
        "model_name": "hist_gradient_boosting",
        "cv_data": str(args.cv_data),
        "cv_rows": int(len(cv)),
        "cv_trips": int(cv["trip_id"].nunique()),
        "full_train_rows": int(len(train)),
        "training_trips": int(train["trip_id"].nunique()),
        "features": features,
        "n_splits": n_splits,
        "seed": seed,
        "study_name": args.study_name,
        "total_trials": len(study.trials),
        "best_trial": int(best.number),
        "best_cv_macro_f1": float(best.value),
        "best_params": best.params,
        "best_fold_macro_f1": best.user_attrs.get("fold_macro_f1"),
        "search_seconds_this_run": search_seconds,
        "full_fit_seconds": fit_seconds,
        "student_test_untuned_metrics": baseline_metrics,
        "student_test_tuned_metrics": tuned_metrics,
    }, indent=2), encoding="utf-8")
    print(f"Best grouped-CV Macro F1: {best.value:.4f} (trial {best.number})")
    print(f"Best parameters: {best.params}")
    print(f"Student-test untuned: {baseline_metrics}")
    print(f"Student-test tuned: {tuned_metrics}")
    print("Saved tuned artifact to models/baseline_tuned.joblib")
    print("Saved comparison to results/tuned_vs_untuned.csv")


if __name__ == "__main__":
    main()
