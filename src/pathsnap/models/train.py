"""Compare candidate classifiers using trip-grouped CV and a labeled dev Parquet.

Expected schema: trip_id, obs_idx, arc_id, label, and numeric feat_* columns.
Run: python -m pathsnap.models.train --train data/processed/train_candidates.parquet
Optional: --student-test data/processed/student_candidates.parquet
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from pathsnap.models.metrics import candidate_miss_rate, log_loss_safe, macro_f1, top1_accuracy
from pathsnap.models.registry import get_models

META_COLUMNS = {"trip_id", "obs_idx", "arc_id", "label", "t", "x", "y"}


def _load_table(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No per-trip Parquet files found in: {path}")
        return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate table not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be a Parquet/CSV file or directory of Parquets")


def _validate(df: pd.DataFrame, name: str) -> list[str]:
    required = {"trip_id", "obs_idx", "arc_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} table missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{name} table is empty")
    if df["label"].isna().any() or not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{name}.label must contain only non-null 0/1 values")
    features = [c for c in df.columns if c.startswith("feat_")]
    if not features:
        raise ValueError("No feat_* feature columns found; wait for P1's feature table")
    nonnumeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if nonnumeric:
        raise ValueError(f"Features must be numeric: {nonnumeric}")
    if df[features].isna().all(axis=None):
        raise ValueError("All feature values are missing")
    return features


def _scores(y, probability, scored):
    predicted = (probability >= 0.5).astype(int)
    scored = scored.copy()
    scored["score"] = probability
    return {
        "candidate_accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": macro_f1(y, predicted),
        "log_loss": log_loss_safe(y, probability),
        "top1_accuracy": top1_accuracy(scored),
        "candidate_miss_rate": candidate_miss_rate(scored),
    }


def _evaluate_cv(name, estimator, X, y, groups, df, n_splits):
    splitter = GroupKFold(n_splits=n_splits)
    fold_rows, oof = [], np.full(len(df), np.nan)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = estimator
        # Every fold receives a fresh pipeline and fits preprocessing on train trips only.
        from sklearn.base import clone
        model = clone(estimator)
        start = time.perf_counter()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        probability = model.predict_proba(X.iloc[valid_idx])[:, 1]
        infer_seconds = time.perf_counter() - start
        oof[valid_idx] = probability
        metrics = _scores(y.iloc[valid_idx], probability, df.iloc[valid_idx])
        fold_rows.append({"model": name, "fold": fold, **metrics,
                          "fit_seconds": fit_seconds, "inference_seconds": infer_seconds,
                          "train_trip_count": int(pd.Series(groups[train_idx]).nunique()),
                          "validation_trip_count": int(pd.Series(groups[valid_idx]).nunique())})
    return fold_rows, oof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/offline/train"))
    parser.add_argument("--student-test", type=Path, default=Path("data/processed/offline/student_test"),
                        help="Labeled development set used for reporting only; never pass evaluator labels")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--models", default=None, help="Optional comma-separated model subset")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    seed = int(config.get("seed", 42))
    train = _load_table(args.train)
    features = _validate(train, "train")
    n_splits = args.n_splits or int(config.get("cv", {}).get("n_splits", 5))
    n_groups = train["trip_id"].nunique()
    if n_splits < 2 or n_groups < n_splits:
        raise ValueError(f"Need at least {n_splits} distinct training trips; found {n_groups}")
    X, y, groups = train[features].reset_index(drop=True), train["label"].astype(int).reset_index(drop=True), train["trip_id"].to_numpy()
    if y.nunique() != 2:
        raise ValueError("Training labels must include both classes 0 and 1")

    output = args.output_dir
    logs_dir, results_dir, models_dir = output / "experiments/logs", output / "results", output / "models"
    for folder in (logs_dir, results_dir, models_dir):
        folder.mkdir(parents=True, exist_ok=True)

    fold_rows, summaries = [], []
    models = get_models(seed)
    if args.models:
        requested = [name.strip() for name in args.models.split(",") if name.strip()]
        unknown = sorted(set(requested) - set(models))
        if unknown:
            parser.error(f"Unknown models {unknown}; choose from {sorted(models)}")
        models = {name: models[name] for name in requested}
    if not models:
        parser.error("Select at least one model")
    for name, estimator in models.items():
        print(f"[{name}] grouped CV on {n_groups} trips ...", flush=True)
        rows, oof = _evaluate_cv(name, estimator, X, y, groups, train, n_splits)
        fold_rows.extend(rows)
        summaries.append({"model": name, **{f"cv_{metric}": float(np.nanmean([r[metric] for r in rows]))
                                              for metric in ("macro_f1", "top1_accuracy", "log_loss", "candidate_miss_rate")},
                          "cv_fit_seconds": float(sum(r["fit_seconds"] for r in rows)),
                          "cv_inference_seconds": float(sum(r["inference_seconds"] for r in rows))})

    comparison = pd.DataFrame(summaries).sort_values(["cv_macro_f1", "cv_top1_accuracy"], ascending=False)
    if args.student_test:
        dev = _load_table(args.student_test)
        _validate(dev, "student-test")
        missing_features = sorted(set(features) - set(dev.columns))
        if missing_features:
            raise ValueError(f"student-test table missing training features: {missing_features}")
        X_dev, y_dev = dev[features], dev["label"].astype(int)
        for row in summaries:
            estimator = get_models(seed)[row["model"]]
            estimator.fit(X, y)
            start = time.perf_counter()
            prob = estimator.predict_proba(X_dev)[:, 1]
            elapsed = time.perf_counter() - start
            for metric, value in _scores(y_dev, prob, dev).items():
                row[f"student_{metric}"] = value
            row["student_inference_seconds"] = elapsed
        comparison = pd.DataFrame(summaries).sort_values(["cv_macro_f1", "cv_top1_accuracy"], ascending=False)

    best_name = str(comparison.iloc[0]["model"])
    final_model = get_models(seed)[best_name]
    start = time.perf_counter()
    final_model.fit(X, y)
    final_fit_seconds = time.perf_counter() - start
    artifact = {"model": final_model, "model_name": best_name, "feature_columns": features,
                "probability_definition": "P(candidate is correct), independent per row; not normalized by obs_idx",
                "positive_label": 1, "seed": seed, "selection_metric": "cv_macro_f1",
                "cv_macro_f1": float(comparison.iloc[0]["cv_macro_f1"]),
                "training_trip_count": int(n_groups), "fit_seconds": final_fit_seconds}
    dump(artifact, models_dir / "baseline.joblib")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(logs_dir / f"model_folds_{stamp}.csv", index=False)
    (logs_dir / f"model_run_{stamp}.json").write_text(json.dumps({
        "created_utc": stamp, "train_path": str(args.train), "student_test_path": str(args.student_test) if args.student_test else None,
        "rows": len(train), "trips": int(n_groups), "features": features, "n_splits": n_splits,
        "seed": seed, "selected_model": best_name, "selection_metric": "cv_macro_f1",
        "models": summaries,
    }, indent=2), encoding="utf-8")
    print(f"Saved comparison to {results_dir / 'model_comparison.csv'}")
    print(f"Saved baseline to {models_dir / 'baseline.joblib'} ({best_name})")


if __name__ == "__main__":
    main()
