"""Refit the CV-selected P2 classifier on the complete labeled train set."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from joblib import dump

from pathsnap.models.registry import get_models
from pathsnap.models.train import _load_table, _scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/offline/train"))
    parser.add_argument("--student-test", type=Path, default=Path("data/processed/offline/student_test"))
    parser.add_argument("--cv-results", type=Path, default=Path("results/model_comparison.csv"))
    parser.add_argument("--model-name", default=None, help="Defaults to highest cv_macro_f1 in --cv-results")
    parser.add_argument("--output", type=Path, default=Path("models/baseline.joblib"))
    args = parser.parse_args()

    comparison = pd.read_csv(args.cv_results)
    if args.model_name is None:
        selected = comparison.sort_values(["cv_macro_f1", "cv_top1_accuracy"], ascending=False).iloc[0]
        model_name = str(selected["model"])
    else:
        rows = comparison.loc[comparison["model"] == args.model_name]
        if rows.empty:
            raise ValueError(f"{args.model_name!r} is not present in {args.cv_results}")
        selected = rows.iloc[0]
        model_name = args.model_name

    train = _load_table(args.train)
    dev = _load_table(args.student_test)
    features = [column for column in train.columns if column.startswith("feat_")]
    if len(features) != 19:
        raise ValueError(f"Expected 19 P1 features, found {len(features)}")
    missing = sorted(set(features) - set(dev.columns))
    if missing:
        raise ValueError(f"Student-test data missing features: {missing}")
    if "label" not in train or "label" not in dev:
        raise ValueError("Train and student-test partitions must be labeled tables")

    config_path = Path("configs/default.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    seed = int(config.get("seed", 42))
    estimator = get_models(seed)[model_name]
    start = time.perf_counter()
    estimator.fit(train[features], train["label"].astype(int))
    fit_seconds = time.perf_counter() - start
    probabilities = estimator.predict_proba(dev[features])[:, 1]
    metrics = _scores(dev["label"].astype(int), probabilities, dev)

    artifact = {
        "model": estimator,
        "model_name": model_name,
        "feature_columns": features,
        "probability_definition": "P(candidate is correct), independent per row; not normalized by obs_idx",
        "positive_label": 1,
        "seed": seed,
        "selection_metric": "mean_group_cv_macro_f1",
        "cv_macro_f1": float(selected["cv_macro_f1"]),
        "cv_top1_accuracy": float(selected["cv_top1_accuracy"]),
        "cv_sample_rows": 250000,
        "training_trip_count": int(train["trip_id"].nunique()),
        "training_row_count": int(len(train)),
        "fit_seconds": fit_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, args.output)

    results_dir = Path("results")
    logs_dir = Path("experiments/logs")
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": model_name, "training_rows": len(train), "features": len(features), **metrics}]).to_csv(
        results_dir / "full_fit_student_test.csv", index=False
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (logs_dir / f"full_refit_{stamp}.json").write_text(json.dumps({
        "created_utc": stamp,
        "model_name": model_name,
        "feature_columns": features,
        "selection_metric": "mean_group_cv_macro_f1",
        "cv_results": str(args.cv_results),
        "cv_sample_rows": artifact["cv_sample_rows"],
        "cv_macro_f1": artifact["cv_macro_f1"],
        "cv_top1_accuracy": artifact["cv_top1_accuracy"],
        "full_train_path": str(args.train),
        "full_train_rows": int(len(train)),
        "training_trips": int(train["trip_id"].nunique()),
        "full_fit_seconds": fit_seconds,
        "student_test_path": str(args.student_test),
        "student_test_metrics": metrics,
    }, indent=2), encoding="utf-8")
    print(f"Refit {model_name} on {len(train):,} labeled rows from {train.trip_id.nunique()} trips.")
    print(f"Student-test metrics: {metrics}")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
