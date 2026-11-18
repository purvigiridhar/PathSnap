"""Stable P2 inference over P1 candidate rows.

The returned values are independent per-row P(candidate is correct) scores;
they are not normalized across observations. Output position i always
corresponds to input row i.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load


def load_model_artifact(path: str | Path = "models/baseline.joblib") -> dict[str, Any]:
    """Load the P2 joblib artifact and validate its prediction contract."""
    artifact = load(path)
    required = {"model", "model_name", "feature_columns", "positive_label"}
    missing = required - set(artifact) if isinstance(artifact, dict) else required
    if missing:
        raise ValueError(f"Invalid P2 model artifact; missing keys: {sorted(missing)}")
    features = artifact["feature_columns"]
    if not features or any(not str(name).startswith("feat_") for name in features):
        raise ValueError("Artifact must define an ordered list of feat_* columns")
    if len(features) != len(set(features)):
        raise ValueError("Artifact contains duplicate feature names")
    return artifact


def predict_candidate_probabilities(
    candidate_rows: pd.DataFrame,
    model: dict[str, Any] | str | Path = "models/baseline.joblib",
) -> np.ndarray:
    """Return one P(candidate is correct) score for each input row, in-place order.

    Candidate identifiers, labels, and row order are never changed. Only the
    saved ordered feature columns are passed to the estimator; an optional
    `label` column in offline tables is ignored. Scores are not normalized per
    `obs_idx`, so P3 may use them as emission likelihoods or normalize them
    according to its decoder design.
    """
    if not isinstance(candidate_rows, pd.DataFrame):
        raise TypeError("candidate_rows must be a pandas DataFrame")
    artifact = load_model_artifact(model) if isinstance(model, (str, Path)) else model
    if not isinstance(artifact, dict):
        raise TypeError("model must be an artifact dictionary or a joblib path")
    features = artifact["feature_columns"]
    missing = [name for name in features if name not in candidate_rows.columns]
    if missing:
        raise ValueError(f"Candidate rows are missing model features: {missing}")
    if len(candidate_rows) == 0:
        return np.empty(0, dtype=np.float64)

    X = candidate_rows.loc[:, features]
    nonnumeric = [name for name in features if not pd.api.types.is_numeric_dtype(X[name])]
    if nonnumeric:
        raise TypeError(f"Model features must be numeric: {nonnumeric}")
    estimator = artifact["model"]
    classes = np.asarray(estimator.classes_)
    positive = np.flatnonzero(classes == artifact["positive_label"])
    if len(positive) != 1:
        raise ValueError(f"Positive label {artifact['positive_label']!r} not present exactly once in model classes {classes}")
    probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(candidate_rows):
        raise RuntimeError("Estimator returned probabilities with an invalid row count or shape")
    scores = probabilities[:, int(positive[0])]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RuntimeError("Estimator returned invalid probability values")
    return scores
