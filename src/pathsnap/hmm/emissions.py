"""Convert classifier probabilities into per-candidate log emissions."""

import numpy as np


def emission_log_scores(
    probabilities: np.ndarray,
    obs_idx: np.ndarray | None = None,
    *,
    log_floor: float = -10.0,
    normalize_per_observation: bool = False,
) -> np.ndarray:
    """Return log probabilities, optionally normalized within each observation.

    Probabilities at or below ``exp(log_floor)`` receive ``log_floor``. This
    bounds the impact of zero or very small classifier probabilities.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional array")
    if not np.isfinite(log_floor) or log_floor > 0:
        raise ValueError("log_floor must be finite and nonpositive")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")

    logs = np.full(values.shape, log_floor, dtype=float)
    positive = values > 0
    logs[positive] = np.maximum(np.log(values[positive]), log_floor)

    if not normalize_per_observation:
        return logs
    if obs_idx is None:
        raise ValueError("obs_idx is required for per-observation normalization")
    groups = np.asarray(obs_idx)
    if groups.shape != values.shape:
        raise ValueError("obs_idx and probabilities must have the same shape")

    normalized = logs.copy()
    for obs in np.unique(groups):
        positions = np.flatnonzero(groups == obs)
        group_logs = logs[positions]
        maximum = float(group_logs.max())
        log_sum_exp = maximum + float(np.log(np.exp(group_logs - maximum).sum()))
        normalized[positions] -= log_sum_exp
    return normalized
