"""Metrics for binary candidate classification and per-observation ranking."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def log_loss_safe(y_true, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return float(log_loss(y_true, p, labels=[0, 1]))


def top1_accuracy(df_scored):
    """Fraction of observations whose highest-scored candidate is labeled true."""
    required = {"trip_id", "obs_idx", "label", "score"}
    missing = required - set(df_scored.columns)
    if missing:
        raise ValueError(f"top1_accuracy missing columns: {sorted(missing)}")
    winners = df_scored.loc[df_scored.groupby(["trip_id", "obs_idx"])["score"].idxmax()]
    valid = winners["label"].notna()
    return float(winners.loc[valid, "label"].astype(int).mean()) if valid.any() else float("nan")


def candidate_miss_rate(df_candidates):
    """Fraction of observations with no candidate marked as the reference arc."""
    required = {"trip_id", "obs_idx", "label"}
    missing = required - set(df_candidates.columns)
    if missing:
        raise ValueError(f"candidate_miss_rate missing columns: {sorted(missing)}")
    found = df_candidates.groupby(["trip_id", "obs_idx"])["label"].max().fillna(0)
    return float((found == 0).mean()) if len(found) else float("nan")


def route_continuity(pred_arcs, arcs):
    """Placeholder for P3's network-aware route continuity metric."""
    raise NotImplementedError("Route continuity depends on P3's road-network contract")
