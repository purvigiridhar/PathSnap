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


def candidate_miss_summary(df_candidates, full_track_observations=None):
    """Summarize reference-arc misses, optionally including zero-candidate GPS points.

    Without full_track_observations, rates use candidate-bearing observations
    only. Supply all GPS keys (trip_id, obs_idx) to compute full-track counts.
    """
    required = {"trip_id", "obs_idx", "label"}
    missing = required - set(df_candidates.columns)
    if missing:
        raise ValueError(f"candidate_miss_summary missing columns: {sorted(missing)}")

    found = df_candidates.groupby(["trip_id", "obs_idx"])["label"].max()
    bearing = len(found)
    misses = int((found.fillna(0) == 0).sum())
    summary = {
        "candidate_bearing_observations": bearing,
        "candidate_bearing_reference_misses": misses,
        "candidate_bearing_reference_miss_rate": float(misses / bearing) if bearing else float("nan"),
    }
    if full_track_observations is None:
        return summary

    universe_required = {"trip_id", "obs_idx"}
    missing = universe_required - set(full_track_observations.columns)
    if missing:
        raise ValueError(f"full_track_observations missing columns: {sorted(missing)}")
    universe = full_track_observations[["trip_id", "obs_idx"]].drop_duplicates()
    statuses = found.rename("reference_candidate").reset_index()
    merged = universe.merge(statuses, on=["trip_id", "obs_idx"], how="left", validate="one_to_one")
    if int(merged["reference_candidate"].notna().sum()) != bearing:
        raise ValueError("Candidate observations include keys absent from the full-track universe")

    total = len(universe)
    zero_candidates = int(merged["reference_candidate"].isna().sum())
    full_track_misses = misses + zero_candidates
    summary.update({
        "full_track_observations": total,
        "zero_candidate_observations": zero_candidates,
        "zero_candidate_rate": float(zero_candidates / total) if total else float("nan"),
        "full_track_reference_misses": full_track_misses,
        "full_track_reference_miss_rate": float(full_track_misses / total) if total else float("nan"),
    })
    return summary


def candidate_miss_rate(df_candidates):
    """Reference-arc miss rate among observations represented by candidate rows."""
    return candidate_miss_summary(df_candidates)["candidate_bearing_reference_miss_rate"]


def route_continuity(pred_arcs, arcs):
    """Placeholder for P3's network-aware route continuity metric."""
    raise NotImplementedError("Route continuity depends on P3's road-network contract")
