"""Log-space Viterbi decoding over candidate road arcs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pyproj import CRS

from pathsnap.hmm.emissions import emission_log_scores
from pathsnap.hmm.transitions import transition_log_score


def load_hmm_config() -> dict[str, Any]:
    """Load HMM settings from the repository's ``configs/hmm.yaml``."""
    config_path = Path(__file__).resolve().parents[3] / "configs" / "hmm.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"HMM configuration must be a mapping: {config_path}")
    return config


def _validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    emission = config.get("emission")
    transition = config.get("transition")
    if not isinstance(emission, dict) or not isinstance(transition, dict):
        raise ValueError("configuration must contain emission and transition mappings")

    log_floor = float(emission.get("log_floor", -10.0))
    if not np.isfinite(log_floor) or log_floor > 0:
        raise ValueError("emission.log_floor must be finite and nonpositive")
    mode = transition.get("mode", "hop")
    if mode not in {"hop", "distance_ratio"}:
        raise ValueError("transition.mode must be 'hop' or 'distance_ratio'")
    beta = float(transition.get("beta", 1.0))
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("transition.beta must be finite and nonnegative")
    same_arc_bonus = float(transition.get("same_arc_bonus", 0.0))
    if not np.isfinite(same_arc_bonus) or same_arc_bonus < 0:
        raise ValueError("transition.same_arc_bonus must be finite and nonnegative")
    hop_penalty = float(transition.get("hop_penalty", 1.0))
    if not np.isfinite(hop_penalty) or hop_penalty < 0:
        raise ValueError("transition.hop_penalty must be finite and nonnegative")
    max_hops = transition.get("max_hops", 5)
    if not isinstance(max_hops, int) or isinstance(max_hops, bool) or max_hops < 1:
        raise ValueError("transition.max_hops must be a positive integer")
    max_distance_m = transition.get("max_distance_m", 5000.0)
    if max_distance_m is not None and (
        not np.isfinite(max_distance_m) or max_distance_m < 0
    ):
        raise ValueError("transition.max_distance_m must be finite and nonnegative")
    unreachable_penalty = float(
        transition.get("unreachable_log_penalty", -10.0)
    )
    if not np.isfinite(unreachable_penalty) or unreachable_penalty > 0:
        raise ValueError(
            "transition.unreachable_log_penalty must be finite and nonpositive"
        )
    gap_threshold_s = transition.get("gap_threshold_s", 30.0)
    if gap_threshold_s is not None and (
        not np.isfinite(gap_threshold_s) or gap_threshold_s < 0
    ):
        raise ValueError("transition.gap_threshold_s must be finite and nonnegative")
    uturn = transition.get("uturn", {})
    if not isinstance(uturn, dict):
        raise ValueError("transition.uturn must be a mapping")
    uturn_penalty = float(uturn.get("penalty", -5.0))
    if not np.isfinite(uturn_penalty) or uturn_penalty > 0:
        raise ValueError("transition.uturn.penalty must be finite and nonpositive")
    return emission, transition


def _validate_network_crs(
    trip_candidates: pd.DataFrame, network: Any
) -> None:
    if "coord_crs" not in trip_candidates.columns or trip_candidates.empty:
        return
    values = trip_candidates["coord_crs"]
    candidate_crs = values.dropna().unique()
    if values.isna().any() or len(candidate_crs) != 1:
        raise ValueError("coord_crs must be nonnull and consistent within a trip")
    network_crs = getattr(network, "coord_crs", None)
    if len(candidate_crs) == 1 and network_crs is not None:
        candidate = CRS.from_user_input(candidate_crs[0])
        if not candidate.equals(CRS.from_user_input(network_crs)):
            raise ValueError("network coord_crs does not match candidate coordinates")


def _observation_metadata(
    frame: pd.DataFrame, obs_values: np.ndarray
) -> dict[int, tuple[float, float, float]]:
    metadata: dict[int, tuple[float, float, float]] = {}
    for obs_idx in np.unique(obs_values):
        rows = np.flatnonzero(obs_values == obs_idx)
        values = frame.iloc[rows][["t", "x", "y"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("candidate timestamps and coordinates must be finite")
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-9):
            raise ValueError(
                f"candidate rows for observation {obs_idx} disagree on t, x, or y"
            )
        metadata[int(obs_idx)] = tuple(float(value) for value in values[0])
    return metadata


def _backtrack_segment(
    obs_layers: list[int],
    row_layers: list[np.ndarray],
    backpointers: list[np.ndarray],
    final_scores: np.ndarray,
    arc_ids: np.ndarray,
    predictions: list[int | None],
    output_start: int,
) -> None:
    """Follow stored argmax predecessors from the best final state."""
    state = int(np.argmax(final_scores))
    for layer_idx in range(len(row_layers) - 1, -1, -1):
        row_idx = int(row_layers[layer_idx][state])
        obs_idx = obs_layers[layer_idx]
        predictions[obs_idx - output_start] = int(arc_ids[row_idx])
        if layer_idx:
            state = int(backpointers[layer_idx - 1][state])


def decode(trip_candidates, scores, network):
    """Return one arc ID or ``None`` per candidate-range observation.

    Scores align with the original candidate-row order. Labels are deliberately
    ignored. Missing interior observations retain ``None`` in the returned
    inclusive ``min(obs_idx)`` through ``max(obs_idx)`` range.
    """
    required = {"obs_idx", "arc_id", "t", "x", "y"}
    if not required.issubset(trip_candidates.columns):
        raise ValueError(f"trip_candidates must contain {sorted(required)}")
    if "trip_id" in trip_candidates.columns:
        trip_ids = trip_candidates["trip_id"].dropna().unique()
        if len(trip_ids) > 1:
            raise ValueError("decode accepts candidate rows from one trip at a time")

    probabilities = np.asarray(scores, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(trip_candidates):
        raise ValueError("scores must be one-dimensional and align with candidate rows")
    if trip_candidates.empty:
        if len(probabilities):
            raise ValueError("empty candidate table requires an empty score array")
        return []

    frame = trip_candidates.reset_index(drop=True)
    obs_numeric = frame["obs_idx"].to_numpy(dtype=float)
    arc_numeric = frame["arc_id"].to_numpy(dtype=float)
    if (
        not np.isfinite(obs_numeric).all()
        or np.any(obs_numeric < 0)
        or np.any(obs_numeric != np.floor(obs_numeric))
    ):
        raise ValueError("obs_idx values must be nonnegative integers")
    if (
        not np.isfinite(arc_numeric).all()
        or np.any(arc_numeric < 0)
        or np.any(arc_numeric != np.floor(arc_numeric))
    ):
        raise ValueError("arc_id values must be nonnegative integers")
    obs_values = obs_numeric.astype(np.int64)
    arc_ids = arc_numeric.astype(np.int64)
    metadata = _observation_metadata(frame, obs_values)
    ordered_obs = sorted(metadata)
    times = [metadata[obs_idx][0] for obs_idx in ordered_obs]
    if np.any(np.diff(times) < 0):
        raise ValueError("candidate-bearing observation timestamps must be nondecreasing")
    _validate_network_crs(frame, network)

    config = load_hmm_config()
    emission_config, transition_config = _validate_config(config)
    log_emissions = emission_log_scores(
        probabilities,
        obs_values,
        log_floor=float(emission_config.get("log_floor", -10.0)),
        normalize_per_observation=bool(
            emission_config.get("normalize_per_observation", False)
        ),
    )

    output_start = ordered_obs[0]
    predictions: list[int | None] = [None] * (ordered_obs[-1] - output_start + 1)
    obs_layers: list[int] = []
    row_layers: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    final_scores = np.empty(0, dtype=float)
    previous_obs: int | None = None
    previous_rows: np.ndarray | None = None
    previous_scores: np.ndarray | None = None

    def finish_segment() -> None:
        if row_layers:
            _backtrack_segment(
                obs_layers,
                row_layers,
                backpointers,
                final_scores,
                arc_ids,
                predictions,
                output_start,
            )
            obs_layers.clear()
            row_layers.clear()
            backpointers.clear()

    gap_threshold_s = transition_config.get("gap_threshold_s", 30.0)
    for obs_idx in ordered_obs:
        rows = np.flatnonzero(obs_values == obs_idx)
        should_restart = (
            previous_obs is not None
            and gap_threshold_s is not None
            and metadata[obs_idx][0] - metadata[previous_obs][0] > gap_threshold_s
        )
        if previous_obs is None or should_restart:
            if should_restart:
                finish_segment()
            obs_layers.append(obs_idx)
            row_layers.append(rows)
            final_scores = log_emissions[rows].copy()
        else:
            assert previous_rows is not None
            assert previous_scores is not None
            _, previous_x, previous_y = metadata[previous_obs]
            _, current_x, current_y = metadata[obs_idx]
            gps_distance_m = float(
                np.hypot(current_x - previous_x, current_y - previous_y)
            )
            next_scores = np.empty(len(rows), dtype=float)
            predecessors = np.empty(len(rows), dtype=np.int64)
            for current_position, current_row in enumerate(rows):
                transition_scores = np.asarray(
                    [
                        transition_log_score(
                            int(arc_ids[previous_row]),
                            int(arc_ids[current_row]),
                            network,
                            gps_distance_m,
                            transition_config,
                            previous_xy=(previous_x, previous_y),
                            current_xy=(current_x, current_y),
                        )
                        for previous_row in previous_rows
                    ],
                    dtype=float,
                )
                # V_t(j) = E_t(j) + max_i(V_(t-1)(i) + T(i, j)).
                predecessor = int(np.argmax(previous_scores + transition_scores))
                predecessors[current_position] = predecessor
                next_scores[current_position] = (
                    previous_scores[predecessor]
                    + transition_scores[predecessor]
                    + log_emissions[current_row]
                )
            obs_layers.append(obs_idx)
            row_layers.append(rows)
            backpointers.append(predecessors)
            final_scores = next_scores

        previous_obs = obs_idx
        previous_rows = rows
        previous_scores = final_scores

    finish_segment()
    return predictions
