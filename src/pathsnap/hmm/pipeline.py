"""Adapters between P1 trip data, P2 probabilities, and the P3 decoder."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS

from pathsnap.hmm.network import ArcGraph
from pathsnap.hmm.viterbi import decode

CandidateProbabilityPredictor = Callable[..., Sequence[float]]
DEFAULT_MODEL = "models/baseline.joblib"


def validate_candidate_scores(
    candidate_rows: pd.DataFrame, scores: Sequence[float]
) -> np.ndarray:
    """Validate one probability per candidate row, preserving row order."""
    probabilities = np.asarray(scores, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(candidate_rows):
        raise ValueError("scores must contain one value per candidate row")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("candidate scores must be finite probabilities in [0, 1]")
    return probabilities


def score_candidate_rows(
    candidate_rows: pd.DataFrame,
    predictor: CandidateProbabilityPredictor,
    model: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Call P2 on P1 rows without labels and validate the aligned scores.

    The predictor must preserve input row order, as required by the P2 contract.
    This adapter never sorts or otherwise reorders candidate rows.
    """
    if candidate_rows.empty:
        return np.empty(0, dtype=float)
    if not callable(predictor):
        raise TypeError("predictor must be P2's probability prediction function")

    inference_rows = candidate_rows.drop(columns=["label"], errors="ignore").copy()
    identity_columns = [
        column
        for column in ("trip_id", "obs_idx", "arc_id")
        if column in inference_rows.columns
    ]
    original_identity = inference_rows[identity_columns].reset_index(drop=True).copy()
    raw_scores = predictor(inference_rows, model=model)
    if not inference_rows[identity_columns].reset_index(drop=True).equals(
        original_identity
    ):
        raise RuntimeError("P2 predictor modified or reordered candidate rows")
    return validate_candidate_scores(candidate_rows, raw_scores)


def decode_candidate_rows(
    candidate_rows: pd.DataFrame,
    scores: Sequence[float],
    network: ArcGraph,
) -> list[int | None]:
    """Pass label-free candidate rows to the unchanged P3 decoder."""
    decoder_rows = candidate_rows.drop(columns=["label"], errors="ignore")
    return decode(decoder_rows, scores, network)


def align_predictions_to_track(
    track: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    candidate_range_predictions: Sequence[int | None],
) -> list[int | None]:
    """Expand decoder output to the full GPS track, filling missing rows with None."""
    predictions = list(candidate_range_predictions)
    if candidate_rows.empty:
        if predictions:
            raise ValueError("empty candidate table requires empty decoder output")
        return [None] * len(track)
    if "obs_idx" not in candidate_rows.columns:
        raise ValueError("candidate_rows must contain obs_idx")

    observation_values = candidate_rows["obs_idx"].to_numpy(dtype=float)
    if (
        not np.isfinite(observation_values).all()
        or np.any(observation_values < 0)
        or np.any(observation_values != np.floor(observation_values))
    ):
        raise ValueError("obs_idx values must be nonnegative integers")
    observation_ids = observation_values.astype(np.int64)
    first_observation = int(observation_ids.min())
    last_observation = int(observation_ids.max())
    if last_observation >= len(track):
        raise ValueError("candidate obs_idx is outside the full GPS track")
    expected_length = last_observation - first_observation + 1
    if len(predictions) != expected_length:
        raise ValueError(
            "decoder output must cover the inclusive candidate observation range"
        )

    aligned: list[int | None] = [None] * len(track)
    aligned[first_observation : last_observation + 1] = predictions
    return aligned


def reconstruct_trip(
    trip_dir: str | Path,
    predictor: CandidateProbabilityPredictor,
    model: str = DEFAULT_MODEL,
) -> list[int | None]:
    """Run label-free P1 candidate generation, scoring, and P3 decoding.

    P1's ``build_candidates_for_trip`` and ``load_projected_network`` are
    imported lazily so P3-only workflows remain usable on branches that have
    not yet incorporated P1's implementation.
    """
    try:
        from pathsnap.build_dataset import build_candidates_for_trip
        from pathsnap.data.loader import load_track
        from pathsnap.data.network import load_projected_network
    except ModuleNotFoundError as error:
        if error.name in {
            "pathsnap.build_dataset",
            "pathsnap.data.network",
        }:
            raise RuntimeError(
                "P1 integration requires build_candidates_for_trip and "
                "load_projected_network from the updated main branch"
            ) from error
        raise

    full_track = load_track(Path(trip_dir))
    candidates = build_candidates_for_trip(Path(trip_dir))
    if candidates.empty:
        return [None] * len(full_track)

    if "coord_crs" not in candidates.columns:
        raise ValueError("P1 candidates must include coord_crs")
    crs_values = candidates["coord_crs"].dropna().unique()
    if candidates["coord_crs"].isna().any() or len(crs_values) != 1:
        raise ValueError("candidate coord_crs must be nonnull and consistent")
    coord_crs = crs_values[0]

    nodes, arcs, network_crs = load_projected_network(
        Path(trip_dir), crs=coord_crs
    )
    if not CRS.from_user_input(coord_crs).equals(
        CRS.from_user_input(network_crs)
    ):
        raise ValueError("P1 network CRS does not match candidate coordinates")
    network = ArcGraph(
        nodes,
        arcs,
        coord_crs=coord_crs,
        nodes_are_projected=True,
    )

    scores = score_candidate_rows(candidates, predictor, model=model)
    candidate_range_predictions = decode_candidate_rows(candidates, scores, network)
    return align_predictions_to_track(
        full_track, candidates, candidate_range_predictions
    )
