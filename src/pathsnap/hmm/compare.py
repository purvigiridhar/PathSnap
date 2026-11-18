"""Compare classifier-only and HMM route reconstruction on student-test trips."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pathsnap.build_dataset import build_candidates_for_trip
from pathsnap.data.alignment import align_reference
from pathsnap.data.loader import (
    list_trips,
    load_arcs,
    load_nodes,
    load_route,
    load_track,
)
from pathsnap.data.network import load_projected_network
from pathsnap.geo import local_metric_crs, project_lonlat
from pathsnap.hmm import viterbi as viterbi_module
from pathsnap.hmm.network import ArcGraph
from pathsnap.hmm.pipeline import (
    align_predictions_to_track,
    decode_candidate_rows,
    score_candidate_rows,
)
from pathsnap.hmm.viterbi import load_hmm_config
from pathsnap.models.predict import load_model_artifact, predict_candidate_probabilities
from pathsnap.viz.folium_maps import make_trip_map


def classifier_only_decode(
    trip_candidates: pd.DataFrame,
    scores,
    track_length: int,
) -> list[int | None]:
    """Choose the highest-scoring candidate independently at each observation.

    Candidate row order breaks score ties. Observations without candidate rows
    remain ``None`` in the full-track output.
    """
    required = {"obs_idx", "arc_id"}
    if not required.issubset(trip_candidates.columns):
        raise ValueError(f"trip_candidates must contain {sorted(required)}")
    probabilities = np.asarray(scores, dtype=float)
    if (
        probabilities.ndim != 1
        or len(probabilities) != len(trip_candidates)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("scores must be finite probabilities aligned with rows")
    if track_length < 0:
        raise ValueError("track_length must be nonnegative")

    observations = trip_candidates["obs_idx"].to_numpy(dtype=float)
    arc_values = trip_candidates["arc_id"].to_numpy(dtype=float)
    if (
        not np.isfinite(observations).all()
        or np.any(observations < 0)
        or np.any(observations != np.floor(observations))
        or np.any(observations >= track_length)
    ):
        raise ValueError("obs_idx values must be valid indices in the full track")
    if (
        not np.isfinite(arc_values).all()
        or np.any(arc_values < 0)
        or np.any(arc_values != np.floor(arc_values))
    ):
        raise ValueError("arc_id values must be nonnegative integers")

    predictions: list[int | None] = [None] * track_length
    observation_ids = observations.astype(np.int64)
    arc_ids = arc_values.astype(np.int64)
    for obs_idx in np.unique(observation_ids):
        rows = np.flatnonzero(observation_ids == obs_idx)
        best_row = rows[int(np.argmax(probabilities[rows]))]
        predictions[int(obs_idx)] = int(arc_ids[best_row])
    return predictions


def evaluate_predictions(
    predictions: list[int | None],
    reference_arc_ids: np.ndarray,
    trip_candidates: pd.DataFrame,
    network: ArcGraph,
    max_hops: int,
) -> dict[str, int | float]:
    """Measure per-observation accuracy, candidate coverage, and continuity.

    Arc accuracy uses all GPS observations as its denominator; missing
    predictions count as incorrect. Conditional accuracy uses only predicted
    observations. Connectivity is evaluated on adjacent original GPS indices
    where both predictions exist. Same-arc pairs count as connected.
    Reachability uses the decoder's configured maximum directed hop count.
    """
    references = np.asarray(reference_arc_ids)
    if references.ndim != 1 or len(predictions) != len(references):
        raise ValueError("predictions and reference arcs must align to the full track")
    if not trip_candidates.empty:
        required = {"obs_idx", "arc_id"}
        if not required.issubset(trip_candidates.columns):
            raise ValueError(f"trip_candidates must contain {sorted(required)}")

    total = len(references)
    predicted_mask = np.fromiter(
        (arc_id is not None for arc_id in predictions), dtype=bool, count=total
    )
    correct_mask = np.zeros(total, dtype=bool)
    for obs_idx, prediction in enumerate(predictions):
        if prediction is not None:
            correct_mask[obs_idx] = int(prediction) == int(references[obs_idx])

    candidate_arcs: dict[int, set[int]] = {}
    if not trip_candidates.empty:
        for obs_idx, arc_id in trip_candidates[
            ["obs_idx", "arc_id"]
        ].itertuples(index=False, name=None):
            candidate_arcs.setdefault(int(obs_idx), set()).add(int(arc_id))
    candidate_reference_found = sum(
        int(reference) in candidate_arcs.get(obs_idx, set())
        for obs_idx, reference in enumerate(references)
    )

    valid_pairs = 0
    directly_connected = 0
    network_reachable = 0
    for previous_idx in range(max(total - 1, 0)):
        previous_arc = predictions[previous_idx]
        current_arc = predictions[previous_idx + 1]
        if previous_arc is None or current_arc is None:
            continue
        previous_arc, current_arc = int(previous_arc), int(current_arc)
        valid_pairs += 1
        if previous_arc == current_arc or current_arc in network.successors(
            previous_arc
        ):
            directly_connected += 1
        if previous_arc == current_arc or network.hop_distance(
            previous_arc, current_arc, max_hops=max_hops
        ) is not None:
            network_reachable += 1

    predicted_count = int(predicted_mask.sum())
    correct_count = int(correct_mask.sum())
    missing_count = total - predicted_count
    incorrect_count = predicted_count - correct_count
    candidate_misses = total - candidate_reference_found
    disconnected = valid_pairs - network_reachable

    def rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "observations": total,
        "predicted": predicted_count,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "missing": missing_count,
        "arc_accuracy": rate(correct_count, total),
        "conditional_arc_accuracy": rate(correct_count, predicted_count),
        "missing_rate": rate(missing_count, total),
        "candidate_reference_found": candidate_reference_found,
        "candidate_reference_misses": candidate_misses,
        "candidate_reference_coverage": rate(candidate_reference_found, total),
        "adjacent_pairs_with_predictions": valid_pairs,
        "directly_connected_pairs": directly_connected,
        "direct_connectivity_rate": rate(directly_connected, valid_pairs),
        "network_reachable_pairs": network_reachable,
        "disconnected_transitions": disconnected,
        "disconnected_transition_rate": rate(disconnected, valid_pairs),
        "route_continuity_rate": rate(network_reachable, valid_pairs),
        "continuity_max_hops": max_hops,
    }


def align_route_reference(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    route: pd.DataFrame,
    coord_crs: str,
) -> np.ndarray:
    """Align the directed ``.route`` sequence to GPS rows using P1's method."""
    route_ids = route["arc_id"].to_numpy(dtype=np.int64)
    if not len(track) or not len(route_ids):
        raise ValueError("Ground-truth evaluation requires track rows and route arcs")
    endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    if np.any(route_ids < 0) or np.any(route_ids >= len(endpoints)):
        raise ValueError("Reference route contains an out-of-range arc ID")
    obs_xy = np.column_stack(
        project_lonlat(track["lon"].to_numpy(), track["lat"].to_numpy(), coord_crs)
    )
    node_xy = nodes[["x", "y"]].to_numpy(dtype=float)
    return align_reference(obs_xy, node_xy, endpoints, route_ids)


def _profiled_hmm_decode(
    candidates: pd.DataFrame,
    scores: np.ndarray,
    network: ArcGraph,
) -> tuple[list[int | None], dict[str, int | float]]:
    """Decode while timing transition and shortest-path work."""
    stats: dict[str, int | float] = {
        "transition_calls": 0,
        "transition_seconds": 0.0,
        "hop_distance_calls": 0,
        "hop_distance_seconds": 0.0,
        "shortest_path_searches": 0,
        "shortest_path_seconds": 0.0,
    }
    original_transition = viterbi_module.transition_log_score
    original_hop = network.hop_distance
    original_shortest = nx.shortest_path_length
    original_bounded_shortest = nx.single_source_shortest_path_length
    original_dijkstra = nx.single_source_dijkstra

    def timed_transition(*args, **kwargs):
        started = perf_counter()
        try:
            return original_transition(*args, **kwargs)
        finally:
            stats["transition_calls"] += 1
            stats["transition_seconds"] += perf_counter() - started

    def timed_hop(arc_a, arc_b, max_distance_m=None, *, max_hops=None):
        started = perf_counter()
        try:
            return original_hop(
                arc_a,
                arc_b,
                max_distance_m=max_distance_m,
                max_hops=max_hops,
            )
        finally:
            stats["hop_distance_calls"] += 1
            stats["hop_distance_seconds"] += perf_counter() - started

    def timed_shortest(*args, **kwargs):
        started = perf_counter()
        try:
            return original_shortest(*args, **kwargs)
        finally:
            stats["shortest_path_searches"] += 1
            stats["shortest_path_seconds"] += perf_counter() - started

    def timed_bounded_shortest(*args, **kwargs):
        started = perf_counter()
        try:
            return original_bounded_shortest(*args, **kwargs)
        finally:
            stats["shortest_path_searches"] += 1
            stats["shortest_path_seconds"] += perf_counter() - started

    def timed_dijkstra(*args, **kwargs):
        started = perf_counter()
        try:
            return original_dijkstra(*args, **kwargs)
        finally:
            stats["shortest_path_searches"] += 1
            stats["shortest_path_seconds"] += perf_counter() - started

    network.hop_distance = timed_hop
    viterbi_module.transition_log_score = timed_transition
    nx.shortest_path_length = timed_shortest
    nx.single_source_shortest_path_length = timed_bounded_shortest
    nx.single_source_dijkstra = timed_dijkstra
    try:
        started = perf_counter()
        predictions = decode_candidate_rows(candidates, scores, network)
        stats["decode_seconds"] = perf_counter() - started
    finally:
        network.hop_distance = original_hop
        viterbi_module.transition_log_score = original_transition
        nx.shortest_path_length = original_shortest
        nx.single_source_shortest_path_length = original_bounded_shortest
        nx.single_source_dijkstra = original_dijkstra

    stats["hop_cache_entries"] = len(network._hop_distance_cache)
    return predictions, stats


def _full_track_saved_predictions(
    path: Path, track_length: int
) -> list[int | None] | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    required = {"obs_idx", "predicted_arc_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Saved predictions must contain {sorted(required)}: {path}")
    if len(frame) != track_length:
        return None
    ordered = frame.sort_values("obs_idx")
    if ordered["obs_idx"].tolist() != list(range(track_length)):
        return None
    return [
        None if pd.isna(value) else int(value)
        for value in ordered["predicted_arc_id"]
    ]


def _rates_from_counts(counts: dict[str, int]) -> dict[str, float]:
    def rate(numerator: str, denominator: str) -> float:
        denominator_value = counts[denominator]
        return (
            float(counts[numerator] / denominator_value)
            if denominator_value
            else float("nan")
        )

    return {
        "arc_accuracy": rate("correct", "observations"),
        "conditional_arc_accuracy": rate("correct", "predicted"),
        "missing_rate": rate("missing", "observations"),
        "candidate_reference_coverage": rate(
            "candidate_reference_found", "observations"
        ),
        "direct_connectivity_rate": rate(
            "directly_connected_pairs", "adjacent_pairs_with_predictions"
        ),
        "route_continuity_rate": rate(
            "network_reachable_pairs", "adjacent_pairs_with_predictions"
        ),
        "disconnected_transition_rate": rate(
            "disconnected_transitions", "adjacent_pairs_with_predictions"
        ),
    }


def _aggregate_metrics(per_trip: pd.DataFrame) -> pd.DataFrame:
    count_columns = [
        "observations",
        "predicted",
        "correct",
        "incorrect",
        "missing",
        "candidate_reference_found",
        "candidate_reference_misses",
        "adjacent_pairs_with_predictions",
        "directly_connected_pairs",
        "network_reachable_pairs",
        "disconnected_transitions",
    ]
    rows = []
    for approach, group in per_trip.groupby("approach", sort=False):
        counts = {
            column: int(group[column].sum()) for column in count_columns
        }
        rows.append(
            {
                "approach": approach,
                **counts,
                **_rates_from_counts(counts),
                "continuity_max_hops": int(group["continuity_max_hops"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _comparison_table(per_trip: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "arc_accuracy",
        "conditional_arc_accuracy",
        "missing_rate",
        "candidate_reference_coverage",
        "direct_connectivity_rate",
        "route_continuity_rate",
        "disconnected_transition_rate",
    ]
    rows = []
    for trip_id in list(per_trip["trip_id"].drop_duplicates()) + ["__aggregate__"]:
        source = aggregate if trip_id == "__aggregate__" else per_trip
        key = "approach"
        group = (
            source
            if trip_id == "__aggregate__"
            else source[source["trip_id"] == trip_id]
        )
        classifier = group[group[key] == "classifier_only"].iloc[0]
        hmm = group[group[key] == "hmm"].iloc[0]
        row: dict[str, Any] = {"trip_id": trip_id}
        for metric in metric_names:
            classifier_value = float(classifier[metric])
            hmm_value = float(hmm[metric])
            row[f"classifier_{metric}"] = classifier_value
            row[f"hmm_{metric}"] = hmm_value
            row[f"hmm_minus_classifier_{metric}"] = (
                hmm_value - classifier_value
                if np.isfinite(hmm_value) and np.isfinite(classifier_value)
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_trip(
    trip_dir: Path,
    model_artifact: dict[str, Any],
    saved_predictions_dir: Path,
    map_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare both decoders on one trip using identical candidates and scores."""
    trip_id = trip_dir.name
    track = load_track(trip_dir)

    started = perf_counter()
    candidates = build_candidates_for_trip(trip_dir)
    candidate_seconds = perf_counter() - started

    if candidates.empty:
        coord_crs = local_metric_crs(
            track["lon"].to_numpy(), track["lat"].to_numpy()
        )
    else:
        if "coord_crs" not in candidates.columns:
            raise ValueError(f"{trip_id}: candidates do not provide coord_crs")
        crs_values = candidates["coord_crs"].dropna().unique()
        if candidates["coord_crs"].isna().any() or len(crs_values) != 1:
            raise ValueError(f"{trip_id}: candidate coord_crs is inconsistent")
        coord_crs = crs_values[0]

    started = perf_counter()
    scores = score_candidate_rows(
        candidates, predict_candidate_probabilities, model=model_artifact
    )
    prediction_seconds = perf_counter() - started

    started = perf_counter()
    nodes, arcs, network_crs = load_projected_network(
        trip_dir, crs=coord_crs
    )
    network = ArcGraph(
        nodes,
        arcs,
        coord_crs=network_crs,
        nodes_are_projected=True,
    )
    graph_seconds = perf_counter() - started

    started = perf_counter()
    classifier_predictions = classifier_only_decode(
        candidates, scores, len(track)
    )
    classifier_seconds = perf_counter() - started

    hmm_range_predictions, profile = _profiled_hmm_decode(
        candidates, scores, network
    )
    hmm_predictions = align_predictions_to_track(
        track, candidates, hmm_range_predictions
    )

    saved_path = saved_predictions_dir / f"{trip_id}_predictions.csv"
    saved_predictions = _full_track_saved_predictions(saved_path, len(track))
    saved_match = (
        None if saved_predictions is None else hmm_predictions == saved_predictions
    )

    evaluation_started = perf_counter()
    # Ground truth is loaded only after both label-free prediction paths finish.
    transition_config = load_hmm_config()["transition"]
    continuity_max_hops = int(transition_config.get("max_hops", 5))
    route = load_route(trip_dir)
    reference = align_route_reference(
        track, nodes, arcs, route, coord_crs
    )

    classifier_metrics = evaluate_predictions(
        classifier_predictions,
        reference,
        candidates,
        network,
        max_hops=continuity_max_hops,
    )
    hmm_metrics = evaluate_predictions(
        hmm_predictions,
        reference,
        candidates,
        network,
        max_hops=continuity_max_hops,
    )
    evaluation_seconds = perf_counter() - evaluation_started
    metrics_rows = []
    for approach, metrics in (
        ("classifier_only", classifier_metrics),
        ("hmm", hmm_metrics),
    ):
        metrics_rows.append(
            {"trip_id": trip_id, "approach": approach, **metrics}
        )

    map_started = perf_counter()
    make_trip_map(
        trip_id,
        track,
        nodes,
        arcs,
        reference_route=route["arc_id"].to_numpy(dtype=int),
        predicted_route=hmm_predictions,
        classifier_route=classifier_predictions,
        out_html=map_path,
    )
    map_seconds = perf_counter() - map_started

    profile_row = {
        "trip_id": trip_id,
        "candidate_generation_and_features_seconds": candidate_seconds,
        "p2_prediction_seconds": prediction_seconds,
        "arc_graph_build_seconds": graph_seconds,
        "classifier_selection_seconds": classifier_seconds,
        "hmm_decode_seconds": profile["decode_seconds"],
        "transition_calls": profile["transition_calls"],
        "transition_seconds": profile["transition_seconds"],
        "hop_distance_calls": profile["hop_distance_calls"],
        "hop_distance_seconds": profile["hop_distance_seconds"],
        "shortest_path_searches": profile["shortest_path_searches"],
        "shortest_path_seconds": profile["shortest_path_seconds"],
        "hop_cache_entries": profile["hop_cache_entries"],
        "candidate_rows": len(candidates),
        "observations": len(track),
        "evaluation_seconds": evaluation_seconds,
        "map_generation_seconds": map_seconds,
        "saved_prediction_match": saved_match,
    }
    return metrics_rows, profile_row


def run_comparison(
    raw_dir: Path,
    model_path: Path,
    saved_predictions_dir: Path,
    output_dir: Path,
    map_dir: Path,
    trip_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run label-free prediction, evaluation, profiling, and map generation."""
    all_trip_ids = list_trips(raw_dir)
    selected_trip_ids = all_trip_ids if trip_ids is None else trip_ids
    unknown = sorted(set(selected_trip_ids) - set(all_trip_ids))
    if unknown:
        raise ValueError(f"Unknown trips in {raw_dir}: {unknown}")
    if not selected_trip_ids:
        raise ValueError(f"No trips found in {raw_dir}")

    started = perf_counter()
    model_artifact = load_model_artifact(model_path)
    model_load_seconds = perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for trip_id in selected_trip_ids:
        print(f"[{trip_id}] candidate generation, scoring, decoding, evaluation")
        trip_metrics, profile = compare_trip(
            raw_dir / trip_id,
            model_artifact,
            saved_predictions_dir,
            map_dir / f"{trip_id}_comparison.html",
        )
        metrics_rows.extend(trip_metrics)
        profile_rows.append(profile)

    per_trip = pd.DataFrame(metrics_rows)
    aggregate = _aggregate_metrics(per_trip)
    comparison = _comparison_table(per_trip, aggregate)
    profile_frame = pd.DataFrame(profile_rows)
    profile_frame["model_load_seconds"] = model_load_seconds
    per_trip.to_csv(output_dir / "per_trip_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    profile_frame.to_csv(output_dir / "profile.csv", index=False)
    return per_trip, aggregate, comparison, profile_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw/student_test")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/baseline.joblib")
    )
    parser.add_argument(
        "--saved-predictions-dir",
        type=Path,
        default=Path("results/hmm/student_test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation/student_test"),
    )
    parser.add_argument(
        "--map-dir", type=Path, default=Path("results/maps/student_test")
    )
    parser.add_argument(
        "--trip-id",
        action="append",
        help="Optional trip ID; repeat to select multiple development trips",
    )
    args = parser.parse_args()

    per_trip, aggregate, comparison, profile = run_comparison(
        args.raw_dir,
        args.model,
        args.saved_predictions_dir,
        args.output_dir,
        args.map_dir,
        trip_ids=args.trip_id,
    )
    print(f"Per-trip metrics:\n{per_trip.to_string(index=False)}")
    print(f"Aggregate metrics:\n{aggregate.to_string(index=False)}")
    print(f"Comparison:\n{comparison.to_string(index=False)}")
    print(f"Profile:\n{profile.to_string(index=False)}")


if __name__ == "__main__":
    main()
