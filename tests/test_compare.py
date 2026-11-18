import numpy as np
import pandas as pd

from pathsnap.hmm import compare
from pathsnap.hmm.network import ArcGraph


def _network_data():
    nodes = pd.DataFrame(
        {
            "lon": [0.0, 0.001, 0.002, 0.003, 0.0, 0.001],
            "lat": [0.0, 0.0, 0.0, 0.0, 0.001, 0.001],
            "x": [0.0, 100.0, 200.0, 300.0, 0.0, 100.0],
            "y": [0.0, 0.0, 0.0, 0.0, 100.0, 100.0],
        }
    )
    arcs = pd.DataFrame(
        {
            "from_node_id": [0, 1, 4, 2],
            "to_node_id": [1, 2, 5, 3],
        }
    )
    network = ArcGraph(
        nodes,
        arcs,
        coord_crs="EPSG:3857",
        nodes_are_projected=True,
    )
    return nodes, arcs, network


def test_classifier_only_preserves_observation_order_and_missing_rows():
    candidates = pd.DataFrame(
        {
            "obs_idx": [1, 1, 3, 3],
            "arc_id": [2, 1, 3, 0],
        }
    )

    predictions = compare.classifier_only_decode(
        candidates, [0.8, 0.2, 0.5, 0.5], track_length=5
    )

    assert predictions == [None, 2, None, 3, None]


def test_evaluation_counts_missing_and_candidate_reference_misses():
    _, _, network = _network_data()
    candidates = pd.DataFrame(
        {"obs_idx": [0, 1, 3], "arc_id": [0, 1, 2]}
    )

    metrics = compare.evaluate_predictions(
        [0, 1, None, 2],
        np.array([0, 1, 1, 2]),
        candidates,
        network,
        max_hops=5,
    )

    assert metrics["observations"] == 4
    assert metrics["correct"] == 3
    assert metrics["arc_accuracy"] == 0.75
    assert metrics["conditional_arc_accuracy"] == 1.0
    assert metrics["missing"] == 1
    assert metrics["candidate_reference_found"] == 3
    assert metrics["candidate_reference_misses"] == 1
    assert metrics["adjacent_pairs_with_predictions"] == 1
    assert metrics["directly_connected_pairs"] == 1
    assert metrics["route_continuity_rate"] == 1.0


def test_evaluation_reports_disconnected_directed_transitions():
    _, _, network = _network_data()
    candidates = pd.DataFrame(
        {"obs_idx": [0, 1, 2, 3], "arc_id": [0, 2, 2, 2]}
    )

    metrics = compare.evaluate_predictions(
        [0, 2, 2, 2],
        np.array([0, 2, 2, 2]),
        candidates,
        network,
        max_hops=5,
    )

    assert metrics["adjacent_pairs_with_predictions"] == 3
    assert metrics["directly_connected_pairs"] == 2
    assert metrics["disconnected_transitions"] == 1
    assert metrics["disconnected_transition_rate"] == 1 / 3


def test_comparison_uses_identical_candidate_rows_and_scores(monkeypatch, tmp_path):
    nodes, arcs, network = _network_data()
    track = pd.DataFrame(
        {
            "lon": [0.0, 0.001, 0.002, 0.003],
            "lat": [0.0, 0.0, 0.0, 0.0],
            "timestamp_s": [0.0, 1.0, 2.0, 3.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "trip_id": ["synthetic"] * 4,
            "obs_idx": [0, 0, 1, 3],
            "arc_id": [0, 2, 1, 3],
            "t": [0.0, 0.0, 1.0, 3.0],
            "x": [0.0, 0.0, 100.0, 300.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "coord_crs": ["EPSG:3857"] * 4,
            "feat_dist_m": [1.0, 2.0, 1.0, 1.0],
            "label": [1, 0, 1, 1],
        }
    )
    scores = np.array([0.2, 0.8, 0.7, 0.6])
    decoder_inputs = {}
    monkeypatch.setattr(compare, "load_track", lambda _: track)
    monkeypatch.setattr(compare, "build_candidates_for_trip", lambda _: candidates)
    monkeypatch.setattr(
        compare,
        "score_candidate_rows",
        lambda candidate_rows, predictor, model: scores,
    )
    monkeypatch.setattr(
        compare,
        "load_projected_network",
        lambda trip_dir, crs: (nodes, arcs, crs),
    )

    def decode_same_inputs(candidate_rows, candidate_scores, candidate_network):
        decoder_inputs["rows"] = candidate_rows.copy()
        decoder_inputs["scores"] = candidate_scores.copy()
        decoder_inputs["network"] = candidate_network
        return [0, 1, None, 3], {
            "decode_seconds": 0.1,
            "transition_calls": 0,
            "transition_seconds": 0.0,
            "hop_distance_calls": 0,
            "hop_distance_seconds": 0.0,
            "shortest_path_searches": 0,
            "shortest_path_seconds": 0.0,
            "hop_cache_entries": 0,
        }

    monkeypatch.setattr(compare, "_profiled_hmm_decode", decode_same_inputs)
    monkeypatch.setattr(
        compare, "load_route", lambda _: pd.DataFrame({"arc_id": [0, 1, 3]})
    )
    monkeypatch.setattr(
        compare, "align_route_reference", lambda *args: np.array([0, 1, 1, 3])
    )
    monkeypatch.setattr(
        compare,
        "make_trip_map",
        lambda *args, **kwargs: None,
    )

    metrics_rows, _ = compare.compare_trip(
        tmp_path / "synthetic",
        model_artifact={},
        saved_predictions_dir=tmp_path / "no_saved_results",
        map_path=tmp_path / "map.html",
    )

    assert decoder_inputs["rows"].equals(candidates)
    np.testing.assert_array_equal(decoder_inputs["scores"], scores)
    assert decoder_inputs["network"].coord_crs.to_string() == "EPSG:3857"
    assert metrics_rows[0]["approach"] == "classifier_only"
    assert metrics_rows[0]["missing"] == 1
    assert metrics_rows[1]["approach"] == "hmm"
    assert metrics_rows[1]["missing"] == 1
