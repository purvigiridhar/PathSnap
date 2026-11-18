"""Focused tests for the P1/P2 adapter boundaries and P3 alignment."""

import numpy as np
import pandas as pd
import pytest
from pyproj import CRS

from pathsnap.hmm.network import ArcGraph
from pathsnap.hmm.pipeline import (
    align_predictions_to_track,
    decode_candidate_rows,
    score_candidate_rows,
    validate_candidate_scores,
)


def _p1_candidates():
    return pd.DataFrame(
        {
            "trip_id": ["synthetic", "synthetic"],
            "obs_idx": [0, 0],
            "arc_id": [1, 0],
            "t": [0.0, 0.0],
            "x": [40.0, 40.0],
            "y": [0.0, 0.0],
            "feat_dist_m": [1.0, 2.0],
            "feat_arc_length_m": [100.0, 100.0],
            "coord_crs": ["EPSG:3857", "EPSG:3857"],
            "label": [1, 0],
        }
    )


def _p1_projected_network():
    nodes = pd.DataFrame(
        {
            "lon": [0.0, 0.001, 0.002],
            "lat": [0.0, 0.0, 0.0],
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, 0.0, 0.0],
        }
    )
    arcs = pd.DataFrame(
        {"from_node_id": [0, 1], "to_node_id": [1, 2]},
        index=[17, 29],
    )
    return nodes, arcs


def test_p1_candidate_schema_and_label_exclusion_preserve_score_order():
    candidates = _p1_candidates()
    received = {}
    deterministic_scores = np.array([0.2, 0.8])

    def deterministic_test_scores(candidate_rows, model):
        received["columns"] = list(candidate_rows.columns)
        received["arc_ids"] = candidate_rows["arc_id"].tolist()
        received["model"] = model
        return deterministic_scores

    scores = score_candidate_rows(candidates, deterministic_test_scores)

    assert "label" not in received["columns"]
    assert received["arc_ids"] == [1, 0]
    assert received["model"] == "models/baseline.joblib"
    np.testing.assert_array_equal(scores, deterministic_scores)


def test_arcgraph_uses_p1_projected_coordinates_and_positional_ids():
    nodes, arcs = _p1_projected_network()
    network = ArcGraph(
        nodes,
        arcs,
        coord_crs="EPSG:3857",
        nodes_are_projected=True,
    )

    assert network.arc_length_m(0) == pytest.approx(100.0)
    assert network.arc_length_m(1) == pytest.approx(100.0)
    assert network.successors(0) == [1]
    assert network.coord_crs.equals(CRS.from_user_input("EPSG:3857"))


def test_candidate_schema_decodes_without_using_reference_labels(monkeypatch):
    candidates = _p1_candidates()
    nodes, arcs = _p1_projected_network()
    network = ArcGraph(
        nodes,
        arcs,
        coord_crs="EPSG:3857",
        nodes_are_projected=True,
    )
    config = {
        "emission": {"log_floor": -10.0, "normalize_per_observation": False},
        "transition": {
            "mode": "hop",
            "same_arc_bonus": 0.0,
            "hop_penalty": 1.0,
            "max_hops": 5,
            "max_distance_m": 1000.0,
            "unreachable_log_penalty": -10.0,
            "gap_threshold_s": 30.0,
            "uturn": {"enabled": False, "penalty": -5.0},
        },
    }
    monkeypatch.setattr("pathsnap.hmm.viterbi.load_hmm_config", lambda: config)

    assert decode_candidate_rows(candidates, np.array([0.2, 0.8]), network) == [0]


def test_decoder_adapter_never_passes_labels(monkeypatch):
    candidates = _p1_candidates()
    received = {}

    def inspect_decoder_rows(decoder_rows, scores, network):
        received["columns"] = list(decoder_rows.columns)
        return [0]

    monkeypatch.setattr("pathsnap.hmm.pipeline.decode", inspect_decoder_rows)

    assert decode_candidate_rows(candidates, [0.2, 0.8], network=None) == [0]
    assert "label" not in received["columns"]


def test_full_track_alignment_restores_leading_internal_and_trailing_gaps():
    track = pd.DataFrame({"timestamp_s": [0, 1, 2, 3, 4]})
    candidates = pd.DataFrame({"obs_idx": [1, 3]})

    aligned = align_predictions_to_track(track, candidates, [7, None, 9])

    assert aligned == [None, 7, None, 9, None]


def test_empty_candidates_skip_predictor_and_align_to_full_track():
    track = pd.DataFrame({"timestamp_s": [0, 1, 2]})
    candidates = pd.DataFrame(
        columns=[
            "trip_id",
            "obs_idx",
            "arc_id",
            "t",
            "x",
            "y",
            "feat_dist_m",
            "feat_arc_length_m",
            "coord_crs",
        ]
    )

    def unexpected_predictor(candidate_rows, model):
        pytest.fail("P2 should not be called when there are no candidates")

    scores = score_candidate_rows(candidates, unexpected_predictor)
    decoded = decode_candidate_rows(candidates, scores, network=None)

    assert validate_candidate_scores(candidates, scores).size == 0
    assert align_predictions_to_track(track, candidates, decoded) == [
        None,
        None,
        None,
    ]


@pytest.mark.parametrize(
    "scores",
    [
        [0.5],
        [[0.5, 0.6]],
        [np.nan, 0.5],
        [-0.1, 0.5],
        [1.1, 0.5],
    ],
)
def test_score_validation_rejects_misaligned_or_invalid_probabilities(scores):
    with pytest.raises(ValueError):
        validate_candidate_scores(_p1_candidates(), scores)


def test_alignment_rejects_candidate_indices_outside_full_track():
    track = pd.DataFrame({"timestamp_s": [0, 1]})
    candidates = pd.DataFrame({"obs_idx": [2]})

    with pytest.raises(ValueError, match="outside the full GPS track"):
        align_predictions_to_track(track, candidates, [4])
