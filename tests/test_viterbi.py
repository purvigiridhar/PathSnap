"""Synthetic tests for directed-network HMM route decoding.

Parallel-street network used by continuity tests:
    Upper: E --2--> F --3--> G
    Lower: A --0--> B --1--> C

Separate U-turn case:
    A --0--> B
    A <--1-- B
"""

from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from pathsnap.hmm.emissions import emission_log_scores
from pathsnap.hmm.network import ArcGraph
from pathsnap.hmm.transitions import transition_log_score
from pathsnap.hmm.viterbi import decode


def _network(points, arc_endpoints):
    inverse = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    x, y = np.asarray(points, dtype=float).T
    lon, lat = inverse.transform(x, y)
    nodes = pd.DataFrame({"lon": lon, "lat": lat})
    arcs = pd.DataFrame(arc_endpoints, columns=["from_node_id", "to_node_id"])
    return ArcGraph(nodes, arcs, coord_crs="EPSG:3857")


def _candidates(rows):
    columns = ["obs_idx", "arc_id", "t", "x", "y"]
    frame = pd.DataFrame(
        [row[:5] for row in rows],
        columns=columns,
    )
    frame["coord_crs"] = "EPSG:3857"
    probabilities = np.asarray([row[5] for row in rows], dtype=float)
    return frame, probabilities


def _config(**transition_overrides):
    transition = {
        "mode": "hop",
        "beta": 1.0,
        "max_distance_m": 1000.0,
        "same_arc_bonus": 0.0,
        "hop_penalty": 1.0,
        "max_hops": 5,
        "unreachable_log_penalty": -10.0,
        "gap_threshold_s": 30.0,
        "uturn": {"enabled": False, "penalty": -5.0},
    }
    transition.update(transition_overrides)
    return {
        "emission": {
            "log_floor": -10.0,
            "normalize_per_observation": False,
        },
        "transition": transition,
    }


def _use_config(monkeypatch, config):
    monkeypatch.setattr("pathsnap.hmm.viterbi.load_hmm_config", lambda: config)


def _parallel_network():
    points = [
        (0, 0), (100, 0), (200, 0),
        (0, 20), (100, 20), (200, 20),
    ]
    arcs = [(0, 1), (1, 2), (3, 4), (4, 5)]
    return _network(points, arcs)


def test_viterbi_prefers_continuous_parallel_street(monkeypatch):
    network = _parallel_network()
    _use_config(monkeypatch, _config())
    candidates, scores = _candidates(
        [
            (0, 0, 0.0, 50.0, 0.0, 0.95),
            (0, 2, 0.0, 50.0, 0.0, 0.05),
            (1, 1, 1.0, 150.0, 0.0, 0.20),
            (1, 3, 1.0, 150.0, 0.0, 0.80),
        ]
    )

    assert decode(candidates, scores, network) == [0, 1]


def test_intermediate_observation_without_candidates_returns_none(monkeypatch):
    network = _parallel_network()
    _use_config(monkeypatch, _config())
    candidates, scores = _candidates(
        [
            (4, 0, 0.0, 50.0, 0.0, 0.8),
            (6, 1, 2.0, 150.0, 0.0, 0.8),
        ]
    )

    assert decode(candidates, scores, network) == [0, None, 1]


def test_large_time_gap_restarts_viterbi_chain(monkeypatch):
    network = _network(
        [(0, 0), (100, 0), (200, 0), (0, 20), (100, 20), (200, 20)],
        [(0, 1), (1, 2), (3, 4), (4, 5)],
    )
    _use_config(monkeypatch, _config(gap_threshold_s=10.0))
    candidates, scores = _candidates(
        [
            (0, 0, 0.0, 50.0, 0.0, 0.99),
            (0, 2, 0.0, 50.0, 0.0, 0.01),
            (1, 1, 100.0, 150.0, 0.0, 0.10),
            (1, 3, 100.0, 150.0, 0.0, 0.90),
        ]
    )

    assert decode(candidates, scores, network) == [0, 3]


def test_disconnected_transition_uses_configured_soft_penalty():
    network = _parallel_network()
    config = _config(
        mode="distance_ratio",
        max_distance_m=500.0,
        unreachable_log_penalty=-7.5,
    )["transition"]

    score = transition_log_score(0, 2, network, 20.0, config)

    assert score == -7.5


def test_viterbi_matches_exhaustive_candidate_path_search(monkeypatch):
    network = _network(
        [
            (0, 0), (100, 0), (200, 0), (300, 0),
            (0, 20), (100, 20), (200, 20), (300, 20),
        ],
        [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)],
    )
    config = _config()
    _use_config(monkeypatch, config)
    candidates, scores = _candidates(
        [
            (0, 0, 0.0, 50.0, 0.0, 0.72),
            (0, 3, 0.0, 50.0, 0.0, 0.28),
            (1, 1, 1.0, 150.0, 0.0, 0.55),
            (1, 4, 1.0, 150.0, 0.0, 0.45),
            (2, 2, 2.0, 250.0, 0.0, 0.20),
            (2, 5, 2.0, 250.0, 0.0, 0.80),
        ]
    )
    obs_idx = candidates["obs_idx"].to_numpy()
    arc_ids = candidates["arc_id"].to_numpy()
    logs = emission_log_scores(scores, obs_idx, log_floor=-10.0)
    rows_by_obs = [np.flatnonzero(obs_idx == obs) for obs in (0, 1, 2)]

    best_score = -np.inf
    best_path = None
    for path in product(*(range(len(rows)) for rows in rows_by_obs)):
        selected = [rows_by_obs[layer][position] for layer, position in enumerate(path)]
        total = sum(logs[row] for row in selected)
        for previous_row, current_row in zip(selected, selected[1:]):
            previous_xy = tuple(
                candidates.loc[previous_row, ["x", "y"]].to_numpy(dtype=float)
            )
            current_xy = tuple(
                candidates.loc[current_row, ["x", "y"]].to_numpy(dtype=float)
            )
            gps_distance = float(
                np.hypot(
                    current_xy[0] - previous_xy[0],
                    current_xy[1] - previous_xy[1],
                )
            )
            total += transition_log_score(
                int(arc_ids[previous_row]),
                int(arc_ids[current_row]),
                network,
                gps_distance,
                config["transition"],
                previous_xy=previous_xy,
                current_xy=current_xy,
            )
        if total > best_score:
            best_score = total
            best_path = [int(arc_ids[row]) for row in selected]

    assert decode(candidates, scores, network) == best_path


def test_weighted_directed_path_distance_and_bound():
    network = _network(
        [
            (-10, 0), (0, 0), (30, 0), (70, 0), (100, 0),
            (0, 100), (100, 100), (110, 0),
        ],
        [
            (0, 1),  # Candidate arc A ends at node 1.
            (4, 7),  # Candidate arc B starts at node 4.
            (1, 2), (2, 3), (3, 4),  # Three-edge, 100 m connector.
            (1, 5), (5, 4),  # Longer two-edge detour.
        ],
    )

    assert network.arc_length_m(2) == pytest.approx(30.0, abs=1e-3)
    assert network.path_length_m(0, 1) == pytest.approx(100.0, abs=1e-3)
    assert network.hop_distance(0, 1) == 3
    assert network.hop_distance(0, 1, max_hops=2) is None
    assert network.hop_distance(0, 1, max_hops=3) == 3
    assert network.hop_distance(0, 1, max_distance_m=101.0) == 4
    assert network.hop_distance(0, 1, max_distance_m=99.0) is None
    assert network.path_length_m(0, 1, max_distance_m=99.0) is None
    assert network.path_length_m(0, 1, max_distance_m=101.0) == pytest.approx(
        100.0, abs=1e-3
    )
    assert network.path_length_m(1, 0) is None


def test_hop_distance_caches_unreachable_directed_pair(monkeypatch):
    network = _network(
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 1), (2, 3)],
    )
    calls = 0
    shortest_path_length = nx.shortest_path_length

    def counted_shortest_path_length(*args, **kwargs):
        nonlocal calls
        calls += 1
        return shortest_path_length(*args, **kwargs)

    monkeypatch.setattr(
        "pathsnap.hmm.network.nx.shortest_path_length",
        counted_shortest_path_length,
    )

    assert network.hop_distance(0, 1) is None
    assert network.hop_distance(0, 1) is None
    assert calls == 1


def test_hop_parameters_are_independent_of_distance_ratio_bound():
    network = _network(
        [
            (-10, 0), (0, 0), (30, 0), (70, 0), (100, 0),
            (0, 100), (100, 100), (110, 0),
        ],
        [
            (0, 1),
            (4, 7),
            (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 4),
        ],
    )
    over_limit = _config(
        max_hops=2,
        max_distance_m=101.0,
        unreachable_log_penalty=-8.0,
    )["transition"]
    allowed = _config(max_hops=3, max_distance_m=1.0)["transition"]

    assert transition_log_score(0, 1, network, 0.0, over_limit) == -8.0
    assert transition_log_score(0, 1, network, 0.0, allowed) == -2.0


def test_same_arc_progress_and_configurable_uturn_penalty():
    network = _network(
        [(0, 0), (100, 0), (0, 20)],
        [(0, 1), (1, 0), (1, 2)],
    )
    assert network.same_arc_distance_m(0, (10.0, 1.0), (60.0, -1.0)) == pytest.approx(
        50.0
    )
    assert network.is_uturn(0, 1)

    disabled = _config()["transition"]
    enabled = _config(
        uturn={"enabled": True, "penalty": -5.0}
    )["transition"]
    same_arc = _config(same_arc_bonus=1.5)["transition"]
    assert transition_log_score(0, 1, network, 10.0, disabled) == 0.0
    assert transition_log_score(0, 1, network, 10.0, enabled) == -5.0
    assert transition_log_score(0, 0, network, 10.0, same_arc) == 1.5


def test_emissions_apply_log_floor_and_optional_normalization():
    probabilities = np.array([0.0, 0.25, 0.75])
    groups = np.array([0, 1, 1])

    unnormalized = emission_log_scores(probabilities, groups, log_floor=-10.0)
    normalized = emission_log_scores(
        probabilities,
        groups,
        log_floor=-10.0,
        normalize_per_observation=True,
    )

    assert unnormalized[0] == -10.0
    assert normalized[1] == pytest.approx(np.log(0.25))
    assert normalized[2] == pytest.approx(np.log(0.75))


def test_decoder_loads_repository_hmm_configuration():
    candidates, scores = _candidates([(0, 0, 0.0, 50.0, 0.0, 0.8)])

    assert decode(candidates, scores, _parallel_network()) == [0]
