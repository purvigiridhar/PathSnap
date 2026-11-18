def test_imports():
    import pathsnap  # noqa: F401


def test_reference_alignment_preserves_route_order():
    import numpy as np

    from pathsnap.data.alignment import align_reference

    nodes = np.array([[0, 0], [1, 0], [10, 0], [11, 0]])
    endpoints = np.array([[0, 1], [2, 3]])
    observations = np.array([[0, 0], [10, 0], [1, 0], [11, 0]])
    aligned = align_reference(observations, nodes, endpoints, np.array([0, 1]))
    assert aligned.tolist() == [0, 0, 0, 1]


def test_heading_distinguishes_reverse_arcs_and_marks_missing_motion(monkeypatch):
    import pandas as pd

    from pathsnap import features

    monkeypatch.setattr(features, "project_lonlat", lambda lon, lat, crs: (lon, lat))
    track = pd.DataFrame({"lon": [0, 10, 100], "lat": [0, 0, 0], "timestamp_s": [0, 1, 20]})
    nodes = pd.DataFrame({"lon": [0, 10], "lat": [0, 0]})
    arcs = pd.DataFrame({"from_node_id": [0, 1], "to_node_id": [1, 0]})
    candidates = pd.DataFrame({
        "obs_idx": [0, 0, 1, 1, 2, 2],
        "arc_id": [0, 1] * 3,
        "feat_dist_m": [0, 0, 0, 0, 90, 90],
    })

    result = features.build_features(candidates, track, nodes, arcs)

    assert result.feat_heading_diff_deg.tolist() == [0, 180, 0, 180, 90, 90]
    assert result.feat_heading_valid.tolist() == [1, 1, 1, 1, 0, 0]
    assert result.feat_projection_ratio.tolist() == [0, 1, 1, 0, 1, 0]
    assert result.feat_gps_speed_mps.tolist() == [0, 0, 10, 10, 0, 0]
    assert result.feat_rank_by_dist.tolist() == [0, 1] * 3
    assert result.feat_prev_dist_m.tolist() == [0, 0, 0, 0, 90, 90]
    assert len(result.filter(regex="^feat_").columns) == 19
