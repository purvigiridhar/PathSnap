import pandas as pd
import pytest

from pathsnap.viz.folium_maps import _route_chains, make_trip_map


def _map_data():
    nodes = pd.DataFrame(
        {
            "lon": [0.0, 0.001, 0.002, 0.0, 0.001],
            "lat": [0.0, 0.0, 0.0, 0.001, 0.001],
        }
    )
    arcs = pd.DataFrame(
        {
            "from_node_id": [0, 1, 3],
            "to_node_id": [1, 2, 4],
        }
    )
    return nodes, arcs


def test_route_geometry_uses_directed_arc_endpoints_and_preserves_gaps():
    nodes, arcs = _map_data()

    chains = _route_chains([0, None, 1, 2], nodes, arcs)

    assert chains == [
        [(0.0, 0.0), (0.0, 0.001)],
        [(0.0, 0.001), (0.0, 0.002)],
        [(0.001, 0.0), (0.001, 0.001)],
    ]


def test_map_saves_standalone_html_with_route_layers(tmp_path):
    nodes, arcs = _map_data()
    track = pd.DataFrame(
        {"lon": [0.0, 0.001, 0.002], "lat": [0.0, 0.0, 0.0]}
    )
    output = tmp_path / "trip.html"

    make_trip_map(
        "synthetic",
        track,
        nodes,
        arcs,
        reference_route=[0, 1],
        classifier_route=[0, None, 1],
        hmm_route=[0, 2],
        out_html=output,
    )

    content = output.read_text(encoding="utf-8")
    assert "GPS trace" in content
    assert "Ground-truth route" in content
    assert "Classifier-only route" in content
    assert "HMM route" in content
    assert "layer_control" in content


def test_map_rejects_route_arcs_with_unknown_node_ids():
    nodes, arcs = _map_data()
    arcs.loc[0, "to_node_id"] = 10
    track = pd.DataFrame({"lon": [0.0], "lat": [0.0]})

    with pytest.raises(ValueError, match="unknown node"):
        make_trip_map("synthetic", track, nodes, arcs, reference_route=[0])
