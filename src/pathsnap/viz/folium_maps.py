"""Folium maps of GPS tracks and directed road-arc routes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import folium
import numpy as np
import pandas as pd


_ROUTE_COLORS = {
    "gps": "#3182bd",
    "reference": "#31a354",
    "classifier": "#fd8d3c",
    "hmm": "#de2d26",
}


def _route_chains(
    route: Iterable[int | None],
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
) -> list[list[tuple[float, float]]]:
    """Make lat/lon line chains without drawing connectors across gaps."""
    longitudes = nodes["lon"].to_numpy(dtype=float)
    latitudes = nodes["lat"].to_numpy(dtype=float)
    endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=float)
    if not np.isfinite(endpoints).all() or np.any(endpoints != np.floor(endpoints)):
        raise ValueError("Arc endpoint IDs must be finite integers")
    chains: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    previous_arc: int | None = None

    def finish_chain() -> None:
        nonlocal current
        if current:
            chains.append(current)
            current = []

    for value in route:
        if value is None or pd.isna(value):
            finish_chain()
            previous_arc = None
            continue
        arc_value = float(value)
        if not np.isfinite(arc_value) or arc_value != np.floor(arc_value):
            raise ValueError(f"Route arc ID must be an integer: {value!r}")
        arc_id = int(arc_value)
        if arc_id < 0 or arc_id >= len(endpoints):
            raise ValueError(f"Route arc ID is out of range: {arc_id}")
        if previous_arc == arc_id:
            continue

        from_node, to_node = endpoints[arc_id].astype(np.int64)
        if (
            from_node < 0
            or to_node < 0
            or from_node >= len(nodes)
            or to_node >= len(nodes)
        ):
            raise ValueError(f"Arc {arc_id} references an unknown node")
        node_coordinates = (
            longitudes[[from_node, to_node]],
            latitudes[[from_node, to_node]],
        )
        if not all(np.isfinite(values).all() for values in node_coordinates):
            raise ValueError(f"Arc {arc_id} references non-finite node coordinates")
        start = (float(latitudes[from_node]), float(longitudes[from_node]))
        end = (float(latitudes[to_node]), float(longitudes[to_node]))
        if current and current[-1] == start:
            current.append(end)
        else:
            finish_chain()
            current = [start, end]
        previous_arc = arc_id

    finish_chain()
    return chains


def _add_route_layer(
    fmap: folium.Map,
    name: str,
    route: Iterable[int | None] | None,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    color: str,
) -> None:
    layer = folium.FeatureGroup(name=name, show=True)
    route_values = [] if route is None else route
    for chain in _route_chains(route_values, nodes, arcs):
        folium.PolyLine(
            chain,
            color=color,
            weight=5,
            opacity=0.9,
            tooltip=name,
        ).add_to(layer)
    layer.add_to(fmap)


def make_trip_map(
    trip_id,
    track,
    nodes,
    arcs,
    reference_route=None,
    predicted_route=None,
    out_html=None,
    *,
    classifier_route=None,
    hmm_route=None,
):
    """Create and optionally save a map of GPS and directed route layers.

    ``track`` and node ``lon``/``lat`` coordinates are WGS84. Each route arc is
    drawn from its ``from_node_id`` to ``to_node_id`` node. Disconnected arcs
    and missing predicted arcs are kept as separate line segments.
    """
    required_track = {"lon", "lat"}
    required_nodes = {"lon", "lat"}
    required_arcs = {"from_node_id", "to_node_id"}
    if not required_track.issubset(track.columns):
        raise ValueError(f"track must contain {sorted(required_track)}")
    if not required_nodes.issubset(nodes.columns):
        raise ValueError(f"nodes must contain {sorted(required_nodes)}")
    if not required_arcs.issubset(arcs.columns):
        raise ValueError(f"arcs must contain {sorted(required_arcs)}")
    if track.empty:
        raise ValueError("Cannot map a trip with no GPS observations")

    gps_lon = track["lon"].to_numpy(dtype=float)
    gps_lat = track["lat"].to_numpy(dtype=float)
    if not np.isfinite(gps_lon).all() or not np.isfinite(gps_lat).all():
        raise ValueError("GPS longitude and latitude must be finite")
    center = [float(np.mean(gps_lat)), float(np.mean(gps_lon))]
    fmap = folium.Map(location=center, zoom_start=13, control_scale=True)

    gps_layer = folium.FeatureGroup(name="GPS trace", show=True)
    folium.PolyLine(
        list(zip(gps_lat, gps_lon)),
        color=_ROUTE_COLORS["gps"],
        weight=2,
        opacity=0.75,
        tooltip=f"{trip_id} GPS trace",
    ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    _add_route_layer(
        fmap,
        "Ground-truth route",
        reference_route,
        nodes,
        arcs,
        _ROUTE_COLORS["reference"],
    )
    _add_route_layer(
        fmap,
        "Classifier-only route",
        classifier_route,
        nodes,
        arcs,
        _ROUTE_COLORS["classifier"],
    )
    selected_hmm_route = hmm_route if hmm_route is not None else predicted_route
    _add_route_layer(
        fmap,
        "HMM route",
        selected_hmm_route,
        nodes,
        arcs,
        _ROUTE_COLORS["hmm"],
    )
    folium.LayerControl(collapsed=False).add_to(fmap)

    legend = """
    <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999;
                background: white; padding: 10px 12px; border: 1px solid #777;">
      <strong>PathSnap routes</strong><br>
      <span style="color: #3182bd;">&#9632;</span> GPS trace<br>
      <span style="color: #31a354;">&#9632;</span> Ground truth<br>
      <span style="color: #fd8d3c;">&#9632;</span> Classifier-only<br>
      <span style="color: #de2d26;">&#9632;</span> HMM
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend))

    if out_html is not None:
        output_path = Path(out_html)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(output_path)
    return fmap
