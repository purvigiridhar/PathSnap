"""Directed road-arc graph with metric geometry and bounded path queries."""

from __future__ import annotations

from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


@lru_cache(maxsize=32)
def _transformer(coord_crs: str) -> Transformer:
    return Transformer.from_crs("EPSG:4326", coord_crs, always_xy=True)


class ArcGraph:
    """NetworkX-backed directed graph indexed by the input ``arc_id`` values.

    Node IDs are their zero-based row positions in ``nodes``; arc IDs are their
    zero-based row positions in ``arcs``. Node coordinates are either taken
    from projected ``x``/``y`` columns or projected from WGS84 longitude and
    latitude into ``coord_crs`` at construction time.
    """

    def __init__(
        self,
        nodes: pd.DataFrame,
        arcs: pd.DataFrame,
        coord_crs: str = "EPSG:32632",
        *,
        nodes_are_projected: bool = False,
    ) -> None:
        required_nodes = {"x", "y"} if nodes_are_projected else {"lon", "lat"}
        required_arcs = {"from_node_id", "to_node_id"}
        if not required_nodes.issubset(nodes.columns):
            raise ValueError(f"nodes must contain {sorted(required_nodes)}")
        if not required_arcs.issubset(arcs.columns):
            raise ValueError(f"arcs must contain {sorted(required_arcs)}")

        crs = CRS.from_user_input(coord_crs)
        if not crs.is_projected or any(
            abs(axis.unit_conversion_factor - 1.0) > 1e-9 for axis in crs.axis_info
        ):
            raise ValueError("coord_crs must be a projected CRS whose units are metres")
        self.coord_crs = crs

        if nodes_are_projected:
            node_x = nodes["x"].to_numpy(dtype=float)
            node_y = nodes["y"].to_numpy(dtype=float)
            if not np.isfinite(node_x).all() or not np.isfinite(node_y).all():
                raise ValueError("projected node coordinates must be finite")
        else:
            lon = nodes["lon"].to_numpy(dtype=float)
            lat = nodes["lat"].to_numpy(dtype=float)
            if not np.isfinite(lon).all() or not np.isfinite(lat).all():
                raise ValueError("node longitude and latitude values must be finite")
            node_x, node_y = _transformer(str(crs)).transform(
                lon, lat, errcheck=True
            )
        self._node_xy = {
            node_id: (float(x), float(y))
            for node_id, (x, y) in enumerate(zip(node_x, node_y))
        }

        self._arc_endpoints: dict[int, tuple[int, int]] = {}
        self._arc_lengths: dict[int, float] = {}
        self._outgoing: dict[int, list[int]] = {
            node_id: [] for node_id in self._node_xy
        }
        self._node_graph = nx.DiGraph()
        self._node_graph.add_nodes_from(self._node_xy)

        endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy()
        if endpoints.size and not np.isfinite(endpoints.astype(float)).all():
            raise ValueError("arc endpoint IDs must be finite")
        for arc_id, (from_value, to_value) in enumerate(endpoints):
            from_node = int(from_value)
            to_node = int(to_value)
            if from_node != from_value or to_node != to_value:
                raise ValueError("arc endpoint IDs must be integers")
            if from_node not in self._node_xy or to_node not in self._node_xy:
                raise ValueError(f"arc {arc_id} references an unknown node")

            self._arc_endpoints[arc_id] = (from_node, to_node)
            self._outgoing[from_node].append(arc_id)
            ax, ay = self._node_xy[from_node]
            bx, by = self._node_xy[to_node]
            length_m = float(np.hypot(bx - ax, by - ay))
            self._arc_lengths[arc_id] = length_m

            existing = self._node_graph.get_edge_data(from_node, to_node)
            if existing is None or length_m < existing["length_m"]:
                self._node_graph.add_edge(from_node, to_node, length_m=length_m)

        for outgoing in self._outgoing.values():
            outgoing.sort()

    def successors(self, arc_id: int) -> list[int]:
        """Return directed arcs leaving the selected arc's destination node."""
        _, to_node = self._arc_endpoints[int(arc_id)]
        return list(self._outgoing[to_node])

    def arc_length_m(self, arc_id: int) -> float:
        """Return the projected straight-segment length for an arc."""
        return self._arc_lengths[int(arc_id)]

    def _connector_path(
        self, arc_a: int, arc_b: int, max_distance_m: float | None
    ) -> tuple[float, list[int]] | None:
        if max_distance_m is not None and (
            not np.isfinite(max_distance_m) or max_distance_m < 0
        ):
            raise ValueError("max_distance_m must be finite and nonnegative")
        _, source = self._arc_endpoints[int(arc_a)]
        target, _ = self._arc_endpoints[int(arc_b)]
        try:
            distance, path = nx.single_source_dijkstra(
                self._node_graph,
                source,
                target=target,
                cutoff=max_distance_m,
                weight="length_m",
            )
        except nx.NetworkXNoPath:
            return None
        return float(distance), path

    def path_length_m(
        self,
        arc_a: int,
        arc_b: int,
        max_distance_m: float | None = None,
    ) -> float | None:
        """Return shortest directed connector length from A.to to B.from.

        The search cutoff is in projected metres. ``None`` means no directed
        path exists within the requested distance bound.
        """
        result = self._connector_path(arc_a, arc_b, max_distance_m)
        return None if result is None else result[0]

    def hop_distance(
        self,
        arc_a: int,
        arc_b: int,
        max_distance_m: float | None = None,
    ) -> int | None:
        """Return shortest directed connector hops plus the final transition.

        A directly adjacent arc pair has distance 1; the same arc has distance
        0. When bounded, only connector paths whose total projected length is
        at most ``max_distance_m`` are considered.
        """
        if max_distance_m is not None and (
            not np.isfinite(max_distance_m) or max_distance_m < 0
        ):
            raise ValueError("max_distance_m must be finite and nonnegative")
        if int(arc_a) == int(arc_b):
            return 0

        _, source = self._arc_endpoints[int(arc_a)]
        target, _ = self._arc_endpoints[int(arc_b)]
        if source == target:
            return 1
        if max_distance_m is None:
            try:
                connector_hops = nx.shortest_path_length(
                    self._node_graph, source, target
                )
            except nx.NetworkXNoPath:
                return None
            return int(connector_hops) + 1

        distances = {source: 0.0}
        for connector_hops in range(1, len(self._node_graph)):
            next_distances: dict[int, float] = {}
            for node_id, distance_m in distances.items():
                for successor in self._node_graph.successors(node_id):
                    next_distance = (
                        distance_m
                        + self._node_graph[node_id][successor]["length_m"]
                    )
                    if next_distance > max_distance_m:
                        continue
                    if next_distance < next_distances.get(successor, np.inf):
                        next_distances[successor] = next_distance
            if target in next_distances:
                return connector_hops + 1
            if not next_distances:
                return None
            distances = next_distances
        return None

    def same_arc_distance_m(
        self,
        arc_id: int,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
    ) -> float:
        """Measure progress between two points projected onto the same arc."""
        from_node, to_node = self._arc_endpoints[int(arc_id)]
        ax, ay = self._node_xy[from_node]
        bx, by = self._node_xy[to_node]
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            return 0.0

        def progress(point: tuple[float, float]) -> float:
            fraction = ((point[0] - ax) * dx + (point[1] - ay) * dy) / length_squared
            return float(np.clip(fraction, 0.0, 1.0) * np.sqrt(length_squared))

        return abs(progress(end_xy) - progress(start_xy))

    def is_uturn(self, arc_a: int, arc_b: int) -> bool:
        """Return true only for arcs with exactly reversed endpoint IDs."""
        if int(arc_a) == int(arc_b):
            return False
        from_a, to_a = self._arc_endpoints[int(arc_a)]
        from_b, to_b = self._arc_endpoints[int(arc_b)]
        return from_a == to_b and to_a == from_b
