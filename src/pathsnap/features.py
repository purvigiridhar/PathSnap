"""Features per GPS observation and candidate road segment."""

import numpy as np

from pathsnap.geo import project_lonlat


def build_features(candidates, track, nodes, arcs, *, crs="EPSG:32632"):
    """Add the length of each candidate segment in projected metres."""
    node_x, node_y = project_lonlat(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), crs)
    start, end = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int).T
    lengths = np.hypot(node_x[end] - node_x[start], node_y[end] - node_y[start])
    return candidates.assign(feat_arc_length_m=lengths[candidates["arc_id"].to_numpy()])
