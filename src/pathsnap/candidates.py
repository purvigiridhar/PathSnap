"""Generate a distance-based shortlist of directed road segments."""

import numpy as np
import pandas as pd

from pathsnap.geo import point_to_segment_distance, project_lonlat


def generate_candidates(track, nodes, arcs, radius_m=None, max_candidates=None):
    """Return candidate rows with zero-based IDs and GPS time and coordinates."""
    obs_x, obs_y = project_lonlat(track["lon"].to_numpy(), track["lat"].to_numpy())
    node_x, node_y = project_lonlat(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    start_ids = arcs["from_node_id"].to_numpy(dtype=int)
    end_ids = arcs["to_node_id"].to_numpy(dtype=int)
    ax, ay = node_x[start_ids], node_y[start_ids]
    bx, by = node_x[end_ids], node_y[end_ids]

    rows = []
    for obs_idx, (t, x, y) in enumerate(zip(track["timestamp_s"], obs_x, obs_y)):
        distances = point_to_segment_distance(x, y, ax, ay, bx, by)

        arc_ids = np.argsort(distances, kind="stable")
        if radius_m is not None:
            arc_ids = arc_ids[distances[arc_ids] <= radius_m]
        if max_candidates is not None:
            arc_ids = arc_ids[:max_candidates]
        rows.extend((obs_idx, arc_id, t, x, y, distances[arc_id]) for arc_id in arc_ids)

    return pd.DataFrame.from_records(
        rows, columns=["obs_idx", "arc_id", "t", "x", "y", "feat_dist_m"]
    ).astype({"obs_idx": "int64", "arc_id": "int64", "t": "float64", "x": "float64",
              "y": "float64", "feat_dist_m": "float64"})
