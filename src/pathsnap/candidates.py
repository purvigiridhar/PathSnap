"""Generate a distance-based shortlist of directed road segments."""

import numpy as np
import pandas as pd
from shapely import STRtree, box

from pathsnap.geo import point_to_segment_distance, project_lonlat


def generate_candidates(
    track, nodes, arcs, radius_m=None, max_candidates=None, *, crs="EPSG:32632"
):
    """Return candidate rows with zero-based IDs and GPS time and coordinates."""
    obs_x, obs_y = project_lonlat(track["lon"].to_numpy(), track["lat"].to_numpy(), crs)
    node_x, node_y = project_lonlat(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), crs)
    start_ids = arcs["from_node_id"].to_numpy(dtype=int)
    end_ids = arcs["to_node_id"].to_numpy(dtype=int)
    ax, ay = node_x[start_ids], node_y[start_ids]
    bx, by = node_x[end_ids], node_y[end_ids]
    min_x, max_x = np.minimum(ax, bx), np.maximum(ax, bx)
    min_y, max_y = np.minimum(ay, by), np.maximum(ay, by)
    index = STRtree(box(min_x, min_y, max_x, max_y))
    all_ids = np.arange(len(arcs))

    rows = []
    for obs_idx, (t, x, y) in enumerate(zip(track["timestamp_s"], obs_x, obs_y)):
        if radius_m is None and (max_candidates is None or len(arcs) <= max_candidates):
            arc_ids = all_ids
            distances = point_to_segment_distance(x, y, ax, ay, bx, by)
        else:
            radius = radius_m if radius_m is not None else 50.0
            while True:
                # Every segment within this distance intersects the search square.
                nearby = index.query(box(x - radius, y - radius, x + radius, y + radius))
                distances = point_to_segment_distance(x, y, ax[nearby], ay[nearby],
                                                      bx[nearby], by[nearby])
                inside = distances <= radius
                arc_ids, distances = nearby[inside], distances[inside]
                if radius_m is not None or len(arc_ids) >= max_candidates:
                    break
                radius *= 2
        order = np.lexsort((arc_ids, distances))
        if max_candidates is not None:
            order = order[:max_candidates]
        rows.extend((obs_idx, arc_ids[i], t, x, y, distances[i]) for i in order)

    return pd.DataFrame.from_records(
        rows, columns=["obs_idx", "arc_id", "t", "x", "y", "feat_dist_m"]
    ).astype({"obs_idx": "int64", "arc_id": "int64", "t": "float64", "x": "float64",
              "y": "float64", "feat_dist_m": "float64"})
