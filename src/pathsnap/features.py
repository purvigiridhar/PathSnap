"""Features per GPS observation and candidate road segment."""

import numpy as np

from pathsnap.geo import point_to_segment_distance, project_lonlat


def _observation_motion(x, y, timestamps):
    dx, dy = np.diff(x), np.diff(y)
    dt = np.diff(timestamps)
    step_length = np.hypot(dx, dy)
    valid_step = (dt > 0) & (dt <= 5)
    speed = np.divide(step_length, dt, out=np.zeros_like(step_length), where=valid_step)
    # Avoid estimating a heading from a GPS gap or a near-stationary step.
    usable = valid_step & (step_length >= 3)
    dx, dy = np.where(usable, dx, 0.0), np.where(usable, dy, 0.0)
    return (np.r_[0.0, dx] + np.r_[dx, 0.0],
            np.r_[0.0, dy] + np.r_[dy, 0.0], np.r_[0.0, speed])


def build_features(candidates, track, nodes, arcs, *, crs="EPSG:32632"):
    """Add deterministic geometry, motion, and road connectivity features."""
    node_x, node_y = project_lonlat(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), crs)
    start, end = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int).T
    arc_ids = candidates["arc_id"].to_numpy(dtype=int)
    obs_idx = candidates["obs_idx"].to_numpy(dtype=int)
    arc_dx = (node_x[end] - node_x[start])[arc_ids]
    arc_dy = (node_y[end] - node_y[start])[arc_ids]
    lengths = np.hypot(arc_dx, arc_dy)
    x, y = project_lonlat(track["lon"].to_numpy(), track["lat"].to_numpy(), crs)
    t = track["timestamp_s"].to_numpy()
    dx, dy, speed = _observation_motion(x, y, t)
    obs_dx, obs_dy = dx[obs_idx], dy[obs_idx]
    heading_valid = (np.hypot(obs_dx, obs_dy) >= 3) & (lengths > 0)
    cross = obs_dx * arc_dy - obs_dy * arc_dx
    dot = obs_dx * arc_dx + obs_dy * arc_dy
    angle = np.degrees(np.arctan2(np.abs(cross), dot))
    start_ids, end_ids = start[arc_ids], end[arc_ids]
    ax, ay = node_x[start_ids], node_y[start_ids]
    bx, by = node_x[end_ids], node_y[end_ids]
    px, py = x[obs_idx], y[obs_idx]
    fraction = np.divide((px - ax) * arc_dx + (py - ay) * arc_dy, lengths ** 2,
                         out=np.zeros_like(lengths), where=lengths > 0)
    speed = speed[obs_idx]
    distance = candidates["feat_dist_m"]
    by_observation = distance.groupby(candidates["obs_idx"])
    rank = by_observation.rank(method="first").astype(np.int32) - 1
    neighbor_features = {}
    for name, step in (("prev", -1), ("next", 1)):
        neighbor = np.clip(obs_idx + step, 0, max(len(track) - 1, 0))
        gap = (t[neighbor] - t[obs_idx]) * step
        neighbor_dist = point_to_segment_distance(x[neighbor], y[neighbor], ax, ay, bx, by)
        neighbor_features[f"feat_{name}_dist_m"] = np.where(
            (gap > 0) & (gap <= 5), neighbor_dist, distance
        )
    return candidates.assign(
        feat_arc_length_m=lengths,
        feat_heading_diff_deg=np.where(heading_valid, angle, 90.0),
        feat_heading_valid=heading_valid.astype(np.int8),
        feat_projection_ratio=np.clip(fraction, 0, 1),
        feat_dist_to_start_m=np.hypot(px - ax, py - ay),
        feat_dist_to_end_m=np.hypot(px - bx, py - by),
        feat_dist_to_midpoint_m=np.hypot(px - (ax + bx) / 2, py - (ay + by) / 2),
        feat_gps_speed_mps=speed,
        feat_heading_cos_sim=np.where(heading_valid, np.cos(np.radians(angle)), 0.0),
        feat_spatial_likelihood=np.exp(-0.5 * (distance / 12) ** 2),
        feat_rank_by_dist=rank,
        feat_is_closest=rank.eq(0).astype(np.int8),
        feat_speed_length_ratio=np.divide(speed, lengths, out=np.zeros_like(speed),
                                          where=lengths > 0),
        feat_dist_above_best_m=distance - by_observation.transform("min"),
        feat_start_in_degree=np.bincount(end, minlength=len(nodes))[start_ids],
        feat_end_out_degree=np.bincount(start, minlength=len(nodes))[end_ids],
        **neighbor_features,
    )
