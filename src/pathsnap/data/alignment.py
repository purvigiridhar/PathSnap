"""Align training GPS observations to an ordered reference route."""

import numpy as np

from pathsnap.geo import point_to_segment_distance


def align_reference(obs_xy, node_xy, endpoints, route_ids):
    """Minimize total segment distance with nondecreasing route positions.

    Route positions can repeat or be skipped. A repeated arc ID can appear at
    several positions in the route, so the recurrence tracks positions.
    """
    if not len(obs_xy) or not len(route_ids):
        raise ValueError("Reference alignment requires observations and route arcs")
    ax, ay = node_xy[endpoints[route_ids, 0]].T
    bx, by = node_xy[endpoints[route_ids, 1]].T
    positions = np.arange(len(route_ids))
    predecessors = np.empty((len(obs_xy), len(route_ids)), dtype=np.int32)
    cost = point_to_segment_distance(*obs_xy[0], ax, ay, bx, by)

    for obs_idx, (x, y) in enumerate(obs_xy[1:], start=1):
        best = np.minimum.accumulate(cost)
        improves = np.r_[True, best[1:] < best[:-1]]
        predecessors[obs_idx] = np.maximum.accumulate(np.where(improves, positions, 0))
        cost = best + point_to_segment_distance(x, y, ax, ay, bx, by)

    aligned = np.empty(len(obs_xy), dtype=route_ids.dtype)
    position = np.argmin(cost)
    for obs_idx in range(len(obs_xy) - 1, -1, -1):
        aligned[obs_idx] = route_ids[position]
        if obs_idx:
            position = predecessors[obs_idx, position]
    return aligned
