"""Log transition scores derived from directed road-network continuity."""

from __future__ import annotations

from typing import Any

import numpy as np


def transition_log_score(
    previous_arc: int,
    current_arc: int,
    network: Any,
    gps_distance_m: float,
    config: dict[str, Any],
    *,
    previous_xy: tuple[float, float] | None = None,
    current_xy: tuple[float, float] | None = None,
) -> float:
    """Score one directed arc transition using the selected configured mode.

    An unreachable pair receives exactly ``unreachable_log_penalty``. Same-arc
    transitions are handled separately so a vehicle can make progress along
    one segment without being treated as a zero-distance network jump.
    """
    mode = config.get("mode", "hop")
    beta = float(config.get("beta", 1.0))
    max_distance_m = config.get("max_distance_m", 5000.0)
    same_arc_bonus = float(config.get("same_arc_bonus", 0.0))
    hop_penalty = float(config.get("hop_penalty", 1.0))
    max_hops = config.get("max_hops", 5)
    unreachable_penalty = float(config.get("unreachable_log_penalty", -10.0))
    uturn_config = config.get("uturn", {})

    if mode not in {"hop", "distance_ratio"}:
        raise ValueError("transition mode must be 'hop' or 'distance_ratio'")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("transition beta must be finite and nonnegative")
    if not np.isfinite(unreachable_penalty) or unreachable_penalty > 0:
        raise ValueError("unreachable_log_penalty must be finite and nonpositive")
    if not np.isfinite(same_arc_bonus) or same_arc_bonus < 0:
        raise ValueError("same_arc_bonus must be finite and nonnegative")
    if not np.isfinite(hop_penalty) or hop_penalty < 0:
        raise ValueError("hop_penalty must be finite and nonnegative")
    if not isinstance(max_hops, (int, np.integer)) or max_hops < 1:
        raise ValueError("max_hops must be a positive integer")
    if max_distance_m is not None and (
        not np.isfinite(max_distance_m) or max_distance_m < 0
    ):
        raise ValueError("max_distance_m must be finite and nonnegative")
    if not np.isfinite(gps_distance_m) or gps_distance_m < 0:
        raise ValueError("gps_distance_m must be finite and nonnegative")

    same_arc = int(previous_arc) == int(current_arc)
    if mode == "hop":
        if same_arc:
            score = same_arc_bonus
        else:
            hops = network.hop_distance(
                previous_arc, current_arc, max_hops=max_hops
            )
            if hops is None or hops > max_hops:
                return unreachable_penalty
            score = -hop_penalty * max(0, hops - 1)
    else:
        if same_arc:
            if previous_xy is None or current_xy is None:
                road_distance_m = gps_distance_m
            else:
                road_distance_m = network.same_arc_distance_m(
                    previous_arc, previous_xy, current_xy
                )
        else:
            road_distance_m = network.path_length_m(
                previous_arc, current_arc, max_distance_m=max_distance_m
            )
            if road_distance_m is None:
                return unreachable_penalty
        score = -beta * abs(float(road_distance_m) - gps_distance_m)

    if (
        not same_arc
        and bool(uturn_config.get("enabled", False))
        and network.is_uturn(previous_arc, current_arc)
    ):
        uturn_penalty = float(uturn_config.get("penalty", -5.0))
        if not np.isfinite(uturn_penalty) or uturn_penalty > 0:
            raise ValueError("uturn.penalty must be finite and nonpositive")
        score += uturn_penalty
    return float(score)
