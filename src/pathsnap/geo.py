"""P1: projection and distance helpers. Do not use Euclidean distance in degrees."""

from functools import lru_cache

import numpy as np
from pyproj import Transformer


@lru_cache
def _transformer(crs: str) -> Transformer:
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def project_lonlat(lon, lat, crs: str = "EPSG:32632"):
    """Project WGS84 longitude and latitude to x and y."""
    return _transformer(crs).transform(lon, lat, errcheck=True)


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Return the distance from P to the closed segment AB in coordinate units."""
    dx = bx - ax
    dy = by - ay
    length_squared = dx * dx + dy * dy
    offset_x = px - ax
    offset_y = py - ay
    dot = offset_x * dx + offset_y * dy
    fraction = np.divide(
        dot, length_squared, out=np.zeros_like(dot, dtype=float), where=length_squared > 0
    )
    fraction = np.clip(fraction, 0, 1)
    return np.hypot(offset_x - fraction * dx, offset_y - fraction * dy)


def segment_bearing(ax, ay, bx, by):
    raise NotImplementedError
