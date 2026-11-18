"""P1: projection and distance helpers. Do not use Euclidean distance in degrees."""

from math import hypot


def project_lonlat(lon, lat, crs: str = "EPSG:32632"):
    raise NotImplementedError


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Return the distance from P to the closed segment AB in coordinate units."""
    dx = bx - ax
    dy = by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return hypot(px - ax, py - ay)

    fraction = ((px - ax) * dx + (py - ay) * dy) / length_squared
    fraction = max(0.0, min(1.0, fraction))
    return hypot(px - (ax + fraction * dx), py - (ay + fraction * dy))


def segment_bearing(ax, ay, bx, by):
    raise NotImplementedError
