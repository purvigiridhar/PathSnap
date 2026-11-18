"""P1: projection and distance helpers. Do not use Euclidean distance in degrees."""


def project_lonlat(lon, lat, crs: str = "EPSG:32632"):
    raise NotImplementedError


def point_to_segment_distance(px, py, ax, ay, bx, by):
    raise NotImplementedError


def segment_bearing(ax, ay, bx, by):
    raise NotImplementedError
