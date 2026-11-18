"""Load a directed road graph in the candidate table's coordinate system."""

from pathlib import Path

from pathsnap.data.loader import load_arcs, load_nodes, load_track
from pathsnap.geo import local_metric_crs, project_lonlat


def load_projected_network(trip_dir, crs=None):
    """Return (nodes, arcs, crs), preserving the zero-based node and arc IDs."""
    trip_dir = Path(trip_dir)
    if crs is None:
        track = load_track(trip_dir)
        crs = local_metric_crs(track["lon"].to_numpy(), track["lat"].to_numpy())
    nodes = load_nodes(trip_dir)
    x, y = project_lonlat(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), crs)
    return nodes.assign(x=x, y=y), load_arcs(trip_dir), crs
