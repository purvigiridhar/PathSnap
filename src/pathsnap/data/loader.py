"""P1: readers for the raw trip files. Verify real column meanings and row counts yourselves (see Dataset Exploration PDF)."""
from pathlib import Path


def list_trips(partition_dir: str | Path) -> list[str]:
    raise NotImplementedError


def load_track(trip_dir: Path):
    """.track: lon, lat, timestamp_s (no header)."""
    raise NotImplementedError


def load_nodes(trip_dir: Path):
    """.nodes: lon, lat (row index = node_id)."""
    raise NotImplementedError


def load_arcs(trip_dir: Path):
    """.arcs: from_node_id, to_node_id (row index = arc_id)."""
    raise NotImplementedError


def load_route(trip_dir: Path):
    """.route: sequence of arc_ids (reference information, may have limitations)."""
    raise NotImplementedError
