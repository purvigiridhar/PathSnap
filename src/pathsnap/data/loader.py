"""P1: readers for the raw trip files. Verify real column meanings and row counts yourselves (see Dataset Exploration PDF)."""
from pathlib import Path

import pandas as pd


def list_trips(partition_dir: str | Path) -> list[str]:
    return sorted(
        trip.name
        for trip in Path(partition_dir).iterdir()
        if trip.is_dir() and (trip / f"{trip.name}.track").is_file()
    )


def _read_trip_file(trip_dir: Path, extension: str, columns: list[str]) -> pd.DataFrame:
    trip_dir = Path(trip_dir)
    path = trip_dir / f"{trip_dir.name}.{extension}"
    return pd.read_csv(path, sep="\t", header=None, names=columns)


def load_track(trip_dir: Path) -> pd.DataFrame:
    """.track: lon, lat, timestamp_s (no header)."""
    return _read_trip_file(trip_dir, "track", ["lon", "lat", "timestamp_s"])


def load_nodes(trip_dir: Path) -> pd.DataFrame:
    """.nodes: lon, lat (row index = node_id)."""
    return _read_trip_file(trip_dir, "nodes", ["lon", "lat"])


def load_arcs(trip_dir: Path) -> pd.DataFrame:
    """.arcs: from_node_id, to_node_id (row index = arc_id)."""
    return _read_trip_file(trip_dir, "arcs", ["from_node_id", "to_node_id"])


def load_route(trip_dir: Path) -> pd.DataFrame:
    """.route: sequence of arc_ids (reference information, may have limitations)."""
    return _read_trip_file(trip_dir, "route", ["arc_id"])
