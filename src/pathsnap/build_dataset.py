"""Build candidate Parquets for one raw trip or a student-data partition."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.candidates import generate_candidates
from pathsnap.data.alignment import align_reference
from pathsnap.data.loader import list_trips, load_arcs, load_nodes, load_route, load_track
from pathsnap.features import build_features
from pathsnap.geo import local_metric_crs, project_lonlat


def build_trip(trip_dir, *, max_candidates, radius_m=None, local_crs=False, labeled=False):
    track, nodes, arcs = (load(trip_dir) for load in (load_track, load_nodes, load_arcs))
    lon, lat = track["lon"].to_numpy(), track["lat"].to_numpy()
    crs = local_metric_crs(lon, lat) if local_crs else "EPSG:32632"
    candidates = generate_candidates(track, nodes, arcs, radius_m, max_candidates, crs=crs)
    candidates = build_features(candidates, track, nodes, arcs, crs=crs)
    candidates.insert(0, "trip_id", trip_dir.name)
    candidates["coord_crs"] = crs

    summary = {
        "trip_id": trip_dir.name,
        "observations": len(track),
        "candidate_rows": len(candidates),
        "observations_without_candidates": len(track) - candidates["obs_idx"].nunique(),
    }
    if labeled:
        route_ids = load_route(trip_dir)["arc_id"].to_numpy(dtype=int)
        endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
        aligned = align_reference(
            np.column_stack(project_lonlat(lon, lat, crs)),
            np.column_stack(project_lonlat(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), crs)),
            endpoints,
            route_ids,
        )
        candidates["label"] = (
            candidates["arc_id"].to_numpy() == aligned[candidates["obs_idx"].to_numpy()]
        ).astype(np.int8)
        misses = len(track) - int(candidates["label"].sum())
        summary.update(candidate_misses=misses, candidate_miss_rate=misses / len(track))
    return candidates, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--radius-m", type=float)
    parser.add_argument("--max-candidates", type=int, required=True)
    parser.add_argument("--partition", action="store_true", help="Input and output are directories")
    parser.add_argument("--local-crs", action="store_true", help="Use a trip-centred metric projection")
    parser.add_argument("--labeled", action="store_true", help="Use reference routes from student data")
    parser.add_argument("--limit", type=int, help="Build only the first N trips of a partition")
    args = parser.parse_args()
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    if args.radius_m is not None and (not np.isfinite(args.radius_m) or args.radius_m < 0):
        parser.error("--radius-m must be finite and nonnegative")
    if args.limit is not None and (not args.partition or args.limit < 1):
        parser.error("--limit requires --partition and a positive trip count")

    if args.partition:
        trips = [args.trip_dir / name for name in list_trips(args.trip_dir)[:args.limit]]
        if not trips:
            parser.error("No trips found in partition")
    else:
        trips = [args.trip_dir]
    summaries = []
    for trip in trips:
        candidates, summary = build_trip(
            trip, max_candidates=args.max_candidates, radius_m=args.radius_m,
            local_crs=args.local_crs, labeled=args.labeled,
        )
        output = args.output / f"{trip.name}.parquet" if args.partition else args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_parquet(output, index=False)
        summaries.append(summary)
        print(f"Wrote {len(candidates):,} candidate rows to {output}", flush=True)

    if args.partition:
        pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
