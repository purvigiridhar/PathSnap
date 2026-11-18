"""Build a candidate Parquet for one raw trip."""

import argparse
from pathlib import Path

from pathsnap.candidates import generate_candidates
from pathsnap.data.loader import load_arcs, load_nodes, load_track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--radius-m", type=float)
    parser.add_argument("--max-candidates", type=int, required=True)
    args = parser.parse_args()
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    if args.radius_m is not None and args.radius_m < 0:
        parser.error("--radius-m must be nonnegative")

    trip_dir = args.trip_dir
    candidates = generate_candidates(
        load_track(trip_dir),
        load_nodes(trip_dir),
        load_arcs(trip_dir),
        radius_m=args.radius_m,
        max_candidates=args.max_candidates,
    )
    candidates.insert(0, "trip_id", trip_dir.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(args.output, index=False)
    print(f"Wrote {len(candidates):,} candidate rows to {args.output}")


if __name__ == "__main__":
    main()
