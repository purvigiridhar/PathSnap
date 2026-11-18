"""Report candidate coverage against every raw GPS observation.

Expected raw layout: data/raw/{train,student_test}/<trip>/<trip>.track
Expected candidates: data/processed/offline/{train,student_test}/
"""
import argparse
from pathlib import Path

import pandas as pd

from pathsnap.data.loader import list_trips, load_track
from pathsnap.models.metrics import candidate_miss_summary
from pathsnap.models.train import _load_table


def observation_universe(raw_split: Path) -> pd.DataFrame:
    rows = []
    for trip_id in list_trips(raw_split):
        track = load_track(raw_split / trip_id)
        rows.extend((trip_id, obs_idx) for obs_idx in range(len(track)))
    if not rows:
        raise FileNotFoundError(f"No raw track observations found under {raw_split}")
    return pd.DataFrame(rows, columns=["trip_id", "obs_idx"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/offline"))
    parser.add_argument("--output", type=Path, default=Path("results/candidate_coverage.csv"))
    args = parser.parse_args()

    rows = []
    for split in ("train", "student_test"):
        candidates = _load_table(args.processed_root / split)
        candidates["trip_id"] = candidates["trip_id"].astype(str)
        universe = observation_universe(args.raw_root / split)
        universe["trip_id"] = universe["trip_id"].astype(str)
        rows.append({"split": split, **candidate_miss_summary(candidates, universe)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(args.output, index=False)
    print(report.to_string(index=False))
    print(f"Saved coverage report to {args.output}")


if __name__ == "__main__":
    main()
