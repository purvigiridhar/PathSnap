"""Create a deterministic grouped-CV sample with complete candidate sets.

Example:
python scripts/make_p2_cv_sample.py --train data/processed/offline/train
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/offline/train"))
    parser.add_argument("--observations", type=int, default=31_250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/processed/offline/p2_cv_sample.parquet"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.observations < 1:
        parser.error("--observations must be positive")
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} already exists; pass --overwrite to replace it")

    files = sorted(args.train.glob("*.parquet")) if args.train.is_dir() else [args.train]
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {args.train}")
    train = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    required = {"trip_id", "obs_idx", "arc_id", "label"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Training data missing required columns: {sorted(missing)}")
    feature_columns = [column for column in train if column.startswith("feat_")]
    if len(feature_columns) != 19:
        raise ValueError(f"Expected 19 P1 features, found {len(feature_columns)}")

    observations = train[["trip_id", "obs_idx"]].drop_duplicates().reset_index(drop=True)
    sample_size = min(args.observations, len(observations))
    rng = np.random.default_rng(args.seed)
    selected = observations.iloc[rng.choice(len(observations), size=sample_size, replace=False)]
    sample = train.merge(
        selected,
        on=["trip_id", "obs_idx"],
        how="inner",
        sort=False,
        validate="many_to_one",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.output, index=False)
    print(
        f"Saved {len(sample):,} rows from {sample_size:,} complete observations "
        f"across {sample['trip_id'].nunique()} trips with seed {args.seed} to {args.output}"
    )


if __name__ == "__main__":
    main()
