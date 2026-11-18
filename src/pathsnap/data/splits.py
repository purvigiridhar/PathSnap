"""P1/P2: group-aware CV. Never split a trip across folds."""

import numpy as np
from sklearn.model_selection import GroupKFold

from pathsnap.config import load_config


def group_kfold_by_trip(trip_ids, n_splits: int | None = None, seed: int | None = None):
    """Yield positional train and validation indices for the training table."""
    config = load_config()
    groups = np.asarray(trip_ids)
    splitter = GroupKFold(
        n_splits=config["cv"]["n_splits"] if n_splits is None else n_splits,
        shuffle=True,
        random_state=config["seed"] if seed is None else seed,
    )
    return splitter.split(np.empty((len(groups), 0)), groups=groups)
