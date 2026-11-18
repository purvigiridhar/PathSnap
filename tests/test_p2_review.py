import numpy as np
import pandas as pd
import pytest

from pathsnap.models.metrics import candidate_miss_summary
from pathsnap.models.predict import predict_candidate_probabilities


class ProbabilityFromFeature:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        positive = frame["feat_probability"].to_numpy(dtype=float)
        return np.column_stack([1.0 - positive, positive])


def test_prediction_preserves_candidate_row_order_and_ignores_labels():
    candidates = pd.DataFrame({
        "obs_idx": [2, 1],
        "label": [0, 1],
        "feat_probability": [0.8, 0.2],
    })
    original = candidates.copy(deep=True)
    artifact = {
        "model": ProbabilityFromFeature(),
        "model_name": "test",
        "feature_columns": ["feat_probability"],
        "positive_label": 1,
    }

    scores = predict_candidate_probabilities(candidates, artifact)

    np.testing.assert_allclose(scores, [0.8, 0.2])
    pd.testing.assert_frame_equal(candidates, original)


def test_candidate_bearing_and_full_track_miss_rates_have_separate_denominators():
    candidates = pd.DataFrame({
        "trip_id": ["a", "a", "a"],
        "obs_idx": [0, 0, 1],
        "label": [1, 0, 0],
    })
    full_track = pd.DataFrame({
        "trip_id": ["a", "a", "a"],
        "obs_idx": [0, 1, 2],
    })

    summary = candidate_miss_summary(candidates, full_track)

    assert summary["candidate_bearing_observations"] == 2
    assert summary["candidate_bearing_reference_misses"] == 1
    assert summary["candidate_bearing_reference_miss_rate"] == pytest.approx(0.5)
    assert summary["zero_candidate_observations"] == 1
    assert summary["full_track_reference_misses"] == 2
    assert summary["full_track_reference_miss_rate"] == pytest.approx(2 / 3)
