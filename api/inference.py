"""P4 (+P2/P3): load model artifacts once at startup; run the full pipeline for a trip_id.
Must NOT import or read reference labels."""


def load_artifacts():
    raise NotImplementedError


def predict_trip(trip_id: str) -> dict:
    raise NotImplementedError
