"""P4: server-side reference labels for scoring only. Path from env EVAL_REFERENCE_DIR.
Used only AFTER prediction, to compute metrics. Never exposed to the model or UI."""
import os

REFERENCE_DIR = os.environ.get("EVAL_REFERENCE_DIR")


def get_reference_route(trip_id: str):
    raise NotImplementedError
