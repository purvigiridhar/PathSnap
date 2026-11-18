"""P4: server-side reference labels for scoring only. Path from env EVAL_REFERENCE_DIR.
Used only AFTER prediction, to compute metrics. Never exposed to the model or UI.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import pandas as pd
import numpy as np

from pathsnap.data.loader import load_track, load_nodes, load_arcs, load_route
from pathsnap.data.network import load_projected_network
from pathsnap.hmm.network import ArcGraph
from pathsnap.hmm.compare import align_route_reference, evaluate_predictions
from pathsnap.build_dataset import build_candidates_for_trip

_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_reference_dir() -> Path:
    """Resolve reference directory from environment or standard data paths."""
    ref_dir = os.environ.get("EVAL_REFERENCE_DIR") or os.environ.get("EVAL_DATA_DIR")
    if ref_dir and os.path.exists(ref_dir):
        return Path(ref_dir)
    student_test = _REPO_ROOT / "data" / "student_test"
    if student_test.exists():
        return student_test
    return _REPO_ROOT / "data" / "train"


def score_trip(
    trip_id: str,
    pred_arcs: List[int | None],
    arcs_df: Optional[pd.DataFrame] = None,
    ref_dir: Optional[str | Path] = None
) -> Dict[str, Any]:
    """
    Score a trip's predictions against the server-side ground truth route.
    Uses the official aligned reference and continuity hop distance evaluator.
    """
    ref_path = Path(ref_dir) if ref_dir else get_reference_dir()
    trip_dir = ref_path / trip_id

    # Check if trip exists in reference directory
    route_file = trip_dir / f"{trip_id}.route"
    if not route_file.exists():
        # Fallback search in data folders
        for d in [_REPO_ROOT / "data" / "student_test" / trip_id, _REPO_ROOT / "data" / "train" / trip_id]:
            if (d / f"{trip_id}.route").exists():
                trip_dir = d
                break

    if not (trip_dir / f"{trip_id}.route").exists():
        return {
            "top1_accuracy": None,
            "route_continuity": None,
            "n_points": len(pred_arcs),
            "note": "Reference route not available for scoring on this partition."
        }

    try:
        track = load_track(trip_dir)
        candidates = build_candidates_for_trip(trip_dir)
        coord_crs = candidates["coord_crs"].dropna().unique()[0]
        nodes, arcs, crs = load_projected_network(trip_dir, crs=coord_crs)
        route = load_route(trip_dir)

        ref_arcs = align_route_reference(track, nodes, arcs, route, coord_crs)
        network = ArcGraph(nodes, arcs, coord_crs=coord_crs, nodes_are_projected=True)

        # Ensure predictions are aligned with track length
        padded_preds = list(pred_arcs)
        if len(padded_preds) < len(track):
            padded_preds.extend([None] * (len(track) - len(padded_preds)))
        elif len(padded_preds) > len(track):
            padded_preds = padded_preds[:len(track)]

        metrics = evaluate_predictions(padded_preds, ref_arcs, candidates, network, max_hops=5)

        return {
            "top1_accuracy": round(float(metrics["arc_accuracy"]), 4),
            "route_continuity": round(float(metrics["route_continuity_rate"]), 4),
            "n_points": int(metrics["observations"]),
            "correct_points": int(metrics["correct"]),
            "missing_points": int(metrics["missing"])
        }

    except Exception as e:
        # Fallback basic count comparison if graph alignment fails
        return {
            "top1_accuracy": 0.0,
            "route_continuity": 0.0,
            "n_points": len(pred_arcs),
            "error": str(e)
        }
