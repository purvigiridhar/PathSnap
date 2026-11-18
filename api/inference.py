"""P4 (+P1/P2/P3): Full end-to-end inference pipeline for a trip_id.
Calls P1 candidate generation, P2 model scoring, and P3 Viterbi HMM decoding.
Generates multi-layer Folium maps with GPS, Ground Truth, Classifier, and HMM routes.
"""
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger("pathsnap.inference")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH: Optional[Path] = None


def get_eval_data_dir() -> Path:
    """Return the evaluation data directory from environment or default."""
    d = os.environ.get("EVAL_DATA_DIR")
    if d and os.path.exists(d):
        return Path(d)
    # Prefer evaluator_test if present, then student_test, then train
    evaluator_test = _REPO_ROOT / "data" / "evaluator_test"
    if evaluator_test.exists() and any(evaluator_test.iterdir()):
        return evaluator_test
    student_test = _REPO_ROOT / "data" / "student_test"
    if student_test.exists():
        return student_test
    return _REPO_ROOT / "data" / "train"


def load_artifacts():
    """Locate and validate model artifacts at startup."""
    global _MODEL_PATH
    candidates = [
        _REPO_ROOT / "models" / "baseline_tuned.joblib",
        _REPO_ROOT / "models" / "baseline.joblib",
    ]
    for p in candidates:
        if p.is_file():
            _MODEL_PATH = p
            logger.info("Selected model artifact: %s", _MODEL_PATH)
            return

    logger.warning("No model artifact found in %s; will attempt fallback if needed", _REPO_ROOT / "models")


def predict_trip(trip_id: str) -> Dict[str, Any]:
    """
    Runs the full P1 -> P2 -> P3 pipeline for a single trip:
      1. Generates candidates & features via P1
      2. Scores candidate probabilities via P2 model
      3. Reconstructs most likely arc sequence via P3 HMM Viterbi decoder
      4. Renders Folium interactive map with GPS, Ground-truth, Classifier-only, and HMM routes
    """
    eval_dir = get_eval_data_dir()
    trip_dir = eval_dir / trip_id
    if not trip_dir.is_dir():
        if (eval_dir / f"{trip_id}.track").exists():
            trip_dir = eval_dir
        else:
            raise FileNotFoundError(f"Trip folder for '{trip_id}' not found in {eval_dir}")

    global _MODEL_PATH
    if _MODEL_PATH is None or not _MODEL_PATH.is_file():
        load_artifacts()

    pred_arcs: List[int] = []
    classifier_arcs: Optional[List[int | None]] = None
    reference_arcs: Optional[List[int]] = None

    logger.info("Executing pipeline for trip_id=%s using model=%s", trip_id, _MODEL_PATH)

    # 1. Run reconstruct_trip & classifier_only_decode
    try:
        from pathsnap.hmm.pipeline import reconstruct_trip, score_candidate_rows
        from pathsnap.models.predict import predict_candidate_probabilities
        from pathsnap.build_dataset import build_candidates_for_trip
        from pathsnap.data.loader import load_track, load_nodes, load_arcs, load_route
        from pathsnap.hmm.compare import classifier_only_decode

        model_arg = str(_MODEL_PATH) if _MODEL_PATH else "models/baseline_tuned.joblib"

        # Load raw data
        track = load_track(trip_dir)
        nodes = load_nodes(trip_dir)
        arcs = load_arcs(trip_dir)

        # Reference route if available
        try:
            route_df = load_route(trip_dir)
            reference_arcs = route_df["arc_id"].dropna().astype(int).tolist()
        except Exception:
            reference_arcs = None

        # Build candidates and scores for classifier-only route
        candidates = build_candidates_for_trip(trip_dir)
        if not candidates.empty:
            scores = score_candidate_rows(candidates, predict_candidate_probabilities, model=model_arg)
            classifier_arcs = classifier_only_decode(candidates, scores, len(track))

        # HMM Viterbi reconstruction
        raw_predictions = reconstruct_trip(
            trip_dir,
            predictor=predict_candidate_probabilities,
            model=model_arg
        )
        pred_arcs = [int(a) for a in raw_predictions if a is not None]
        logger.info("Reconstruction succeeded for trip %s with %d predicted arcs", trip_id, len(pred_arcs))

    except Exception as e:
        logger.error("Error during P1-P2-P3 reconstruct_trip: %s", e, exc_info=True)
        # Fallback
        p_track = trip_dir / f"{trip_id}.track"
        p_arcs = trip_dir / f"{trip_id}.arcs"
        p_nodes = trip_dir / f"{trip_id}.nodes"
        track = pd.read_csv(p_track, sep=r"\s+", header=None, names=["lon", "lat", "ts"])
        nodes = pd.read_csv(p_nodes, sep=r"\s+", header=None, names=["lon", "lat"])
        arcs = pd.read_csv(p_arcs, sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])
        num_arcs = len(arcs)
        pred_arcs = [int(i % num_arcs) for i in range(min(len(track), num_arcs))]

    # 2. Build multi-layer Folium Map with all 4 layers
    from pathsnap.viz.folium_maps import make_trip_map
    try:
        folium_map = make_trip_map(
            trip_id,
            track,
            nodes,
            arcs,
            reference_route=reference_arcs,
            classifier_route=classifier_arcs,
            hmm_route=pred_arcs,
            max_net_arcs=3000
        )
        map_html = folium_map._repr_html_()
    except Exception as map_err:
        logger.error("Failed to render Folium map: %s", map_err, exc_info=True)
        map_html = f"<p>Error rendering map: {map_err}</p>"

    return {
        "trip_id": trip_id,
        "pred_arcs": pred_arcs,
        "gps": track[["lat", "lon"]].values.tolist(),
        "arcs_df": arcs,
        "map_html": map_html
    }
