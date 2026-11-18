"""P4: FastAPI app. Log every request (evaluators will inspect the API logs)."""
import os
import time
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.inference import load_artifacts, predict_trip, get_eval_data_dir
from api.reference_store import score_trip

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap Map-Matching API", version="0.1.0")


class PredictRequest(BaseModel):
    trip_id: str


@app.on_event("startup")
def startup_event():
    log.info("Starting PathSnap API server...")
    load_artifacts()
    log.info("Evaluation data directory: %s", get_eval_data_dir())


@app.get("/")
def root():
    """Root endpoint welcoming evaluators and linking to docs."""
    return {
        "message": "PathSnap Map-Matching API is running",
        "docs": "/docs",
        "health": "/health",
        "trips": "/trips"
    }


@app.get("/health")
def health():
    """Health check endpoint to test connectivity between Laptop 1 and Laptop 2."""
    log.info("GET /health -> status: ok")
    return {"status": "ok"}


@app.get("/trips")
def get_trips():
    """Returns the list of evaluation trip IDs available."""
    eval_dir = get_eval_data_dir()
    if not os.path.exists(eval_dir):
        log.error("Evaluation directory not found: %s", eval_dir)
        raise HTTPException(status_code=500, detail="Evaluation directory not found.")

    trip_ids = []
    for item in sorted(os.listdir(eval_dir)):
        item_path = os.path.join(eval_dir, item)
        if os.path.isdir(item_path):
            trip_ids.append(item)
        elif item.endswith(".track"):
            trip_ids.append(item.replace(".track", ""))

    trip_ids = sorted(list(set(trip_ids)))
    partition_name = os.path.basename(str(eval_dir))
    log.info("GET /trips -> found %d trips in %s", len(trip_ids), partition_name)
    return {"trips": trip_ids, "count": len(trip_ids), "partition": partition_name}


def _handle_predict(trip_id: str):
    start_time = time.time()
    log.info("Incoming prediction request for trip_id: %s", trip_id)

    try:
        # 1. Run inference
        inf_result = predict_trip(trip_id)
        pred_arcs = inf_result["pred_arcs"]
        arcs_df = inf_result.get("arcs_df")
        map_html = inf_result.get("map_html")

        # 2. Server-side evaluation scoring
        metrics = score_trip(trip_id, pred_arcs, arcs_df)

        latency_s = round(time.time() - start_time, 3)
        log.info(
            "COMPLETED trip_id=%s | latency=%.3fs | top1_acc=%s | continuity=%s",
            trip_id,
            latency_s,
            metrics.get("top1_accuracy"),
            metrics.get("route_continuity")
        )

        return {
            "trip_id": trip_id,
            "predicted_route": pred_arcs,
            "metrics": metrics,
            "latency_s": latency_s,
            "map_html": map_html
        }

    except FileNotFoundError as fnf:
        log.warning("Trip not found: %s", fnf)
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        log.error("Error processing trip_id %s: %s", trip_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/{trip_id}")
def predict_path(trip_id: str):
    """Path-parameter predict endpoint (used by Streamlit UI)."""
    return _handle_predict(trip_id)


@app.post("/predict")
def predict_body(req: PredictRequest):
    """JSON-body predict endpoint (supports test contracts / standard POST)."""
    return _handle_predict(req.trip_id)
