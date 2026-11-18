"""P4: FastAPI app. Log every request (evaluators will inspect the API logs)."""
import logging
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pathsnap.api")
app = FastAPI(title="PathSnap API")


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    # TODO: return the ten evaluation trip IDs
    raise NotImplementedError


@app.post("/predict")
def predict(req: PredictRequest):
    log.info("predict request trip_id=%s", req.trip_id)
    # TODO: inference.predict_trip -> reference_store scoring -> response
    raise NotImplementedError
