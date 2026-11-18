# Interface contracts (agree on these in the first 30 minutes, then edit here only with team consent)

## 1. Candidate table (P1 produces, P2 and P3 consume)
One row per (GPS observation, candidate arc). Saved as parquet in `data/processed/`.

| column | meaning |
|---|---|
| trip_id | trip identifier (group key for CV) |
| obs_idx | 0-based index of the GPS point within the trip |
| arc_id | candidate arc (0-based row index in the trip's `.arcs`) |
| t | GPS observation timestamp in seconds, copied from `.track` |
| x | GPS observation easting in metres (EPSG:32632) |
| y | GPS observation northing in metres (EPSG:32632) |
| feat_* | engineered features (document each in `docs/features.md`) |
| label | 1 if arc_id equals the reference arc for that observation, else 0 (train/dev only) |

Open question to settle: how obs_idx is aligned to the `.route` reference (the route is a sequence of arcs, not one per GPS point). Record the decision in `docs/data_quality_log.md`.

## 2. Model (P2 produces, P3 and API consume)
`model.predict_proba(X_features) -> np.ndarray` of shape (n_rows,) = P(candidate is correct).
Saved with joblib along with the fitted preprocessing. Models that don't give probabilities need a documented adaptation.

## 3. Decoder (P3 produces, API consumes)
`decode(trip_candidates, scores, network) -> list[int]` one predicted arc_id per observation (or None if no candidate).

## 4. API (P4)
`GET /trips` -> list of the ten evaluation trip IDs
`POST /predict {"trip_id": "..."}` -> `{predicted_route, per_obs_arcs, metrics, latency_ms}`
Ground truth is looked up server-side only.
