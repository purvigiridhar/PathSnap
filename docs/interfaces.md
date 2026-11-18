# Interface contracts (agree on these in the first 30 minutes, then edit here only with team consent)

## 1. Candidate table (P1 produces, P2 and P3 consume)
One row per (GPS observation, candidate arc). Saved as parquet in `data/processed/`.

| column | meaning |
|---|---|
| trip_id | trip identifier (group key for CV) |
| obs_idx | 0-based index of the GPS point within the trip |
| arc_id | candidate arc (0-based row index in the trip's `.arcs`) |
| t | GPS observation timestamp in seconds |
| x, y | GPS observation coordinates projected into metres |
| coord_crs | Optional target CRS for x/y; when present it must match the network CRS |
| feat_* | engineered features (document each in `docs/features.md`) |
| label | Optional training/dev target; never used by the decoder |

P1's merged candidate builder currently projects x/y to EPSG:32632. If a
candidate producer supplies `coord_crs`, it must identify the projected metric
CRS used for x/y. The decoder requires scores aligned one-to-one with the
candidate rows in their original order and ignores `label`.

## 2. Model (P2 produces, P3 and API consume)
`model.predict_proba(X_features) -> np.ndarray` of shape (n_rows,) = P(candidate is correct).
Saved with joblib along with the fitted preprocessing. Models that don't give probabilities need a documented adaptation.

## 3. Decoder (P3 produces, API consumes)
`decode(trip_candidates, scores, network) -> list[int]` one predicted arc_id per observation (or None if no candidate).

`network` is a directed `ArcGraph`, constructed from the trip's nodes and arcs
using the same target CRS as the candidate x/y coordinates. The result covers
the inclusive range from the minimum through maximum `obs_idx` in the candidate
table. An observation with no candidate row between those bounds returns
`None`. Leading and trailing observations with no candidate rows cannot be
recovered without the original GPS track.

The Stage 2 `reconstruct_trip()` adapter loads the full track and expands this
candidate-range result to the original track length, using `None` for leading,
internal, and trailing observations with no candidates. It invokes P1's
label-free `build_candidates_for_trip()` and `load_projected_network()` using
the candidate table's `coord_crs`. P2's probability function is passed into the
adapter and receives the P1 candidate rows with any `label` column removed.
Scores must be one-dimensional, finite probabilities in `[0, 1]`, and remain
aligned with the original candidate-row order. The adapter also removes
`label` before calling the P3 decoder.

## 4. API (P4)
`GET /trips` -> list of the ten evaluation trip IDs
`POST /predict {"trip_id": "..."}` -> `{predicted_route, per_obs_arcs, metrics, latency_ms}`
Ground truth is looked up server-side only.
