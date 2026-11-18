# Interface contracts (agree on these in the first 30 minutes, then edit here only with team consent)

## 1. Candidate table (P1 produces, P2 and P3 consume)
One row per (GPS observation, candidate arc). Saved as parquet in `data/processed/`.

| column | meaning |
|---|---|
| trip_id | trip identifier (group key for CV) |
| obs_idx | Original 0-based GPS observation row index in the trip's `.track` |
| arc_id | Original 0-based row index in the trip's `.arcs`; each row is a directed segment, with each direction represented by its own arc ID when both directions occur |
| t | GPS observation timestamp in seconds, copied from `.track` |
| x, y | GPS observation projected coordinates in metres |
| coord_crs | CRS shared by the trip's projected GPS coordinates and network coordinates; use it to project network nodes for HMM transitions |
| feat_* | engineered features (document each in `docs/features.md`) |
| label | Reference-candidate target: 1 when `arc_id` matches the aligned reference arc for the observation, otherwise 0. Present only in offline training and student-test development data; never used by the decoder |

P1 aligns observations in file order to nondecreasing positions in the
reference route by minimizing total point-to-segment distance. Observations
may repeat a route position or skip forward. P1 generates candidates and
features from GPS and network files before attaching labels. A candidate miss
produces no candidate row (and therefore no positive row) for that observation;
partition summaries count misses over all observations, including those with
zero candidates.

The GPS positions, network nodes, and metric features for a trip use the same
projected, metre-based CRS. `coord_crs` identifies that CRS and must agree
between candidate coordinates and the network used by the decoder. Candidate
rows and scores must remain aligned one-to-one in their original row order;
attach scores before sorting or grouping by `obs_idx`. The decoder ignores
`label`. For inference, use P1's label-free `build_candidates_for_trip()`;
`load_projected_network(trip_dir, crs)` returns projected nodes, directed arcs,
and their CRS while preserving original file-row node and arc IDs.

Fit models on training trips only. Student-test labels are for offline
development evaluation; prediction must use the label-free candidate builder.
Evaluator references are reserved for final scoring.

## 2. Model (P2 produces, P3 and API consume)

`pathsnap.models.predict.predict_candidate_probabilities(candidate_rows, model="models/baseline.joblib")` returns a one-dimensional NumPy array with exactly one `P(candidate is correct)` score per input row. Output position `i` always corresponds to input row `i`; the function does not sort, filter, or drop candidates. Scores are independent per row and are not normalized across an observation.

The joblib artifact stores the fitted preprocessing/model pipeline, positive label, and ordered `feat_*` feature columns. Prediction passes only those stored feature columns to the estimator and ignores any `label` column. P3 may use these scores as emission likelihoods or normalize them according to its decoder design.

The selected baseline is trained on labeled training candidates only. Student-test labels are used for development metrics only; evaluator labels are not used for training, model selection, or tuning.
## 3. Decoder (P3 produces, API consumes)
Pass the full GPS track separately from the candidate table. The decoder returns
one arc ID per original track row, using `None` where there are no candidates.
`load_projected_network(trip_dir, crs)` returns projected nodes, directed arcs,
and their CRS. Node and arc IDs are their original file row indices.

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

Use `build_candidates_for_trip(trip_dir)` for label-free prediction inputs. It uses the same deterministic feature calculations and configured shortlist as the batch build. Any learned preprocessing belongs in the saved P2 model pipeline.
