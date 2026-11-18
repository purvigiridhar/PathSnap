# Interface contracts (agree on these in the first 30 minutes, then edit here only with team consent)

## 1. Candidate table (P1 produces, P2 and P3 consume)
One row per (GPS observation, candidate arc). Saved as parquet in `data/processed/`.

| column | meaning |
|---|---|
| trip_id | trip identifier (group key for CV) |
| obs_idx | 0-based index of the GPS point within the trip |
| arc_id | candidate arc (0-based row index in the trip's `.arcs`) |
| t | GPS observation timestamp in seconds, copied from `.track` |
| x | GPS observation projected x coordinate in metres |
| y | GPS observation projected y coordinate in metres |
| coord_crs | CRS definition shared by every row in the trip. Use the same CRS to project network nodes for HMM transitions. |
| feat_* | engineered features (document each in `docs/features.md`) |
| label | 1 if arc_id equals the reference arc for that observation, else 0. Present only in offline train and student-test development evaluation tables. |

P1 baseline for team review. Align observations in file order to nondecreasing positions in the reference route by minimizing total point-to-segment distance. Observations can repeat a position or skip forward. Generate candidates and features from GPS and network files, then attach the label. A candidate miss produces no positive row for that observation. The partition summary counts misses over all observations, including those with zero candidates.

Fit models on training trips only. Use student-test labels for offline development evaluation. Prediction uses the label-free build. Evaluator references are reserved for final scoring.

The GPS positions, road-network nodes, and metric features use the same trip CRS. Each row of `.arcs` is a directed segment. If both directions occur, each direction has its own arc ID.

## 2. Model (P2 produces, P3 and API consume)
`model.predict_proba(X_features) -> np.ndarray` of shape (n_rows,) = P(candidate is correct).
Saved with joblib along with the fitted preprocessing. Models that don't give probabilities need a documented adaptation.

Score at position `i` belongs to candidate row at position `i`. Attach scores before sorting or grouping by `obs_idx`.

Use `group_kfold_by_trip(train_table["trip_id"])` from `pathsnap.data.splits` to obtain the shared training/validation row positions. Reuse those folds for each model and fit model preprocessing within each training fold.

## 3. Decoder (P3 produces, API consumes)
Pass the full GPS track separately from the candidate table. The decoder returns
one arc ID per original track row, using `None` where there are no candidates.
`load_projected_network(trip_dir, crs)` returns projected nodes, directed arcs,
and their CRS. Node and arc IDs are their original file row indices.

## 4. API (P4)
`GET /trips` -> list of the ten evaluation trip IDs
`POST /predict {"trip_id": "..."}` -> `{predicted_route, per_obs_arcs, metrics, latency_ms}`
Ground truth is looked up server-side only.

Use `build_candidates_for_trip(trip_dir)` for label-free prediction inputs. It uses the same deterministic feature calculations and configured shortlist as the batch build. Any learned preprocessing belongs in the saved P2 model pipeline.
