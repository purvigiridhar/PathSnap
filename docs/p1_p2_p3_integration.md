# P1–P3 integration

`pathsnap.hmm.pipeline.reconstruct_trip()` connects the trip-level P1 inputs to
the unchanged P3 `decode(trip_candidates, scores, network)` interface.

## Pipeline contract

The pipeline uses P1's label-free `build_candidates_for_trip(trip_dir)` and
`load_track(trip_dir)` functions. The candidate builder's rows are passed to
the score adapter without sorting. The P1 candidate schema includes
`trip_id`, `obs_idx`, `arc_id`, `t`, `x`, `y`, `feat_dist_m`,
`feat_arc_length_m`, and `coord_crs`; candidate-generation columns retain P1's
zero-based row IDs. The builder does not add `label`.

| Column | P1 output type | Meaning |
|---|---|---|
| `trip_id` | string | Trip identifier |
| `obs_idx` | `int64` | Original GPS track row index |
| `arc_id` | `int64` | Original `.arcs` row index |
| `t` | `float64` | GPS timestamp in seconds |
| `x`, `y` | `float64` | GPS coordinates in projected metres |
| `feat_dist_m` | `float64` | Point-to-candidate-segment distance in metres |
| `feat_arc_length_m` | `float64` | Candidate segment length in metres |
| `coord_crs` | string | Projected metric CRS used for candidate coordinates |

The graph comes from P1's `load_projected_network(trip_dir, crs=coord_crs)`.
That helper returns node `x`/`y` coordinates, directed arcs, and the CRS.
`ArcGraph(..., nodes_are_projected=True)` consumes those projected coordinates
directly, preserving node and arc row positions as IDs. The pipeline checks
that the returned CRS matches the candidates' single, non-null `coord_crs`.

## P2 probability adapter

P2's function is not present on this branch yet. Once available, pass it to
`reconstruct_trip()` or `score_candidate_rows()`:

```python
from pathsnap.hmm.pipeline import reconstruct_trip

# Import predict_candidate_probabilities from P2's published module.
arc_ids = reconstruct_trip(
    trip_dir,
    predictor=predict_candidate_probabilities,
    model="models/baseline.joblib",
)
```

The adapter calls the supplied function as
`predictor(candidate_rows, model=model)`, removing `label` first if present.
The predictor must return one finite probability in `[0, 1]` per input row and
must preserve input row order. The adapter checks output dimensionality,
length, value range, and that the predictor did not mutate or reorder the
candidate identity columns. As the P2 API returns only a vector, the adapter
cannot independently detect a predictor that returns a same-length vector in
the wrong order; preserving row order is part of P2's contract.

No classifier, synthetic probability model, or fallback scores are provided.
The real P2 function and its model artifact are still required for end-to-end
inference.

## Full-track alignment and missing candidates

P1 omits observations that have no candidate arcs. P3's decoder returns one
prediction per observation from the minimum through maximum candidate
`obs_idx`; its internal gaps are `None`. The wrapper places that range into a
list whose length is the full GPS track, leaving leading, internal, and
trailing zero-candidate positions as `None`. An entirely empty candidate table
returns a full-length list of `None` without invoking P2 or constructing a
network.

Neither reference routes nor `label` values are read by this inference path.
`label`, if present in caller-provided candidate rows, is excluded before P2
scoring and removed before P3 decoding.

## Availability and tests

The P1 APIs were verified on `origin/main`, but `src/pathsnap/build_dataset.py`
and `src/pathsnap/data/network.py` are not present in the current `p3-hmm`
checkout. Their imports are lazy, so P3-only tests remain usable; calling
`reconstruct_trip()` on this branch reports that the P1 integration
dependencies are unavailable. The integration unit tests cover the P1 row
schema, projected-network handoff, score boundary, empty candidates, and
full-track alignment with synthetic inputs. They do not execute P1's raw-trip
builder or a P2 model.

Run the available integration tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hmm_integration.py -v
```

After P1's implementation and P2's prediction function/model are available on
the same branch, add and run a trip-level smoke test using development data
only. Do not pass reference routes or labels to inference.
