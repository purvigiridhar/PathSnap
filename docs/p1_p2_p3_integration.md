# P1–P3 integration

`pathsnap.hmm.pipeline.reconstruct_trip()` connects the trip-level P1 inputs to
the unchanged P3 `decode(trip_candidates, scores, network)` interface.

## Pipeline contract

The pipeline uses P1's label-free `build_candidates_for_trip(trip_dir)` and
`load_track(trip_dir)` functions. The candidate builder's rows are passed to
the score adapter without sorting. The P1 candidate schema includes
`trip_id`, `obs_idx`, `arc_id`, `t`, `x`, `y`, all engineered `feat_*`
columns, and `coord_crs`; candidate-generation columns retain P1's zero-based
row IDs. The builder does not add `label`.

| Column | P1 output type | Meaning |
|---|---|---|
| `trip_id` | string | Trip identifier |
| `obs_idx` | `int64` | Original GPS track row index |
| `arc_id` | `int64` | Original `.arcs` row index |
| `t` | `float64` | GPS timestamp in seconds |
| `x`, `y` | `float64` | GPS coordinates in projected metres |
| `feat_*` | numeric | Engineered candidate features documented in `docs/features.md` |
| `coord_crs` | string | Projected metric CRS used for candidate coordinates |

The graph comes from P1's `load_projected_network(trip_dir, crs=coord_crs)`.
That helper returns node `x`/`y` coordinates, directed arcs, and the CRS.
`ArcGraph(..., nodes_are_projected=True)` consumes those projected coordinates
directly, preserving node and arc row positions as IDs. The pipeline checks
that the returned CRS matches the candidates' single, non-null `coord_crs`.

## P2 probability adapter

The published P2 predictor and baseline artifact are available on this branch.
Pass the function to `reconstruct_trip()` or `score_candidate_rows()`:

```python
from pathsnap.hmm.pipeline import reconstruct_trip
from pathsnap.models.predict import predict_candidate_probabilities

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

The adapter does not provide a substitute classifier or fallback scores. The
comparison command loads `models/baseline.joblib` once and passes that artifact
to the actual P2 predictor for every trip.

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

## Evaluation, comparison, and maps

`pathsnap.hmm.compare` compares independent per-observation classifier argmax
against Viterbi using the same P1 candidate DataFrame and P2 score vector. The
classifier preserves row-order tie breaking. Neither prediction path receives
`label` or the reference route. The `.route` reference is loaded only after
both prediction sequences have been produced and is aligned to GPS rows with
P1's `align_reference()` for evaluation.

The evaluation reports per-trip and aggregate results in
`results/evaluation/student_test/`:

| Metric | Definition |
|---|---|
| `arc_accuracy` | Correct directed arc IDs / all GPS observations; missing predictions count as incorrect |
| `conditional_arc_accuracy` | Correct predictions / nonmissing predictions |
| `missing_rate` | Missing predictions / all GPS observations |
| `candidate_reference_coverage` | Observations whose aligned reference arc exists among candidates / all observations |
| `direct_connectivity_rate` | Adjacent observation pairs with predictions whose arcs are equal or immediate directed successors / adjacent pairs with predictions |
| `route_continuity_rate` | Pairs with predictions connected by a directed path within configured `transition.max_hops` / adjacent pairs with predictions |
| `disconnected_transition_rate` | Pairs without such a bounded directed path / adjacent pairs with predictions |

Empty denominators produce `NaN`. The continuity metrics use the HMM
configuration's hop bound (currently 5); direct connectivity is reported
separately. These metrics do not bridge observation gaps when either prediction
is missing.

Standalone Folium maps are written to `results/maps/student_test/`. They show
the GPS trace, reference `.route` sequence, classifier-only route, and HMM route
as separate controllable layers. Arc lines use each directed `.arcs` row's
`from_node_id`/`to_node_id` and the original WGS84 node coordinates. The raw
network files contain endpoint nodes rather than detailed road polylines, so
the road display is a straight endpoint-to-endpoint representation of each
arc, not a higher-resolution road shape. Missing predictions and disconnected
arcs create separate line segments.

Run the development comparison from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m pathsnap.hmm.compare `
  --raw-dir data\raw\student_test `
  --model models\baseline.joblib `
  --saved-predictions-dir results\hmm\student_test `
  --output-dir results\evaluation\student_test `
  --map-dir results\maps\student_test
```

The existing saved HMM predictions are read only for row-aligned parity
checking; they are not overwritten. Use `--trip-id ID` (repeatable) to run a
subset of development trips. The command writes `per_trip_metrics.csv`,
`aggregate_metrics.csv`, `comparison.csv`, and `profile.csv` alongside the
maps.

Focused tests cover candidate score order and label exclusion, missing rows,
evaluation denominators, directed connectivity, comparison input identity,
directed map geometry, and standalone HTML generation:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_compare.py tests\test_folium_maps.py `
  tests\test_hmm_integration.py tests\test_viterbi.py -q
```
