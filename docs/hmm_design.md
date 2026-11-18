# P3 HMM/Viterbi design

## Inputs and outputs

`decode(trip_candidates, scores, network)` accepts candidate rows for one trip,
one classifier probability per row in the same order, and a directed `ArcGraph`.
The decoder reads only `obs_idx`, `arc_id`, `t`, `x`, and `y`; optional labels
are not read. It returns an arc ID or `None` for each integer observation index
from the minimum through maximum index in the candidate table.

Candidate tables omit observations with no candidate arcs. Such an interior
observation is returned as `None`; its absence does not advance or reset the
Viterbi scores. The transition to the next candidate-bearing observation uses
the previous candidate-bearing observation's timestamp and coordinates.
Leading and trailing observations omitted from the table cannot be recovered
without the original GPS track.

## Emissions

For candidate probability `p`, the emission is `log(max(p, exp(log_floor)))`.
The default `log_floor` is `-10.0`. Optional per-observation normalization
subtracts log-sum-exp across that observation's candidates; it does not change
their within-observation ranking.

## Directed ArcGraph and projected distances

`ArcGraph(nodes, arcs, coord_crs="EPSG:32632")` treats node and arc IDs as
zero-based row positions, matching the raw `.nodes` and `.arcs` files. Nodes
are read as WGS84 longitude/latitude and projected with pyproj's
`Transformer.from_crs("EPSG:4326", coord_crs, always_xy=True)`. The CRS must be
projected in metres. If candidate rows include `coord_crs`, it must match the
graph CRS. This also supports a trip-specific projected CRS without assuming
EPSG:32632.

NetworkX stores directed node connectivity with projected segment lengths.
`successors(arc_id)` returns arcs leaving that arc's `to_node`. `arc_length_m`
returns the projected straight-segment length. For distinct candidate arcs,
`path_length_m(A, B)` searches from `A.to_node` to `B.from_node`; `max_distance_m`
is a distance cutoff in metres for distance-ratio mode. Hop mode uses the
fewest directed connector hops, bounded separately by `max_hops`, so a directly
adjacent arc pair has one hop. Same-arc transitions are handled separately: in
distance-ratio mode, GPS points are projected onto the segment and their
along-segment progress is compared.

## Transitions

The `transition.mode` setting selects:

- `hop`: `same_arc_bonus` for the same arc; otherwise
  `-hop_penalty * max(0, hops - 1)`. A transition exceeding `max_hops` is
  assigned the shared unreachable penalty.
- `distance_ratio`: `-beta * abs(d_road - d_GPS)`. For distinct arcs, `d_road`
  is the shortest directed connector length from `A.to_node` to `B.from_node`;
  for the same arc it is the difference in projected along-segment progress.

If no connector is found within the applicable `max_hops` or `max_distance_m`
bound, the transition receives the single configured
`transition.unreachable_log_penalty`. It is a soft penalty, not an impossible
transition. U-turns are identified only when the two arcs have exactly
reversed endpoint IDs. They remain possible; when `transition.uturn.enabled`
is true, the configured U-turn penalty is added. No reverse arcs are
synthesized.

If the elapsed time between candidate-bearing observations exceeds
`transition.gap_threshold_s`, decoding starts a new independent Viterbi chain
at the later observation. The configured default is 30 seconds; set it to
`null` to disable time-based restarts.

These hop parameters apply only to hop mode. Distance-ratio mode continues to
use `beta` and the `max_distance_m` road-distance bound.

## Viterbi recurrence and backtracking

For observation `t`, candidate state `j`, emission `E_t(j)`, and transition
score `T(i, j)`, the decoder calculates:

```text
V_t(j) = E_t(j) + max_i [V_(t-1)(i) + T(i, j)]
```

The maximizing predecessor index is stored for every state at every
candidate-bearing observation. Once a chain ends (at a configured time-gap
restart or the end of the trip), the state with the highest final score is
selected. Backtracking follows the stored predecessor indices in reverse
observation order to recover the globally highest-scoring candidate sequence.
Missing observation indices are then left as `None` in the output.

## Configuration

`configs/hmm.yaml` defines emission flooring/normalization, hop-mode limits and
scores, distance-ratio weight and bound, the shared unreachable penalty,
time-gap threshold, and optional U-turn handling. The initial defaults are for
deterministic Stage 1 behavior; they have not been tuned on real or evaluator
data.

## Classifier-only comparison and evaluation

Run the development evaluation from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pathsnap.hmm.compare `
  --raw-dir data\raw\student_test `
  --model models\baseline.joblib `
  --saved-predictions-dir results\hmm\student_test `
  --output-dir results\evaluation\student_test `
  --map-dir results\maps\student_test
```

Both approaches receive the same label-free P1 candidate rows and P2 scores.
Classifier-only independently selects the highest score per observation,
breaking ties by original candidate row order. HMM uses the existing Viterbi
decoder. Reference `.route` files are read only after both predictions and
aligned to GPS observations using P1's `align_reference()` for evaluation.

The evaluation writes per-trip and aggregate CSVs under
`results/evaluation/student_test/`. Arc accuracy is correct directed arc IDs
divided by all GPS observations, so a missing prediction counts as incorrect.
Conditional accuracy uses only nonmissing predictions. Candidate reference
coverage is the fraction of all observations whose P1-aligned reference arc is
among that observation's candidates. Direct connectivity counts adjacent
predicted observations where arcs are the same or immediate directed
successors. Route continuity uses the configured maximum hop count to test for
a directed path between the predicted arcs; disconnected-transition rate is
its complement. Both connectivity rates use only adjacent original observation
pairs with predictions at both ends. Missing observations are never bridged.
An undefined zero-denominator rate is written as `NaN`.

### Student-test results

The measured student-test run includes 10 trips and 19,160 observations.
Classifier-only achieved 16,452 correct arcs (85.866%); HMM achieved 16,934
(88.382%), an increase of 482 correct observations or 2.516 percentage points.
Both produced zero missing predictions. P1 candidate reference coverage was
99.645% (68 of 19,160 observations had no reference-matching candidate).
Direct connectivity was 92.762% for classifier-only and 97.645% for HMM.
Continuity within the configured five-hop bound was 97.493% and 99.896%,
respectively, with 480 versus 20 disconnected transitions.

These are results on the provided student-test development references, not
evaluator data or an independent ground-truth source. The comparison measures
agreement with the route-to-observation alignment produced by P1; it should
not be interpreted as a guarantee of geographic route accuracy. In particular,
zero missing predictions alone is not evidence of correctness.

The run reported exact row-for-row parity between newly decoded HMM outputs
and all ten existing files in `results/hmm/student_test/`. Those PR #8
prediction files are read-only inputs and are not overwritten. New outputs
include `per_trip_metrics.csv`, `aggregate_metrics.csv`, `comparison.csv`, and
`profile.csv`. Folium HTML maps are stored separately under
`results/maps/student_test/`.

Maps contain togglable GPS, reference `.route`, classifier-only, and HMM
layers. Route geometry is assembled from directed `.arcs` endpoints and
original WGS84 `.nodes` coordinates. Since the dataset stores arc endpoints
rather than detailed polylines, the map draws endpoint-to-endpoint segments;
it does not claim higher-resolution road shape. Missing predictions and
disconnected arcs remain visually separated.

## Performance investigation

The profile isolates candidate generation/features, P2 scoring, ArcGraph
construction, classifier selection, Viterbi decoding, transition calls,
shortest-path work, evaluation, and map generation in `profile.csv`.
For trip `00000056` (3,958 observations, 31,664 candidates), the current
profile measured candidate generation at 0.472 s, P2 scoring at 0.046 s,
ArcGraph construction at 0.265 s, and HMM decoding at 8.403 s. The decoder
made 253,248 transition calls and 226,886 hop-distance calls; memoization
resulted in 35,221 underlying shortest-path searches and 38,348 cached pair
entries. HMM output matched the saved PR #8 result exactly.

An earlier unoptimized profile of this same trip measured 143.04 s in the
decoder, with 131.05 s in repeated hop-distance queries. `ArcGraph` now caches
directed hop queries (including unreachable pairs), and hop-mode queries honor
the configured `max_hops` bound. Focused decoder tests passed and the saved
prediction parity check remained exact after the optimization. Runtime values
are machine-dependent; the CSV profile is the source for the specific run.

## P4 handoff

`reconstruct_trip(trip_dir, predictor, model)` remains the trip-level
integration entry point. It returns one directed arc ID or `None` for each
original GPS observation index. P4 should consume this full-track sequence;
reference routes and candidate labels are not part of inference. The current
HTTP contract remains documented in `docs/interfaces.md`.
