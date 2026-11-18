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
