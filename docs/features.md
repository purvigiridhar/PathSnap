# Candidate features

| Feature | Calculation | Purpose |
|---|---|---|
| `feat_dist_m` | Shortest distance from the projected observation to the closed line segment between the projected arc endpoints | Measures proximity to the candidate road |
| `feat_arc_length_m` | Distance between the projected arc endpoints | Gives the model the scale of the candidate segment |
| `feat_heading_diff_deg` | Unsigned angle from 0° to 180° between the GPS displacement and the arc's from-node to to-node direction | Distinguishes opposite directions on the same road |
| `feat_heading_valid` | 1 when a displacement of at least 3 m and a nonzero arc length support a heading comparison, otherwise 0 | Identifies rows where the heading difference defaults to 90° |

Use the `feat_*` columns as model inputs. The identifiers, `t`, `x`, `y`, and `coord_crs` support grouping and decoding. Both distances are in the trip's projected metres.

| Feature | Calculation and purpose |
|---|---|
| `feat_projection_ratio` | Position along the directed segment, clipped to [0, 1]. Captures progress toward the end node. |
| `feat_dist_to_start_m`, `feat_dist_to_end_m`, `feat_dist_to_midpoint_m` | Distances to three landmarks on the segment. Describe where the GPS point lies relative to the segment. |
| `feat_gps_speed_mps` | Distance from the preceding GPS point divided by elapsed seconds. Defaults to zero at trip start or across gaps over five seconds or nonpositive intervals. |
| `feat_heading_cos_sim` | Cosine of the heading difference, or zero when heading is unavailable. Measures directional agreement. |
| `feat_spatial_likelihood` | Gaussian-shaped proximity score with a 12 m scale. |
| `feat_rank_by_dist`, `feat_is_closest` | Zero-based distance rank and a flag for rank zero. Ties follow candidate row order. |
| `feat_speed_length_ratio` | Speed divided by arc length, in inverse seconds. Defaults to zero for zero-length arcs. |
| `feat_dist_above_best_m` | Distance minus the closest candidate distance. Measures the candidate's margin against its alternatives. |
| `feat_start_in_degree`, `feat_end_out_degree` | Counts of directed arcs entering the start node and leaving the end node. Summarise local connectivity. |
| `feat_prev_dist_m`, `feat_next_dist_m` | Distances from adjacent GPS readings to the same candidate segment. Use the current distance when a neighbor is unavailable or the time gap is outside (0, 5] seconds. |

P1 uses deterministic geometry features with no fitted scaler or encoder. P2 can fit and save any model-specific preprocessing with the model.

GPS heading uses the sum of the incoming and outgoing GPS displacements when
both are usable. Each step must cover at least 3 m in more than zero and at most
five seconds. Endpoints and observations beside a GPS gap can use the single
usable step. The same calculation runs in offline builds and whole-trip inference.
