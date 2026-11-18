# Candidate features

| Feature | Calculation | Purpose |
|---|---|---|
| `feat_dist_m` | Shortest distance from the projected observation to the closed line segment between the projected arc endpoints | Measures proximity to the candidate road |
| `feat_arc_length_m` | Distance between the projected arc endpoints | Gives the model the scale of the candidate segment |

Use the `feat_*` columns as model inputs. The identifiers, `t`, `x`, `y`, and `coord_crs` support grouping and decoding. Both distances are in the trip's projected metres.

P1 uses deterministic geometry features with no fitted scaler or encoder. P2 can fit and save any model-specific preprocessing with the model.
