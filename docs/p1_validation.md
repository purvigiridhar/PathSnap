# P1 candidate build validation

The README commands use the eight nearest directed segments per GPS observation
and a trip-centred azimuthal equidistant projection. The builds write one
Parquet per trip and a per-trip `summary.csv` in each output directory. The
[per-trip results](p1_validation_by_trip.csv) combine the four run summaries.

| Output under `data/processed/` | Trips completed | GPS observations | Candidate rows | Observations without candidates | Candidate misses |
|---|---:|---:|---:|---:|---:|
| `offline/sample` | 5 of 5 | 11,488 | 91,904 | 0 | 699 (6.08%) |
| `offline/train` | 80 of 80 | 208,417 | 1,667,336 | 0 | 18,905 (9.07%) |
| `offline/student_test` | 10 of 10 | 19,160 | 153,280 | 0 | 68 (0.35%) |
| `prediction/student_test` | 10 of 10 | 19,160 | 153,280 | 0 | — |

A candidate miss means the aligned reference arc was absent from the shortlist.
Training trips `00000033` and `00000054` had miss rates of 66.32% and 67.66%,
respectively. The prediction build skips reference routes, so it has no miss
metric.

The candidate build uses the same CRS for GPS positions, road nodes and metric
features. For P3, project nodes with the table's `coord_crs`. Reverse directions
are stored as separate arc rows when both exist. In training trip `00000002`,
arc `0` is node `0` to node `1492`, and arc `3181` is the reverse.

The partition rules are described in Dataset Exploration, section 7, and Models
and Training Guidelines, section 4. The 80 training trips are for fitting and
cross-validation. The 10 student-test trips are for development evaluation and
model comparison. The reserved evaluator trips and references are for final
scoring.
