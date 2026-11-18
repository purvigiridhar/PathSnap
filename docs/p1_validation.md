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

## Preprocessing and candidate selection for report section 2

Read longitude before latitude, preserve the original row-based IDs and GPS
timestamps, and project GPS observations and road nodes into the same
trip-centred CRS. Row counts for training trip `00000002` were 7,243 track
observations, 7,936 nodes, 16,139 arcs, and 262 reference arcs. All observed
arc endpoints and reference arc IDs were within the zero-based bounds of
their target tables in the training partition.

The STRtree indexes segment bounding boxes. The default query begins at 50 m
and doubles the search radius until it can return the exact eight nearest
segments. It then sorts by point-to-segment distance and breaks ties by arc ID.
Across training observations, the nearest segment distances had a median of
5.0 m and a 95th percentile of 83.5 m. This motivated checking fixed cutoffs
against the expanding search.

| Training shortlist cap | 25 m cutoff | 50 m cutoff | 100 m cutoff | Expanding search |
|---|---:|---:|---:|---:|
| 2 | 19.47% | 18.32% | 17.82% | 17.74% |
| 4 | 15.09% | 13.71% | 13.16% | 13.08% |
| 8 | 13.53% | 11.92% | 11.13% | 9.07% |

These are miss rates against the aligned training reference. The smaller
shortlists and fixed cutoffs were evaluated by filtering the eight-nearest
build. Eight with an expanding search gave the best coverage among these tested
settings and bounds the decoder to eight states per observation. The 9.07%
training miss rate places a 90.93% ceiling on top-1 agreement with the aligned
reference for a decoder choosing only among these candidates.

See [data quality and alignment limitations](data_quality_log.md) and
[feature definitions](features.md).

## Heading feature comparison

For a development check, fit the same histogram gradient boosting classifier
twice with 40 iterations and seed 42, using the same 250,000 randomly sampled
candidate rows from the 80 training trips. The two runs used geometry alone or
geometry plus the heading difference and validity flag. On the 10 student-test
trips, choosing the highest-scoring arc per observation gave **50.23%** and
**78.54%** top-1 agreement with the aligned reference, respectively.

Among student-test observations whose reference arc was shortlisted, the
fraction with a negative candidate identical across all model features dropped
from **26.30%** to **0.91%**. The candidate lists and miss counts stayed the same.
The label-free batch and single-trip inference builds produced identical feature
values on the checked trip.

## Expanded feature comparison

Repeating the same controlled comparison with the four heading-baseline
features and then all 19 features gave **78.54%** and **85.79%** student-test
top-1 agreement, respectively. Both fits used the same 250,000 sampled training
rows, 40 histogram gradient boosting iterations, and seed 42. The rebuilt
training and student-test tables retained their previous candidate miss counts.
The checked single-trip inference output matched its label-free batch table.
