# Data-quality issues, assumptions and decisions
Record every finding here with the partition used. This feeds report section 2.

| Date | Finding | Partition | Decision / assumption |
|---|---|---|---|
| 2026-11-18 | No missing, non-finite, or out-of-range GPS rows in 208,417 training and 19,160 student-test observations. Training had 7,286 timestamp gaps over one second and student-test had 264. All within-trip timestamp steps were positive. | Train and student-test | Preserve track row order and original timestamps for downstream gap detection. |
| 2026-11-18 | EPSG:32632 fails for training trip `00000057` and the dataset spans multiple UTM zones. | Train | Use a trip-centred metric CRS for batch and API builds. Record it in `coord_crs`. |
| 2026-11-18 | A reference route gives ordered arcs without observation timestamps. The distance-only monotone alignment can be ambiguous at intersections, parallel roads, and repeated arcs. Reference errors or GPS offsets can also produce incorrect labels. | Train and student-test | Allow observations to repeat or skip forward through the reference route positions. Treat the result as derived labels and record misses without inserting reference arcs into the candidate list. |
| 2026-11-18 | Training trips `00000033` and `00000054` had candidate miss rates of 66.32% and 67.66%. Their median aligned reference distances among missed observations were 133.3 m and 161.7 m. | Train | Keep both trips in the baseline and report them as outliers. |
