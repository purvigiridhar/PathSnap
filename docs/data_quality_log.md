# Data Quality Log — PathSnap EDA
**Team:** PathSnap | **Dataset:** UE24CS352A Hackathon GPS map-matching dataset
**Scripts used:** `eda/01_inventory_eda.py` and `pathsnap.build_dataset` on `data/train` and `data/student_test` (evaluator data never touched)

---

## 1. Dataset Inventory

| Partition | Trips | GPS pts (min / mean / max) | Nodes (min / mean / max) | Arcs (min / mean / max) | Route arcs (min / mean / max) |
|---|---|---|---|---|---|
| train | 80 | 361 / 2,605 / 19,313 | 299 / 24,169 / 155,003 | 649 / 48,461 / 307,392 | 59 / 392 / 2,266 |
| student_test | 10 | 262 / 1,916 / 3,958 | 1,770 / 9,069 / 26,635 | 3,577 / 18,504 / 56,027 | 95 / 348 / 1,323 |

**Total: 90 trips** (80 train + 10 student_test). Evaluator partition not accessed during training/EDA.

---

## 2. Findings

| Date | Finding | Partition | Decision / Assumption |
|---|---|---|---|
| 2026-11-18 | No missing, non-finite, or out-of-range GPS rows in 208,417 training and 19,160 student-test observations. Training had 7,286 timestamp gaps over one second and student-test had 264. All within-trip timestamp steps were positive. | Train and student-test | Preserve track row order and original timestamps for downstream gap detection. |
| 2026-11-18 | EPSG:32632 fails for training trip `00000057` and the dataset spans multiple UTM zones. | Train | Use a trip-centred metric CRS for batch and API builds. Record it in `coord_crs`. |
| 2026-11-18 | A reference route gives ordered arcs without observation timestamps. The distance-only monotone alignment can be ambiguous at intersections, parallel roads, and repeated arcs. Reference errors or GPS offsets can also produce incorrect labels. | Train and student-test | Allow observations to repeat or skip forward through the reference route positions. Treat the result as derived labels and record misses without inserting reference arcs into the candidate list. |
| 2026-11-18 | Training trips `00000033` and `00000054` had candidate miss rates of 66.32% and 67.66%. Their median aligned reference distances among missed observations were 133.3 m and 161.7 m. | Train | Keep both trips in the baseline and report them as outliers. |
| 2026-11-18 | Zero duplicate rows in any trip's `.track` file; all lon/lat values in valid ranges. | Both | No deduplication or coordinate sanitisation needed. |
| 2026-11-18 | No isolated nodes in any trip's road network. | Both | Network graph is connected per trip. |
| 2026-11-18 | 97% of time steps are exactly 1 second apart (nominal 1 Hz GPS sampling). | Both | Time intervals are reliable. |
| 2026-11-18 | 4 train trips have GPS gaps exceeding 1,000 s (see table below). | Train | Flag to P3 — HMM must handle large gaps; do not interpolate blindly. |
| 2026-11-18 | 11 trips have reference route breaks (arc i does not chain to arc i+1). | Train (10) + student_test (1) | Known label limitation, not a pipeline error. Documented in report. |

---

## 3. Anomalous Trips — Large Time Gaps (train only)

| Trip ID | GPS pts | Max gap (s) | Gaps > 10 s | Action |
|---|---|---|---|---|
| 00000021 | 15,783 | 2,161 | 2 | Flagged to P3 for HMM gap handling |
| 00000069 | 2,067 | 2,658 | 2 | Flagged to P3 for HMM gap handling |
| 00000078 | 2,909 | 52,810 | 25 | Flagged to P3 for HMM gap handling |
| 00000090 | 536 | 153,064 | 6 | Flagged to P3 for HMM gap handling |

---

## 4. Reference Route Quality

11 trips contain at least one route break (arc _i_.v ≠ arc _i+1_.u):
- **Train:** 00000003, 00000013, 00000021, 00000032, 00000043, 00000055, 00000069, 00000078, 00000090, 00000098
- **Student test:** 00000009

---

## 5. Assumptions and Decisions

- Evaluator partition (`data/evaluator_test`) was never accessed during EDA.
- Only `train` and `student_test` partitions were used for all analysis.
- Road network is **per-trip local** — coordinate projection is performed per trip using trip-centred metric CRS.
- The 4 anomalous high-gap trips are retained in training with a gap flag passed to P3.
- Reference route breaks are treated as label noise, not pipeline errors.
