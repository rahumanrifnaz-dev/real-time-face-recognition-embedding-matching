# Reproducible Experiments

## Before collecting data

Obtain informed consent, use only local participants, and decide whether images may remain private or may be shown. Keep the camera, background, and participant consistent when changing one factor. Do not reuse enrollment frames as test frames. For each capture, add a manifest row with relative image path, exact expected identity (or `Unknown`), and one condition label. Repeat each condition several times if practical and report `n`; these are small exploratory experiments, not statistically general claims.

The stored-image runtime benchmark is complete and traceable to
`outputs/metrics/performance_metrics.csv`. Recognition and robustness observations
still require separate held-out images. Run
`python evaluate.py --manifest data/test/manifest.csv` only after collecting them.

## Current collection status

The enrollment database is not evaluation evidence. `data/test/` currently contains
no labeled test images, so every experiment remains incomplete.

| Experiment | Status |
|---|---|
| Stored-image runtime benchmark | Completed (20 permitted enrollment images; speed only) |
| Threshold too strict | Not yet collected |
| Threshold too lenient | Not yet collected |
| Normal lighting | Not yet collected |
| Dim lighting | Not yet collected |
| Bright lighting | Not yet collected |
| Near distance | Not yet collected |
| Far distance | Not yet collected |
| Frontal pose | Not yet collected |
| Left/right pose | Not yet collected |
| Blurred frame | Not yet collected |
| Enrollment count | Not yet collected |
| Frame resizing | Not yet collected |
| Processing cadence | Not yet collected |
| Unknown rejection | Not yet collected |

| # | Experiment | What to change | Hypothesis | Metric to observe | Expected learning outcome |
|---:|---|---|---|---|---|
| 1 | Threshold too strict | Evaluate lower distance thresholds, e.g. 0.40–0.50 | More enrolled samples become `Unknown` | Known recognition and false rejection rates | See the cost of requiring very close matches |
| 2 | Threshold too lenient | Evaluate higher thresholds, e.g. 0.65–0.80 | More unknown samples are accepted | Unknown rejection and false acceptance rates | See why nearest does not mean correct |
| 3 | Normal lighting | Frontal face in ordinary indoor light | Baseline should be strongest | Correct count, distance, latency | Establish a comparison baseline |
| 4 | Dim lighting | Reduce room illumination without changing camera/pose | Shadows/noise may increase distance or miss detection | Detection count, correct count, distance | Separate detection and matching effects |
| 5 | Bright lighting | Use strong room lighting without deliberate overexposure | Highlights may reduce useful detail | Correct count and distance | Observe illumination sensitivity |
| 6 | Near distance | Move nearer while keeping pose frontal | Larger face boxes may retain detail | Box width/height, distance, correctness | Relate face resolution to matching |
| 7 | Far distance | Move farther away | Smaller face boxes may increase misses/distance | Box size, detection, distance | Find the practical webcam range |
| 8 | Frontal pose | Look toward the camera | Frontal enrollment should match well | Correct count and distance | Establish pose baseline |
| 9 | Left/right pose | Use moderate left and right turns separately | Distances may increase asymmetrically | Correct count by `left`/`right` | Understand pose variation |
| 10 | Blurred frame | Introduce controlled motion blur in test captures | Blur may hurt detection/encoding | Blur score, detection, distance | Understand lost-detail failures |
| 11 | Enrollment count | Rebuild with 1, 3, 5, then 10 good references | More varied references may improve recall until diminishing returns | Known recognition, DB size/build time | Evaluate multiple-reference value |
| 12 | Frame resizing | Run scale 1.0, 0.75, and 0.5 on the same scene | Smaller frames improve speed but may hurt small faces | Latency/FPS and recognition | Measure accuracy-speed trade-off |
| 13 | Processing cadence | Compare every frame with every second frame | Skipping work improves display FPS but labels update less often | FPS, processing latency, visual lag | Distinguish throughput from responsiveness |
| 14 | Unknown rejection | Test consenting, non-enrolled people | Appropriate thresholds return `Unknown` | Closest identity, distance, final decision | Demonstrate open-set recognition |

Optional extensions include moderate `up`/`down` pose, glasses when naturally available, and multi-face scenes containing only consenting participants.

## Collection checklist

1. Enroll each known participant with 8–15 clear, distinct images.
2. Build embeddings, then capture separate test images for normal, dim, bright, near, medium, far, frontal, left, right, up, and down conditions.
3. Ask one or more consenting non-enrolled participants to provide test images. Put them under `data/test/Unknown/`; never enroll them.
4. Record qualitative labels honestly. Use numeric metres only if measured.
5. Run evaluation at the default threshold, inspect failures, then analyze alternative thresholds on the same saved distances.
6. Run the benchmark on the same machine while noting hardware and background load.
7. Keep conclusions proportional to sample count and never fabricate missing runs.

## CSV outputs

`evaluation_results.csv` records image, expected and predicted identity, condition, closest identity, Euclidean distance, threshold, latency, bounding-box size, faces detected, and correctness. `evaluation_summary.json` records aggregate rates. `performance_metrics.csv` records stored-image average/median latency and approximate FPS. The notebook reads these files and gracefully stops with a data-collection message when they do not exist.
