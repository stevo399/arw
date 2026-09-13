# Which motion estimate predicts where a storm goes? (2026-09-13)

**Question:** now that ARW keeps several scans per radar, can storm motion be measured from the storms
themselves, and how many scans does that take?

**Method:** `scripts/investigations/motion_prediction_eval.py`, run on code with the geographic
placement fixes (`fa3aaae`). Raw numbers are in `2026-09-13-motion-prediction-evaluation.json`.
- Every cached back-to-back window (25 windows, 213 scans, 9 radars) runs through the production
  tracker.
- For every tracked position, each candidate predicts it from the track's earlier positions.
- The error is the distance from the predicted to the observed centre (7,488 predictions; scans
  about 4 to 10 minutes apart).
- A target is **structural** when the step into it was a merge, split or reacquisition, because its
  centre then reflects a reshaped storm.

**Candidates:**

| Candidate | How it predicts |
|---|---|
| stationary | no motion |
| scene_field | today's blended scene and local estimate |
| neighbours | median recent velocity of other tracks within 60 km that have 3 or more positions |
| lifetime | fit over all earlier positions |
| last_N | fit over the last N positions |
| last_M_min | fit over the last M minutes |
| clean_steps_K | median velocity of the last K steps that did not end in a merge, split or reacquisition |

## Results (median / p90 error, km)

| Earlier positions | stationary | scene_field | neighbours | best track-based |
|---|---|---|---|---|
| 1 (new storm) | 2.42 / 9.61 | -- | **1.85 / 8.57** | -- |
| 2 | 2.20 / 7.57 | 1.94 / **7.43** | **1.83** / 8.10 | last_2: 1.98 / 10.57 |
| 3 | 2.37 / 9.04 | 2.06 / **8.83** | 2.02 / 9.92 | clean_steps_2: **1.73** / 9.44 |
| 4 or more | 2.18 / 6.89 | 1.88 / 6.65 | 1.69 / 7.58 | clean_steps_3: **1.30 / 6.57** |
| 4 or more, clean target | 2.07 / 6.02 | 1.75 / 5.82 | 1.50 / 6.97 | clean_steps_3: **1.13 / 5.68** |
| 4 or more, structural target | 2.94 / 10.22 | 2.74 / 10.53 | 2.71 / 10.89 | clean_steps_3: **2.38 / 9.06** |

## Findings

1. **Centres jitter by about 2 km between scans.** Even with no motion assumed, the median error
   over one scan interval is 2.2 km, so a 5-minute displacement is the same size as the noise. A
   one-step velocity carries roughly 25 km/h of noise.
2. **Two positions are not enough to measure motion.** A velocity from the last two positions has
   a worse 90th-percentile error than assuming no motion (10.6 against 7.6 km with 2 earlier
   positions; 13.3 against 9.0 with 3). This is where the reported outliers come from.
3. **Merges, splits and reacquisitions corrupt centre tracks.** Excluding those steps and taking
   the median of the last three clean step velocities is the best estimator: 1.13 km median error
   on clean targets, against 1.54 for the whole-track fit and 2.07 for no motion. It also has the
   lowest tail.
4. **Nearby storms' motion is the best estimate for a new storm** (1.85 against 2.42 km median),
   supporting the owner's decision for one-position storms. It is also competitive at 2 positions.
5. **Longer is not automatically better.** The whole-track fit (issue #1) beats short fits in the
   tail but not in the median; the clean-step median beats both.
6. **The scene estimate's modest tail is not accuracy.** It is usually a 1 to 2 km radial shift
   (compact-scan-history report, section 7c), which behaves almost like assuming no motion.

## Not yet measured

- Whether a pattern-matching displacement per storm (correlating each storm's reflectivity between
  aligned scans on the ground) beats centre steps. It would not be corrupted by reshaping.
  Centre-based error cannot judge it fairly, because the targets themselves carry about 2 km of
  jitter.
- Neighbour motion built from neighbours' clean-step medians rather than their last-3 fits.
