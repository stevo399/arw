# Measuring storm motion by matching reflectivity patterns (2026-09-13)

Follows `2026-09-13-motion-prediction-evaluation.md`, where the best estimate from storm centres was
the median of the last three clean steps. Owner decision: test pattern matching before building
the motion redesign.

## Method

**Scripts:**
- `scripts/investigations/pattern_motion_eval.py` (pass 1)
- `scripts/investigations/pattern_motion_eval_v2.py` (pass 2)
- `scripts/investigations/pattern_motion_summary.py`, which writes the two summaries
  `2026-09-13-pattern-motion-evaluation-pass1.json` and `-pass2.json`

**Pattern velocity.** For every tracked storm at every scan:
- its reflectivity is resampled to a 0.5 km grid on the ground, using each scan's own azimuths so
  beam order does not matter;
- it is matched against the previous scan by normalized cross-correlation within a 150 km/h search
  radius, with sub-pixel peak refinement;
- the displacement divided by the time between scans is the velocity.

A storm first detected in this scan still gets one, because the previous scan's field exists.

**Scoring.** Every cached back-to-back window (25 windows, 213 scans) runs through the production
tracker. Each storm observed at scans k-1 and k is predicted at k using only information up to k-1.
Two measures are used:
- **Centre error:** distance from predicted to observed centre.
- **Outline IoU:** overlap of the k-1 outline, moved by the prediction, with the k outline. This
  measure is not dominated by centre jitter.

Comparisons are head to head on identical targets. A **clean** target is one whose step was not a
merge, split or reacquisition.

**Matcher checks.** Synthetic storms moved a known distance were recovered within 67 m, including a
storm straddling due north in a sweep starting at 137 degrees and a three-storm scene.

## Pass 1: unguarded matching of the storm alone

| Clean targets | Stationary (median / p90 km, IoU) | Pattern, last velocity |
|---|---|---|
| All (5,497) | 2.12 / 7.31, 0.359 | 2.17 / **14.89**, 0.358 |
| New storms (1,140) | 2.21 / 8.01, 0.246 | **8.54 / 19.17**, 0.140 |

**What went wrong:**
- Bad matches (centre error over 8 km) had a median implied speed of 129 km/h, near the search
  limit. The matcher had locked onto another echo, mostly in dense scenes (652 of the 1,021 were
  KJAX).
- A storm that has just formed is often not in the previous scan at all.

**What worked:** where the matched speed was under 40 km/h, matching beat stationary (1.06 against
1.48 km under 20 km/h; 1.18 against 2.30 km at 20-40 km/h).

## Pass 2: guards

- **Edge rejection:** a correlation peak on the border of the search window is not a match.
- **Context template:** the template holds every echo within 10 km of the storm, not only the
  storm, so a lone blob cannot match any similar blob.
- **Neighbour consistency (`ctx_guarded`):** a velocity is accepted only when it is within 25 km/h
  of the vector median of other storms' velocities within 60 km, whenever at least three such
  neighbours exist.

| Clean targets, head to head | Baseline (median / p90 km, IoU) | Guarded pattern |
|---|---|---|
| New storm (1 prior): neighbours' guarded velocity vs stationary (1,101) | 2.22 / 8.17, 0.243 | **1.36** / 8.64, **0.327** |
| New storm: its own guarded velocity vs stationary (929) | 2.36 / 8.43, 0.238 | 1.51 / 9.83, 0.316 |
| 2 priors: own last guarded velocity vs stationary (1,223) | 2.10 / 7.30, 0.354 | **1.18 / 6.90, 0.449** |
| 3 priors: own last guarded velocity vs stationary (844) | 2.16 / 8.20, 0.367 | **1.17** / 8.27, **0.463** |
| 4+ priors: median of last 3 guarded vs clean_steps_3 (1,777) | 1.13 / 5.68, 0.505 | **1.01 / 5.27, 0.537** |
| 4+ priors: last guarded vs clean_steps_3 (1,753) | 1.11 / 5.66, 0.506 | **1.01 / 5.35, 0.538** |
| Structural targets: own last guarded vs stationary (1,038) | 2.86 / 9.54, 0.354 | **2.39** / 9.72, **0.409** |

**Coverage:** a guarded estimate (own or neighbours') exists for 86% of clean targets; centre steps
exist for 29%.

**Noise:** a storm's single guarded velocity changes between consecutive scans by a median of
3.8 km/h (p75 7.9, p90 15.3); a median of three changes by 1.3 km/h. Today's "nearly stationary"
threshold of 2 km/h is inside that noise.

## Findings

1. **Guarded pattern matching is the most accurate motion measurement tested**, and it is available
   for three times as many storms as centre steps. With two positions it beats assuming no motion
   in both median and tail, which no centre-based estimate did.
2. **Each guard matters.**
   - Without the context template: new-storm median error 7.59 km.
   - With the context template but no neighbour check: new-storm p90 12.47 km.
   - With both: 1.51 km median, 9.83 km p90.
3. **Nearby storms' motion is the best estimate for a new storm.** Its median is much better than
   assuming no motion, with a slightly worse tail (8.64 against 8.17 km p90). This supports the
   owner's decision, and speech should say the motion is nearby storms'.
4. **Merges and splits do not corrupt pattern motion** the way they corrupt centre tracks, because
   the match follows the reflectivity pattern.

## Consequence for the motion redesign

Measure motion by guarded pattern matching:
- **Established storms:** the median of the storm's last three guarded velocities.
- **Storms with fewer:** their last guarded velocity.
- **New storms, or storms with no accepted velocity:** nearby storms' guarded velocity, spoken as
  such.
- **Otherwise:** motion not yet known.

The "nearly stationary" threshold must come from the measured noise above. Verify by rerunning this
evaluation against the production tracker's reported motion.
