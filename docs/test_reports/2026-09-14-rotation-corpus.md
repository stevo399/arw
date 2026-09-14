# Rotation validation corpus v2 (2026-09-14)

**Where things are:**
- Plan: `docs/superpowers/plans/2026-09-14-rotation-detection-truth.md`.
- Manifest: `docs/validation/rotation-signal-corpus-v2.json`.
- Builder: `scripts/build_rotation_corpus.py`.

**Built from** SPC reports on 2026-07-12, 07-14, 09-07, 09-11 and 09-12 (SPC days): 9 tornado, 78 hail and
359 wind reports.

## Cases

Detections = NWS mesocyclone detections (NMD) counted across the case's products.

| Case | Label | Radar (km) | Volumes | NMD products | Products with detections | Detections | Report |
|---|---|---|---|---|---|---|---|
| tornado-20260712-1945-0 | tornado | KAMA (16) | 7 | 9 | 1 | 1 | 3 NNE Lake Tanglewood TX 19:45Z |
| tornado-20260715-1020-1 | tornado | KEWX (52) | 4 | 6 | 0 | 0 | 2 ENE San Antonio TX 10:20Z |
| tornado-20260907-1939-2 | tornado | KLIX (67) | **0** | 0 | 0 | 0 | 2 W Hammond LA 19:39Z |
| tornado-20260907-2335-3 | tornado | KMBX (80) | 5 | 7 | 0 | 0 | 6 S Strawberry Lake ND 23:35Z |
| tornado-20260912-1940-7 | tornado | KJAX (106) | 6 | 7 | 0 | 0 | 5 WNW Palm Coast FL 19:40Z |
| tornado-20260912-2046-8 | tornado | KAKQ (20) | 5 | 7 | 0 | 0 | 1 ESE Spring Grove VA 20:46Z |
| hail-20260715-0431-0 | hail | KTYX (87) | 3 | 6 | 4 | 5 | Hermon NY 04:31Z, 2.75 in |
| hail-20260715-0018-1 | hail | KTFX (125) | 3 | 0 | 0 | 0 | 1 N Hobson MT 00:18Z, 2.50 in |
| hail-20260912-0259-2 | hail | KEWX (34) | 2 | 4 | 0 | 0 | 3 SE Converse TX 02:59Z, 2.50 in |
| hail-20260715-0418-3 | hail | KTYX (90) | 4 | 6 | 1 | 1 | De Peyster NY 04:18Z, 2.00 in |
| hail-20260715-0428-4 | hail | KTYX (87) | 3 | 5 | 3 | 3 | De Kalb NY 04:28Z, 2.00 in |
| hail-20260913-0010-5 | hail | KDDC (72) | 2 | 4 | 4 | 6 | 6 N Coldwater KS 00:10Z, 2.00 in |
| wind-20260715-0155-0 | wind | KBLX (113) | 3 | 5 | 0 | 0 | 12 SSE Forestgrove MT 01:55Z, 86 kt |
| wind-20260713-0100-2 | wind | KBLX (117) | 3 | 6 | 0 | 0 | 9 SSE Grass Range MT 01:00Z, 77 kt |
| wind-20260912-2305-3 | wind | KDDC (80) | 3 | 5 | 1 | 1 | 8 ESE Acres KS 23:05Z, 75 kt |
| wind-20260715-0230-4 | wind | KBLX (66) | 2 | 4 | 0 | 0 | 7 N Lavina MT 02:30Z, 74 kt |
| clear-air-KIWA-20260712 | clear_air | KIWA | 6 | 0 | 0 | 0 | 0.01-0.02% of gates >= 35 dBZ |
| quiet-KAMA-20260712 | clear_air | KAMA | 1 | 5 | -- | -- | 0.69% of gates >= 35 dBZ |
| quiet-KEWX-20260714 | null_storm | KEWX | 1 | 5 | -- | -- | 17.97% of gates >= 35 dBZ |
| quiet-KMBX-20260907 | null_storm | KMBX | 1 | 5 | -- | -- | 4.95% of gates >= 35 dBZ |
| quiet-KJAX-20260912 | null_storm | KJAX | 1 | 5 | -- | -- | 3.04% of gates >= 35 dBZ |
| quiet-KAKQ-20260912 | clear_air | KAKQ | 1 | 5 | -- | -- | 0.05% of gates >= 35 dBZ |

## Findings while building

1. **Few tornadoes can be scored.**
   - 3 of 9 reports lie beyond 150 km of every radar: Spencer NE, 172 km; Alberta MN, 184 km;
     Artichoke MN, 182 km.
   - 1 has no data (below).
   - That leaves 5 scorable tornado cases.
   - The speak rule needs at least 2 hits; with 5 events that is a low bar, so the per-case table
     matters more than the pooled rate.
2. **NWS mesocyclone detection saw almost none of these tornadoes.** Only 1 of 36 tornado-case NMD
   products (KAMA) contains a detection. These are weak tornadoes, including a tropical-environment
   tornado near Palm Coast and a Virginia tornado, the kind radar algorithms rarely detect. A
   detector that also misses them is not necessarily wrong. Missing them does not justify a speech
   claim either way.
3. **KLIX is retired, and ARW's site database still uses it.** KLIX has no Level II data on
   2026-09-07, 2026-01-15 or 2025-06-01. Its replacement, KHDC (Hammond, LA), has data on all three
   dates. `src/sites.py` lists KLIX and not KHDC, so ARW routes New Orleans-area users to a radar
   with no data. Filed as its own issue; the Hammond tornado cannot be scored until the site list
   is corrected.
4. **"No reports nearby" did not mean clear air.** Three noon volumes chosen for having no reports
   within 200 km and 3 hours held widespread rain (3-18% of gates at 35 dBZ or more). The builder
   now applies the spec's clear-air definition (under 1% at 35 dBZ or more). Those three volumes
   became `null_storm` cases, scored only by null-storm probes.

## Sweep and selection (Task 7)

**Files:**
- Sweep: `docs/test_reports/2026-09-14-rotation-sweep.json`, from
  `scripts/evaluate_rotation_corpus_v2.py OUT --sweep`.
- Baseline only: `docs/test_reports/2026-09-14-rotation-baseline.json`.
- Selector: `scripts/select_rotation_config.py`.

**What was run:** 68 volumes; the baseline plus 24 configurations. Every swept configuration uses both
physics corrections: azimuthal pairs only, and ground-overlap merging.

**Columns:**
- Hits r1/r2/r3 are tornado events hit at evidence rank 1 (any), 2 (vertically confirmed or better) and
  3 (persistent).
- "Null r1" counts storm centroids more than 50 km from any report or NMD detection that carry a
  rank-1 assessment.
- "Clear r1" counts rank-1 assessments in clear-air volumes.
- "Speak" lists the ranks that pass the pre-registered speak rule.
- "Sec" is detector plus association time across all 68 volumes.

| # | shear | side | diam | fold | hits r1 | r2 | r3 | null r1 | NMD | clear r1 | speak | sec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 15 | 0 | none | no | 4/5 | 3 | 2 | 910/5144 (0.177) | 6 | 16 | 3 | 403 |
| 0 | 15 | 0 | none | no | 3/5 | 3 | 3 | 840/5144 (0.163) | 6 | 44 | 3 | 316 |
| **1** | **15** | **0** | **none** | **yes** | **3/5** | **3** | **3** | **827/5144 (0.161)** | **14** | **42** | **3** | **315** |
| 2 | 15 | 0 | 10 | no | 3/5 | 3 | 3 | 841/5144 (0.163) | 10 | 44 | 3 | 317 |
| 3 | 15 | 0 | 10 | yes | 3/5 | 3 | 3 | 828/5144 (0.161) | 14 | 42 | 3 | 318 |
| 4 | 15 | 10 | none | no | 2/5 | 2 | 2 | 278/5144 (0.054) | 5 | 10 | 3 | 111 |
| 5 | 15 | 10 | none | yes | 2/5 | 2 | 2 | 257/5144 (0.050) | 12 | 6 | 3 | 101 |
| 6 | 15 | 10 | 10 | no | 2/5 | 2 | 2 | 279/5144 (0.054) | 9 | 10 | 3 | 111 |
| 7 | 15 | 10 | 10 | yes | 2/5 | 2 | 2 | 257/5144 (0.050) | 12 | 6 | 3 | 101 |
| 8 | 20 | 0 | none | no | 2/5 | 2 | 2 | 606/5144 (0.118) | 4 | 26 | 3 | 201 |
| 9 | 20 | 0 | none | yes | 2/5 | 2 | 2 | 590/5144 (0.115) | 12 | 26 | 3 | 200 |
| 10 | 20 | 0 | 10 | no | 2/5 | 2 | 2 | 607/5144 (0.118) | 8 | 26 | 3 | 202 |
| 11 | 20 | 0 | 10 | yes | 2/5 | 2 | 2 | 590/5144 (0.115) | 12 | 26 | 3 | 200 |
| 12 | 20 | 10 | none | no | 2/5 | 2 | 2 | 278/5144 (0.054) | 5 | 10 | 3 | 110 |
| 13 | 20 | 10 | none | yes | 2/5 | 2 | 2 | 257/5144 (0.050) | 12 | 6 | 3 | 100 |
| 14 | 20 | 10 | 10 | no | 2/5 | 2 | 2 | 279/5144 (0.054) | 9 | 10 | 3 | 111 |
| 15 | 20 | 10 | 10 | yes | 2/5 | 2 | 2 | 257/5144 (0.050) | 12 | 6 | 3 | 101 |
| 16 | 25 | 0 | none | no | 2/5 | 2 | 2 | 430/5144 (0.084) | 4 | 15 | 3 | 145 |
| 17 | 25 | 0 | none | yes | 2/5 | 2 | 2 | 413/5144 (0.080) | 12 | 15 | 3 | 143 |
| 18 | 25 | 0 | 10 | no | 2/5 | 2 | 2 | 431/5144 (0.084) | 8 | 15 | 3 | 146 |
| 19 | 25 | 0 | 10 | yes | 2/5 | 2 | 2 | 413/5144 (0.080) | 12 | 15 | 3 | 143 |
| 20 | 25 | 10 | none | no | 2/5 | 2 | 2 | 246/5144 (0.048) | 5 | 10 | 3 | 104 |
| 21 | 25 | 10 | none | yes | 2/5 | 2 | 2 | 226/5144 (0.044) | 12 | 6 | 3 | 93 |
| 22 | 25 | 10 | 10 | no | 2/5 | 2 | 2 | 247/5144 (0.048) | 9 | 10 | 3 | 103 |
| 23 | 25 | 10 | 10 | yes | 2/5 | 2 | 2 | 226/5144 (0.044) | 12 | 6 | 3 | 93 |

**Selected by the pre-registered rule: configuration 1.**
- Rule: most rank-1 tornado hits with a null fraction no higher than the baseline's (0.177). Ties
  go to the lower null fraction.
- Configuration 1 settings: shear 15 m/s, no side minimum, no diameter limit, fold rejection,
  azimuthal pairs only, ground-overlap merge.
- These are now the `RotationDetectorConfig` defaults. `RotationDetectorConfig.baseline()` keeps the
  earlier rules.

### Selected configuration by evidence rank

| Rank | Tornado hits | Null false alarms | Clear-air false alarms | NMD matched | Speakable |
|---|---|---|---|---|---|
| 1 (any) | 3/5 | 827/5144 (0.161) | 42 | 14 | no |
| 2 (vertically confirmed) | 3/5 | 373/5144 (0.073) | 14 | 10 | no |
| 3 (persistent) | 3/5 | 240/5144 (0.047) | **0** | 7 | **yes** |

Baseline, for comparison: rank 3 hit 2/5 at 219/5144 (0.043). Rank 3 was its only speakable rank too.

### Trade-offs

1. **The corrections lose one weak tornado at rank 1: San Antonio, KEWX, 10:20Z.**
   - The baseline reached it only through pairs along the beam or 10 km merging. Those rules are
     physically wrong: opposite signs along a ray are convergence or divergence, not rotation.
   - At rank 3 the selected configuration hits 3 events against the baseline's 2.
2. **Clear-air rank-1 assessments rose from 16 to 42.**
   - quiet-KAMA accounts for 37 and KIWA for 4. quiet-KAMA's noon volume holds scattered
     convection (0.69% of gates at 35 dBZ or more).
   - Ranks 1 and 2 fail the speak rule for this reason, and rank 3 has none.
3. **Agreement with NWS mesocyclone detections improved from 6 to 14 at rank 1.**
   - Most of the gain comes from fold rejection: configuration 0, without it, stays at 6.
   - The cause was not traced. Likely, but unverified: a fold candidate merged with a real one
     and moved the merged centroid away from the circulation.
4. **Configurations with a 10 m/s side minimum (4-7, 12-15, 20-23) cut false alarms by about 70%.**
   They also lose Spring Grove (KAKQ) at rank 1, and hit one fewer event at ranks 2 and 3. The pre-registered rule ranks hits
   first, so they were not selected. They remain the obvious candidates if a larger corpus shows the
   extra hit is chance.
5. **The corpus is small.**
   - 5 scorable tornado events, all weak.
   - A 1-event difference is within chance.
   - Nothing here shows the detector is good at rank 1 or 2. It shows only that persistent evidence
     beats report-free storms by the pre-registered margin.

### Strong-rotation check: fold rejection on Moore 2013 (EF5, KTLX)

Fold rejection treats any gate pair jumping by at least 0.8 × 2 × Nyquist, with both sides above half
Nyquist, as an aliasing fold. On raw velocity a very strong real couplet looks the same. The corpus
holds only weak tornadoes, so the sweep could not measure that cost. Two cached Moore volumes were
checked on the live path (raw velocity, Nyquist 26.1 m/s):

**Volume KTLX20130520_195527:**
- SPC report: Newcastle, 19:56Z.
- Baseline and physics-only: 52 m/s, both sides saturated at ±26, 3.8-4.0 km from the report, on 3
  tilts.
- With fold rejection: that pair is dropped. The circulation is still found 4.8 km from the report
  at 41 m/s, on 3 tilts (`vertically_confirmed`).

**Volume KTLX20130520_200356:** it has no tornado position report to score against, so it is not
used as evidence.

**Conclusion:**
- Fold rejection did not erase this EF5 circulation, so the selection stands.
- On raw velocity no shear above twice Nyquist can be measured, with or without the rule.
- Detection of the strongest couplets therefore depends on the neighbouring, less-extreme pairs
  surviving, as they did here.
- One case is not a guarantee. More strong-tornado volumes belong in the next corpus.

Synthetic test consequence: `test_detect_rotation_multi_sweep_increases_sweep_count` used ±22 m/s
beside each other (Nyquist 26.2), which falls inside the fold band. Its peaks were lowered to ±18-20
m/s because the test concerns sweep counting, not folds.

## Live pipeline (Task 10)

### Cost

`analyze_velocity` uses the selected configuration.
- **Densest cached volumes, before speedups:** KJAX 7.8 s, KEMX 16.2 s, KTLX 10.1 s. Region merging
  (full-grid masks), azimuth alignment, association and shear-component extraction dominated.
- **After the speedups (commit `65e3f15`):** 1.5-3.1 s on KJAX, KEMX, two KTLX volumes, KAMA and KIWA.
  Output is byte-identical under both the selected and baseline configurations (12 of 12 compared).
- **Added to a full live scan:** 30.4 -> 34.5 s on KJAX 2026-09-14 08:09Z (129 storms); 5.2 -> 8.9 s
  on KAMA 08:17Z (1,036 candidates). This includes the quality-advisory refresh that now receives
  assessments.

### Wiring

- **Live scans:** they now run velocity analysis.
- **Persistence:** when a scan is tracked, `RadarHistory.add_live_scan` promotes rotation seen again
  on the same tracked storm to `persistent`. The promoted level goes into the scan, its storms and the
  tracker's history entry.
- **Compact scans:** they now keep velocity regions. Without this, `/velocity` returned none for
  retained scans.

**Known limit, not yet fixed:** the quality advisory is refreshed before tracking, so it sees
assessments before promotion.

### Real-data checks

**Proof** (`tests/e2e/test_proof_rotation_corpus.py`):
- The recorded selection equals the detector defaults.
- The six KIWA clear-air volumes, run through the live path, produce no spoken-level assessment and
  no rotation in any summary or map description.
- The Spring Grove tornado (KAKQ) gets persistent evidence within 10 km of its report.

**Other cases through the live path:**
- KAMA 2026-07-12: persistent evidence 8.2 km from the tornado report.
- KMBX 2026-09-07: 8.0 km.
- Spring Grove: 0.6 km.

These are the same three hits as the corpus evaluation.

**Tracking benchmark** (`docs/test_reports/2026-09-14-benchmark-after-rotation.json`): identical to
`2026-09-13-benchmark-after-measured-motion.json` except 29 summary texts. Each difference removes
spoken unconfirmed couplets. Every KSOX and KEYX window summary had been saying "strong unconfirmed
velocity couplet".

### Live check: persistent evidence is not ready to speak

Setup: server on port 8011 with a fresh history directory, KJAX, KAMA and KIWA, 2026-09-14
08:09-08:24Z.

**First scans:** unconfirmed and vertically confirmed assessments only; nothing spoken.

**Second scans:** spoken with no severe reports anywhere after 04Z on the SPC day:
- **KJAX 08:17Z:** "Weak persistent rotation evidence 13 miles S of the radar" and "Moderate
  persistent rotation evidence 171 miles SSW".
- **KAMA 08:24Z:** "Moderate persistent rotation evidence 14 miles E" and "26 miles ENE".

**The KJAX persistent assessments are noise:**
- 1-2 gate couplets, 0.2-0.5 km across;
- about ±11 m/s;
- mostly on one tilt;
- inside a 34 dBZ echo 18-23 km from the radar.

**Cause:** persistence accepts any couplet on the same tracked storm in the previous scan. The
allowed separation is the two couplets' mean diameter plus 120 km/h × gap, about 16 km for an 8-minute
gap. On a large storm with many noise couplets, that is met almost by chance. The corpus measured
4.7% of report-free storms carrying persistent evidence. The speak rule allowed that, and with 5 weak
tornado events it could not show how often such sentences occur on a busy radar.

**Owner decision (2026-09-14):**
- Speak no rotation at any level: `SPOKEN_ROTATION_EVIDENCE` is empty.
- Assessments stay in the data, labelled with their evidence level.
- The speech wording and gating remain tested with a patched level.
- The detector work stops here for now.

**What a next attempt needs:**
- A persistence test that requires the same circulation, not the same storm: a displacement bound
  from the storm's measured motion, plus a minimum size and tilt count.
- A larger corpus that includes tonight's false alarms and more tornado days, including strong ones.
- Pre-registration before it is scored.
