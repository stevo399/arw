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
