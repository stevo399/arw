# Level III companion products as an ingest precondition

Date: 2026-09-15.

**Status (2026-09-15):** proposal, not started. Extends the owner decisions in
`2026-09-14-detection-truth-design.md`, which already commit to cross-checking the classifier
against NWS Level III hydrometeor classification and using Level III mesocyclone/TVS as an
operational cross-check. This spec asks whether that cross-check moves from offline corpus
tooling onto the live serving path, and what it costs.

Existing machinery: `src/ingest.py` already has `LEVEL3_BUCKET`, `list_level3_keys`,
`level3_key_time` and `download_level3`. `src/validation/nmd.py` already parses product 141
tabular text. Nothing in the serving path imports either — today they are reached only from
`scripts/evaluate_rotation_corpus_v2.py`, `scripts/build_rotation_corpus.py` and tests.

## The owner constraint that shapes everything

> No situation where an Audiom map of a scan loads before the product it is based on is fully
> ready, or where ARW emits GeoJSON before all analysis that would alter it is complete.

The strong form of this is a versioning property, not just a freshness property. WeatherKitten
treats `scanTimestamp` as an immutable version key — see the comments in its `radar_snapshot`
builder — and Audiom's dynamic loader caches a source by that identity. So:

**Invariant: `scanTimestamp` → GeoJSON content must be a function.** If a volume is served with
merged objects at T and split objects at T+1 under the same timestamp, Audiom may pin either, and
which one a user gets is a race. That is worse than being slow.

This rules out the obvious design (fetch Level III concurrently, join if it arrives, serve without
it if not). Level III must be a **precondition of ingestion**, not an enrichment of output.

## Measured facts (2026-09-15, KIWA)

Publication timing, `unidata-nexrad-level3`, 29 NST objects across 2026-09-16 00:00–02:17Z:

```
volume cadence:                    mean 272s  (min 247, max 291)
L3 publish lag after volume START: mean 277s  (min 250, max 317)
```

Level III lands within ~5s of the volume completing; worst observed case trailed completion by
~45s. Since `nexradaws` only ever lists completed volumes, ARW is already 4.5–9 minutes behind
volume start by construction. Adding ≤45s of worst-case staleness is inside the noise.

Concrete case: ARW served `scanTimestamp 2026-09-16T01:34:20Z`. That volume's NST published at
01:39:18Z. WeatherKitten received the GeoJSON around 01:47Z. **Level III had been available for
eight minutes.** Zero wait would have been required.

Sizes for one volume (hour 01Z):

| Product | Mean bytes | Form | Published for KIWA |
|---|---|---|---|
| NST (58, storm tracking) | 3 628 | tabular | yes |
| NMD (141, mesocyclone) | 150 | tabular | yes |
| EET (enhanced echo tops) | 3 431 | raster | yes |
| NVL (VIL) | 1 419 | raster | yes |
| DVL (digital VIL) | 8 406 | raster | yes |
| HHC (hybrid hydrometeor class) | 6 140 | raster | yes |
| NCR (composite reflectivity) | 22 215 | raster | yes |
| NHI (hail index) | — | — | **no** |
| NSS (storm structure) | — | — | **no** |
| NTV (TVS) | — | — | **no** |
| NET (legacy echo tops) | — | — | **no** |
| RCM (radar coded message) | — | — | **no** |

Checked across 2026-09-10 → 2026-09-16. NHI, NSS, NET and RCM appear in the bucket's historical
key space but have nothing in that window; treat them as unavailable. NTV is plausibly
event-driven rather than retired — absence over a Phoenix monsoon week is not evidence either way,
and it should not be depended on.

All seven published products together are ~45 KB, roughly 0.5% of one Level II volume.

## Design

**1. A volume is ineligible for analysis until all of its inputs exist.** Level II *and* its
declared Level III companions. `latest` means *latest fully-analyzed volume*. If a request
arrives during the window where Level II has landed and a companion has not, serve the previous
volume — complete, final, correctly labelled. WeatherKitten already displays the actual scan time,
so nothing is concealed. No request ever blocks on a companion fetch.

This makes the invariant hold by construction: a volume is published exactly once, fully analyzed.
No window exists in which Audiom could cache a partial source.

**2. Deadline, then freeze.** A hard precondition with no escape hatch means one missing companion
strands a volume forever and ARW sits on an ageing scan. The invariant is not "always has Level
III", it is *once published, never changes*. So: wait a bounded interval past volume completion;
when it expires, publish without the missing companion and **never revisit that volume**. Record
which companions were present in the scan record so the decision is auditable and so tracking can
see the boundary.

**3. Level III enters at detection/segmentation, upstream of tracking.** STI cell positions used as
segmentation seeds change object count and identity. Applying them after `tracking/association.py`
has run would perturb track IDs retroactively. This is not a serialization-time enrichment.

**4. Network calls stay in the Ingest Manager.** Per CLAUDE.md, only ingest touches the network.
`download_level3` is already there. Parsers move out of `src/validation/` into the parser layer;
`validation/nmd.py` keeps working against the same parser.

**5. Analysis versioning across the retained-scan ring.** WeatherKitten retains 5 scans and
tracking reads across them. A mid-flight change in segmentation policy leaves a mix of seeded and
unseeded scans in one track history. Version the analysis and do not let tracks span the boundary.

## What this changes about the map

Measured today: ARW's storm GeoJSON carries 40 properties per feature. Rich on intensity, QC,
geometry and beam physics. The gaps, in priority order:

**A. No motion reaches the map at all.** `summary.py` has `MotionVector`, `_get_motion_for_object`,
heading labels and confidence-downgrade logic. `map_layer.py` emits no motion property. A user
walking the map is told "58 miles WNW" — bearing from the radar, static — and nothing about where
the storm is going. **This is an existing capability that is not plumbed, not a Level III gap**,
and it is the single largest improvement available. Level III's contribution here is corroboration
and ready-made 15/30/45/60-minute forecast positions, not the motion itself.

**B. Object count disagrees with NWS where cells are adjacent.** On the 01:34:20Z volume, NEXRAD
SCIT resolved W2 (az 257/56NM) and H3 (az 250/57NM) as two cells ~8 mi apart; ARW emitted one
63 mi² object covering both. A user walking that shape hears one storm where NWS reports two.
SCIT centroids as segmentation seeds address exactly this.

**C. No vertical extent.** `class_fractions` and `peak_dbz` describe a column's composition and
peak but not its depth. EET (3.4 KB) distinguishes a 50 kft convective tower from shallow
precipitation with the same surface reflectivity. For a walkable audio map those should not sound
alike.

**D. Rotation confidence is self-referential.** `rotation_evidence_level` has its own ladder
(`corroborated`, `vertically_confirmed`, `persistent`). NMD gives NWS's answer on the same volume
for 150 bytes. Agreement justifies speaking the evidence level; disagreement justifies withholding
it — which is precisely the unresolved item in the 2026-09-14 spec, where persistent evidence
passed the corpus but failed the live check.

C and D are close and their order is a judgement call: depth applies to every feature in every
scan, rotation is rarer but higher-stakes. Rank them by which failure you would rather not have.

**E. Hail has no per-cell product.** NHI is not published. Hail must come from HHC raster sampling,
or from ARW's own dual-pol classifier, which `class_fractions` shows already exists. HHC is a
cross-check on that classifier — matching owner decision 1 of 2026-09-14 — not a new capability,
and costs a raster sample rather than a regex.

**F. No shared vocabulary with other sources.** STI cell IDs persist across volumes. Adopting them
lets a user correlate what ARW says with any other NWS-derived source.

## Risks

- **Staleness on the newest scan**, bounded at ~45s by measurement, inside existing 4.5–9 min lag.
- **Companion outage** strands a volume — mitigated by deadline-and-freeze.
- **Segmentation change perturbs tracking history** — mitigated by analysis versioning.
- **Seeds are points, footprints are areas.** STI centroids cannot improve shape geometry and must
  never be allowed to. `shape_truth.py` measures whether a shape lies when walked on; splitting a
  region by seed proximity must preserve total covered area exactly, or that metric regresses.
- **Level III is itself an algorithm, not ground truth.** SCIT has its own false alarms. Using it
  to gate ARW's claims imports its biases. Validate against it before depending on it.
- **STI reports motion direction FROM, ARW reports direction TOWARD.** Measured on the KIWA
  `02:45:06Z` volume: STI printed `MOVEMENT 172/18`, while ARW reported the same cells moving `N`
  at 18-19 mph. That looks like a 180-degree error and is not one. STI's own forecast positions
  (az/range at +15/30/45/60 min) resolve to a true track of **350.6 degrees at 17.5 kts** — so
  172 is the direction the storm comes from, and ARW's heading agrees to within ~10 degrees and
  ~1 mph. **Any corroboration code must add 180 to the printed STI movement before comparing**, and
  should assert against the forecast positions rather than the printed degrees. A naive comparison
  would report every cell as reversed and invite someone to "fix" correct behaviour.

## Decisions needed

1. Which companions become preconditions: NST only, NST+NMD, or the raster set too?
2. Deadline length past volume completion before publishing without a companion.
3. Does `latest` serving the previous volume during the gap need surfacing in WeatherKitten's copy,
   or is the existing scan-time display sufficient?
4. Do STI seeds change emitted object identity, or only annotate (`nws_cell_ids: [...]`) without
   splitting? Annotation is far cheaper and preserves `shape_truth` by construction.
5. Is motion plumbed to the map independently of this work? It is the biggest win and has no
   Level III dependency.

## Test plan

- Unit: STI tabular parse against the fixtures already captured for 2026-09-16 01:25–01:43Z.
- Unit: deadline-and-freeze — a volume published without a companion is never re-published.
- Property: for any volume, repeated GeoJSON requests return byte-identical content.
- Integration: `latest` during a synthetic companion gap serves the previous volume, never partial.
- Regression: `shape_truth` false-positive/negative rates do not move when seeds split an object.
- Corpus: extend the rotation corpus pattern to detection/segmentation against SCIT.
