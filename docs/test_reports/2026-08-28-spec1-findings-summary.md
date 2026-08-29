# Spec 1 findings summary

**Date:** 2026-08-28
**Branch:** `spec1-data-layer`
**Suite:** 324 passing, 4 strict expected-failures

This is the one-page account. The design is in
`docs/superpowers/specs/2026-08-22-radar-data-layer-correctness-design.md` (read
its two amendments), the benchmark comparison in
`docs/test_reports/2026-08-28-spec1-rebaseline.md`.

## What prompted this

The question was whether ARW interprets radar as well as the established
applications — RadarScope, RadarOmega, GRLevel3 — and whether its GeoJSON
reflects true storm shape and extent.

The answer that emerged: **the gap was never in the algorithms. It was in
verification.** Every serious defect found here was invisible to a green test
suite of 195 tests and a project record full of "validated".

## The defects, and why each was invisible

| Defect | Consequence | Why no test caught it |
|---|---|---|
| Centroid interpolation across the 0°/360° azimuth seam | A storm due north reported **due south**, ~164 km out | Nothing pinned a centroid to a known geographic position |
| `scipy.ndimage.label` does not wrap ray 0 to ray N−1 | A storm straddling due north became **two objects**, two wrong centroids, two wrong shapes | No test placed a storm on the seam |
| Velocity sweeps selected by array index | Two of three carried **zero** velocity; multi-sweep rotation confirmation could never engage | Fixtures built sweeps that all carried velocity — a VCP structure that does not exist |
| `_merge_cross_sweep_rotations` counted couplets, not sweeps | `sweep_count` reached **198** from 3 sweeps, published as a confidence signal | Nothing asserted the invariant |
| Velocity paired with reflectivity by array index | Two different antenna revolutions, up to 21° apart — **37 km** of lateral error at 100 km, on the heaviest-weighted clutter discriminator | Every fixture used one sweep where the grids trivially matched |
| Beam height used the true earth radius | Site selection rejected everything beyond 306 km instead of 345 km | `devspec/06` specified the wrong radius, so code and spec agreed with each other |

## What the reference applications do differently

They do not extract objects. RadarScope, RadarOmega and GRLevel3 render every
gate faithfully and overlay interpretation the NWS already computed. They carry
no "should this echo be deleted?" decision, so they cannot get it wrong.

ARW must summarise — it cannot speak 1.3 million gates — but **summarising does
not require deleting.** That realisation drove the architecture change below.

## The finding that changed the design

Proof 4 measured the classifier against the confirmed Newcastle–Moore EF5
tornado of 2013-05-20. Within 10 km of the SPC damage path, on 4,596 gates
peaking at 54.5 dBZ:

| Class | Gates | Share |
|---|---|---|
| precipitation | 2,599 | 56.5% |
| biological | 662 | 14.4% |
| ground clutter | 597 | 13.0% |
| debris | 525 | 11.4% |
| hail | 213 | 4.6% |

**1,259 gates — 27.4% — were condemned as clutter or birds.** All 1,259 survived
only because protection rescued them.

### The coupling, and why sequencing mattered

Two defects were cancelling each other:

- the classifier wrongly condemns real tornado debris
- an over-sensitive shear detector (121–132 spurious signatures per clear-air
  scan) over-protects everything, masking it

Fixing either alone would have converted a hidden defect into an active, silent,
life-safety failure — in an application whose users cannot see that anything went
wrong.

**The resolution was not to sequence the fixes. It was to remove the
deletion.** Quality control now classifies and annotates; it deletes nothing.
The classifier's errors became a quality problem rather than a safety one, and
rotation sensitivity can now be corrected independently.

## What remains wrong

- **The classifier condemns EF5 debris** (27.4%). Strict xfail.
- **Polarimetric discrimination does not work.** With velocity excluded,
  biological-vs-clutter separation is 1.05×. RhoHV, ZDR, reflectivity and
  texture separate birds from buildings essentially not at all; only velocity
  does real work. Strict xfail.
- **All 18 membership parameters are unvalidated placeholders.** Sourcing six
  published Park et al. 2009 values made classification measurably *worse*
  (velocity separation 2.70× → 1.76×), because those breakpoints were fitted
  for a different algorithm — 1-D radial texture, velocity as a hard threshold,
  Z-dependent ZDR breakpoints, a different class set. Conforming to the HCA
  means porting its **structure**, not its constants. Preserved in
  `PARK2009_REFERENCE`.
- **Shapes are still convex hulls.** A crescent of rain around a city renders as
  a solid disc covering it. Spec 2.

## What a future developer must know

1. **A test that cannot fail is not evidence.** Nine test gaps were found here,
   every one originating in a plan written by the same author as the code.
   Each fix was required to be *proven to fail* against the broken version
   before acceptance — restoring the old logic from git and observing the
   failure, not reasoning about it. That discipline is what found these.
2. **Verification by hand protects nothing.** Measurements taken once in a shell
   and pasted into a report do not survive. Anything that matters gets pinned in
   a test.
3. **The two strict xfails are the gate.** They start passing when the classifier
   is calibrated. Until then, no hazard call from this system should be trusted,
   and `PROGRESS.md` should not claim otherwise.
4. **Provenance is what makes conformance checkable.** Every membership parameter
   declares where it came from. That is the only reason it was possible to
   discover that "cited" and "correct" had diverged.
