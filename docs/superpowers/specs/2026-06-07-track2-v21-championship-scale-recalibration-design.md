# Track2 v21 Championship-Scale Recalibration Design

## Goal

Build a Track2-only v21 candidate for the second-to-last Codabench submission. The target is no longer conservative stability; it is a high-upside classification recalibration large enough to plausibly move the official score toward the first-place band.

v20 changed only 28 labels, all `content->calm`. Even if every change were correct, the overall score upside is too small. v21 must create a larger, structured candidate with at least 80 accepted label changes while preserving the strongest available description text.

## Current Evidence

- Our visible public score is around overall `0.84`, classification `0.72`, description `0.95`.
- First place is around overall `0.89`, classification `0.78`, description near `1.00`.
- The largest visible gap is emotion accuracy: our row `0.57`, first place `0.80`.
- `781601` proved broad same-quadrant reshuffling is unsafe.
- `v20` proved directional `content->calm` evidence exists, but 28 changes are too small to matter.
- `v15_desc_expand300` remains the best text anchor; v21 should not rewrite text unless a later description-only pass explicitly does that.

## Strategy

v21 is a classification-scale candidate built from existing v17 evidence. It accepts changes in three tiers:

1. **Reference tier**
   - exact or near public duplicate evidence with strong support,
   - may include same-quadrant and limited cross-quadrant changes.

2. **Consensus tier**
   - at least three model families plus public-style support,
   - score threshold lower than v20 because the point is scale,
   - protects rare current classes and caps transition families.

3. **Expansion tier**
   - at least two model families with strong confidence or four total sources,
   - allowed only after reference/consensus rows are selected,
   - capped by transition family, quadrant, and projected distribution.

The candidate ladder should produce:

- `v21_precision80`: about 80 changes.
- `v21_champion120`: about 120 changes.
- `v21_lastshot160`: up to 160 changes for final-shot use only.
- `v21_calmshift90`: up to 90 high-evidence `content->calm` changes, added after first v21 results showed the mixed profile was too noisy and too cross-quadrant heavy.

## Risk Controls

- No formal submission overwrite.
- No automatic Codabench upload.
- Preserve v15 description fields.
- Repair valence/arousal from emotion after every accepted change.
- Keep all 12 emotions present.
- Cap top emotion share at `0.58`.
- Cap cross-quadrant changes:
  - precision80: 12
  - champion120: 24
  - lastshot160: 36
- Cap any exact transition:
  - precision80: 40
  - champion120: 55
  - lastshot160: 70
  - calmshift90: 90, but only for `content->calm`
- Do not remove an emotion class below 4 samples.

## Success Criteria

- Generate four side-path candidate JSON/ZIP files.
- Recommended candidate must have at least 80 accepted changes, no label consistency issues, no missing emotions, and preserved description anchor.
- Validator passes for all v21 candidates.
- Track2 unit tests pass.
- The final report must state this is a high-variance competition probe, not a reconstructed hidden scorer.
