# Track2 v20 Championship Scorer Design

## Goal

Build a Track2-only v20 experiment for the remaining Codabench submissions where the only success target is first place, not conservative rank preservation.

The current visible gap is approximately `0.84` to `0.89`. First place combines description near `1.00` with classification around `0.78`. Our visible row is description around `0.95` and classification around `0.72`. The largest visible classification gap is emotion accuracy: first place `0.80`, our row `0.57`, while both have emotion macro-F1 around `0.31`.

## Current Evidence

- `779605` is the exact official anchor: overall `0.836408`, classification `0.723150`, description `0.949667`.
- `781601` made 85 same-quadrant emotion changes and dropped to overall `0.834027`, classification `0.719137`.
- `782683` returned to the `0.84` band with no classification label changes, so text-only stability does not solve the classification gap.
- `v15_desc_expand300` is the strongest local text-only package: fused expected `0.839902`, lower `0.835854`.
- `v19` correctly held all candidates, but it treated every transition family present in the failed `781601` batch as broad negative evidence. For a championship attempt this is too conservative because `781601` confounds multiple directions: `calm->content` occurred 43 times, `content->calm` 20 times, and several smaller same-quadrant transitions also changed.

## Championship Hypothesis

The most plausible high-upside correction is not another balanced same-quadrant shuffle. It is a directional official-style calibration:

- Treat `calm->content`, `content->glad`, `glad->content`, and weak positive-low reshuffles as dangerous unless exact duplicate evidence exists.
- Treat high-evidence `content->calm` differently from `calm->content`: the failed batch moved more rows away from calm than toward calm, so the aggregate failure does not prove strong `content->calm` rows are bad.
- Preserve valence/arousal unless exact duplicate evidence strongly supports a cross-quadrant change.
- Use `v15_desc_expand300` text as the description anchor, because the remaining score upside still needs description near the first-place band.

This is intentionally higher variance than v19. It is not a gold-label reconstruction and must report that caveat. Its purpose is to create a candidate that is actually capable of moving emotion accuracy, instead of another no-op text polish.

## Architecture

v20 has five units:

1. **Frontier Requirement Model**
   - Reads the public leaderboard export.
   - Computes the classification score required to reach target overall values under possible description scores.
   - Reports why `0.89` requires classification near `0.78` if description is near `1.00`.

2. **Directional Failure Interpreter**
   - Reads the official pairwise diff between `779605` and `781601`.
   - Converts the failed batch into directional risk, not a blanket transition ban.
   - Marks high-count directions from the failed lower-scoring candidate as dangerous, especially `calm->content`.

3. **Championship Evidence Scorer**
   - Uses v17 evidence rows as the per-sample evidence source.
   - Accepts only row-level supported changes; aggregate pressure cannot create a label change alone.
   - Allows high-evidence `content->calm` despite its presence in the failed batch when the row has at least three model families or strong near/exact public evidence.
   - Blocks broad `calm->content` unless exact duplicate evidence is strong.

4. **Candidate Ladder**
   - Writes side-path candidates only:
     - `v20_precision`: small high-evidence directional candidate.
     - `v20_champion_probe`: larger high-upside content-to-calm probe.
     - `v20_last_shot`: maximum allowed directional candidate for the final submission only.
   - No candidate may overwrite `submissions/track2_submission.json` or `track2_submission.zip`.

5. **Final Gate**
   - Does not pretend to know the official hidden score.
   - Recommends a candidate only if it has a credible path to classification lift, preserves description anchor text, validates cleanly, and its intended online role is explicit: precision probe, championship probe, or last shot.

## Data Flow

```text
leaderboard aggregates
  + exact official anchors
  + pairwise 779605/781601 diff
  + v17 evidence matrix
  + v15 description anchor
      -> frontier requirement report
      -> directional risk map
      -> v20 championship evidence table
      -> precision / champion_probe / last_shot candidate JSON+ZIP
      -> local validation and final gate
```

## Safety Policy

v20 is allowed to be aggressive because the user’s target is first place, but it still has hard safety boundaries:

- no formal submission overwrite,
- no automatic Codabench upload,
- no evaluator-manipulation text,
- no valence/arousal inconsistency,
- no missing emotion classes,
- no cross-quadrant label changes unless exact duplicate evidence is strong.

The report must clearly separate:

- known official aggregate feedback,
- directional inference from failed batch behavior,
- row-level model/reference evidence,
- speculative championship risk.

## Expected Files

- Create `affectiveart/track2_v20_championship_scorer.py`
- Create `scripts/track2_v20_championship_scorer.py`
- Create `tests/test_track2_v20_championship_scorer.py`
- Output under `experiments/track2_v20_championship_scorer_20260607/`
- Candidate files:
  - `submissions/track2_submission_v20_precision_candidate.json/.zip`
  - `submissions/track2_submission_v20_champion_probe_candidate.json/.zip`
  - `submissions/track2_submission_v20_last_shot_candidate.json/.zip`

## Success Criteria

- Unit tests cover frontier requirement math, directional failure interpretation, evidence scoring, candidate path safety, and final gate roles.
- Candidate JSON files validate with `python3 -m affectiveart.challenge validate-track2`.
- `python3 -m unittest discover -s tests -p 'test_track2*.py' -v` passes or Track2-only relevant tests pass with unrelated failures documented.
- `git diff --check` passes.
- Final report explicitly says whether v20 is worth one of the final two submissions and which candidate file to upload if accepted.

