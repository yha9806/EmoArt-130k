# Track2 v19 Aggregate-Calibrated Classification Design

## Goal

Build a Track2-only v19 experiment that targets the visible emotion-accuracy gap without spending a Codabench submission on another weak same-quadrant probe.

The current public leaderboard shows our strongest visible version at about `0.84` overall, with classification about `0.72`, description about `0.95`, emotion accuracy `0.57`, and emotion macro-F1 `0.31`. First place is about `0.89` overall, with classification `0.78`, description `1.00`, emotion accuracy `0.80`, and emotion macro-F1 `0.31`. The main gap is therefore not rare-class macro-F1; it is official-style 12-way emotion accuracy.

## Current Evidence

- `779605` remains the exact official anchor: overall `0.836408`, classification `0.723150`, description `0.949667`.
- `781601` made 85 same-quadrant emotion changes and dropped to overall `0.834027`, classification `0.719137`. This proves same-quadrant emotion changes are not automatically safe.
- `782683` changed no classification labels and returned to the public `0.84` band. It did not fix emotion accuracy.
- `v15_desc_expand300` is the best local text-only candidate: local fused overall expected `0.839902`, lower `0.835854`.
- `v17` and `v18` conservative label-change candidates did not beat `v15` lower bound. Their accepted label changes were too few or too risky to explain a jump toward `0.89`.

## Approaches Considered

### A. Description-Only Final Polish

This is useful for score stability but insufficient for first place. Even a perfect description score from `0.95` to `1.00` contributes about `+0.025` overall. The remaining gap still requires classification lift.

### B. Per-Sample Evidence Gate

This is what v17/v18 mostly did: only accept changes with multi-model or duplicate evidence. It is safe, but it produces too few changes and underfits the visible problem: emotion accuracy is low across the majority region, not only in a handful of obvious duplicate cases.

### C. Aggregate-Aware Classification Calibration

This is the recommended route. It combines local per-sample evidence with leaderboard-derived aggregate constraints. The method does not know hidden labels, and it must not pretend to reconstruct official gold. Instead, it searches for candidates whose label distribution and transition profile are more consistent with the public leaderboard gap while still requiring row-level model evidence.

## Architecture

v19 has six components:

1. **Official Aggregate Target Builder**
   - Reads the public leaderboard export and our local submission distributions.
   - Derives feasible target bands rather than exact target labels.
   - Focuses on matching first-tier behavior: higher emotion accuracy, similar macro-F1, strong valence/arousal, description above `0.95`.

2. **Public Validation Confusion Calibrator**
   - Uses available public/validation prediction files to estimate per-source confusion tendencies.
   - Produces per-emotion reliability and transition reliability for SigLIP2, CLIP, DINOv2, public-style classifiers, and existing specialist outputs.
   - Treats `781601` as negative official feedback for broad `calm/content/glad/happy` reshuffling.

3. **Aggregate-Constrained Search**
   - Starts from `v15_desc_expand300`.
   - Considers only rows from the existing evidence matrix or recomputed prediction evidence.
   - Builds nested candidates by optimizing a local objective:
     - increase emotion accuracy proxy,
     - keep macro-F1 proxy from collapsing,
     - keep valence/arousal unchanged or improved,
     - move distribution away from excessive `calm/content` concentration only when supported by row-level evidence,
     - preserve description fields from v15.

4. **Candidate Ladder**
   - Writes three side-path candidates:
     - `v19_precision`: fewest changes, only high row evidence and aggregate-positive transitions.
     - `v19_balanced`: moderate changes, constrained by transition and target distribution bands.
     - `v19_probe`: one deliberately stronger aggregate probe for local analysis, not default submission.
   - All candidates must retain all 12 emotions and pass label consistency checks.

5. **Shadow Scoring Upgrade**
   - Adds a v19 classification proxy that reports separate components:
     - emotion accuracy proxy,
     - emotion macro-F1 proxy,
     - valence task proxy,
     - arousal task proxy,
     - aggregate distribution penalty,
     - historical official-feedback penalty.
   - The scorer must reproduce the ordering `779605 > 781601` and must not rank a massive same-quadrant batch above the anchor without a clear aggregate explanation.
   - The scorer must include an explicit `781601_regression_guard` section that explains why v19 is not repeating the 85-row same-quadrant failure mode.

6. **Final Gate**
   - Compares `v19_precision`, `v19_balanced`, `v19_probe`, `v15_desc_expand300`, `782683`, and `779605`.
   - Recommends submission only if a v19 candidate has:
     - validator OK,
     - label consistency issues `0`,
     - missing emotions none,
     - description lower not below v15 by more than `0.003`,
     - classification expected and classification lower above v15,
     - official-feedback risk below a fixed threshold,
     - no broad unsupported same-quadrant reshuffle.

## Data Flow

```text
official leaderboard aggregates
  + official exact anchors and pairwise failed diffs
  + local submission distributions
  + v17/v18 evidence matrix
  + model/public/reference predictions
      -> v19 aggregate target bands
      -> v19 calibrated evidence table
      -> v19 precision/balanced/probe candidates
      -> validate-track2
      -> v19 classification proxy + fused shadow scorer
      -> final gate report
```

## Safety Policy

v19 may use public EmoArt data to learn labeling style, train calibrators, and identify exact or near-exact reference evidence. It must not blindly copy labels from broad style similarity. Reports must distinguish:

- exact or near-exact same-work evidence,
- same-series or same-style evidence,
- model-only evidence,
- aggregate distribution pressure,
- official-feedback penalty.

Aggregate pressure can rank candidates but cannot by itself change a label. Every changed row still needs row-level support.

## Submission Policy

There are only two remaining Codabench submissions. v19 should not auto-submit. A candidate becomes worth one submission only if it clears a stricter local gate than v18:

- local fused lower bound above `v15_desc_expand300` lower bound,
- classification proxy expected above `0.74`,
- classification lower above the current anchor `0.723150`,
- emotion accuracy proxy improves by at least `0.05` on local validation/proxy slices, not just VA preservation,
- emotion macro-F1 proxy does not regress by more than `0.005`,
- description stays near `0.95+`,
- transition report explains why this is not another `781601` failure mode.

If no candidate clears this gate, the correct action is to hold the submission and continue offline.

## Expected Files

- Create `affectiveart/track2_v19_aggregate_calibration.py`
- Create `scripts/track2_v19_aggregate_calibration.py`
- Create `tests/test_track2_v19_aggregate_calibration.py`
- Output under `experiments/track2_v19_aggregate_calibration_20260607/`
- Candidate files:
  - `submissions/track2_submission_v19_precision_candidate.json/.zip`
  - `submissions/track2_submission_v19_balanced_candidate.json/.zip`
  - `submissions/track2_submission_v19_probe_candidate.json/.zip`

## Success Criteria

Minimum local success:

- Track2 validator OK for all candidates.
- `python3 -m unittest discover -s tests -p 'test_track2*.py' -v` passes.
- v19 scorer preserves historical ordering: `779605` above `781601`.
- v19 scorer includes a regression report for the `781601` failure mode.
- v19 aggregate calibration includes at least one ablation comparing no aggregate calibration, global aggregate calibration, and transition-family constrained calibration.
- v19 final gate clearly recommends one candidate or clearly holds all.
- No candidate overwrites formal submission files.

Competition success target:

- A submitted v19 candidate should plausibly move classification from `0.72` toward `0.75+` while keeping description around `0.95+`. A candidate that only improves local expected overall by less than `0.003` is not worth one of the last two submissions.
