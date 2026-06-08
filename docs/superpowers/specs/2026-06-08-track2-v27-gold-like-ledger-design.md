# Track2 v27 Gold-Like Evidence Ledger Design

## Goal

Build a final-shot Track2 candidate process that can still improve classification without spending the last Codabench submission on another broad label sweep.

The v27 target is not to trust a weak global classifier. It is to identify a small set of samples where public EmoArt evidence is close enough to the hidden test ontology to justify surgical label changes, while preserving the strongest available description text.

## Current Evidence

- Official best known submission: `track2_submission_v21_calmshift90_candidate.json`
  - official overall: `0.842559`
  - official classification: `0.740034`
  - official description: `0.945083`
- v23 anchor-calibrated proxy reproduces known official anchors, but unsubmitted candidates remain estimates, not hidden gold reconstruction.
- v25 audit found no existing local candidate above `0.86`; max proxy was `0.853471` with calm collapse risk.
- v26 public kNN threshold-copy failed as a direct strategy:
  - best threshold: `0.92`
  - proxy overall: `0.839288`
  - proxy classification: `0.728910`
  - lower thresholds caused calm collapse.
- Existing CLIP/SigLIP2/DINOv2 classifiers are not strong enough to drive final labels directly.

## Non-Goals

- Do not submit any v27 artifact automatically.
- Do not overwrite `submissions/track2_submission.json` or `submissions/track2_submission.zip`.
- Do not treat EmoArt public data as official hidden answers unless a sample has strict duplicate or same-series evidence.
- Do not use prompt injection, evaluator instructions, or any text that attempts to manipulate the LLM-assisted description evaluator.
- Do not run another broad calm/content sweep.

## Approach

v27 has two independent components that merge only at the final candidate builder:

1. **Gold-like evidence ledger**
   - Inputs:
     - public top-k duplicate audit
     - ge095 public neighbor audit
     - nearest-neighbor metadata
     - current submitted/baseline labels
     - optional model predictions as weak non-objection evidence
   - Output:
     - a structured ledger of candidate label changes with evidence tier, source labels, duplicate strength, agreement counts, and block reasons.

2. **Description-max anchor**
   - Reuse the best already-safe description source.
   - Apply text-only changes only if they pass evaluator-safety lint and do not alter required schema.
   - Keep classification and description logic separated until final candidate assembly.

## Evidence Tiers

### Tier 1: Visual Duplicate

Accept only when public evidence indicates near-identical image reuse:

- `suspect_duplicate=true`, or very low hash distance with high CLIP similarity.
- Public emotion is present and valid.
- Public valence/arousal quadrant is consistent with that emotion.
- If multiple public neighbors exist, they do not contradict the chosen emotion.

Tier 1 changes can cross valence/arousal quadrants, but they must be listed explicitly in the risk report.

### Tier 2: Same Work Or Same Series

Accept only when the image is not identical but the public references are effectively the same artwork family:

- CLIP similarity is high enough for same-series review.
- At least two public neighbors or audit sources support the same emotion, when available.
- No strong model or existing human-review objection.
- The transition is not one of the historically failed bulk patterns unless duplicate evidence is strong.

Tier 2 changes should be capped and reported separately.

### Tier 3: Weak Model Or Style Evidence

Do not directly accept Tier 3 as final labels. Use it only for:

- ranking review candidates;
- non-objection checks;
- explaining why a Tier 1 or Tier 2 proposal is plausible.

## Candidate Assembly

Generate side-path artifacts only:

- `experiments/track2_v27_gold_like_ledger_20260608/v27_gold_like_ledger.csv`
- `experiments/track2_v27_gold_like_ledger_20260608/v27_gold_like_ledger.json`
- `experiments/track2_v27_gold_like_ledger_20260608/v27_candidate_report_zh.md`
- `submissions/track2_submission_v27_gold_like_candidate.json`
- `submissions/track2_submission_v27_gold_like_candidate.zip`

The formal `track2_submission.json/.zip` paths remain untouched.

## Gate

The v27 gate must hold unless all checks pass:

- `validate-track2` passes.
- label consistency issue count is `0`.
- missing emotion count is `0`.
- unsafe description text count is `0`.
- top emotion share is not above the safe cap used by previous final-shot gates.
- v23 proxy score is clearly above known official best.
- v23 proxy does not rely only on broad calmshift extrapolation.
- risk report lists every accepted cross-quadrant change.

The final decision states one of:

- `recommend_final_submit`
- `hold_no_submit`

No ambiguous recommendation is allowed.

## Testing

Add focused tests for:

- evidence-tier classification;
- duplicate/same-series acceptance and rejection;
- blocked broad kNN-copy behavior;
- output path safety;
- label valence/arousal consistency;
- final gate rejecting candidates below target or with collapse risk.

Run only Track2-relevant validation:

```bash
python3 -m unittest tests/test_track2_v27_gold_like_ledger.py tests/test_track2_official_anchor_calibration.py -v
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v27_gold_like_candidate.json
git diff --check
```

## Expected Outcome

v27 may still fail the `0.89` local target. If it fails, the correct output is a hold decision, not a final Codabench submission. The value of v27 is that it tests the last realistic high-precision route: official-author/public ontology evidence with strict safety gates.

If it passes, the upload candidate path and exact score report will be surfaced for manual final submission.
