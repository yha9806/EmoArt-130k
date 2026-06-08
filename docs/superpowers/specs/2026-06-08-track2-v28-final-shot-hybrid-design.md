# Track2 v28 Final-Shot Hybrid Design

## Goal

Build the last-submission decision system for AffectiveArt Track2. The target is not another small safe repair. The target is to find a candidate that can plausibly reach first-place territory: local proxy overall at or above `0.89`, with classification and description both improved.

The output must be a recommendation, not an automatic upload. If no candidate clears the final gate, the system must return `hold_no_submit`.

## Current Evidence

- Best known official submission: `track2_submission_v21_calmshift90_candidate.json`
  - official overall: `0.842559`
  - official classification: `0.740034`
  - official description: `0.945083`
- Public first-place target is approximately:
  - overall: `0.89`
  - classification: about `0.78`
  - description: about `1.00`
- Required classification for `0.89` depends on description:
  - if description is `1.00`, classification must be `0.78`
  - if description is `0.95`, classification must be `0.83`
  - if description is `0.945`, classification must be `0.835`
- v27 gold-like ledger did not provide enough coverage:
  - `frontier`: overall `0.840171`, classification `0.729460`, description `0.950882`
  - `allaccepted` sensitivity: overall `0.841701`, classification `0.732520`, description `0.950882`
- v25 found a proxy candidate at `0.853471`, but it carried calm-collapse risk and remained below target.
- v26 blind public kNN copying failed: best overall `0.839288`.

## Non-Goals

- Do not submit automatically to Codabench.
- Do not overwrite `submissions/track2_submission.json` or `submissions/track2_submission.zip`.
- Do not treat the local v23 proxy as hidden-gold reconstruction.
- Do not build a final candidate only because it beats the failed second submission or a weak local baseline.
- Do not make evaluator-directed text, prompt injection, or description content that asks for a score.
- Do not run Track1 tests or modify Track1 files.

## Core Strategy

v28 is a hybrid of two necessary moves:

1. **Classification reconstruction**
   - Start from the official-best v21 label set.
   - Search beyond the submitted `content->calm` 90-change direction.
   - Explore controlled ladders that combine:
     - more `content->calm` changes;
     - rare-class floor protection;
     - calm/content/glad boundary changes only when supported by multiple evidence families;
     - very limited non-calm same-quadrant transitions;
     - zero or near-zero cross-quadrant changes unless evidence is visual-duplicate strength.
   - Reject candidates with top-emotion collapse, missing classes, or broad public-kNN behavior.

2. **Description-max merge**
   - Merge the safest high-description text source onto promising classification candidates.
   - Run unsafe-text lint on every text field.
   - Keep text changes separate from label changes in reports so score gains can be diagnosed.

## Candidate Families

Generate side-path candidates only:

- `v28_descmax_on_v21`
  - v21 official-best labels plus best safe description source.
  - Purpose: measure description-only ceiling.
- `v28_calm_ladder_120`
  - v21 plus additional high-evidence `content->calm`, capped at 120 total `content->calm` changes versus anchor.
  - Purpose: test whether v21 trend continues without collapse.
- `v28_calm_ladder_150`
  - same as above, capped at 150.
  - Purpose: aggressive but still interpretable continuation.
- `v28_balanced_frontier`
  - high-evidence label changes with rare-class protection and top-emotion cap.
  - Purpose: classification reconstruction, not pure calmshift.
- `v28_hybrid_best`
  - best-scoring profile after gates, with best safe description merge.
  - Purpose: only candidate that may be recommended for the final submission.

Every candidate must write both JSON and ZIP under explicit side-path names such as:

- `submissions/track2_submission_v28_descmax_on_v21_candidate.json`
- `submissions/track2_submission_v28_descmax_on_v21_candidate.zip`
- `submissions/track2_submission_v28_hybrid_best_candidate.json`
- `submissions/track2_submission_v28_hybrid_best_candidate.zip`

## Evidence Inputs

Use only repo-local artifacts:

- official anchors from `affectiveart.track2_official_anchor_calibration`
- v21 official-best submission JSON
- v23 scorer
- v25 candidate pool and model-signal audit
- v26 public-kNN probe output
- v27 gold-like ledger output
- v17 evidence matrix
- existing description-max or description-audited candidate JSONs

Do not depend on Downloads. Do not require network access for the core candidate sweep.

## Scoring Model

Use the current v23 scorer for comparability with existing anchor reports, but report its limitations explicitly:

- exact official anchors are reproduced by fingerprint;
- unsubmitted candidates are proxy estimates;
- estimates above the known official range are high extrapolation risk.

The v28 report must include:

- profile name;
- candidate JSON and ZIP path;
- label changes versus v21 and versus 779605 anchor;
- transition counts;
- top emotion and top-emotion share;
- missing emotions;
- description text rows changed;
- unsafe text count;
- v23 expected overall, classification, and description;
- gate decision and reasons.

## Final Gate

The final recommendation can only be `recommend_final_submit` if all hard gates pass:

- `validate-track2` passes.
- label consistency issue count is `0`.
- missing emotion count is `0`.
- unsafe description text count is `0`.
- top emotion share is at or below the configured safe cap.
- classification proxy is at least `0.78`.
- description proxy is at least `0.97`.
- overall proxy is at least `0.89`.
- candidate does not rely on blind public-kNN copying.
- candidate report lists every cross-quadrant change.

If any hard gate fails, the recommendation is `hold_no_submit`.

## Expected Failure Modes

- **Calm collapse:** too many images become `calm`, hurting macro-F1 even if accuracy rises.
- **Description ceiling:** text score may stay around `0.95`, making `0.89` unreachable unless classification becomes unrealistically high.
- **Proxy overfitting:** v23 may reward patterns that hidden scoring does not reward.
- **Rare-class erosion:** moving too many samples out of low-frequency emotions can reduce macro-F1.
- **Ontology mismatch:** public EmoArt labels may not exactly match hidden test labels even when images are similar.

## Testing

Run only Track2-relevant checks:

```bash
python3 -m unittest tests/test_track2_v28_final_shot_hybrid.py tests/test_track2_official_anchor_calibration.py -v
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v28_hybrid_best_candidate.json
git diff --check
```

If no v28 candidate clears the final gate, validation should still run on the highest-scoring side-path candidate, and the report should state that it is not recommended for upload.

## Success Criteria

The work is successful if it produces one of two clear outcomes:

1. `recommend_final_submit`
   - local proxy overall `>= 0.89`;
   - classification `>= 0.78`;
   - description `>= 0.97`;
   - all safety gates pass;
   - exact ZIP path is surfaced for manual upload.

2. `hold_no_submit`
   - no candidate passes the final gate;
   - report explains why;
   - final Codabench attempt is preserved.

