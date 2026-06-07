# Track2 v17 Classification Calibration Design

## Goal

Build a Track2-only v17 experiment that improves 12-way emotion classification on top of the fixed `v15_desc_expand300` text base.

The target is not to rewrite descriptions or re-run broad Gemini visual review. The target is to find a small or medium set of emotion-label changes that are more likely to match the official hidden labels than the current anchor, while preserving valence, arousal, and description quality.

## Current Evidence

- Current stable official anchor: `779605`, overall `0.836408`, classification `0.723150`, description `0.949667`.
- Latest page result: `782683`, displayed overall `0.84`, classification about `0.72`, description about `0.95`, with no classification-label changes.
- Current best local text base: `v15_desc_expand300`, local fused expected `0.839902`, lower `0.835854`, with zero label changes.
- Failed aggressive official submission: `781601`, overall `0.834027`, classification `0.719137`, description `0.948917`.
- v16 RAG teacher probes found no reliable classification lift:
  - top48 accepted only 3 same-quadrant `content -> calm` changes and did not beat v15 lower bound.
  - non-content top24 accepted 0 changes at `min-confidence=0.85`.

The public leaderboard diagnosis shows the gap is mostly emotion accuracy: our emotion accuracy is about `0.57`, first place is about `0.80`, while macro-F1 is similar. This points to official label-style mismatch rather than only rare-class weakness.

## Non-Goals

- Do not touch Track1.
- Do not overwrite `submissions/track2_submission.json` or `submissions/track2_submission.zip`.
- Do not use public near-duplicates as direct hidden-test answers unless the evidence is exact or near-exact same work and passes a safety gate.
- Do not submit automatically.
- Do not optimize by inserting evaluator-facing instructions or invalid content.
- Do not keep expanding the current v16 Gemini-RAG queue; it is overconcentrated in `content -> calm` and has already produced weak evidence.

## Approaches Considered

### A. Continue Gemini-RAG Review

This is not recommended. It is easy to run, but v16 top48 and non-content top24 both showed low confidence and minimal accepted changes. It is unlikely to close a 0.05 overall gap.

### B. Train A Single Stacked Head

This is partially useful but insufficient by itself. Existing stacked-head sweeps underperformed on public hard96, with macro-F1 around `0.2649`, below the existing selective/rubric challenger. A plain replacement classifier is too risky.

### C. Official-Feedback-Calibrated Selective Fusion

This is the recommended route. Use stacked/supervised/public-reference/Gemini-specialist outputs as proposers, then calibrate acceptance using official submission feedback, public validation performance, transition priors, and strict safety gates. The system should produce nested candidates rather than a single all-in replacement.

## Architecture

v17 has five components:

1. **Evidence Builder**
   - Inputs:
     - fixed base JSON: `submissions/track2_submission_v15_desc_expand300_candidate.json`
     - public/validation predictions from SigLIP2, CLIP, DINOv2, existing stacked heads, selective hard96 outputs, and Gemini specialist predictions
     - public duplicate and near-duplicate audits
     - official score anchors and pairwise submission diffs
   - Output:
     - one row per proposed sample-label change with model votes, public-reference evidence, confidence, transition type, and known official-risk features.

2. **Calibration Evaluator**
   - Evaluates each proposer family on available public validation/hard-case splits.
   - Computes per-emotion and per-transition reliability.
   - Separates emotion accuracy proxy from macro-F1 proxy; same-quadrant changes must not be treated as safe by default.
   - Uses official feedback to penalize failed transition families, especially bulk `calm/content/glad` reshuffling from `781601`.

3. **Selective Candidate Builder**
   - Starts from `v15_desc_expand300`.
   - Applies only label changes that pass acceptance rules.
   - Writes side-path candidates:
     - `submissions/track2_submission_v17_safe_candidate.json/.zip`
     - `submissions/track2_submission_v17_balanced_candidate.json/.zip`
     - `submissions/track2_submission_v17_aggressive_probe_candidate.json/.zip`
   - Repairs valence/arousal mechanically from accepted emotion labels.

4. **Local Scoring And Gate**
   - Runs `validate-track2`.
   - Runs classification-only local shadow scorer.
   - Runs fused scorer against:
     - `official_779605_moe_v2_anchor`
     - `official_782683_v12_stable_probe`
     - `v15_desc_expand300`
     - all v17 candidates
   - Recommends submission only if a v17 candidate beats `v15_desc_expand300` on lower bound or has a clearly documented official-probe rationale.

5. **Review Artifacts**
   - Generates Markdown and CSV reports for:
     - accepted changes
     - blocked changes
     - transition distribution
     - emotion distribution shift
     - public validation support
     - official-feedback risk
   - Optional HTML review can be generated for the final small set of accepted changes, but human review is not the primary scoring mechanism.

## Acceptance Rules

An emotion change may enter `safe` only if all are true:

- At least two independent non-Gemini model families support the proposed label, or exact/near-exact public duplicate evidence supports it.
- Public validation reliability for that proposer/transition is nonnegative.
- The change is not from a known failed bulk transition family unless evidence is exact duplicate or unusually strong.
- It does not create valence/arousal inconsistency.
- It does not increase top-emotion collapse.
- It does not remove a rare class below coverage unless the current label has strong negative evidence.

`balanced` may allow weaker but still multi-source evidence, capped by transition family.

`aggressive_probe` may include higher-risk changes only for local analysis or, later, an explicit official-score probe. It must not be treated as the default submission package.

## Data Flow

```text
base v15 JSON
  + model prediction files
  + public validation reports
  + duplicate/reference audits
  + official score feedback
      -> v17 evidence matrix
      -> per-proposer/per-transition calibration
      -> safe/balanced/aggressive candidate JSON/ZIP
      -> validate-track2
      -> local shadow + fused scorer
      -> final gate report
```

## Safety And Leakage Policy

Public EmoArt data can inform label-style calibration and exact/near-duplicate evidence, but v17 must not blindly copy labels from broad style similarity. Candidate reports must distinguish:

- exact or near-exact same work evidence
- same-series or same-style evidence
- model-only evidence
- official-feedback-derived transition penalty

Only exact/near-exact evidence can override otherwise weak model support.

## Testing

Run only Track2-relevant checks:

```bash
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v17_safe_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v17_balanced_candidate.json
git diff --check
```

New tests should cover:

- evidence matrix aggregation from multiple prediction formats
- official-feedback penalties for failed transition families
- acceptance/rejection rules for safe, balanced, and aggressive candidates
- formal submission overwrite protection
- valence/arousal repair
- local scorer integration and final gate decision

## Success Criteria

Minimum success for local promotion:

- validator OK
- label consistency issues = 0
- missing emotions = none
- v17 lower bound beats `v15_desc_expand300` lower bound, or the report explicitly marks the package as an official-score probe
- no broad `content -> calm` batch reshuffle
- no description regression, because v17 starts from `v15_desc_expand300`

Submission recommendation:

- Prefer no submission over spending a remaining Codabench attempt on a candidate whose lower bound is below v15.
- If no v17 candidate beats v15 locally, keep `v15_desc_expand300` as the next safest package.

## Expected Output Paths

- Implementation module: `affectiveart/track2_v17_classification_calibration.py`
- CLI wrapper: `scripts/track2_v17_classification_calibration.py`
- Tests: `tests/test_track2_v17_classification_calibration.py`
- Experiment root: `experiments/track2_v17_classification_calibration_20260607/`
- Candidate files:
  - `submissions/track2_submission_v17_safe_candidate.json/.zip`
  - `submissions/track2_submission_v17_balanced_candidate.json/.zip`
  - `submissions/track2_submission_v17_aggressive_probe_candidate.json/.zip`

## Implementation Boundary

The first implementation should generate and score candidates locally only. It should not upload to Codabench and should not modify formal submission files.
