# Track2 V6 Risk-Calibrated Specialist Selector Design

Date: 2026-06-05
Status: approved design direction; implementation plan not started

## Goal

Build a Track2 classification v6 route that can improve the current best local candidate without spending an online Codabench submission.

The v6 route must not become another full-test relabeler. Previous full supervised and stacked-head attempts underperformed the current selective fusion strategy:

- SigLIP2 full public embedding: holdout macro-F1 `0.2497`
- CLIP full public embedding: holdout macro-F1 `0.2415`
- DINOv2 full public embedding: holdout macro-F1 `0.2181`
- local SigLIP2 cap500 head-only fine-tune: macro-F1 `0.3047`
- stacked embedding head on hard96: macro-F1 `0.2508` to `0.2649`
- current selective hard96 reference: macro-F1 `0.4004`

Therefore v6 will be a risk-calibrated specialist selector: it decides whether an already proposed label delta is safe to accept. It does not predict a fresh label for every test image.

## Current Baseline

The main baseline is:

- JSON: `submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json`
- ZIP: `submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.zip`

Known properties:

- 85 same-quadrant classification changes
- 0 cross-quadrant classification changes
- 192 Gemini 3.5 Flash multimodal description audits
- 39 accepted text rewrites
- 226 text fields changed
- Track2 validator OK
- local description audit issue count 0
- local shadow expected overall `0.8639085`
- local shadow classification expected `0.778150`
- local shadow description proxy `0.949667`

This baseline is close to the visible leaderboard target, but local evidence does not prove it beats first place. If official Description reached `1.0`, its implied overall would be about `0.889075`, still only near the visible `0.89` target.

## Design Principles

1. Preserve submission chances.
   v6 must produce local evidence before any online submission is considered.

2. Do not trust any single model as ground truth.
   Gemini, Vulca, public references, supervised heads, kNN, and human review are evidence sources, not final answers.

3. Prefer small, auditable deltas.
   The output should be a candidate ladder with a few explainable changes, not broad replacement.

4. Treat cross-quadrant changes as dangerous.
   Cross-quadrant deltas can help classification, but they also damage valence/arousal and lower the local risk bound when wrong.

5. Avoid hard96 overfitting.
   hard96 is a useful public stress set, not hidden ground truth. v6 must use stability checks, not just a single hard96 score.

## Architecture

### Evidence Builder

`EvidenceBuilder` creates one normalized delta table from existing artifacts.

Inputs:

- `experiments/track2_score_calibrated_champion_20260605/v1/evidence_matrix.csv`
- current candidate rows from `v3_mid_gemini35_desc_192`
- full supervised predictions from CLIP, SigLIP2, and DINOv2
- stacked-head predictions where available
- MoE specialist dry-run decisions
- scaled human review gates
- public reference and duplicate-audit outputs
- Gemini 3.5 Flash arbitration ledgers
- local shadow evaluator row-risk outputs

Output:

- one row per candidate delta with fields such as:
  - `sample_id`
  - `current_emotion`, `current_valence`, `current_arousal`
  - `proposed_emotion`, `proposed_valence`, `proposed_arousal`
  - `transition`
  - `same_quadrant`
  - `cross_quadrant_risk`
  - `supporting_family_count`
  - `supporting_source_count`
  - `evidence_score`
  - `gemini35_objection`
  - `vulca_objection`
  - `public_reference_support`
  - `public_reference_contradiction`
  - `human_accept_support`
  - `hard96_transition_stats`

The builder should be deterministic and should write CSV/JSON outputs for review.

Missing or malformed evidence must not become implicit support. If an input source is unavailable, duplicated, corrupted, or missing required columns, the selector must either drop that source from the family count or default the affected delta to `hold_review`. It must never promote a delta because evidence is absent.

### Specialist Selector

`SpecialistSelector` decides whether each delta should be:

- `accept_safe`
- `accept_cross_micro`
- `hold_review`
- `block`

The first implementation should be deterministic rules, not a learned model. The rule selector is easier to audit and less likely to overfit.

The initial rule shape:

- same-quadrant deltas may be accepted when multi-source evidence is strong and no major objection exists;
- cross-quadrant deltas are blocked by default;
- cross-quadrant deltas may enter `cross_micro` only when they pass stricter evidence and arbitration thresholds;
- known dangerous patterns such as broad `calm/content -> tired/bored/frustrated/annoyed` remain hold unless external evidence is unusually strong;
- public duplicate/reference contradictions override model votes.
- conflicting high-authority evidence defaults to hold, for example when Gemini supports a delta but Vulca, public reference, or human gate contradicts it.

A later learned selector may be added only after the rule selector baseline is measured. If added, it should be highly regularized and interpretable, such as L1 logistic regression or a shallow decision tree. It must report feature weights or rules.

### Hard96 Stability Gate

The hard96 gate should evaluate whether selector rules are stable, not merely whether one final candidate looks good.

Required checks:

- leave-one-out or k-fold evaluation over hard96 deltas where labels are available;
- per-transition net gain/loss;
- per-emotion recall and precision movement;
- valence and arousal movement;
- sensitivity sweep over evidence thresholds;
- explicit warning when a rule only works because of one or two hard96 examples.

The gate should fail a selector when:

- macro-F1 improvement is concentrated in one fragile transition;
- valence or arousal accuracy drops below the existing reference gate;
- minority classes such as `glad`, `aroused`, `excited`, `annoyed`, `frustrated`, or `tired` collapse;
- the selector accepts broad cross-quadrant deltas without independent arbitration.

### Candidate Ladder Writer

`CandidateLadderWriter` writes side-path candidates only.

Candidate ladder:

- `v6_safe_sameq`: accepts only stable same-quadrant deltas beyond the v3 baseline.
- `v6_cross_micro`: starts from `v6_safe_sameq` and adds only the strongest one to three cross-quadrant deltas.
- `v6_desc_plus`: applies the selected v6 classification deltas on top of the `desc_192` text-enhanced baseline.

Output paths should follow this pattern:

- `submissions/track2_submission_v6_safe_sameq_candidate.json`
- `submissions/track2_submission_v6_safe_sameq_candidate.zip`
- `submissions/track2_submission_v6_cross_micro_candidate.json`
- `submissions/track2_submission_v6_cross_micro_candidate.zip`
- `submissions/track2_submission_v6_desc_plus_candidate.json`
- `submissions/track2_submission_v6_desc_plus_candidate.zip`

The writer must reject formal submission paths:

- `submissions/track2_submission.json`
- `submissions/track2_submission.zip`

### V6 Reports

Each run should produce:

- normalized evidence table CSV/JSON;
- selector decision report CSV/JSON/Markdown;
- hard96 stability report JSON/Markdown;
- candidate report JSON/Markdown;
- local shadow evaluator outputs;
- optional HTML review for human inspection.

The Markdown report must lead with a submit/hold recommendation and explain why.

The JSON report must include a machine-readable top-level decision field:

- `recommend_submit`
- `recommend_hold`
- `diagnostic_only`
- `invalid`

## Acceptance Rules

### Same-Quadrant Deltas

Same-quadrant deltas can be accepted only when all are true:

- label consistency remains valid;
- no Gemini/Vulca objection;
- no public reference contradiction;
- evidence comes from at least two independent families or a strong human-confirmed gate;
- distribution remains stable;
- hard96 stability does not show transition-level loss.

### Cross-Quadrant Deltas

Cross-quadrant deltas are hold by default.

They may enter `v6_cross_micro` only when all are true:

- evidence score is high relative to the existing matrix;
- supporting family count and source count are high;
- Gemini 3.5 Flash or equivalent image arbitration prefers proposed;
- confidence and fit margin exceed the cross threshold;
- Vulca or public reference does not contradict proposed;
- hard96 stability accepts the transition family;
- local shadow lower bound does not degrade beyond the configured tolerance.

The first configured tolerance is intentionally strict:

- for `v6_safe_sameq`, `overall_lower` must be at least the `v3_mid_gemini35_desc_192` lower bound;
- for `v6_cross_micro`, `overall_expected` must improve by at least `0.002000` over `v3_mid_gemini35_desc_192`;
- for `v6_cross_micro`, `overall_lower` may drop by at most `0.003000` versus `v3_mid_gemini35_desc_192`;
- any larger lower-bound drop makes the candidate `diagnostic_only` or `recommend_hold`.

The current v5 finding is a useful example:

- `track2_0547 annoyed -> calm` passed image arbitration but `v5_cross1` still received `recommend_hold` from the local shadow evaluator due cross-quadrant uncertainty.
- The score=7 cross-quadrant group did not pass the strict gate.

This means v6 must not assume cross-quadrant expansion is automatically beneficial.

## Success Criteria

A v6 candidate is allowed to be considered for online submission only if all checks pass:

- `python3 -m affectiveart.challenge validate-track2 <candidate.json>` returns OK;
- Track2 label consistency issue count is 0;
- missing emotions is none;
- no category collapse;
- hard96 stability gate passes;
- local shadow expected is greater than `v3_mid_gemini35_desc_192`;
- local shadow lower bound satisfies the explicit tolerance for its ladder type;
- description audit has no evaluator-manipulation or contradiction issues;
- formal submission files are not overwritten;
- the report explicitly lists changed rows and risk reasons.

If v6 cannot beat the v3 baseline locally, the correct output is a hold report, not a new submission candidate.

## Non-Goals

V6 will not:

- submit to Codabench;
- overwrite formal submission files;
- train a full replacement classifier as the first step;
- accept all Gemini or Vulca suggestions;
- use public duplicate labels as hidden test answers;
- optimize text fields in this classification pass except where needed to keep changed labels self-consistent.

## Testing Plan

Run only Track2-relevant checks:

- `python3 -m unittest discover -s tests -p 'test_track2*.py' -v`
- `python3 -m affectiveart.challenge validate-track2 <candidate.json>`
- local shadow evaluator on all v6 candidates
- hard96 stability gate
- `git diff --check`

Unit tests should cover:

- deterministic evidence normalization;
- missing/corrupted evidence defaults to hold;
- conflicting evidence defaults to hold;
- same-quadrant accept path;
- cross-quadrant diagnostic path;
- formal submission path rejection;
- candidate ladder output names;
- machine-readable report decisions.

Do not run full test discovery for this Track2 session because unrelated Track1 work can fail or add noise.

## Implementation Shape

Likely files:

- `affectiveart/track2_v6_specialist_selector.py`
- `scripts/track2_v6_specialist_selector.py`
- `tests/test_track2_v6_specialist_selector.py`

The implementation should reuse existing helpers from:

- `affectiveart.track2_score_calibrated_champion`
- `affectiveart.track2_local_shadow_evaluator`
- `affectiveart.track2_audit`
- `affectiveart.track2_repair`
- `affectiveart.challenge`

## First Implementation Slice

The first implementation should build:

1. evidence loading and normalization;
2. deterministic selector rules;
3. candidate ladder writer;
4. hard96 stability summary from available public validation artifacts;
5. reports and tests.

Learned selectors and new model training are deferred until the deterministic selector establishes whether more classification gains are still visible locally.
