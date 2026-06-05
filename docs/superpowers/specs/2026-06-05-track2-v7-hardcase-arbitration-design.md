# Track2 V7 Hard-Case Arbitration Design

**Date:** 2026-06-05

## Goal

Build a Track2-only v7 hard-case label arbitration pass for aggressive classification exploration after the v3/v6 passes. The pass must create side-path candidates only, must not overwrite `submissions/track2_submission.json` or `submissions/track2_submission.zip`, and must not submit to Codabench.

## Context

The current online score anchor is:

- Overall: `0.836408`
- Classification: `0.723150`
- Description: `0.949667`

The current strongest local candidate is `track2_submission_v3_mid_gemini35_desc_192_candidate`, with local proxy expected overall `0.8639085`. The old shadow evaluator already caps same-quadrant classification gain at this point, so v7 cannot improve the old proxy by simply adding more calm/content boundary changes. Any plausible upside must come from a very small number of hard cross-quadrant corrections that the hidden gold may reward.

## Design

V7 reads frozen artifacts:

- Baseline labels/text: `submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json`
- Absolute shadow anchor: `submissions/track2_submission_moe_v2_accept5_candidate.json`
- Evidence matrix: `experiments/track2_score_calibrated_champion_20260605/v1/evidence_matrix.csv`
- Gemini 3.5 cross-case arbitration summary: `experiments/track2_score_calibrated_champion_20260605/v1/v5_cross_arbitration/cross_score7_arbitration_summary.json`
- Label-aligned text override source: `submissions/track2_submission_v5_cross1_gemini35_desc_192_candidate.json`

The selector accepts a cross-quadrant proposal only when:

- Gemini arbitration prefers the proposed label;
- strict gate is explicitly true for `v7_cross1` and `v7_cross3`;
- confidence, fit margin, evidence score, source count, and family count exceed thresholds;
- the baseline current label matches the evidence current label;
- the proposed emotion is a valid Track2 emotion and valence/arousal match the official mapping;
- the proposed row has label-aligned text overrides for all text fields;
- no Gemini/public-reference contradiction is present;
- dangerous Positive/Low to non-Positive/Low shifts are blocked unless the strict gate passes.

## Candidate Ladder

- `v7_cross1`: strict accepted rows, cap 1.
- `v7_cross3`: strict accepted rows, cap 3.
- `v7_hardcase_push`: strict rows plus non-dangerous borderline proposed rows, cap 5. This is diagnostic only.

## Expected Behavior

On the current frozen artifacts, v7 should accept only `track2_0547 annoyed->calm` because it is the only reviewed cross-case with strict Gemini support and a label-aligned text override. All calm-to-tired/bored/aroused score-7 rows remain held.

## Success Criteria

- Side-path v7 JSON/ZIP candidates are written.
- Formal submission paths are not overwritten.
- Track2 validator passes for all v7 candidates.
- Label consistency issue count is zero.
- The HTML/reports clearly show accepted and held rows.
- The final recommendation distinguishes review-worthy v7 cross probes from safer v3 submission candidates.
