# Track2 Score-Calibrated Champion Pass Design

**Date:** 2026-06-05

## Summary

This design defines the next Track2-only push after the first Codabench feedback for AffectiveArt Challenge 2026. The current submitted MoE v2 accept-5 package scored:

- Overall: `0.836408`
- Classification Score: `0.723150`
- Description Score: `0.949667`
- Emotion accuracy / macro-F1: `0.570000` / `0.309338`
- Valence accuracy / macro-F1: `0.883000` / `0.823380`
- Arousal accuracy / macro-F1: `0.913000` / `0.840184`

The visible Track2 leader is around `0.89` overall with about `0.78` classification and `1.00` Description Score. The largest actionable gap is not emotion macro-F1 alone. It is official-style emotion accuracy plus the remaining Description Score gap.

The next candidate must therefore optimize expected official score, not only local macro-F1 or human-review confidence. It must remain side-path only and must not overwrite `submissions/track2_submission.json` or `submissions/track2_submission.zip`.

## Goal

Build a score-calibrated Track2 candidate generator that combines existing local evidence, public-style label priors, human/Gemini/Vulca audits, and description alignment into submit-ready side-path candidates. The first submission produced by this pass should be a `v3_safe_plus` candidate, not a broad `v3_push`, unless local gates show the aggressive candidate is clearly safer than expected.

## Why This Changes The Previous Strategy

The first submitted package was too conservative for classification. It changed only five rows:

- `track2_0112`: `content -> calm`
- `track2_0347`: `content -> calm`
- `track2_0745`: `content -> calm`
- `track2_0798`: `content -> calm`
- `track2_0960`: `content -> calm`

This preserved Description Score but left emotion accuracy far behind the leaders. The leaderboard also shows that the best team has emotion macro-F1 near ours but much higher emotion accuracy, which means the hidden gold labels likely follow a dominant official annotation style. The system should prefer high-confidence official-style changes over broad class-balancing moves.

## Design Choice

Use a deterministic score-calibrated selector, not a new end-to-end trained model.

Existing training results are not strong enough to become final authority:

- SigLIP2 full public embedding: holdout accuracy `0.4659`, macro-F1 `0.2497`
- CLIP full public embedding: macro-F1 `0.2415`
- DINOv2 full public embedding: macro-F1 `0.2181`
- local SigLIP2 cap500 fine-tune: macro-F1 `0.3047`
- hard96 selective fusion: macro-F1 `0.4004`

These models are useful proposers and evidence sources. They are not safe full-test relabelers. The final selector should accept a change only when multiple independent signals point to the same label and the expected official-score gain exceeds the risk.

## Inputs

The pass reads existing frozen artifacts:

- Current submitted candidate:
  - `submissions/track2_submission_moe_v2_accept5_candidate.json`
- Safe baseline before accept-5:
  - `submissions/final_track2_20260602_scaled_human_gate_gemini35_guarded_v1.json`
- Public-style clean/inclusive outputs:
  - `experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/clean_vs_inclusive_report.json`
  - `experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/clean_predictions.json`
  - `experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/inclusive_predictions.json`
- Larger historical candidates:
  - `submissions/track2_submission_teacher_review_top120_research_gated_v3.json`
  - `submissions/track2_submission_champion_candidate_selective_v2.json`
  - `submissions/track2_submission_champion_candidate_balanced.json`
  - `submissions/track2_submission_champion_candidate_max_macro.json`
  - `submissions/track2_submission_champion_candidate_label_rubric_high_precision_v1_from_auto_v3.json`
- Description repair evidence:
  - `experiments/track2_champion_runs/20260512_final_review/description_score_v2_report.json`
  - `experiments/track2_scaled_human_gate_20260602/description_candidate_report.json`
- Human and guard evidence:
  - `human_review/returns/20260602_scaled/track2_scaled_gate_recommendations.csv`
  - `experiments/track2_scaled_human_gate_20260602/gemini35_guarded_v1/rollback_report.json`
- Public reference and duplicate evidence, treated as supporting style evidence rather than official gold answers:
  - `experiments/track2_emoart130k_clip/overlap_reference/ge095_all/`
  - `experiments/track2_emoart130k_clip/deep_duplicate_audit_20260511/`

If any input is missing, the pass should write a blocked report that names the missing path and should not build a submit-ready ZIP.

## Evidence Matrix

Each proposed change becomes one row in a normalized evidence matrix:

- `sample_id`
- `current_emotion`, `current_valence`, `current_arousal`
- `proposed_emotion`, `proposed_valence`, `proposed_arousal`
- `transition`, such as `content->calm`
- `source_support_count`
- `supporting_sources`
- `opposing_sources`
- `same_quadrant`
- `public_style_agreement`
- `duplicate_or_series_support`
- `human_accept_support`
- `gemini35_objection`
- `vulca_description_risk`
- `caption_alignment_required`
- `cross_quadrant_risk`
- `category_collapse_risk`
- `score_calibrated_rank`
- `decision`: `accept`, `hold`, `block`, or `rewrite_only`

The matrix is the main debug artifact. Every accepted change must be explainable from this table.

## Score-Calibrated Selection

The selector estimates expected official-score value per edit. It should not claim to predict the hidden score exactly; it is a ranking heuristic that protects the daily submission quota.

Positive evidence:

- Clean and inclusive public-style models agree on the proposed label.
- Historical strong candidates independently proposed the same label.
- Human scaled review accepted the same label with confidence at least 4.
- Gemini 3.5 did not object to the same change or explicitly supported it.
- The change is same-quadrant, especially `content -> calm` or `calm -> content`.
- Duplicate or same-series public reference supports the same broad annotation style.
- The target label improves likely official-style accuracy without removing any class.

Risk evidence:

- Cross-quadrant flip without strong support.
- Gemini 3.5 rollback previously objected.
- Public-style model is the only supporting source.
- Change creates label-text contradiction.
- Change increases top emotion share or collapses `calm/content`.
- Change over-corrects toward a minority class for macro-F1 while likely hurting accuracy.
- The row was previously manually held, low confidence, or conflicted.

The selector should default to `hold` when evidence is mixed.

## Candidate Ladder

The pass creates three side-path candidates. All candidates must preserve all 12 emotion labels and pass validation before review.

### v3_safe_plus

Purpose: first submission candidate if gates pass.

Expected size:

- 30-45 label changes from accept-5.
- Mostly same-quadrant changes.
- Prioritize high-confidence `content -> calm`, `calm -> content`, and a small number of clear `content -> glad` or similar positive-low boundary repairs.
- Apply prior validated description repairs and rewrite descriptions for all changed labels.

Submission logic:

- Prefer this as the next Codabench submission because it tests the full pipeline while limiting downside.
- Do not submit if it cannot beat the current package on local risk gates and description audit.

### v3_mid

Purpose: second candidate for stronger leaderboard jump if `v3_safe_plus` is coherent.

Expected size:

- 60-85 label changes.
- Same-quadrant changes remain dominant.
- Allows a limited number of cross-quadrant repairs only when at least three independent sources agree.

Submission logic:

- Submit only after `v3_safe_plus` score or offline audits show the selector is calibrated.
- May become the next-day submission if `v3_safe_plus` improves classification but still trails 0.86.

### v3_push

Purpose: diagnostic upper bound, not the default daily submission.

Expected size:

- 90-125 label changes.
- Includes more public-style agreement rows and selective-fusion proposals.

Submission logic:

- Build and audit, but do not submit first unless the safer candidates fail local gates for a clear reason and `v3_push` has no category-collapse or description-risk warning.

## Description Score Pass

Because Description Score is half of the official score, every candidate must run a label-aware description pass after labels are selected.

Minimum text policy:

- Apply the previously validated 28 `overall_caption` repairs where still label-consistent.
- Rewrite every row whose emotion label changes.
- Rewrite any row where the existing caption or attributes contradict the selected emotion, valence, or arousal.
- Preserve concrete image details and avoid generic template prose.
- Do not include evaluator-directed instructions, scoring language, hidden prompts, or any attempt to manipulate the evaluator.

Gemini 3.5 Flash may be used for targeted rewrite and audit. Vulca/vulca-emnlp should be used as a post-hoc quality audit on changed and high-risk rows. If Gemini or Vulca flags a row as visually unsupported or emotionally contradictory, the row is held or rolled back.

## Gates

No candidate can be recommended unless all gates pass:

- `python3 -m affectiveart.challenge validate-track2 <candidate.json>` passes.
- Track2 label consistency issues are `0`.
- All 12 emotions are present.
- Missing emotions are none.
- Candidate ZIP payload matches candidate JSON.
- No formal submission path is overwritten.
- No evaluator manipulation or invalid response text is present.
- Description audit finds no systematic image-text or emotion-text drift.
- Top emotion share stays within the configured bound for that candidate tier.
- `calm/content` combined share does not increase by broad collapse.
- Cross-quadrant changes have at least three independent supporting signals.
- Rows with Gemini 3.5 high-confidence objection remain held unless later evidence explicitly resolves the objection.

## Submission Strategy

There is one submission available today and four total remaining. The rational strategy is:

1. Build all three candidates locally.
2. Reject any candidate with validation, distribution, or description issues.
3. Prefer `v3_safe_plus` for the next submission if it contains enough high-confidence changes and a description pass.
4. Choose `v3_mid` only if `v3_safe_plus` is too weak to plausibly improve and `v3_mid` passes the same gates cleanly.
5. Do not use `v3_push` for the first next submission unless the evidence matrix shows low risk across changed rows.

The target for the next submission is to move from `0.836408` toward `0.86+`. Reaching first place likely requires both:

- Description Score near `0.98-1.00`.
- Classification Score near `0.78`, especially emotion accuracy closer to the official dominant style.

## Outputs

All outputs are side-path artifacts under:

- `experiments/track2_score_calibrated_champion_20260605/`

Expected artifacts:

- `evidence_matrix.csv`
- `evidence_matrix.json`
- `candidate_report_v3_safe_plus.md`
- `candidate_report_v3_mid.md`
- `candidate_report_v3_push.md`
- `html_review/track2_score_calibrated_v3_review.html`

Side-path submissions:

- `submissions/track2_submission_v3_safe_plus_candidate.json`
- `submissions/track2_submission_v3_safe_plus_candidate.zip`
- `submissions/track2_submission_v3_mid_candidate.json`
- `submissions/track2_submission_v3_mid_candidate.zip`
- `submissions/track2_submission_v3_push_candidate.json`
- `submissions/track2_submission_v3_push_candidate.zip`

## Non-Goals

- Do not work on Track1 in this pass.
- Do not train a new large model before the selector is built.
- Do not treat public EmoArt labels as official test answers.
- Do not let Gemini, Vulca, or any single model become the final arbiter.
- Do not optimize macro-F1 by creating broad category shifts that likely hurt official emotion accuracy.
- Do not submit another package until a candidate has passed local gates and the user explicitly approves the exact ZIP.

## Gemini-Agent Critique Incorporated

Gemini-agent independently flagged four risks:

- limited remaining Codabench submissions;
- possible Description Score degradation from broad LLM rewrites;
- weak authority of the current local visual models;
- label-description mismatch risk.

This design incorporates that critique by making `v3_safe_plus` the default next submission, making broader candidates diagnostic until gates pass, and requiring strict description alignment after any label change.

## Success Criteria

The design is ready for implementation when:

- the evidence matrix can explain every accepted label change;
- the candidate ladder is side-path only;
- `v3_safe_plus`, `v3_mid`, and `v3_push` can be generated and compared without touching formal paths;
- Description Score protection is a first-class gate;
- the user can inspect changed rows in HTML before approving a submission;
- one exact ZIP path can be recommended with a risk note.
