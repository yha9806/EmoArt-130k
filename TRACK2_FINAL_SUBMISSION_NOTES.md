# Track2 Final Submission Notes

## 2026-06-03 Final Freeze Update

Recommended official Track2 package is now frozen to the exact formal filenames:

- JSON: `submissions/track2_submission.json`
- ZIP: `submissions/track2_submission.zip`

These files are byte-identical to the post-human-review guarded candidate:

- JSON: `submissions/final_track2_20260602_scaled_human_gate_gemini35_guarded_v1.json`
- ZIP: `submissions/final_track2_20260602_scaled_human_gate_gemini35_guarded_v1.zip`

Previous formal package was preserved as:

- JSON: `submissions/track2_submission_legacy_20260510_before_guarded_20260603.json`
- ZIP: `submissions/track2_submission_legacy_20260510_before_guarded_20260603.zip`

Final validation on 2026-06-03:

- `python3 -m affectiveart.challenge validate-track2 submissions/track2_submission.json`: OK
- `python3 scripts/affectiveart_challenge.py validate-track2 submissions/track2_submission.json`: OK
- ZIP members: `submission.json`
- ZIP payload matches formal JSON: yes
- Formal JSON matches guarded candidate JSON: yes
- Formal ZIP matches guarded candidate ZIP: yes
- Rows: 1000
- Post-manual-review final gate: `safe_to_freeze=true`
- Label consistency issues: 0
- Missing emotions: none
- Suspected overlap label changes: 0
- Track2-only tests: 104 OK

SHA-256:

- `submissions/track2_submission.json`: `b8e170f67e90d5edf64d0d0d8dbf808192f5e33c253cfe8014d1a0c74b1dc10d`
- `submissions/track2_submission.zip`: `e34c3b069e58bffa1b45edcc2032d089a55380fde0de937f6c19450414773932`

Human-review closeout:

- Protocol: `docs/track2_human_review_protocol_zh.md`
- Acceptance record: `experiments/track2_scaled_human_gate_20260602/gemini35_guarded_v1/html_review/track2_simplified_manual_review_acceptance_zh.md`
- Final gate report: `experiments/track2_scaled_human_gate_20260602/gemini35_guarded_v1/final_gate_post_manual_review/track2_final_gate_final_track2_20260602_scaled_human_gate_gemini35_guarded_v1_post_manual_review.json`

The older 2026-05-12 notes below are retained for provenance.

Date: 2026-05-12

## Final Frozen Candidate

Recommended Track2 submission package:

- JSON: `submissions/final_track2_20260512_description_v2_vulca_audited.json`
- ZIP: `submissions/final_track2_20260512_description_v2_vulca_audited.zip`

Source candidate:

- JSON: `submissions/final_track2_20260512_description_aligned.json`
- ZIP: `submissions/final_track2_20260512_description_aligned.zip`

Do not overwrite `submissions/track2_submission.json` or `submissions/track2_submission.zip` unless the official submission instructions explicitly require those exact names.

## Decision

Freeze `final_track2_20260512_description_v2_vulca_audited` as the main Track2 candidate.

This candidate combines:

- `vulca_emnlp_repaired` as the original safe baseline.
- `selective_v2` for conservative macro-F1-oriented candidate repair.
- `public_reference_auto_v3` for exact/near-duplicate public-reference alignment.
- `label_rubric_high_precision_v1` for high-precision low-arousal boundary repairs.
- Human-reviewed approval for the two suspected-overlap label changes.
- Text-only description alignment after official scoring clarified that Description Score is 50% of Track2.
- Vulca-EMNLP/local Description Score audit to remove the remaining formulaic caption suffixes without changing labels.

The final pipeline report marks this candidate as `safe_to_freeze=true`.

## Final Gate Evidence

Primary final gate report:

- `experiments/track2_champion_runs/20260512_final_review/final_gate_description_v2/track2_final_gate_final_track2_description_v2_vulca_audited.json`
- `experiments/track2_champion_runs/20260512_final_review/final_gate_description_aligned/track2_final_gate_final_track2_description_aligned.json`
- `experiments/track2_champion_runs/20260512_final_review/final_gate/track2_final_gate_label_rubric_high_precision_v1_from_auto_v3.json`
- `experiments/track2_champion_runs/20260512_final_review/candidate_scoreboard.md`

Final gate summary:

- Rows: 1000
- Label consistency issues: 0
- Missing emotions: none
- ZIP payload matches JSON: yes
- Validator: OK
- Suspected overlap label changes: 2
- Human-approved suspected overlap changes: 2
- Unapproved suspected overlap changes: 0
- Risk summary: none
- Classification labels changed by Description v2 pass: 0
- Text rows changed by Description v2 pass: 28
- Evaluator manipulation risk scan: 0 hits

Freeze manifest:

- `experiments/track2_champion_runs/20260512_final_review/track2_final_freeze_manifest.md`
- `experiments/track2_champion_runs/20260512_final_review/track2_final_freeze_manifest.json`

## Human Review

Human review record:

- `experiments/track2_champion_runs/20260512_final_review/human_review_label_rubric_high_precision_v1.json`

Rejected high-risk teacher-review changes:

- `track2_0999`: rejected Gemini `alarmed / Negative / High` -> `glad / Positive / Low`; manual review keeps `alarmed / Negative / High` because the image is a figure restraining or confronting a winged sea-dragon-like creature, and the struggle/threat cues outweigh the composed pose.

Approved overlap changes:

- `track2_0326`: `tired / Negative / Low` -> `calm / Positive / Low`
- `track2_0841`: `content / Positive / Low` -> `calm / Positive / Low`

Approved high-precision label-rubric changes:

- `track2_0153`: `sad` -> `calm`
- `track2_0286`: `tired` -> `calm`
- `track2_0318`: `tired` -> `calm`
- `track2_0452`: `tired` -> `calm`
- `track2_0605`: `sad` -> `calm`
- `track2_0772`: `tired` -> `calm`
- `track2_0829`: `bored` -> `calm`

HTML review page:

- `experiments/track2_champion_runs/20260512_final_review/html_review/track2_final_review.html`

## Description Score Alignment

Official scoring gives 50% weight to Description Score, covering artwork consistency, artistic attribute quality, and overall caption quality. After this clarification, the package first received text-only edits on 10 rows to reduce label/text contradictions such as `calm` rows still mentioning fatigue, exhaustion, boredom, or overtly somber mood.

The v2 pass then removed remaining template-like emotional-atmosphere suffixes from 28 captions. It does not rewrite labels or artistic attribute fields; it only reduces repetition and possible LLM-evaluator template penalty in `overall_caption`.

Description alignment report:

- `experiments/track2_champion_runs/20260512_final_review/description_alignment_report.json`
- `experiments/track2_champion_runs/20260512_final_review/description_score_v2_report.json`
- `experiments/track2_champion_runs/20260512_final_review/description_score_v2_report.md`

Description v2 audit summary:

- Formulaic suffix issues: 28 before -> 0 after
- Classification label changes: 0
- Label consistency issues: 0
- Vulca-EMNLP sample audit: 20 rows, local fallback judge available through `/Users/yhryzy/dev/vulca-emnlp2026/packages/vulca-framework`

This update does not change emotion, valence, or arousal labels.

## Training Decision

Do not make new training the main Track2 path for the current submission.

Training can continue only as a parallel backup experiment. A trained candidate may replace the frozen package only if it passes the same final gate and shows clear public-validation improvement without introducing class collapse, broad `content -> calm` drift, or regressions on the manually reviewed samples.

## Submission Readiness

Use the v2 Vulca-audited frozen ZIP unless the official instructions ask for raw JSON or a different archive layout.

If official instructions require a different filename, copy the frozen package to the required name after preserving this frozen artifact unchanged.

Latest dry-run verification on 2026-05-12:

- `python3 -m affectiveart.challenge validate-track2 submissions/final_track2_20260512_description_v2_vulca_audited.json`: OK
- `python3 -m unittest discover -s tests -p 'test_track2*.py' -v`: 88 tests OK
- `python3 scripts/track2_final_gate.py ...`: `safe_to_freeze=true`, `overlap_changed_count=2`
- ZIP members: `submission.json`
- ZIP payload matches frozen JSON: yes
- `track2_0999` in frozen package: `alarmed / Negative / High`
- `git diff --check`: OK
