# Track2 Official Scorer Probing Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the remaining 3 Codabench submissions as scarce, controlled calibration probes to improve Track2 classification performance without sacrificing the already-strong description score.

**Architecture:** Treat each online submission as an aggregate black-box measurement, not as a direct source of hidden labels. Locally, split the system into three layers: candidate generation, calibrated shadow scoring, and final submit gating. VULCA remains a description and self-consistency guard; emotion-label changes must be justified by calibrated transition evidence.

**Tech Stack:** Python, `unittest`, Codabench detailed result pages, `affectiveart.track2_local_shadow_evaluator`, `affectiveart.track2_fused_shadow_evaluator`, `affectiveart.track2_vulca_entailment_gate`, Track2 JSON validator.

---

## Current Evidence

We have two official anchors:

| Submission | Candidate | Overall | Classification | Description | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `779605` | `track2_submission_moe_v2_accept5_candidate.zip` | `0.836408` | `0.723150` | `0.949667` | Reliable current anchor |
| `781601` | `track2_submission_v3_mid_gemini35_d.zip` | `0.834027` | `0.719137` | `0.948917` | Aggressive same-quadrant label changes underperformed |

The failure mode is classification, specifically 12-way emotion classification:

| Metric | `779605` | `781601` | Delta |
| --- | ---: | ---: | ---: |
| Emotion Accuracy | `0.570000` | `0.547000` | `-0.023000` |
| Emotion Macro F1 | `0.309338` | `0.308260` | `-0.001078` |
| Valence Score Inputs | unchanged | unchanged | `0` |
| Arousal Score Inputs | unchanged | unchanged | `0` |

The v3/v9 route changed 85 emotion labels, all within the same valence/arousal quadrant. The local scorer treated those same-quadrant changes as positive, but official scoring showed net negative classification impact. Therefore the old local scorer is directionally wrong for large same-quadrant calm/content/glad relabeling.

## What Aggressive Submissions Can And Cannot Tell Us

Aggressive online submissions can be useful probes if they isolate one hypothesis at a time. They can tell us:

- Whether a transition family is likely beneficial in aggregate, such as `content -> calm`.
- Whether a specific evidence gate is useful, such as public-reference duplicate support or high-confidence human review.
- Whether description remains stable under a candidate.
- Whether the local scorer's lower bound is calibrated enough to trust.

They cannot tell us:

- The official gold label for any individual test image.
- Which exact changed rows were correct or wrong.
- Whether a high description score came from every row being good or from averaging out weak rows.
- Whether one broad candidate failed because every subgroup failed or because one large subgroup dominated the loss.

This is legitimate aggregate calibration, not evaluator manipulation. We must not insert evaluator-directed instructions into captions or attributes, and we must not try to exploit the LLM-assisted description protocol.

## Unknowns That Still Matter

1. **Per-transition gold tendency:** We do not know whether `content -> calm`, `calm -> content`, `content -> glad`, and `glad -> content` have the same expected gain. The 85-change submission only tells us the combined batch was bad.
2. **Which v9 rows were correct:** Some of the 85 v9 changes may still be correct. We only know the batch lowered official classification.
3. **Public-reference reliability:** Near-duplicate and same-work public references helped us reason manually, but we do not know how often official hidden labels match public labels.
4. **Description evaluator variance:** Description stayed high, but we do not know whether small VULCA text edits reliably improve the official LLM-assisted score.
5. **Leaderboard target stability:** The current first-place public score is around `0.89`, but the board can move before the phase ends.
6. **Submission opportunity value:** The page shows 2 total submissions used out of 5, so we should assume 3 total attempts remain. The daily limit also matters, so each attempt needs a clear experimental purpose.

## Strategic Decision

Do not submit `v9_vulca_entailment` as-is.

Reason:

- `v9_vulca_entailment` has the same classification labels as `781601`.
- Official classification for that label set is `0.719137`, below the old anchor `0.723150`.
- Its VULCA text edits may help description slightly, but description is already near `0.95`; text gains cannot compensate for weak emotion classification.

Instead, build a calibrated v9 subset. The goal is to retain only rows with independent evidence, while blocking large uncalibrated calm/content relabeling.

## Remaining Submission Budget

Use the 3 remaining submissions as:

| Attempt | Purpose | Candidate Type | Submit Only If |
| --- | --- | --- | --- |
| 3rd total | Stable recovery/probe | Low-risk candidate based on `779605` or `public_style_review11_vulca` | Local calibrated lower bound beats `0.836408`; description safe |
| 4th total | Controlled transition probe | One isolated transition family, preferably `content -> calm` only if evidence-backed | It tests exactly one hypothesis and has <= 20 label changes |
| 5th total | Final best | Best calibrated fusion after attempts 3 and 4 | Strongest official-calibrated expected score |

Do not spend a submission on a broad 85-change candidate again.

## Task 1: Calibrate The Local Scorer With `781601`

**Files:**
- Modify: `affectiveart/track2_local_shadow_evaluator.py`
- Modify: `affectiveart/track2_fused_shadow_evaluator.py`
- Test: `tests/test_track2_local_shadow_evaluator.py`
- Test: `tests/test_track2_fused_shadow_evaluator.py`
- Output: `experiments/track2_scorer_calibration_20260605/`

- [ ] **Step 1: Add a regression test for the v3/v9 failure**

Create a test asserting that an 85-row same-quadrant batch must not receive a positive classification delta after `781601` calibration.

Run:

```bash
python3 -m unittest tests.test_track2_local_shadow_evaluator -v
```

Expected before implementation: a failing test showing the scorer still over-rewards large same-quadrant batches.

- [ ] **Step 2: Change scoring from count-based to transition-aware**

Replace the positive same-quadrant formula with transition-family penalties and caps:

```text
calm -> content: default negative unless duplicate/human/multi-model evidence exists
content -> calm: neutral by default, positive only with evidence
content -> glad: default hold
glad -> content: default hold unless text strongly contradicts glad
happy <-> excited: neutral or hold
sad <-> tired: neutral or hold
large same-quadrant batches: uncertainty penalty grows nonlinearly
```

- [ ] **Step 3: Verify local scorer no longer ranks v9 as a high-confidence submit**

Run:

```bash
python3 scripts/track2_fused_shadow_evaluator.py score \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate moe_v2_anchor=submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate v9_vulca_entailment=submissions/track2_submission_v9_vulca_entailment_candidate.json \
  --candidate public_style_review11_vulca=submissions/track2_submission_v10_public_style_review11_vulca_fused_candidate.json \
  --out-dir experiments/track2_scorer_calibration_20260605/post_781601_fused_compare \
  --expected-row-count 1000
```

Expected: `v9_vulca_entailment` is not ranked as a safe high-confidence candidate.

## Task 2: Decompose The 85 v9 Label Changes

**Files:**
- Create: `experiments/track2_scorer_calibration_20260605/v9_transition_ablation/`
- Create: `submissions/track2_submission_v12_*_candidate.json`
- Create: `submissions/track2_submission_v12_*_candidate.zip`

- [ ] **Step 1: Export v9 transition groups**

Generate per-transition manifests for:

```text
calm_to_content
content_to_calm
content_to_glad
glad_to_content
negative_low_swaps
positive_high_swaps
```

- [ ] **Step 2: Build small ablation candidates**

Create candidates that isolate one transition family at a time:

```text
v12_content_to_calm_only
v12_no_calm_to_content
v12_public_duplicate_only
v12_human_high_confidence_only
v12_multisource_consensus_only
```

- [ ] **Step 3: Score all candidates locally**

Run the fused scorer across all ablations and write a ranking report:

```bash
python3 scripts/track2_fused_shadow_evaluator.py score \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate moe_v2_anchor=submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate public_style_review11_vulca=submissions/track2_submission_v10_public_style_review11_vulca_fused_candidate.json \
  --candidate v12_content_to_calm_only=submissions/track2_submission_v12_content_to_calm_only_candidate.json \
  --candidate v12_no_calm_to_content=submissions/track2_submission_v12_no_calm_to_content_candidate.json \
  --candidate v12_public_duplicate_only=submissions/track2_submission_v12_public_duplicate_only_candidate.json \
  --candidate v12_human_high_confidence_only=submissions/track2_submission_v12_human_high_confidence_only_candidate.json \
  --candidate v12_multisource_consensus_only=submissions/track2_submission_v12_multisource_consensus_only_candidate.json \
  --out-dir experiments/track2_scorer_calibration_20260605/v9_transition_ablation/fused_compare \
  --expected-row-count 1000
```

Expected: no candidate is recommended only because it changes many same-quadrant labels.

## Task 3: Build The Next Submission Candidate

**Files:**
- Create: `submissions/track2_submission_v12_stable_probe_candidate.json`
- Create: `submissions/track2_submission_v12_stable_probe_candidate.zip`
- Output: `experiments/track2_scorer_calibration_20260605/v12_stable_probe/`

- [ ] **Step 1: Select base**

Use this base unless the calibrated scorer proves another candidate is safer:

```text
submissions/track2_submission_v10_public_style_review11_vulca_fused_candidate.json
```

- [ ] **Step 2: Apply only high-evidence label changes**

Accept a label change only when at least two of these evidence classes agree:

```text
public duplicate or near-duplicate support
high-confidence human review
multi-model agreement
VULCA label-text consistency
no known-dangerous transition penalty
```

- [ ] **Step 3: Validate candidate**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v12_stable_probe_candidate.json
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
git diff --check
```

Expected:

```text
validate-track2: OK
Track2 tests: pass
git diff --check: no output
```

## Task 4: Final Submit Gate For The 3rd Total Attempt

**Files:**
- Read: `experiments/track2_scorer_calibration_20260605/v12_stable_probe/`
- Read: `submissions/track2_submission_v12_stable_probe_candidate.zip`

- [ ] **Step 1: Apply scarce-budget gate**

Submit only if all conditions hold:

```text
overall lower bound >= 0.836408
classification expected does not depend on same-quadrant count alone
description expected >= 0.945
cross-quadrant changes = 0
label changes <= 20 unless every change has explicit high-confidence evidence
safety_issue_codes is empty
VULCA pseudo-understanding risk does not increase
```

- [ ] **Step 2: Present exact upload path**

Use this exact text:

```text
Upload only this ZIP if you confirm spending one Codabench attempt:
/Users/yhryzy/dev/emoart-130k/submissions/track2_submission_v12_stable_probe_candidate.zip
```

Do not upload or click submit without explicit confirmation of the exact ZIP.

## Recommended Conversation Decision

The next discussion should choose which online probe philosophy to use:

1. **Conservative probe:** Submit `v12_stable_probe` first to try to beat `0.836408` safely.
2. **Diagnostic probe:** Submit one isolated transition family to learn official scorer tendency, accepting that it may not improve rank.
3. **Final-score-first:** Hold submissions until local calibration produces a candidate with both expected and lower-bound improvement.

Given only 3 attempts remain, the recommended choice is **Conservative probe first**, then **Diagnostic probe only if the conservative probe gives useful positive signal**.

