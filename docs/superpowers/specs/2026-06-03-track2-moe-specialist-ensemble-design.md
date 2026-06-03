# Track2 MoE-Like Specialist Ensemble Design

**Date:** 2026-06-03

**Goal:** Build a Track2-only, MoE-like specialist ensemble that improves classification decisions only where local evidence is strong, while preserving the current safe submission and protecting Description Score.

## Scope

This design is limited to AffectiveArt Challenge Track2. It does not touch Track1, does not overwrite `submissions/track2_submission.json`, and does not overwrite `submissions/track2_submission.zip`.

The current safe baseline remains:

- `submissions/final_track2_20260602_scaled_human_gate_gemini35_guarded_v1.json`
- `submissions/final_track2_20260602_scaled_human_gate_gemini35_guarded_v1.zip`

All new outputs must be written under:

- `experiments/track2_moe_specialist_ensemble_20260603/`
- optional candidate files named `submissions/final_track2_20260603_moe_specialist_*.json/.zip`

## Motivation

Naive full-public training is not strong enough to become the final Track2 labeler. The strongest local single-backbone public results are still low macro-F1:

- SigLIP2 full public embedding: holdout accuracy `0.4659`, macro-F1 `0.2497`
- CLIP full public embedding: macro-F1 `0.2415`
- DINOv2 full public embedding: macro-F1 `0.2181`
- I-JEPA medium smoke: macro-F1 `0.0364`

The best local hard-case evidence is a selective fusion strategy with hard96 macro-F1 `0.4004`. This means trained models are useful as proposers and reviewers, but not safe as fully automatic replacement labelers.

The official Track2 score is also not classification-only. It combines classification and Description Score with equal weight. Any label change that makes the caption, attribute analysis, valence, or arousal inconsistent can lose score even if the emotion label is better.

## Design Choice

Use a controlled MoE-like system, not an end-to-end MoE.

An end-to-end MoE would train a shared trunk, multiple experts, and a learned router. That is risky here because the hidden test set has no labels, the public labels are highly imbalanced, and a free router can overfit public long-tail quirks or collapse into head classes.

Instead, use an external specialist ensemble:

1. Each expert is trained or computed independently.
2. A deterministic gate decides when an expert is allowed to propose a change.
3. A change is accepted only if it passes classification, valence/arousal, distribution, duplicate/leakage, and description-alignment checks.

## Expert Roles

### Global Visual Expert

Uses existing cached SigLIP2, CLIP, and DINOv2 evidence. It provides broad visual priors and per-class probabilities.

It should not directly override labels. Its role is to identify low-margin rows, model agreement, and strong disagreement with the current baseline.

### Tail Rescue Expert

Focuses on minority or under-recalled classes:

- `glad`
- `aroused`
- `bored`
- `tired`
- `annoyed`
- `alarmed`

This expert may use class-balanced public training, weighted loss, calibrated logistic heads, or Vertex-trained image classifiers. It must report both macro-F1 gain and accuracy loss on public validation. Tail rescue is rejected if it improves minority recall by broadly damaging head-class accuracy.

### Boundary Expert

Handles the main subjective boundaries:

- `calm` / `content` / `glad`
- `frustrated` / `aroused` / `annoyed` / `alarmed`
- `sad` / `bored` / `tired`
- `excited` / `happy` / `aroused`

This expert is the first priority because most current uncertainty is boundary ambiguity rather than fully new class discovery.

### Valence/Arousal Expert

Checks whether the proposed emotion is compatible with binary valence and arousal. It may veto an emotion change but must not force broad emotion rewrites by itself.

The gate rejects any candidate row with emotion, valence, and arousal inconsistency.

### Description/VLM Expert

Uses Gemini 3.5 Flash and Vulca/vulca-emnlp as post-hoc reviewers for:

- artwork-caption consistency;
- attribute specificity;
- whether the text matches the proposed emotion;
- whether a proposed label change would make the existing caption misleading.

This expert is required because Description Score is 50% of the official score. If a label changes but the text does not remain aligned, the change is held.

## Gate Inputs

The first implementation should process only high-risk or high-yield rows:

- 56 clean/inclusive disagreement rows from `experiments/track2_final_push_20260603/clean_inclusive_disagreement/`
- scaled human review accept/hold rows;
- Gemini 3.5 rollback rows;
- low-margin rows from supervised sources;
- rows with emotion/valence/arousal conflicts;
- rows where multiple public-trained models disagree;
- suspected duplicate or near-duplicate rows, but only as leakage evidence, not as answer copying.

The gate should not run over all 1000 rows as a free automatic relabeler until public validation proves that doing so improves both macro-F1 and accuracy without category collapse.

## Acceptance Rules

A proposed label change can enter a challenger candidate only when all of these are true:

1. At least two independent sources support the proposed emotion, or one specialist source supports it with very high confidence and no strong opposing source.
2. The proposed emotion, valence, and arousal are internally consistent.
3. The top emotion share and top-two emotion share do not increase into category collapse.
4. No emotion class disappears from the 1000-row candidate.
5. Public validation shows a plausible net benefit in macro-F1 without unacceptable accuracy loss.
6. Description/VLM audit finds no obvious text-emotion contradiction.
7. Suspected public/test overlap evidence is not used to copy public labels into the answer.
8. The Track2 validator passes.

Default behavior is hold, not change.

### Quantitative Defaults

The first implementation should use explicit public-validation thresholds instead of subjective tuning:

- broad public validation emotion macro-F1 improves by at least `0.015`, or a documented hard-case validation slice improves by at least `0.030`;
- broad public validation emotion accuracy drops by no more than `0.010`;
- valence and arousal accuracy each drop by no more than `0.005`;
- no class with validation support of at least five examples has zero recall after the change;
- candidate top-emotion share does not increase by more than two percentage points over the current safe baseline.

These thresholds are launch gates for a challenger candidate, not proof that the candidate will win. If a threshold fails, the gate may still produce a diagnostic report, but it must not generate a submit-ready candidate.

## Rejection Rules

Reject or hold a proposed change when any of these occur:

- it changes only because one public-trained model predicts a different label;
- it converts many calm/content rows into one minority class without strong visual evidence;
- it improves emotion but breaks valence or arousal;
- it makes caption or attribute fields inconsistent with the submitted label;
- it relies on a suspected public duplicate label as direct ground truth;
- it comes from a model with known category collapse on public validation;
- it touches the formal submission path.

## Data Flow

1. Normalize existing expert predictions into a common row schema: `sample_id`, `source`, `emotion`, `confidence`, `margin`, `top3`, optional `class_scores`.
2. Build a gate queue from hard-case sources.
3. Attach expert evidence to each queued row.
4. Run deterministic gate rules and produce one decision per row: `accept_change`, `keep_current`, `hold`, or `needs_description_rewrite`.
5. For accepted changes, verify valence/arousal and text consistency.
6. Write a side-path challenger JSON/ZIP only if all final gates pass.
7. Render an HTML review packet for changed or held rows before any formal submission decision.

## Dry-Run First

The first executable milestone should be dry-run only. It should write:

- a normalized expert-evidence matrix;
- one gate decision per queued sample;
- a Markdown/JSON report with acceptance and rejection reasons;
- an HTML review packet for human inspection.

Dry-run mode must not write any submission JSON or ZIP. Candidate generation should be a separate explicit command after the dry-run report looks coherent.

## Local First, Vertex Second

The first milestone should run locally using existing cached embeddings and current scripts. Vertex or larger Google spend should start only after the local gate proves that expert fusion improves public validation or clearly identifies a small set of high-value rows.

Recommended order:

1. Local cached expert matrix and gate report.
2. Local boundary and tail specialist heads.
3. Description/VLM audit on the rows the gate wants to change.
4. Side-path challenger candidate and HTML review.
5. Vertex training only if local results show a credible gain over the current safe candidate.

## Validation

Run only Track2 checks:

```bash
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
python3 -m affectiveart.challenge validate-track2 submissions/final_track2_20260603_moe_specialist_candidate.json
git diff --check
```

For design-only changes, `git diff --check` is sufficient.

## Success Criteria

The design is ready for implementation when:

- the gate can explain every accept/hold/reject decision;
- no formal submission file is modified;
- the current safe candidate remains the fallback;
- Description Score protection is a first-class gate;
- local public validation is used before Vertex spending;
- the implementation plan is narrow enough to build and verify in Track2-only tests.

## Non-Goals

- Do not build a full end-to-end MoE model first.
- Do not run automatic full-test relabeling as the next step.
- Do not use public duplicate labels as official test answers.
- Do not optimize only macro-F1 while ignoring accuracy and Description Score.
- Do not let Gemini or any single model become the final arbiter without deterministic gates.

## Research Anchors

This design follows current long-tail and emotion-recognition trends without copying a heavy architecture too early:

- IJCAI 2025 class-wise adaptation and mixture-of-experts for long-tail visual recognition.
- CVPR 2025 dynamic emotion experts for emotion recognition.
- CVPR 2025 cross-domain visual emotion recognition.
- NeurIPS 2025 two-stage and model-rebalancing approaches for long-tail recognition.
- ICCV 2025 multi-granularity semantics for long-tailed classification.
- ICLR 2026 long-tailed distribution-aware routing for LVLM MoE as a conceptual reference, not an implementation dependency.
