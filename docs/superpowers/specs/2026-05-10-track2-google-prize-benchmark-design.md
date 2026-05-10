# Track2 Google Prize Benchmark Design

**Date:** 2026-05-10

**Goal:** Build a Track2-only benchmark gate that uses Google models and cloud-ready artifacts to decide whether a Google/VLM route can beat the current supervised baseline before we spend Vertex AI training budget or replace the current submission.

## Scope

This work is limited to AffectiveArt Track2. It does not modify Track1 generation, does not regenerate the current Track2 submission, and does not overwrite `submissions/track2_submission.json` or `submissions/track2_submission.zip`.

The benchmark uses public EmoArt-130k annotations for validation and test-set artifacts only for optional inference ledgers. Public validation is the decision gate. Hidden Track2 test predictions are never used as training labels, pseudo-labels, or validation feedback.

## Current Baseline

The current local baseline is a strict ensemble built from CLIP, SigLIP2, and DINOv2 evidence. The best single supervised model recorded locally is SigLIP2 with holdout accuracy `0.4659`, macro-F1 `0.2497`, and weighted-F1 `0.5324`.

The benchmark must report whether each Google route beats this reference on a deterministic public validation split. A model is not eligible for expensive training or final submission changes unless it improves macro-F1 and does not collapse minority classes.

## Data Inputs

- Public annotations: `/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k/Annotation.json`
- Public image archives: `/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k/*.tar.gz`
- Test set ZIP: `/Users/yhryzy/dev/emoart-130k/data/raw/Track2_testset.zip`
- Current safe submission: `/Users/yhryzy/dev/emoart-130k/submissions/track2_submission.json`
- Existing leakage exclusion: `/Users/yhryzy/dev/emoart-130k/experiments/track2_emoart130k_clip/suspected_public_test_overlap.json`

The public split builder excludes suspected public/test overlaps before sampling. It records exact request IDs, styles, labels, and file locations so later runs are reproducible.

## Validation Splits

The benchmark creates two deterministic validation views with seed `20260510`.

1. **Style-stratified validation:** Samples across all available styles while preserving emotion coverage as much as possible. This measures broad performance against the public distribution.
2. **Leave-style-out validation:** Holds out selected styles from training/evidence selection and evaluates only those styles. This measures cross-style generalization, which matters because the official hidden set is distributionally distinct from public training data.

Each split stores a manifest JSONL containing `request_id`, `style`, `emotion`, `valence`, `arousal`, `tar_path`, and `member`. The split manifest is the source of truth for all model runs.

## Google Model Roles

- `gemini-3-flash-preview`: primary high-throughput VLM judge for public validation and optional full Track2 inference.
- `gemini-3.1-pro-preview`: expensive arbiter for hard examples, disagreement cases, minority classes, and final audit samples.
- `gemini-3.1-flash-lite-preview`: cheap quality audit model for caption/attribute/emotion self-consistency.
- `gemini-embedding-2`: multimodal embedding backbone for kNN evidence, clustering, style retrieval, and category-collapse detection.

Gemini 3.x is treated as inference-time teacher/reviewer, not as the primary fine-tuned student. Vertex Gemini supervised tuning currently supports Gemini 2.5-series models, so the later training route should target open students such as Gemma 4 or Qwen3-VL if the benchmark justifies it.

## Architecture

The feature adds a small benchmark layer beside the existing Track2 code.

- `affectiveart/track2_public_eval.py` owns public split construction, label normalization, metric calculation, per-class breakdown, per-style breakdown, and report serialization.
- `affectiveart/track2_google_benchmark.py` owns Gemini request construction, response normalization into the existing Track2 schema, JSONL ledgers, retry-safe resume behavior, and result aggregation.
- `scripts/track2_public_eval.py` exposes split creation and offline metric evaluation.
- `scripts/track2_google_benchmark.py` exposes Google model runs against a split manifest or the Track2 test ZIP.
- Tests cover deterministic splitting, leakage exclusion, metric math, Gemini response normalization, and no-overwrite behavior for current submissions.

This keeps the existing `affectiveart/gemini_track2.py` intact for direct Track2 inference while adding a public-validation gate that can compare multiple Google models consistently.

## Data Flow

1. Build a public validation split manifest from EmoArt-130k.
2. Run one or more Google models against the split and append normalized predictions to a model-specific JSONL ledger.
3. Evaluate each ledger against public labels using emotion accuracy, emotion macro-F1, emotion weighted-F1, valence accuracy/macro-F1, arousal accuracy/macro-F1, per-class recall/F1, and per-style macro-F1.
4. Create a benchmark report under `/Users/yhryzy/dev/emoart-130k/experiments/track2_google_prize_benchmark/`.
5. Optionally run the same prompt over Track2 test images to produce an inference ledger, but do not convert it into a final candidate unless a later implementation plan adds an explicit reviewed candidate step.

## Output Files

The benchmark writes only experiment artifacts:

- `/Users/yhryzy/dev/emoart-130k/experiments/track2_google_prize_benchmark/splits/*.jsonl`
- `/Users/yhryzy/dev/emoart-130k/experiments/track2_google_prize_benchmark/predictions/*.jsonl`
- `/Users/yhryzy/dev/emoart-130k/experiments/track2_google_prize_benchmark/reports/*.json`
- `/Users/yhryzy/dev/emoart-130k/experiments/track2_google_prize_benchmark/reports/*.md`

It must not write to `/Users/yhryzy/dev/emoart-130k/submissions/track2_submission.json` or `/Users/yhryzy/dev/emoart-130k/submissions/track2_submission.zip`.

## Error Handling

The Gemini runner appends each successful prediction immediately. If a process stops, the next run reads the ledger and skips completed IDs.

API failures are retried with bounded attempts. Quota and rate-limit errors are reported with the model name, sample ID, and any retry delay returned by the API. Invalid JSON responses are saved as raw failure records in a sidecar error JSONL so they can be reviewed without losing progress.

The evaluator refuses to score incomplete ledgers unless `--allow-partial` is explicit. Partial scoring reports the number and IDs of missing predictions.

## Privacy and Leakage Controls

Public EmoArt examples are safe for training and validation under the dataset license and EULA. Registered Track2 test images are private challenge material. Test images may be sent to Google APIs only if the team accepts that private cloud inference under the team account satisfies the challenge restriction against sharing or distributing the data.

The benchmark does not train on test images. It does not add test predictions back into public training data. It records the exclusion list used for suspected public/test overlaps in every report.

## Success Criteria

The first benchmark milestone is complete when:

- A deterministic public validation split can be regenerated byte-for-byte from the same inputs and seed.
- Offline metric tests pass for known prediction fixtures.
- A small Gemini smoke run on public validation produces a valid JSONL ledger and metric report.
- Reports include enough detail to compare against SigLIP2 macro-F1 `0.2497`.
- No current submission file is modified.

The route advances to Vertex training only if at least one Google/VLM path improves public validation macro-F1 over the current SigLIP2 reference and shows no obvious class collapse in per-class/per-style breakdowns.

## References

- Gemini model list: https://ai.google.dev/gemini-api/docs/models
- Gemini pricing and Batch/Flex tradeoffs: https://ai.google.dev/gemini-api/docs/pricing
- Gemini Batch API behavior: https://ai.google.dev/gemini-api/docs/batch-api
- Gemini media resolution: https://ai.google.dev/gemini-api/docs/media-resolution
- Gemini supervised tuning support: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning
- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
- AffectiveArt 2026 proposal: https://openreview.net/pdf?id=LbbHX8ofXZ
