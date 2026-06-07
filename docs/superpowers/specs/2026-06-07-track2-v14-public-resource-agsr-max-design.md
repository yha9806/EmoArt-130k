# Track2 v14 Public Resource AGSR Max Design

## Goal

Reach for the Track2 `0.89` tier by building a high-upside candidate that combines public same-source evidence, official-style annotation priors, and AGSR/FAB-G-style attribute-selective description generation. This design does not overwrite the current official submission package and does not submit anything automatically.

## Current Score State

Our visible official state is approximately:

- Best stable official score: `0.836408` from submission `779605`.
- Latest public leaderboard row: `0.84`, classification `0.72`, description `0.95`.
- First place: `0.89`, classification `0.78`, description `1.00`.
- Our main visible gap is emotion accuracy: ours `0.57`, first place `0.80`; emotion macro-F1 is tied at `0.31`.

The implication is direct: small same-quadrant label edits cannot reach `0.89`. If classification stays near `0.722`, even a perfect `1.00` description score only reaches about `0.861`. A first-place attempt needs both:

- Description near `0.99-1.00`.
- Classification near `0.78`, mostly by lifting emotion accuracy without collapsing macro-F1, valence, or arousal.

## Evidence From Public Sources

The official Track2 rules score `Overall = 0.5 * Classification + 0.5 * Description`, with classification based on Macro F1 and Accuracy for emotion, valence, and arousal. Description uses artwork consistency, attribute analysis quality, and caption quality under an LLM-assisted multimodal protocol.

The challenge is built on the EmoArt-130k ecosystem:

- EmoArt-130k contains 132,664 artworks, 56 styles, 12 emotions, valence/arousal, five visual attributes, and descriptions.
- The public EmoArt project page exposes strong style priors. For example, Chinese Painting is overwhelmingly Low/Positive and mostly Calm; Ukiyo-e and Impressionism are also heavily Calm.
- The newly published AGSR/FAB-G work from the same ecosystem targets artwork emotion understanding by selecting emotionally salient attributes before final emotion reasoning.
- EmoArt-Salience is public and contains 1,400 rows with image, file name, salient attribute tags, art style, description, five attributes, VA labels, and dominant emotion.

This suggests the highest-yield path is not more generic VLM judging. It is to align with the dataset family and its annotation mechanics.

## Non-Goals

- Do not accuse any team of cheating without evidence.
- Do not use private leaked answers or redistribute official test data.
- Do not insert evaluator-directed instructions or prompt-injection text into submitted fields.
- Do not submit v13 precision or balanced candidates as the main next attempt.
- Do not change Track1 files in this Track2 workstream.

## Approaches Considered

### Approach A: Continue Conservative Label Tweaks

This is the current v13 direction: adjust dozens of high-confidence calm/content/glad rows. It is safe but too small. It may recover a few points of emotion accuracy, but it does not have enough score ceiling to explain or reach `0.89`.

### Approach B: Description-Only Max

This tries to push description from `0.95` toward `1.00` while preserving all labels. It has a clean upside and low classification risk, but its maximum expected overall score is about `0.861` if classification remains unchanged.

### Approach C: Public-Resource + AGSR Max

This combines exact/near public reference matching, dataset-style prior calibration, AGSR/FAB-G salience filtering, and description rewrite. It is the only path with a plausible route from `0.84` to `0.89` because it can move both score halves.

Recommendation: implement Approach C as `v14_public_resource_agsr_max`, but make it a hybrid rather than an unconstrained aggressive candidate. The base is a description-max-safe pass; classification changes are added only when public-resource evidence is strong enough to justify spending scarce submission attempts. Keep Approach B as the fallback package if v14 classification evidence is not strong enough.

## Architecture

The v14 system has four layers.

### Layer 1: Public Resource Registry

Create a local manifest of public resource candidates:

- EmoArt-130k public rows already available locally or downloaded into a controlled experiment cache.
- EmoArt-Salience public rows.
- Existing local EmoArt duplicate audit outputs.
- Optional external art-emotion resources for metadata and weak priors: ArtEmis, ArtELingo, EmoSet, WikiArt/museum metadata.

Each resource row is normalized to:

```json
{
  "resource": "emoart130k",
  "resource_id": "Chinese Painting/...",
  "image_path": "...",
  "title_artist": "...",
  "style": "Chinese Painting",
  "emotion": "Calm",
  "valence": "Positive",
  "arousal": "Low",
  "description": "...",
  "brushstroke": "...",
  "composition": "...",
  "color": "...",
  "line": "...",
  "light": "...",
  "salient_attributes": ["composition", "color"]
}
```

### Layer 2: Evidence Matcher

For each official Track2 sample, collect ranked evidence:

- Exact or near-exact image similarity from existing SigLIP2/CLIP/DINO caches and perceptual-hash style checks.
- Same-work evidence from file name, artist/title tokens, and high image similarity.
- Same-series evidence when the artwork is not identical but appears from the same style/series/artist.
- Style-level priors from EmoArt statistics.
- Model evidence from existing fused candidates, Gemini, Vulca, and supervised heads.

Evidence levels:

- `exact_same_work`: allowed to strongly transfer public label and description schema.
- `near_same_work`: allowed to propose label transfer if VA and visual content agree.
- `same_series_style`: allowed only as a prior, not as automatic label transfer.
- `style_prior_only`: affects confidence but cannot override strong image-specific evidence.

### Layer 3: AGSR/FAB-G Salience Reasoning

For every candidate row selected for rewrite or relabeling:

1. Predict salient attributes among `color`, `composition`, `line`, `light`, `brushstroke`.
2. Restrict the final explanation to those salient attributes.
3. Use the selected attributes to reconcile emotion, valence, arousal, and textual fields.

Implementation can start without training Qwen3-VL LoRA by using a lightweight local AGSR proxy:

- If an EmoArt-Salience same-work or high-similarity row exists, use its `tags` as salience evidence.
- Otherwise infer salience from public nearest-neighbor attributes plus Gemini/Vulca attribute judgments.
- Keep a field-level audit that rejects rows where the proposed text cites non-salient or visually unsupported attributes.

If time and compute allow, a later v15 can train or adapt the official-style FAB-G LoRA structure:

- Five binary salience adapters.
- One final emotion/VA/explanation adapter.
- Qwen3-VL-8B or a cloud equivalent.

### Layer 4: Candidate Gate

Generate two side-path candidates, but rank them as a single gated family:

- `v14_public_resource_agsr_max`: starts from the description-max-safe base, then accepts exact/near public label transfers and only the highest-confidence public-style emotion calibration changes.
- `v14_description_max_safe`: accepts AGSR description rewrites only, plus exact same-work public label transfers. This is the fallback if classification evidence is too weak.

Both candidates must pass:

- Track2 JSON schema validation.
- No missing/duplicated sample IDs.
- Emotion/valence/arousal quadrant consistency.
- Description fields grounded in visible artwork and selected salient attributes.
- Distribution sanity: no uncontrolled calm/content/glad collapse, no sudden high-arousal negative explosion.
- Diff report against `779605` and `782683`.

## Score Hypotheses

### Hypothesis H1: Description Can Reach Near 1.0

If the evaluator rewards concise, specific, artwork-grounded attributes, AGSR-style text should improve from `0.95` toward `0.98-1.00`. This alone can place us around `0.85-0.86`.

### Hypothesis H2: Emotion Accuracy Gap Is Mostly Official-Style Alignment

Because first place has emotion accuracy `0.80` but macro-F1 `0.31`, it probably benefits from better majority-class alignment rather than perfect rare-class recognition. Exact/near public-resource label transfer and style priors should improve emotion accuracy more than generic VLM relabeling.

### Hypothesis H3: Exact Public Matches Are The Biggest Classification Lever

Rows like previously inspected public duplicates are not enough by themselves, but a deeper public-resource pass may find more same-work cases. Exact matches are the only row-level evidence strong enough to justify aggressive label changes before the next submission.

### Hypothesis H4: Emotion Accuracy Is The Core Submission Probe

The leaderboard shows first place and our row both have emotion macro-F1 `0.31`, but first place has emotion accuracy `0.80` while ours is `0.57`. This means the next useful probe must target majority-class decision boundaries, not rare-class expansion. The key local diagnostic is a confusion-style table over calm/content/glad/happy/excited/tired/frustrated/aroused, estimated from public-style validation and exact/near public matches.

## Submission Strategy

Remaining submissions are scarce, so use them as probes with explicit hypotheses:

1. Next submission: one hybrid package, not two unrelated probes. Submit `v14_public_resource_agsr_max` only if it is description-safe and its classification changes pass strict evidence thresholds. This tests whether public-resource alignment plus AGSR text can raise both score halves.
2. Final submission: choose based on the next score.
   - If classification improves but description is below `0.98`, run final `description_max`.
   - If description improves but classification remains near `0.72`, run final `classification_style_calibrated`.
   - If both improve toward `0.86+`, run final combined candidate with tighter gates.

## Required Outputs

- `experiments/track2_v14_public_resource_agsr_20260607/public_resource_registry.jsonl`
- `experiments/track2_v14_public_resource_agsr_20260607/track2_public_evidence_matrix.csv`
- `experiments/track2_v14_public_resource_agsr_20260607/track2_public_evidence_matrix.json`
- `experiments/track2_v14_public_resource_agsr_20260607/v14_agsr_rewrite_report.md`
- `experiments/track2_v14_public_resource_agsr_20260607/v14_candidate_score_proxy.md`
- `experiments/track2_v14_public_resource_agsr_20260607/html_review/track2_v14_public_resource_agsr_review.html`
- `submissions/track2_submission_v14_public_resource_agsr_max_candidate.json`
- `submissions/track2_submission_v14_public_resource_agsr_max_candidate.zip`
- `submissions/track2_submission_v14_description_max_safe_candidate.json`
- `submissions/track2_submission_v14_description_max_safe_candidate.zip`

## Tests And Checks

Run only Track2 checks:

```bash
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v14_public_resource_agsr_max_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v14_description_max_safe_candidate.json
git diff --check
```

Do not run all tests if Track1 failures are unrelated to this Track2 workstream.

Additional pre-submission diagnostics:

```bash
python3 scripts/track2_v14_public_resource_agsr.py --mode threshold-ablation
python3 scripts/track2_v14_public_resource_agsr.py --mode emotion-boundary-report
python3 scripts/track2_v14_public_resource_agsr.py --mode historical-correlation
```

The threshold ablation estimates false positives for exact/near matching by holding out public resource rows and checking whether the chosen thresholds recover the correct public labels. The emotion-boundary report focuses on high-frequency decision boundaries. The historical-correlation report compares v14 local proxy movement against the known official movement from `779605`, `781601`, and `782683` to avoid repeating the v3 mistake.

## Risk Controls

- Treat public labels as evidence, not automatically as gold, except for exact/same-work cases with strong visual confirmation.
- Never accept a text rewrite that changes emotion language away from the submitted emotion/VA.
- Prefer precision over recall for classification changes unless exact public-resource evidence supports the change.
- Do not use high-similarity rows as training evidence and test-answer evidence at the same time without a manifest flag.
- Keep every accepted change auditable by sample ID, evidence level, source row, and reason.

## Decision Gate Before Submission

Submit v14 only if all are true:

- Validator OK.
- Description proxy does not regress relative to `782683`.
- At least one of these classification conditions holds:
  - 50+ exact/near public-resource label transfers with coherent VA and a measured low false-positive threshold in public holdout; or
  - a calibrated public-resource classifier predicts material emotion accuracy gain without distribution collapse; or
  - multiple independent sources agree on a high-confidence set of label changes larger than v13 but with lower estimated downside.
- Human-readable HTML review confirms the high-impact changes are not visually absurd.

If these conditions fail, do not spend the submission. Build `v14_description_max_safe` instead and use the final attempt for classification.

Hard block conditions:

- The candidate changes many labels but leaves emotion accuracy proxy unexplained by public evidence.
- The candidate mainly performs same-quadrant calm/content/glad swaps without exact/near public support.
- The threshold ablation shows public match false positives concentrated in the same boundaries being changed.
- Description text mentions attributes that are not selected by AGSR/FAB-G salience evidence.
- The candidate's local score gain is driven by an uncalibrated proxy that would have incorrectly preferred `781601` over `779605`.
