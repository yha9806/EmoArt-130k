# Track2 v18 Champion-Shaped Hybrid Design

## Goal

Build the next Track2 candidate under a hard budget of two remaining Codabench submissions.

The v18 candidate must be a real score attempt, not a pure probe. It should combine:

- description-score maximization, because first place is near perfect on all three description sub-scores;
- high-confidence emotion-accuracy calibration, because our largest visible gap is 12-way emotion accuracy;
- strict gates that prevent another wide label-change regression like `781601` or the locally rejected v17 candidates.

## Current Official Evidence

Current public leaderboard snapshot:

| Metric | First place `N&T小分队` | Current `vulcaart` |
| --- | ---: | ---: |
| Overall | 0.89 | 0.84 |
| Classification | 0.78 | 0.72 |
| Description | 1.00 | 0.95 |
| Visual Grounding | 0.99 | 0.96 |
| Attribute Specificity | 1.00 | 0.95 |
| Overall Caption | 0.99 | 0.94 |
| Emotion Accuracy | 0.80 | 0.57 |
| Emotion Macro F1 | 0.31 | 0.31 |
| Valence Accuracy / Macro F1 | 0.90 / 0.86 | 0.88 / 0.82 |
| Arousal Accuracy / Macro F1 | 0.94 / 0.87 | 0.91 / 0.84 |

Known official anchors:

- `779605`: overall `0.836408`, classification `0.723150`, description `0.949667`.
- `781601`: overall `0.834027`, classification `0.719137`, description `0.948917`; it changed 85 same-quadrant emotion labels and scored worse.
- `782683`: page score `0.84`, classification about `0.72`, description about `0.95`; it did not change classification labels.

Interpretation:

- Description is strong but not champion-level. There may be about `0.02-0.025` overall headroom if description approaches `0.99-1.00`.
- Classification must also improve to reach `0.89`. Text-only improvement is not enough unless the leaderboard score is heavily rounded or the evaluator variance is unusually favorable.
- The critical classification weakness is emotion accuracy, not emotion macro-F1. First place has similar emotion macro-F1 but much higher emotion accuracy.
- Same-quadrant changes are not automatically safe. Official 12-way emotion accuracy penalizes `calm/content/glad/happy` swaps even when valence and arousal are unchanged.

## Non-Goals

- Do not touch Track1.
- Do not overwrite formal `submissions/track2_submission.json` or `submissions/track2_submission.zip`.
- Do not submit automatically.
- Do not spend a Codabench attempt on a pure diagnostic package.
- Do not create broad v17-style classification rewrites.
- Do not insert evaluator-facing instructions or manipulative content.
- Do not copy public labels from broad style similarity. Public references are evidence, not hidden-test gold.

## Recommended Approach

### A. Pure Description Max

This is safe but not enough. It can possibly move overall from about `0.84` to `0.86`, but likely cannot reach `0.89` because classification would remain around `0.72`.

### B. Wide Classification Calibration

This has the necessary upside but is too risky with only two submissions. The official `781601` result and v17 local gate both show that batch emotion changes can reduce score even when they look plausible locally.

### C. Champion-Shaped Hybrid

This is the recommended v18 route.

Use the current best official-safe anchor as the base. Apply a description-max pass broadly, then apply only a small to moderate set of emotion changes that are supported by multiple evidence sources and are shaped toward the visible first-place profile:

- higher emotion accuracy;
- similar or non-degraded emotion macro-F1;
- slightly stronger valence and arousal;
- description near `0.99`.

## Architecture

v18 has six components.

### 1. Official Anchor Loader

Inputs:

- stable anchor candidate JSONs and official score records;
- public leaderboard CSV and derived score table;
- pairwise diffs for `779605`, `781601`, `782683`, `v15`, `v16`, and `v17`.

Output:

- selected base JSON;
- score targets and guard thresholds.

Base selection rule:

- Prefer the highest exact official anchor if classification labels are unchanged and the file is available.
- Use `v15_desc_expand300` only if its text is strictly preferred and the classification labels match the highest official anchor.

### 2. Description-Max Rewriter

Scope:

- `overall_caption`
- `brushstroke`
- `composition`
- `color`
- `line`
- `light`

The rewriter should improve specificity and naturalness without changing labels. It should prioritize rows that currently limit:

- Visual Grounding;
- Attribute Specificity;
- Overall Caption.

Text constraints:

- concise, image-grounded, no boilerplate;
- no evaluator instructions;
- no unsupported named artist claims unless the artwork identity is already known from project evidence;
- avoid repeating the same wording across many samples.

### 3. Emotion-Accuracy Evidence Matrix

Inputs:

- public duplicate and near-duplicate audits;
- clean/inclusive public-style model outputs;
- SigLIP2, CLIP, DINOv2, stacked head, hard96, MoE, and RAG teacher predictions where available;
- Gemini/Vulca audit signals as reviewers, not sole labelers;
- transition outcomes from failed official submissions.

Output:

- one row per proposed label change with fields for current label, proposed label, valence/arousal, model votes, public-reference evidence, transition family, risk flags, and expected effect.

The matrix must explicitly separate:

- exact or near-exact same artwork evidence;
- same-series evidence;
- model consensus;
- text-emotion inconsistency evidence;
- official-feedback penalties.

### 4. Champion-Shape Gate

Accept a classification change only if it passes all required gates:

- no schema or label consistency issue;
- no unsupported cross-quadrant jump;
- no top-emotion collapse increase;
- no large `calm/content/glad` reshuffle;
- at least two independent non-Gemini visual/model evidence families support it, or exact/near-exact reference evidence supports it;
- not in a transition family that official feedback already penalized, unless evidence is exact/near-exact or unusually strong.

The gate should prefer transitions that may improve emotion accuracy without sacrificing macro-F1:

- `calm -> content`
- `content -> calm`
- `content/calm -> glad`
- `happy <-> excited`
- `aroused <-> excited`
- `sad <-> tired`
- `annoyed <-> frustrated`

These are candidates, not automatic rules. Each row still needs evidence.

### 5. Local Scoring And Ranking

Run both local scorers, but treat them as risk controls, not official replicas:

- local classification scorer;
- fused shadow scorer.

The v18 report must show:

- official-anchor comparison;
- classification proxy;
- emotion-accuracy proxy;
- description proxy;
- number of changed labels;
- number of changed text rows;
- transition counts;
- distribution shift.

Submission gate:

- validator OK;
- label consistency issue count = 0;
- no missing emotions;
- description proxy not below the current anchor;
- classification lower bound not below the current anchor by more than a small documented tolerance;
- emotion-accuracy proxy must improve materially over the current anchor;
- manual/Vulca/Gemini review sample has no obvious semantic drift.

### 6. Two-Submission Decision Policy

There are only two remaining Codabench submissions.

First remaining submission:

- submit v18 only if it is both a plausible score improvement and a controlled official-feedback experiment;
- do not submit a pure probe;
- do not submit a package whose local lower bound is materially below the current best official anchor.

Final remaining submission:

- if v18 improves description but not classification, use the last attempt for stronger classification micro-calibration;
- if v18 improves classification but drops description, use the last attempt for text recovery;
- if v18 improves both, expand the same evidence families conservatively;
- if v18 drops overall, revert to the best exact official anchor and only apply near-zero-risk text fixes.

## Data Flow

```text
official anchors + leaderboard breakdown
  + base submission JSON
  + public/reference evidence
  + model prediction files
  + Gemini/Vulca review signals
      -> v18 evidence matrix
      -> description-max candidate
      -> champion-shape classification gate
      -> v18 candidate JSON/ZIP
      -> validate-track2
      -> local/fused shadow scoring
      -> HTML/Markdown review
      -> submit/no-submit decision
```

## Expected Output Paths

- Design: `docs/superpowers/specs/2026-06-07-track2-v18-champion-shaped-hybrid-design.md`
- Implementation module: `affectiveart/track2_v18_champion_hybrid.py`
- CLI wrapper: `scripts/track2_v18_champion_hybrid.py`
- Tests: `tests/test_track2_v18_champion_hybrid.py`
- Experiment root: `experiments/track2_v18_champion_hybrid_20260607/`
- Candidate files:
  - `submissions/track2_submission_v18_champion_hybrid_candidate.json`
  - `submissions/track2_submission_v18_champion_hybrid_candidate.zip`

## Testing

Run only Track2-relevant checks:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v18_champion_hybrid_candidate.json
git diff --check
```

If Gemini/Vulca live audits are used, their outputs must be saved under the experiment root with model names, timestamps, and sampled row IDs. They remain reviewer evidence, not human-confirmed labels.

## Success Criteria

Minimum local success:

- candidate validates;
- no label consistency issues;
- no formal submission overwrite;
- description proxy is at least current-anchor level;
- emotion-accuracy proxy rises versus current anchor;
- no broad label distribution collapse;
- final report explains every accepted classification change.

Codabench submission recommendation:

- Submit v18 only if it is the best available candidate by the two-submission policy.
- If v18 does not clear the local gate, keep the current highest official anchor as the safest package and do not spend a submission.

