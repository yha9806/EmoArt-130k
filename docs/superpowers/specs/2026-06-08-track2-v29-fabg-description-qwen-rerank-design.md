# Track2 v29 FAB-G Description And Qwen Rerank Design

## Goal

Build the next Track2 final-shot system around the only path that can plausibly reach `0.89+` locally before the last Codabench submission: raise the Description proxy toward `0.98-1.00` while improving classification toward the public first-place shape (`classification ~= 0.78`, `description ~= 1.00`).

The system must produce side-path candidate files, a local scorer report, and a final `recommend_final_submit` or `hold_no_submit` decision. It must not upload to Codabench and must not overwrite the formal `track2_submission.json/.zip` files.

## Root-Cause Finding

The v28 approach is insufficient because it searches mostly inside the existing label-change neighborhood. That neighborhood tops out near `0.85` in local proxy scoring and cannot reach `0.89` unless classification jumps unrealistically high.

The official public leaderboard shows the first-place pattern:

- overall: `0.89`
- classification: `0.78`
- description: `1.00`
- emotion accuracy: `0.80`
- emotion macro-F1: `0.31`
- valence/arousal: both strong

Our best exact official anchor is:

- overall: `0.842559`
- classification: `0.740034`
- description: `0.945083`

The gap is therefore not a pure 12-way macro-F1 problem. The practical gap is:

1. description needs a large improvement;
2. valence/arousal need smaller but real gains;
3. emotion accuracy needs improvement without destroying macro-F1;
4. local scorer must stop treating Description Score as a nearly fixed historical constant for unsubmitted candidates.

## Public Resource Assessment

Use these resources as method evidence, not hidden labels:

- EmoArt-130k public dataset: ontology, public annotation style, visual-attribute language, class priors.
- EmoArt-Salience: 1,400 examples with salient formal attributes, emotion, valence, and arousal.
- FAB-G public code/paper: a two-stage method that first selects emotionally operative attributes, then performs cue-constrained emotion analysis.
- EmoVIT: an auxiliary visual-emotion instruction-tuning teacher; useful for review/proposals but not directly compatible with the 12-class AffectiveArt label space.
- Qwen3-VL-Embedding/Reranker: a better multimodal retrieval/reranking route than blind CLIP/SigLIP public kNN.

Do not treat near-duplicates from public EmoArt as official hidden answers unless they are already human-confirmed exact visual duplicates. Public labels are distribution and method evidence, not gold.

## Chosen Approach

v29 has three independent modules that feed a final gate.

### 1. Description-Max Scorer

Create a local Description proxy that can score beyond historical anchor constants. It should estimate:

- visual grounding;
- attribute specificity;
- overall caption quality.

The scorer should combine:

- structural lint: no evaluator-directed text, no missing fields, no copied boilerplate patterns;
- EmoArt/FAB-G style checks: each attribute field names concrete visual evidence;
- salience consistency: the strongest emotional explanation relies on a small set of operative attributes;
- optional Gemini/Vulca audit when credentials and policy allow.

This scorer is not the official evaluator, but it must rank candidates by the same observable rubric rather than inheriting old official Description values.

### 2. FAB-G-Lite Description Rewrite

Rewrite Track2 text fields using a salience bottleneck:

- predict or infer salient attributes among brushstroke, composition, color, line, light;
- keep every text field grounded in visible image evidence;
- make the caption naturally summarize the artwork's emotional atmosphere;
- avoid score-seeking language and evaluator instructions;
- keep label fields unchanged unless the classification module explicitly changes them.

The rewrite must be side-path only and reversible. It should produce:

- JSON candidate;
- ZIP candidate;
- per-row text-diff report;
- unsafe-text lint report;
- Description proxy report.

### 3. Qwen-Rerank Classification Evidence

Replace blind public kNN with a reranked evidence table:

- retrieve public EmoArt candidates from existing local public inventory or cached embeddings;
- rerank with a multimodal model/prompt that considers image similarity, style, visible content, visual attributes, emotion, valence, and arousal;
- use public references as evidence votes, not direct answers;
- prefer changes that improve valence/arousal and emotion accuracy while preserving rare-class coverage.

If Qwen3-VL-Embedding/Reranker cannot run locally in time, use a fallback reranker prompt through Gemini on a bounded high-risk set. The fallback must be recorded as external-model evidence and must not send credentials or hidden test data outside the already permitted challenge workflow.

Runtime selection is explicit:

- local Qwen rerank is preferred when the model is already available or can be started without blocking the last-shot timeline;
- bounded Gemini fallback is allowed only for candidate-level evidence review when project policy permits sending the relevant image/content to Gemini;
- if neither path is available, v29 must still run the deterministic Description proxy and existing local evidence, then report that rerank evidence is missing rather than fabricating confidence.

## Candidate Families

Generate these side-path candidates:

- `v29_descmax_on_v21`: official-best v21 labels with FAB-G-lite rewritten text.
- `v29_descmax_on_v24_frontier`: highest local classification proxy from v24 with FAB-G-lite rewritten text.
- `v29_qwen_va_rescue`: only label changes with strong valence/arousal evidence and no rare-class erosion.
- `v29_qwen_balanced`: label changes supported by Qwen rerank plus current official-anchor trends, capped by top-emotion share and rare-class floor.
- `v29_hybrid_best`: best gated combination of rewritten text and classification changes.

All outputs must use side-path names under `submissions/`, such as:

- `submissions/track2_submission_v29_hybrid_best_candidate.json`
- `submissions/track2_submission_v29_hybrid_best_candidate.zip`

## Final Gate

`recommend_final_submit` is allowed only if all hard gates pass:

- Track2 validator OK.
- label consistency issue count is `0`.
- missing emotion count is `0`.
- unsafe description text rows are `0`.
- top emotion share is within configured cap.
- no broad public-kNN copying.
- Description proxy is at least `0.98`; target is `1.00`.
- Classification proxy is at least `0.78`.
- Overall proxy is at least `0.89`.
- report lists all label transitions and all cross-quadrant changes.

If any hard gate fails, output `hold_no_submit` and preserve the last Codabench attempt.

## File Boundaries

Create focused modules:

- `affectiveart/track2_v29_description_proxy.py`: local Description proxy and text lint.
- `affectiveart/track2_v29_fabg_lite.py`: salience inference and text rewrite helpers.
- `affectiveart/track2_v29_qwen_rerank.py`: evidence schema and rerank/fallback integration.
- `affectiveart/track2_v29_final_shot.py`: candidate assembly, scoring, gates, and reports.
- `scripts/track2_v29_final_shot.py`: CLI entry point.

Create focused tests:

- `tests/test_track2_v29_description_proxy.py`
- `tests/test_track2_v29_fabg_lite.py`
- `tests/test_track2_v29_qwen_rerank.py`
- `tests/test_track2_v29_final_shot.py`

## Verification

Run only Track2 checks:

```bash
python3 -m unittest tests/test_track2_v29_description_proxy.py tests/test_track2_v29_fabg_lite.py tests/test_track2_v29_qwen_rerank.py tests/test_track2_v29_final_shot.py tests/test_track2_official_anchor_calibration.py -v
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v29_hybrid_best_candidate.json
git diff --check
```

If no v29 candidate clears `0.89`, validation may run on the highest-scoring side-path candidate, but the report must mark it `hold_no_submit`.

## Success Criteria

The work is successful only if it produces one of two outcomes:

1. A gated final candidate:
   - local overall proxy `>= 0.89`;
   - classification proxy `>= 0.78`;
   - Description proxy `>= 0.98`;
   - exact JSON and ZIP paths;
   - all safety checks pass.

2. A clear stop decision:
   - no candidate reaches the hard gate;
   - report explains which component blocks `0.89`;
   - final submission attempt is preserved.
