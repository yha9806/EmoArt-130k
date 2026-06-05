# Track1 Full Reference Assets Pipeline - 2026-06-05

## What Changed

- Built a full reference asset index from local official EmoArt-130k archives plus prior curated official-reference assets.
- Bound reference assets to all 1000 Track1 distribution routes.
- Rebuilt reference-bound MoE prompt packets for the 76 high-priority replacement samples.
- Dry-ran 228 generation jobs and verified every job has a generated reference board.
- Added generation fallback: if Gemini blocks a request with a reference image, retry the same prompt without the reference image and mark metadata.

## Key Outputs

- Full reference index:
  - `experiments/track1_reference_conditioned_pilot_20260603/full_reference_assets_20260605/track1_full_reference_assets_index.json`
- Bound routes:
  - `experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_full_20260605/track1_distribution_routes_with_reference_assets.json`
  - Coverage: 1000 / 1000 routes with reference assets.
- Prompt packets:
  - `experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_full_reference_assets_20260605/track1_moe_prompt_packets.jsonl`
  - 228 prompt packets across 76 samples.
- Dry-run manifest:
  - `experiments/track1_reference_conditioned_pilot_20260603/moe_full_reference_assets_dryrun_20260605/candidate_manifest.json`
  - 228 / 228 planned jobs have reference boards.
- Review HTML:
  - `experiments/track1_reference_conditioned_pilot_20260603/moe_full_reference_assets_dryrun_20260605/track1_full_reference_prompt_review_zh.html`
- Smoke sheet:
  - `experiments/track1_reference_conditioned_pilot_20260603/moe_full_reference_assets_dryrun_20260605/track1_full_reference_smoke_with_0803_fallback_sheet_zh.jpg`

## Smoke Result

- `track1_0063`: 3 / 3 candidates generated with reference image.
- `track1_0077`: 3 / 3 candidates generated with reference image.
- `track1_0803`: 3 / 3 candidates generated after fallback support.
  - `aas_safe` and `reference_style` initially blocked with the reference board, then succeeded without the reference image.
  - `fid_diverse` succeeded with the reference image.

## Validation

- `python3 -m pytest tests/test_track1_reference_asset_bindings.py tests/test_track1_distribution_router.py tests/test_track1_moe_prompt_packets.py tests/test_track1_moe_generation.py -q`
  - `45 passed, 12 subtests passed`
- `git diff --check` on touched code/tests passed.
- Champion package guard:
  - `git diff -- submissions/track1_submission.json submissions/track1_submission.zip submissions/track1/images | wc -l`
  - Result: `0`

## Notes

- The large generated reference images, reference boards, and smoke outputs remain in the workspace for review, but are not intended to be committed wholesale.
- Current champion submission remains immutable.
- Fallback-generated candidates must be marked for stricter review because they did not use the reference image on the successful provider call.
