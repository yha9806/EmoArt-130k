# Track1 Reference Selector V4 Design

## Goal

Build a diversity-aware Track1 reference selector that reduces avoidable reference-board repetition without weakening caption/style fidelity, and makes the remaining reference-bank ceiling visible before any new Gemini generation.

## Context

The current `reference_asset_routes_media_v3_20260605` output fixed an important media mismatch: poster captions no longer blindly use oil-painting or landscape references when poster-like references exist. However, local statistics show that the current "full reference assets" index is still a small curated bank, not the official 130k corpus:

- 86 unique reference files in the index.
- 78 unique files used across 3996 route reference slots.
- Several large style groups reuse the same four references hundreds of times.
- Poster/media routes cover 54 samples but use only nine unique references.

This repetition is risky for Track1 because the official score is 50% FID and 50% AAS. Overly repetitive reference conditioning can improve some local AAS cases while narrowing the generated distribution and harming FID.

## Design

V4 adds a conservative selector mode named `diversity_balanced`.

The mode keeps the existing priority order intact:

1. Sample-specific references remain highest priority.
2. Poster captions still use media-matched poster references.
3. V3 family-poster preservation remains active for poster captions.
4. Non-poster captions keep family/style matching and do not cross-contaminate styles.

Within an already valid candidate pool, V4 applies a deterministic reuse penalty so the same reference image is not always selected first when more candidates exist than the route can carry. It records per-route candidate-pool size, selected reference identities, and whether the route is diversity-limited because the candidate pool is no larger than `max_assets_per_route`.

V4 does not pretend to solve missing data. If a style only has four available references and each route needs four assets, it reports `reference_diversity_limited=true` rather than mixing unrelated styles. This is the correct behavior for AAS/FID balance: expose the bank gap instead of hiding it behind bad references.

## Outputs

V4 writes a separate experiment output only:

- `experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_media_v4_20260605/`
- `experiments/track1_reference_conditioned_pilot_20260603/reference_media_diversity_v4_20260605/`

It must not mutate:

- `submissions/track1_submission.json`
- `submissions/track1_submission.zip`
- `submissions/track1/images/`

## Acceptance Criteria

- Existing v3 behavior remains the default when no selection mode is passed.
- `diversity_balanced` rotates among equally valid candidates when the candidate pool has more files than the route can use.
- `caption_media_reranked` keeps at least the first two preserved family poster references when available, then diversity-fills remaining slots.
- Reports include enough metrics to answer whether repeated boards are caused by selector behavior or by a small reference bank.
- Unit tests cover default stability, diversity rotation, family-poster protection, and diversity-ceiling reporting.

## Next Separate Work

After V4 audit is reviewed, build a larger official/local reference bank. That is a separate task because it requires deciding which local official training/reference images are allowed, extracting image embeddings or captions, and adding style/medium/content metadata.
