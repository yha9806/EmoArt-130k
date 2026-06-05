# Track1 Official Reference Bank V5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track1 official EmoArt-130k reference bank from local official annotation/tar archives and route each Track1 sample to official retrieved references.

**Architecture:** Add a standalone official-bank builder module that reads `Annotation.json`, scores official artworks by style and caption, extracts only selected references from tar archives, and writes a `route_references` index. Extend `track1_reference_asset_bindings` to prioritize `route_references` as `official_retrieval`, then reuse the existing v4 selector and audit pages.

**Tech Stack:** Python standard library (`json`, `tarfile`, `pathlib`, `collections`), PIL for tests, existing Track1 route/audit scripts.

---

### Task 1: Route Reference Protocol

**Files:**
- Modify: `/Users/yhryzy/dev/emoart-130k/affectiveart/track1_reference_asset_bindings.py`
- Modify: `/Users/yhryzy/dev/emoart-130k/tests/test_track1_reference_asset_bindings.py`

- [x] Add a failing test proving `route_references` overrides legacy `references`.
- [x] Implement validation for `route_references`.
- [x] In `_candidate_items_for_route`, read `route_references[sample_id]` before legacy sample references and normalize with source `official_retrieval`.
- [x] Verify the focused test passes.

### Task 2: Official Bank Builder

**Files:**
- Create: `/Users/yhryzy/dev/emoart-130k/affectiveart/track1_official_reference_bank.py`
- Create: `/Users/yhryzy/dev/emoart-130k/scripts/track1_build_official_reference_bank.py`
- Create: `/Users/yhryzy/dev/emoart-130k/tests/test_track1_official_reference_bank.py`

- [x] Add tests with temporary style tar archives and `Annotation.json`.
- [x] Implement annotation loading grouped by official style.
- [x] Implement style inference from Track1 caption, including automatic matching over the official 56 style names.
- [x] Implement caption/annotation/filename scoring with poster boosts.
- [x] Extract selected top references from tar archives into `reference_assets/`.
- [x] Write a compatible index with `route_references`.
- [x] Add CLI for reproducible generation.

### Task 3: Generate V5 Official Outputs

**Commands:**

```bash
python3 scripts/track1_build_official_reference_bank.py \
  --routes-json experiments/track1_reference_conditioned_pilot_20260603/distribution_router_antitemplate_20260605/track1_distribution_routes.json \
  --emoart-root /Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k \
  --out-dir experiments/track1_reference_conditioned_pilot_20260603/official_reference_bank_v5_20260605 \
  --out-index-json experiments/track1_reference_conditioned_pilot_20260603/official_reference_bank_v5_20260605/track1_official_reference_asset_index.json \
  --out-summary-json experiments/track1_reference_conditioned_pilot_20260603/official_reference_bank_v5_20260605/track1_official_reference_asset_summary.json \
  --max-candidates-per-route 4
```

Then:

```bash
python3 scripts/track1_attach_reference_assets.py \
  --selection-mode diversity_balanced \
  --routes-json experiments/track1_reference_conditioned_pilot_20260603/distribution_router_antitemplate_20260605/track1_distribution_routes.json \
  --reference-assets-index-json experiments/track1_reference_conditioned_pilot_20260603/official_reference_bank_v5_20260605/track1_official_reference_asset_index.json \
  --reference-assets-dir experiments/track1_reference_conditioned_pilot_20260603/official_reference_bank_v5_20260605/reference_assets \
  --out-json experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_official_v5_20260605/track1_distribution_routes_with_reference_assets.json \
  --out-csv experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_official_v5_20260605/track1_distribution_routes_with_reference_assets.csv \
  --out-md experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_official_v5_20260605/track1_distribution_routes_with_reference_assets.md
```

Finally:

```bash
python3 scripts/track1_reference_media_audit.py \
  --old-routes-json experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_media_v4_20260605/track1_distribution_routes_with_reference_assets.json \
  --new-routes-json experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_official_v5_20260605/track1_distribution_routes_with_reference_assets.json \
  --partial-candidate-image-dir experiments/track1_reference_conditioned_pilot_20260603/moe_full_reference_assets_candidates_20260605/images \
  --out-dir experiments/track1_reference_conditioned_pilot_20260603/reference_media_official_v5_20260605
```

### Task 4: Verification And Commit

- [x] Run:

```bash
python3 -m pytest tests/test_track1_official_reference_bank.py tests/test_track1_reference_asset_bindings.py tests/test_track1_reference_media_audit.py -q
```

- [x] Run:

```bash
git diff --check
git diff -- submissions/track1_submission.json submissions/track1_submission.zip submissions/track1/images | wc -l
```

- [ ] Commit only code, tests, spec, and plan. Do not commit extracted official bank outputs unless explicitly requested.

## Self-Review

- Spec coverage: official source, route references, extraction strategy, CLI, and audit generation are covered.
- No placeholders remain.
- Type names are consistent: `route_references`, `official_retrieval`, `track1_official_reference_bank`.
