# Track1 Reference Selector V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative diversity-aware Track1 reference selector and generate a v4 audit page that shows whether repeated boards are caused by selector ordering or by a small reference bank.

**Architecture:** Extend `affectiveart.track1_reference_asset_bindings` with an optional `selection_mode` parameter. Keep the stable v3 path as default, and add a stateful `diversity_balanced` path that rotates valid candidates by prior usage while preserving semantic anchors for reranked poster routes. Reuse existing route report and reference media audit scripts for separate v4 experiment outputs.

**Tech Stack:** Python standard library, existing Track1 scripts, PIL-backed tests, pytest/unittest.

---

### Task 1: Add Selection Mode API And RED Tests

**Files:**
- Modify: `/Users/yhryzy/dev/emoart-130k/tests/test_track1_reference_asset_bindings.py`
- Modify: `/Users/yhryzy/dev/emoart-130k/affectiveart/track1_reference_asset_bindings.py`

- [ ] **Step 1: Write failing tests**

Add tests that call `attach_reference_assets_to_routes(..., selection_mode="diversity_balanced")` and assert:

```python
def test_diversity_balanced_rotates_candidates_when_pool_exceeds_slots(self):
    # Six same-style assets, three routes, two slots each.
    # Stable mode would always select asset_0 and asset_1.
    # Diversity mode should use more than two unique assets.
```

```python
def test_diversity_balanced_reports_limited_pool_when_all_candidates_are_required(self):
    # Two assets, two slots, repeated routes.
    # The route should include reference_candidate_pool_size=2 and reference_diversity_limited=True.
```

```python
def test_diversity_balanced_keeps_family_poster_anchors_before_media_fill(self):
    # Reranked poster pool has two family poster anchors plus four media posters.
    # With max_assets_per_route=4, selected refs must include both family anchors.
```

- [ ] **Step 2: Run RED tests**

Run:

```bash
python3 -m pytest tests/test_track1_reference_asset_bindings.py -q
```

Expected: FAIL because `selection_mode` and v4 metadata do not exist yet.

### Task 2: Implement Minimal V4 Selector

**Files:**
- Modify: `/Users/yhryzy/dev/emoart-130k/affectiveart/track1_reference_asset_bindings.py`
- Modify: `/Users/yhryzy/dev/emoart-130k/scripts/track1_attach_reference_assets.py`

- [ ] **Step 1: Add API surface**

Add `selection_mode: str = "stable"` to `attach_reference_assets_to_routes`.

Supported values:

```python
VALID_SELECTION_MODES = {"stable", "diversity_balanced"}
```

Invalid values raise:

```python
ValueError("selection_mode must be one of: diversity_balanced, stable")
```

- [ ] **Step 2: Add stateful diversity selection**

For `diversity_balanced`, keep a `Counter[str]` of selected reference identities across the route loop. Before selecting candidates, resolve existing files and sort non-protected candidates by:

```python
(usage_count[reference_identity], deterministic_sample_offset, original_index)
```

Use `_reference_identity(path.name)` as the reference identity.

- [ ] **Step 3: Protect reranked family poster anchors**

For candidate items whose note includes:

```text
preserved family poster/print reference
```

keep up to two existing assets at the front before diversity-filling remaining slots.

- [ ] **Step 4: Add per-route metadata**

Each output route gets:

```python
reference_selection_mode
reference_candidate_pool_size
reference_selected_identities
reference_diversity_limited
```

`reference_diversity_limited` is true when the candidate pool has at most `max_assets_per_route` existing assets for non-sample routes.

- [ ] **Step 5: Add CLI flag**

Add to `scripts/track1_attach_reference_assets.py`:

```python
parser.add_argument(
    "--selection-mode",
    choices=["stable", "diversity_balanced"],
    default="stable",
)
```

Pass it to `attach_reference_assets_to_routes`.

### Task 3: Extend Reports And Verify

**Files:**
- Modify: `/Users/yhryzy/dev/emoart-130k/affectiveart/track1_reference_asset_bindings.py`
- Test: `/Users/yhryzy/dev/emoart-130k/tests/test_track1_reference_asset_bindings.py`

- [ ] **Step 1: Extend summary**

Add summary fields:

```python
reference_asset_slots
unique_reference_assets
diversity_limited_routes
top_reference_reuse
```

`top_reference_reuse` is a list of up to 20 `{file, count}` records.

- [ ] **Step 2: Run GREEN tests**

Run:

```bash
python3 -m pytest tests/test_track1_reference_asset_bindings.py -q
```

Expected: PASS.

- [ ] **Step 3: Run related regression tests**

Run:

```bash
python3 -m pytest tests/test_track1_reference_asset_bindings.py tests/test_track1_reference_media_audit.py tests/test_track1_moe_generation.py tests/test_track1_moe_prompt_packets.py -q
```

Expected: PASS.

### Task 4: Generate V4 Audit Outputs

**Files:**
- Generate only under `/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/`

- [ ] **Step 1: Build v4 route binding**

Run:

```bash
python3 scripts/track1_attach_reference_assets.py \
  --selection-mode diversity_balanced \
  --routes-json experiments/track1_reference_conditioned_pilot_20260603/distribution_router_antitemplate_20260605/track1_distribution_routes.json \
  --reference-assets-index-json experiments/track1_reference_conditioned_pilot_20260603/full_reference_assets_20260605/track1_full_reference_assets_index.json \
  --reference-assets-dir experiments/track1_reference_conditioned_pilot_20260603/full_reference_assets_20260605/reference_assets \
  --out-json experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_media_v4_20260605/track1_distribution_routes_with_reference_assets.json \
  --out-csv experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_media_v4_20260605/track1_distribution_routes_with_reference_assets.csv \
  --out-md experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_media_v4_20260605/track1_distribution_routes_with_reference_assets.md
```

- [ ] **Step 2: Build v3-v4 audit page**

Run:

```bash
python3 scripts/track1_reference_media_audit.py \
  --old-routes-json experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_media_v3_20260605/track1_distribution_routes_with_reference_assets.json \
  --new-routes-json experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_media_v4_20260605/track1_distribution_routes_with_reference_assets.json \
  --partial-candidate-image-dir experiments/track1_reference_conditioned_pilot_20260603/moe_full_reference_assets_candidates_20260605/images \
  --out-dir experiments/track1_reference_conditioned_pilot_20260603/reference_media_diversity_v4_20260605
```

- [ ] **Step 3: Open browser**

Open:

```text
http://127.0.0.1:51929/experiments/track1_reference_conditioned_pilot_20260603/reference_media_diversity_v4_20260605/track1_reference_media_audit_zh.html
```

### Task 5: Final Verification And Commit

- [ ] **Step 1: Check formatting**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 2: Protect champion package**

Run:

```bash
git diff -- submissions/track1_submission.json submissions/track1_submission.zip submissions/track1/images | wc -l
```

Expected: `0`.

- [ ] **Step 3: Commit code, tests, spec, and plan**

Do not stage generated experiment directories unless explicitly requested.

Run:

```bash
git add affectiveart/track1_reference_asset_bindings.py scripts/track1_attach_reference_assets.py tests/test_track1_reference_asset_bindings.py docs/superpowers/specs/2026-06-05-track1-reference-selector-v4-design.md docs/superpowers/plans/2026-06-05-track1-reference-selector-v4.md
git commit -m "feat: add diversity-aware track1 reference selector"
```

## Self-Review

- Spec coverage: selection mode, diversity rotation, reranked poster anchor protection, metadata, reports, audit output, and champion immutability are covered.
- Placeholder scan: no TODO/TBD placeholders remain.
- Type consistency: `selection_mode`, `reference_candidate_pool_size`, `reference_diversity_limited`, and `reference_selected_identities` are consistently named.
