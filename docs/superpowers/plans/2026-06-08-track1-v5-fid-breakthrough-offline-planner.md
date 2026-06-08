# Track1 v5 FID Breakthrough Offline Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a no-API full-1000 Track1 v5 route/prompt planner and Chinese HTML review page for FID-first generation planning.

**Architecture:** Add a focused planner module that joins existing domain contracts, expert routes, and official reference routes. The planner assigns one primary v5 route per sample, keeps high-AAS-risk samples protected, writes provider prompts and reports, and renders an HTML dashboard for human review before any Gemini image generation.

**Tech Stack:** Python standard library, existing `affectiveart` Track1 contract/router artifacts, pytest.

---

### Task 1: Add v5 planner module

**Files:**
- Create: `affectiveart/track1_v5_fid_breakthrough.py`
- Test: `tests/test_track1_v5_fid_breakthrough.py`

- [x] **Step 1: Write route classification tests**

Create tests that cover:

- text/landmark/relation risk routes to `caption_faithful_guard`
- generic artwork with overused references routes to `anti_template_diversifier`
- generic low-risk artwork routes to `reference_family_primary`
- provider prompt contains the official caption and no sample id

- [x] **Step 2: Implement planner functions**

Implement:

- `classify_v5_route(contract, expert_route, official_route, reference_reuse_counts)`
- `build_v5_provider_prompt(row)`
- `build_v5_plan(contracts, expert_routes, official_routes)`
- `summarize_v5_plan(rows)`

- [x] **Step 3: Run tests**

Run:

```bash
python3 -m pytest tests/test_track1_v5_fid_breakthrough.py -q
```

Expected: tests pass.

### Task 2: Add CLI and reports

**Files:**
- Create: `scripts/track1_v5_fid_breakthrough_plan.py`
- Modify: `affectiveart/track1_v5_fid_breakthrough.py`

- [x] **Step 1: Add CLI**

The CLI reads:

- `track1_domain_contract_1000.json`
- `track1_expert_routes_1000.json`
- `track1_distribution_routes_with_reference_assets.json`

It writes JSON, JSONL, CSV, Markdown, provider prompt files, and HTML under:

`experiments/track1_v5_fid_breakthrough_20260608/offline_plan/`

- [x] **Step 2: Add HTML dashboard**

The HTML must show:

- route counts
- style family counts
- aspect counts
- top overused reference assets
- v7 family id versus v5 route
- current/v3 thumbnail when available
- four official reference thumbnails per sample
- risk tags and provider prompt excerpt

- [x] **Step 3: Run CLI**

Run:

```bash
python3 scripts/track1_v5_fid_breakthrough_plan.py
```

Expected: 1000 rows and no API calls.

### Task 3: Verify and commit

**Files:**
- Verify generated files under `experiments/track1_v5_fid_breakthrough_20260608/offline_plan/`

- [x] **Step 1: Validate champion immutability**

Run:

```bash
git diff -- submissions/track1_submission.json submissions/track1_submission.zip submissions/track1/images | wc -l
```

Expected: `0`

- [x] **Step 2: Run focused tests**

Run:

```bash
python3 -m pytest tests/test_track1_v5_fid_breakthrough.py -q
```

Expected: pass.

- [x] **Step 3: Commit verified files**

Commit only the planner code, tests, plan doc, and generated offline review artifacts needed for handoff.

---

## 2026-06-08 Smoke Follow-up

**New goal:** Before full 1000-image generation, run a 48-sample v5 smoke set that covers hard human-review samples, routes, aspect/support types, style families, and FID-risk tags.

**Implementation added:**

- `affectiveart.track1_v5_smoke_queue`
- `scripts/track1_v5_smoke_queue.py`
- `scripts/track1_v5_smoke_review.py`
- `tests/test_track1_v5_smoke_queue.py`

**Smoke queue result:**

- total: `48`
- routes: `caption_faithful_guard=22`, `reference_family_primary=16`, `anti_template_diversifier=10`
- aspects: `portrait_poster=17`, `square_artwork=27`, plus vertical scroll, album spread, horizontal scroll, and panel story coverage
- planned models after route repair: `gemini-3-pro-image=25`, `gemini-3.1-flash-image=23`

**Runtime finding:**

`imagen-4-ultra` is not a valid `generateContent` model for the current Gemini image provider path. The available Imagen entries are `predict/generate_images` models and do not preserve the current reference-board conditioning path. For this v5 pipeline, `imagen-*` recommendations are now mapped to Gemini image models, preferring `gemini-3-pro-image` when provided.

**Generated smoke result:**

- generated/cached candidates: `48/48`
- missing current/candidate/reference board in review HTML: `0`
- actual models: `gemini-3-pro-image=27`, `gemini-3.1-flash-image-preview=21`
- reference fallback without reference: `4`
- review HTML: `experiments/track1_v5_smoke_20260608/review/track1_v5_smoke_generated_review_zh.html`
- contact sheets: `experiments/track1_v5_smoke_20260608/review/track1_v5_smoke_contact_sheet_page_01.jpg` through `_06.jpg`

**Qualitative read:**

The smoke supports the main v5 hypothesis: reference-grounded generation is materially stronger for non-poster reference-family distribution, especially Gongbi, ink-wash, ukiyo-e, Baroque, and abstract drawing cases. Poster cases are mixed: many are more poster-like, but text correctness, relation semantics, and requested object fidelity still require conservative gates. `track1_0747` remains a high-risk hold despite better poster surface.

**Next full-run implication:**

Do not submit a blind 1000-image v5 replacement. Use v5 as a candidate-generation layer, then build a hybrid package using:

- high-confidence reference-family wins,
- conservative poster accepts only after visual/text/relation review,
- current champion fallback for uncertain cases,
- local FID/style-family proxy to avoid degrading distribution.
