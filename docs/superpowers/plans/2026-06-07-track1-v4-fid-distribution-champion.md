# Track1 V4 FID Distribution Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conservative Track1 v4 candidate package that attacks the official FID gap at batch scale while keeping AAS near the current 0.98-0.99 ceiling.

**Architecture:** Do not mutate the current champion package. Use existing Track1 tools to compare whole candidate packages against the official EmoArt-130K reference distribution, then select 100-300 replacements from already generated candidates and only regenerate samples where the candidate pool is weak. Gemini remains the final image provider; Vulca/Track1 gates are used as prompt compiler, AAS guard, reference-role guard, and final rejection layer.

**Tech Stack:** Python 3, existing `affectiveart` Track1 modules, Gemini image API through Vulca, local EmoArt-130K reference corpus at `/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k`, Inception FID-like sanity, SigLIP/CLIP embedding proxy, human review manifests, Codabench official score feedback.

---

## Current Evidence

- Official latest v3 gate7 result: Overall `0.77`, FID `80.92`, FID Score `0.55`, AAS `0.98`.
- Visible leaderboard top is around Overall `0.80`, FID `66.x`, FID Score `0.60`, AAS `0.99`.
- The gap is mostly FID/style distribution, not single-image caption alignment.
- `track1_v3_gate7_replacement_manifest.json` contains only 7 accepted replacements, so it cannot materially move FID.
- `official_reference_bank_v7_20260605` already extracted `1638` official EmoArt-130K reference assets for all `1000` Track1 routes.
- `reference_asset_routes_official_v7_20260605` uses official retrieval for all `1000` routes, but only 4 reference-board assets per route.
- `reference_asset_routes_media_v4_20260605` confirms the older curated bank is too small: `969/1000` routes are diversity-limited.
- First local FID-like smoke on `2026-06-07` ranked `current` best: current `58.526207`, partial757 `62.414763`, v3_gate7 `63.010429`, full1000_no_fallback `63.098811`, strict_subset `63.287051`, safe_subset `63.308289`.
- Therefore do not submit `full1000_no_fallback`, `partial757`, `safe_subset`, or `strict_subset` directly. Existing broad generated packages are useful as a candidate/error pool, not as final distribution packages.

## Proxy Calibration Warnings

- Local FID-like results are decision aids, not official scores.
- Treat a package as promising only if its ordering is stable across reference seeds and feature dimensions.
- Do not optimize FID at the expense of caption content, surface/support fidelity, relation logic, or style attributes.
- Do not present any local proxy result as proof that the package will win; official Codabench remains the only ground truth.
- If local FID proxy and human/AAS gates disagree, do not submit the risky package directly. Build a smaller conservative package or regenerate weak samples.

## Non-Negotiables

- Never modify:
  - `submissions/track1_submission.json`
  - `submissions/track1_submission.zip`
  - `submissions/track1/images/`
- Do not submit another micro-fix package under 50 replacements.
- Do not accept a candidate solely because a VLM says it is aligned.
- Do not use test images or hidden labels as training labels.
- All generated outputs go under:
  - `experiments/track1_v4_fid_distribution_champion_20260607/`
  - `submissions/track1_candidate_v4_conservative80_20260607/`
  - `submissions/track1_candidate_v4_balanced180_20260607/`

## Package Strategy

Produce three packages, then submit at most one after local gates:

1. `v4_conservative_80`
   - 50-100 replacements.
   - Use only high-confidence candidate wins.
   - Target: AAS unchanged or +0.01, no local FID regression.

2. `v4_balanced_180`
   - 120-220 replacements.
   - Main target package.
   - Target: improve official-style distribution without repeating the failed full1000 template.

3. `v4_regen_outlier_300`
   - Regenerate 200-300 current outliers first, then accept only clean wins.
   - This package is built only after the current-preserving candidate pool cannot reach at least 120 clean replacements.

---

### Task 1: Freeze Baseline And Candidate Inventory

**Files:**
- Read: `/Users/yhryzy/dev/emoart-130k/submissions/track1_submission.json`
- Read: `/Users/yhryzy/dev/emoart-130k/submissions/track1_candidate_v3_gate7_20260606/submission.json`
- Read: `/Users/yhryzy/dev/emoart-130k/submissions/track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605/submission.json`
- Read: `/Users/yhryzy/dev/emoart-130k/submissions/track1_candidate_full1000_aas_safe_official_v7_partial757_20260605/submission.json`
- Read: `/Users/yhryzy/dev/emoart-130k/submissions/track1_candidate_v2_safe_subset_20260606/submission.json`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/package_inventory.json`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/package_inventory_zh.md`

- [ ] **Step 1: Record manifest-level replacement counts**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

out = Path("experiments/track1_v4_fid_distribution_champion_20260607")
out.mkdir(parents=True, exist_ok=True)
manifest_paths = {
    "v3_gate7": "experiments/track1_reference_conditioned_pilot_20260603/v3_balanced_style_fid_20260606/track1_v3_gate7_replacement_manifest.json",
    "full1000_no_fallback": "experiments/track1_reference_conditioned_pilot_20260603/full1000_aas_safe_official_v7_20260605/candidate_package_full1000_no_fallback_20260605/replacement_manifest_full1000_no_fallback.json",
    "partial757": "experiments/track1_reference_conditioned_pilot_20260603/full1000_aas_safe_official_v7_20260605/candidate_package_partial757_20260605/replacement_manifest_partial757.json",
    "flash31_safe_subset": "experiments/track1_reference_conditioned_pilot_20260603/full1000_aas_safe_official_v7_20260605/fallback99_flash31_v2_20260606/replacement_manifest_flash31_fallback99_v2_safe_subset.json",
    "flash31_strict_subset": "experiments/track1_reference_conditioned_pilot_20260603/full1000_aas_safe_official_v7_20260605/fallback99_flash31_v2_20260606/replacement_manifest_flash31_fallback99_v2_strict_subset.json",
}
rows = []
for name, raw_path in manifest_paths.items():
    path = Path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("replacements") or payload.get("decisions") or payload.get("rows") or []
    accepted = []
    for row in items:
        status = str(row.get("decision") or row.get("status") or row.get("recommendation") or row.get("action") or "").lower()
        if not status or any(token in status for token in ("accept", "replace", "accepted")):
            if row.get("sample_id"):
                accepted.append(str(row["sample_id"]))
    rows.append({"package": name, "manifest": raw_path, "rows": len(items), "accepted_like": len(accepted), "sample_ids": accepted})
report = {"rows": rows}
(out / "package_inventory.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
lines = ["# Track1 V4 Package Inventory", ""]
for row in rows:
    lines.append(f"- `{row['package']}` manifest_rows=`{row['rows']}` accepted_like=`{row['accepted_like']}`")
(out / "package_inventory_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out / "package_inventory_zh.md")
PY
```

Expected:

- `v3_gate7` has 7 accepted replacements.
- `full1000_no_fallback` has 1000 replacements.
- `partial757` has 757 replacements.
- `flash31_safe_subset` has about 27 accepted-like replacements.
- `flash31_strict_subset` has about 23 accepted-like replacements.

- [ ] **Step 2: Verify champion package remains untouched**

Run:

```bash
git diff -- submissions/track1_submission.json submissions/track1_submission.zip submissions/track1/images | wc -l
```

Expected: `0`.

### Task 2: Run Local FID-Like Package Sweep

**Files:**
- Read: `/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/fid_sanity_8192_dim512.json`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/fid_sanity_8192_dim512.md`

- [ ] **Step 1: Run broad all-style FID proxy**

Run:

```bash
python3 scripts/track1_inception_fid_sanity.py \
  --reference-root /Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k \
  --reference-limit 8192 \
  --reference-seed 20260607 \
  --fid-feature-dim 512 \
  --device auto \
  --batch-size 32 \
  --out-json experiments/track1_v4_fid_distribution_champion_20260607/fid_sanity_8192_dim512.json \
  --out-md experiments/track1_v4_fid_distribution_champion_20260607/fid_sanity_8192_dim512.md \
  --package current=submissions/track1_submission.json=submissions/track1/images \
  --package v3_gate7=submissions/track1_candidate_v3_gate7_20260606/submission.json=submissions/track1_candidate_v3_gate7_20260606/images \
  --package full1000_no_fallback=submissions/track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605/submission.json=submissions/track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605/images \
  --package partial757=submissions/track1_candidate_full1000_aas_safe_official_v7_partial757_20260605/submission.json=submissions/track1_candidate_full1000_aas_safe_official_v7_partial757_20260605/images \
  --package safe_subset=submissions/track1_candidate_v2_safe_subset_20260606/submission.json=submissions/track1_candidate_v2_safe_subset_20260606/images \
  --package strict_subset=submissions/track1_candidate_v2_strict_subset_20260606/submission.json=submissions/track1_candidate_v2_strict_subset_20260606/images
```

Expected:

- The report ranks candidate packages by local FID-like distance.
- If `full1000_no_fallback` beats `current` strongly but human/AAS is risky, it becomes the source pool, not a direct submission.
- If every full package is worse than `current`, stop broad replacement and move to candidate-level reranking only.

- [ ] **Step 2: Repeat broad FID proxy with another reference seed**

Run:

```bash
python3 scripts/track1_inception_fid_sanity.py \
  --reference-root /Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k \
  --reference-limit 8192 \
  --reference-seed 20260608 \
  --fid-feature-dim 512 \
  --device auto \
  --batch-size 32 \
  --out-json experiments/track1_v4_fid_distribution_champion_20260607/fid_sanity_8192_dim512_seed20260608.json \
  --out-md experiments/track1_v4_fid_distribution_champion_20260607/fid_sanity_8192_dim512_seed20260608.md \
  --package current=submissions/track1_submission.json=submissions/track1/images \
  --package v3_gate7=submissions/track1_candidate_v3_gate7_20260606/submission.json=submissions/track1_candidate_v3_gate7_20260606/images \
  --package full1000_no_fallback=submissions/track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605/submission.json=submissions/track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605/images \
  --package partial757=submissions/track1_candidate_full1000_aas_safe_official_v7_partial757_20260605/submission.json=submissions/track1_candidate_full1000_aas_safe_official_v7_partial757_20260605/images \
  --package safe_subset=submissions/track1_candidate_v2_safe_subset_20260606/submission.json=submissions/track1_candidate_v2_safe_subset_20260606/images \
  --package strict_subset=submissions/track1_candidate_v2_strict_subset_20260606/submission.json=submissions/track1_candidate_v2_strict_subset_20260606/images
```

Expected: the top package ordering is similar to the first seed. If not, local FID is too noisy for large replacement decisions.

- [ ] **Step 3: Optional high-fidelity repeat**

Run only if Step 1 gives a plausible winner:

```bash
python3 scripts/track1_inception_fid_sanity.py \
  --reference-root /Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k \
  --reference-limit 8192 \
  --reference-seed 20260607 \
  --fid-feature-dim 2048 \
  --device auto \
  --batch-size 16 \
  --out-json experiments/track1_v4_fid_distribution_champion_20260607/fid_sanity_8192_dim2048.json \
  --out-md experiments/track1_v4_fid_distribution_champion_20260607/fid_sanity_8192_dim2048.md \
  --package current=submissions/track1_submission.json=submissions/track1/images \
  --package v3_gate7=submissions/track1_candidate_v3_gate7_20260606/submission.json=submissions/track1_candidate_v3_gate7_20260606/images \
  --package full1000_no_fallback=submissions/track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605/submission.json=submissions/track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605/images \
  --package partial757=submissions/track1_candidate_full1000_aas_safe_official_v7_partial757_20260605/submission.json=submissions/track1_candidate_full1000_aas_safe_official_v7_partial757_20260605/images
```

Expected: same ordering as the 512-dim runs. If ordering flips, treat the local FID proxy as unstable.

### Task 3: Diagnose Why Existing Broad Packages Lose FID

**Files:**
- Read: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/fid_sanity_smoke2048_dim512.json`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/broad_package_failure_analysis_zh.md`

- [ ] **Step 1: Write failure analysis**

Write a short analysis with these conclusions:

```text
current is the local FID-like winner among existing packages
partial/full1000 generated packages move away from EmoArt-130K all-style distribution
v3_gate7 did not improve local FID despite only 7 manifest replacements, likely because package-level image encoding/rebuild differs from current
existing broad packages should not be submitted directly
next package must be current-preserving and candidate-level, not broad replacement by default
```

Expected: a decision document that blocks direct full1000/partial submission unless official feedback contradicts local proxy.

### Task 4: Build Candidate-Level Metric Reports

**Files:**
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/metric_proxy_full1000.json`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/metric_proxy_full1000.md`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/metric_proxy_partial757.json`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/metric_proxy_partial757.md`

- [ ] **Step 1: Export current and candidate embeddings when missing**

Run:

```bash
python3 scripts/track1_export_embeddings.py \
  --submission-json submissions/track1_submission.json \
  --image-dir submissions/track1/images \
  --out-npz experiments/track1_v4_fid_distribution_champion_20260607/current_siglip2_full.npz \
  --backend hf \
  --model google/siglip2-base-patch16-224 \
  --device auto \
  --batch-size 32

python3 scripts/track1_export_embeddings.py \
  --submission-json submissions/track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605/submission.json \
  --image-dir submissions/track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605/images \
  --out-npz experiments/track1_v4_fid_distribution_champion_20260607/full1000_siglip2_full.npz \
  --backend hf \
  --model google/siglip2-base-patch16-224 \
  --device auto \
  --batch-size 32

python3 scripts/track1_export_embeddings.py \
  --submission-json submissions/track1_candidate_full1000_aas_safe_official_v7_partial757_20260605/submission.json \
  --image-dir submissions/track1_candidate_full1000_aas_safe_official_v7_partial757_20260605/images \
  --out-npz experiments/track1_v4_fid_distribution_champion_20260607/partial757_siglip2_full.npz \
  --backend hf \
  --model google/siglip2-base-patch16-224 \
  --device auto \
  --batch-size 32
```

Expected: each `.npz` contains 1000 image embeddings and no failures.

- [ ] **Step 2: Run metric proxy on full1000 and partial757**

Run:

```bash
python3 scripts/track1_metric_proxy.py \
  --submission-json submissions/track1_submission.json \
  --current-image-dir submissions/track1/images \
  --candidate-image-dir submissions/track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605/images \
  --current-embedding-npz experiments/track1_v4_fid_distribution_champion_20260607/current_siglip2_full.npz \
  --candidate-embedding-npz experiments/track1_v4_fid_distribution_champion_20260607/full1000_siglip2_full.npz \
  --expert-manifest experiments/track1_reference_conditioned_pilot_20260603/v3_balanced_style_fid_20260606/real_run_top50_v2/track1_v3_top50_augmented_review_manifest.json \
  --out-json experiments/track1_v4_fid_distribution_champion_20260607/metric_proxy_full1000.json \
  --out-md experiments/track1_v4_fid_distribution_champion_20260607/metric_proxy_full1000.md

python3 scripts/track1_metric_proxy.py \
  --submission-json submissions/track1_submission.json \
  --current-image-dir submissions/track1/images \
  --candidate-image-dir submissions/track1_candidate_full1000_aas_safe_official_v7_partial757_20260605/images \
  --current-embedding-npz experiments/track1_v4_fid_distribution_champion_20260607/current_siglip2_full.npz \
  --candidate-embedding-npz experiments/track1_v4_fid_distribution_champion_20260607/partial757_siglip2_full.npz \
  --expert-manifest experiments/track1_reference_conditioned_pilot_20260603/v3_balanced_style_fid_20260606/real_run_top50_v2/track1_v3_top50_augmented_review_manifest.json \
  --out-json experiments/track1_v4_fid_distribution_champion_20260607/metric_proxy_partial757.json \
  --out-md experiments/track1_v4_fid_distribution_champion_20260607/metric_proxy_partial757.md
```

Expected: the reports identify per-sample candidates with positive distribution/perceptual proxy and no surface/artifact penalty.

### Task 5: Select V4 Replacement Sets

**Files:**
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/replacement_manifest_conservative80.json`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/replacement_manifest_balanced180.json`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/replacement_manifest_regen_outlier300.json`

- [ ] **Step 1: Use hard gates before score gates**

Accept a replacement only when all are true:

```text
candidate image exists and is a raster image
no provider fallback without reference
no gallery/mockup/watermark/sample-id artifact
no unrequested visible text
caption-requested support/surface is preserved
required relation/spatial logic is not worse than current
human veto list does not hold/reject the sample
```

- [ ] **Step 2: Use score gates after hard gates**

For each candidate, compute:

```text
candidate_score = 0.50 * distribution_proxy + 0.25 * perceptual_proxy + 0.25 * aas_proxy
accept_conservative if candidate_score - current_score >= 0.06 and local FID package smoke does not regress
accept_balanced if candidate_score - current_score >= 0.04 and local FID package smoke does not regress
accept_regen_outlier if candidate_score - current_score >= 0.03 and the sample was selected as a current outlier
```

Expected replacement counts:

- `conservative80`: 50-100.
- `balanced180`: 120-220.
- `regen_outlier300`: build from 200-300 generated candidates, but accept fewer if gates are strict.

If `balanced180` cannot reach at least 120 clean replacements without local FID regression, regenerate current outliers instead of relaxing gates.

- [ ] **Step 3: Add semantic/AAS sanity after score gates**

Before building a package, inspect the accepted set distribution:

```text
poster captions keep poster/print/surface requirements
scroll and album captions keep mounted support/page structure
real landmarks/symbols do not become generic fantasy motifs
visible text is either caption-requested or visually minor
accepted set does not collapse into one repeated poster or decorative template
```

Expected: no systematic failure pattern. If a pattern appears, lower the replacement count and regenerate only affected families.

### Task 6: Regenerate Current Outliers Only

**Files:**
- Read: `/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_official_v7_20260605/track1_distribution_routes_with_reference_assets.json`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/moe_prompt_packets/`
- Create: `/Users/yhryzy/dev/emoart-130k/experiments/track1_v4_fid_distribution_champion_20260607/generated_candidates/`

- [ ] **Step 1: Build distribution prompt packets**

Run:

```bash
python3 scripts/track1_moe_prompt_packets.py \
  --routes-json experiments/track1_reference_conditioned_pilot_20260603/expert_router_20260604/track1_expert_routes_1000.json \
  --contract-json experiments/track1_reference_conditioned_pilot_20260603/final_contract_20260603/track1_domain_contract_1000.json \
  --distribution-routes-json experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_official_v7_20260605/track1_distribution_routes_with_reference_assets.json \
  --reference-text-bank-jsonl experiments/track1_reference_conditioned_pilot_20260603/reference_text_bank_full_v2.jsonl \
  --strategy-mode distribution \
  --max-strategies-per-sample 3 \
  --out-dir experiments/track1_v4_fid_distribution_champion_20260607/moe_prompt_packets
```

Expected: prompt packets include `aas_safe`, `reference_style`, and `fid_diverse`.

- [ ] **Step 2: Generate only missing high-value candidates**

Run after building a sample-id list of weak high-value rows:

```bash
while read -r sample_id; do
  python3 scripts/track1_moe_generate_candidates.py \
    --packets-jsonl experiments/track1_v4_fid_distribution_champion_20260607/moe_prompt_packets/track1_moe_prompt_packets.jsonl \
    --out-dir experiments/track1_v4_fid_distribution_champion_20260607/generated_candidates \
    --sample-id "$sample_id" \
    --max-candidates-per-sample 2 \
    --image-model-override gemini-3-pro-image
done < experiments/track1_v4_fid_distribution_champion_20260607/regenerate_sample_ids.txt
```

Expected: regenerate only samples where current is likely causing distribution/AAS loss. Do not regenerate all 1000.

### Task 7: Build And Validate Candidate Packages

**Files:**
- Create: `/Users/yhryzy/dev/emoart-130k/submissions/track1_candidate_v4_conservative80_20260607/`
- Create: `/Users/yhryzy/dev/emoart-130k/submissions/track1_candidate_v4_conservative80_20260607.zip`
- Create: `/Users/yhryzy/dev/emoart-130k/submissions/track1_candidate_v4_balanced180_20260607/`
- Create: `/Users/yhryzy/dev/emoart-130k/submissions/track1_candidate_v4_balanced180_20260607.zip`

- [ ] **Step 1: Build conservative and balanced packages**

Run:

```bash
python3 scripts/track1_build_candidate_proxy.py \
  --baseline-submission-json submissions/track1_submission.json \
  --baseline-image-dir submissions/track1/images \
  --replacement-manifest experiments/track1_v4_fid_distribution_champion_20260607/replacement_manifest_conservative80.json \
  --out-dir submissions/track1_candidate_v4_conservative80_20260607 \
  --out-zip submissions/track1_candidate_v4_conservative80_20260607.zip

python3 scripts/track1_build_candidate_proxy.py \
  --baseline-submission-json submissions/track1_submission.json \
  --baseline-image-dir submissions/track1/images \
  --replacement-manifest experiments/track1_v4_fid_distribution_champion_20260607/replacement_manifest_balanced180.json \
  --out-dir submissions/track1_candidate_v4_balanced180_20260607 \
  --out-zip submissions/track1_candidate_v4_balanced180_20260607.zip
```

- [ ] **Step 2: Validate packages**

Run:

```bash
python3 scripts/affectiveart_challenge.py validate-track1 \
  submissions/track1_candidate_v4_conservative80_20260607/submission.json \
  --check-files

python3 scripts/affectiveart_challenge.py validate-track1 \
  submissions/track1_candidate_v4_balanced180_20260607/submission.json \
  --check-files

git diff -- submissions/track1_submission.json submissions/track1_submission.zip submissions/track1/images | wc -l
```

Expected:

- Both validation commands pass.
- Protected champion diff is `0`.

### Task 8: Submit Decision Gate

- [ ] **Step 1: Pick one package**

Submit `v4_balanced180` only if:

```text
local FID-like score does not regress against current and improves against v3_gate7/rebuilt packages
metric proxy sweep does not show broad AAS/perceptual collapse
replacement count is at least 120
manual spot-check page has no obvious systematic artifact
```

Submit `v4_conservative80` if:

```text
balanced180 has visible AAS risk or template drift
conservative80 preserves local FID-like score
```

Do not submit `v4_regen_outlier_300` unless:

```text
official submission budget allows a risky probe
balanced180 is clearly insufficient
local FID-like improvement is stable across seeds and feature dimensions
```

## Self-Review

- Spec coverage: public-resource strategy, FID bottleneck, AAS protection, reference bank, candidate package building, validation, proxy calibration warnings, and champion immutability are covered.
- Placeholder scan: no TODO/TBD placeholders remain.
- Type consistency: package names, output directories, and manifest names are consistent across tasks.
