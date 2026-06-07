# Track2 v16 Official-Style RAG Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track2 v16 classification experiment that can produce a small, evidence-backed emotion-label candidate on top of the v15 description-expanded text base.

**Architecture:** v16 must separate official-score calibration from label proposal. The system first encodes what the official submissions proved, then builds a retrieval-augmented teacher queue for only high-impact uncertain rows, and finally creates nested ablation candidates that can be scored before any Codabench submission.

**Tech Stack:** Python standard library, existing `affectiveart.challenge`, existing Track2 JSON schema, Gemini 3.5 Flash via existing Keychain access, existing fused shadow evaluator.

---

## Current Evidence

- Best exact official anchor: `779605`, overall `0.836408`, classification `0.723150`, description `0.949667`.
- Failed aggressive calibration: `781601`, overall `0.834027`, classification `0.719137`, description `0.948917`.
- v12 stable probe: official page `0.84`, no classification label changes, description-only improvement.
- v15 expanded: local best text candidate, no classification changes, local expected `0.839902`, local lower `0.835854`.
- Public first-place target: around overall `0.89`, classification `0.78`, description `1.00`.

The bottleneck is emotion accuracy. v16 must not reuse same-quadrant count as a proxy for classification gain.

## Files

- Create: `affectiveart/track2_v16_official_style_rag.py`
  - Official anchor loading.
  - Pairwise submission delta analysis.
  - RAG queue construction.
  - Candidate construction from accepted v16 decisions.
- Create: `scripts/track2_v16_official_style_rag.py`
  - CLI entrypoints: `calibrate`, `build-queue`, `apply`, `ablation-score`.
- Create: `tests/test_track2_v16_official_style_rag.py`
  - Unit tests for calibration penalties, queue sorting, acceptance rules, and no formal overwrite.
- Output: `experiments/track2_v16_official_style_rag_20260607/`
  - `official_counterfactual_report.json/.md`
  - `rag_teacher_queue_top*.csv/.json/.md`
  - `v16_candidate_report_*.json/.md`
  - `fused_shadow_compare/`
- Output candidates only under `submissions/track2_submission_v16_*_candidate.json/.zip`
  - Never overwrite `submissions/track2_submission.json/.zip`.

## Task 1: Official Counterfactual Calibration

**Files:**
- Create: `affectiveart/track2_v16_official_style_rag.py`
- Create: `scripts/track2_v16_official_style_rag.py`
- Test: `tests/test_track2_v16_official_style_rag.py`

- [ ] **Step 1: Write tests for official-score delta calibration**

```python
from affectiveart.track2_v16_official_style_rag import (
    OfficialSubmissionScore,
    classify_transition_risk,
    official_delta_summary,
)


def test_official_delta_summary_marks_781601_as_negative_batch():
    anchor = OfficialSubmissionScore(
        submission_id="779605",
        overall=0.836408,
        classification=0.723150,
        description=0.949667,
    )
    failed = OfficialSubmissionScore(
        submission_id="781601",
        overall=0.834027,
        classification=0.719137,
        description=0.948917,
    )
    summary = official_delta_summary(anchor=anchor, candidate=failed, changed_rows=85)
    assert summary["classification_delta"] < 0
    assert summary["classification_delta_per_changed_row"] < 0
    assert summary["batch_verdict"] == "negative_official_evidence"


def test_classify_transition_risk_blocks_failed_bulk_boundaries():
    risk = classify_transition_risk(
        transition="calm->content",
        evidence_sources={"same_quadrant_batch"},
        official_failed_transition_count=43,
    )
    assert risk == "blocked_by_failed_official_batch"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest tests/test_track2_v16_official_style_rag.py -v
```

Expected: import errors for missing v16 module.

- [ ] **Step 3: Implement calibration dataclasses and helpers**

Implement:

```python
@dataclass(frozen=True)
class OfficialSubmissionScore:
    submission_id: str
    overall: float
    classification: float
    description: float


def official_delta_summary(*, anchor: OfficialSubmissionScore, candidate: OfficialSubmissionScore, changed_rows: int) -> dict[str, Any]:
    changed = max(1, int(changed_rows))
    classification_delta = float(candidate.classification) - float(anchor.classification)
    return {
        "anchor_submission_id": anchor.submission_id,
        "candidate_submission_id": candidate.submission_id,
        "classification_delta": classification_delta,
        "classification_delta_per_changed_row": classification_delta / changed,
        "overall_delta": float(candidate.overall) - float(anchor.overall),
        "description_delta": float(candidate.description) - float(anchor.description),
        "changed_rows": changed_rows,
        "batch_verdict": "negative_official_evidence" if classification_delta < 0 else "nonnegative_official_evidence",
    }
```

Implement transition risk rule:

```python
def classify_transition_risk(
    *,
    transition: str,
    evidence_sources: set[str],
    official_failed_transition_count: int = 0,
) -> str:
    if official_failed_transition_count >= 10 and evidence_sources <= {"same_quadrant_batch"}:
        return "blocked_by_failed_official_batch"
    if "exact_public_duplicate" in evidence_sources:
        return "allow_exact_duplicate"
    if {"rag_teacher", "multibackbone_consensus"} <= evidence_sources:
        return "allow_strong_consensus"
    return "hold_needs_more_evidence"
```

- [ ] **Step 4: Add CLI `calibrate`**

Command:

```bash
python3 scripts/track2_v16_official_style_rag.py calibrate \
  --official-scores experiments/track2_official_results_20260606/track2_known_official_exact_scores_from_ledger_20260606.csv \
  --pairwise-diffs experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv \
  --out-dir experiments/track2_v16_official_style_rag_20260607/calibration
```

Expected outputs:

- `official_counterfactual_report.json`
- `official_counterfactual_report.md`

- [ ] **Step 5: Verify and commit**

Run:

```bash
python3 -m unittest tests/test_track2_v16_official_style_rag.py -v
python3 scripts/track2_v16_official_style_rag.py calibrate \
  --official-scores experiments/track2_official_results_20260606/track2_known_official_exact_scores_from_ledger_20260606.csv \
  --pairwise-diffs experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv \
  --out-dir experiments/track2_v16_official_style_rag_20260607/calibration
git diff --check -- affectiveart/track2_v16_official_style_rag.py scripts/track2_v16_official_style_rag.py tests/test_track2_v16_official_style_rag.py
```

Commit:

```bash
git add affectiveart/track2_v16_official_style_rag.py scripts/track2_v16_official_style_rag.py tests/test_track2_v16_official_style_rag.py experiments/track2_v16_official_style_rag_20260607/calibration
git commit -m "feat: add track2 v16 official counterfactual calibration"
```

## Task 2: Build RAG Teacher Queue

**Files:**
- Modify: `affectiveart/track2_v16_official_style_rag.py`
- Modify: `scripts/track2_v16_official_style_rag.py`
- Test: `tests/test_track2_v16_official_style_rag.py`

- [ ] **Step 1: Write tests for queue priority**

```python
from affectiveart.track2_v16_official_style_rag import build_rag_queue_rows


def test_build_rag_queue_prioritizes_disagreement_and_blocks_failed_transitions():
    current_rows = [
        {"sample_id": "track2_0001", "emotion": "calm"},
        {"sample_id": "track2_0002", "emotion": "content"},
    ]
    prediction_rows = {
        "track2_0001": {"target_emotion": "content", "sources": ["same_quadrant_batch"], "confidence": 0.9},
        "track2_0002": {"target_emotion": "glad", "sources": ["rag_teacher", "multibackbone_consensus"], "confidence": 0.8},
    }
    queue = build_rag_queue_rows(
        current_rows=current_rows,
        prediction_rows=prediction_rows,
        failed_transition_counts={"calm->content": 43},
    )
    by_id = {row["sample_id"]: row for row in queue}
    assert by_id["track2_0001"]["risk_gate"] == "blocked_by_failed_official_batch"
    assert by_id["track2_0002"]["risk_gate"] == "allow_strong_consensus"
```

- [ ] **Step 2: Implement queue construction**

Queue inputs:

- Current base: `submissions/track2_submission_v15_desc_expand300_candidate.json`
- Public duplicate evidence:
  - `experiments/track2_emoart130k_clip/deep_duplicate_audit_20260511/track2_deep_duplicate_top10_audit.json`
  - `experiments/track2_emoart130k_clip/overlap_reference/ge095_all/ge095_public_neighbor_audit.json`
- Prediction sources:
  - `experiments/track2_emoart130k_siglip2/predictions.json`
  - `experiments/track2_emoart130k_clip/predictions.json`
  - `experiments/track2_emoart130k_dinov2/predictions.json`
  - `experiments/track2_moe_specialist_ensemble_20260603/dry_run_v2/gemini35_vlm_specialist_predictions.json`

Queue row fields:

- `sample_id`
- `current_emotion`
- `proposed_emotion`
- `transition`
- `risk_gate`
- `priority_score`
- `sources`
- `public_neighbor_labels`
- `same_valence`
- `same_arousal`
- `reason`

Priority rule:

```text
priority = 4.0 * exact_duplicate
         + 2.5 * multibackbone_agreement
         + 2.0 * rag_needed_boundary_case
         + 1.0 * public_neighbor_agreement
         - 5.0 * blocked_by_failed_official_batch
```

- [ ] **Step 3: Add CLI `build-queue`**

Run:

```bash
python3 scripts/track2_v16_official_style_rag.py build-queue \
  --base-json submissions/track2_submission_v15_desc_expand300_candidate.json \
  --out-dir experiments/track2_v16_official_style_rag_20260607/rag_queue \
  --limit 160
```

Expected outputs:

- `rag_teacher_queue_top160.csv`
- `rag_teacher_queue_top160.json`
- `rag_teacher_queue_top160.md`
- `sample_ids_top160.txt`

- [ ] **Step 4: Verify and commit**

Run:

```bash
python3 -m unittest tests/test_track2_v16_official_style_rag.py -v
python3 scripts/track2_v16_official_style_rag.py build-queue \
  --base-json submissions/track2_submission_v15_desc_expand300_candidate.json \
  --out-dir experiments/track2_v16_official_style_rag_20260607/rag_queue \
  --limit 160
git diff --check -- affectiveart/track2_v16_official_style_rag.py scripts/track2_v16_official_style_rag.py tests/test_track2_v16_official_style_rag.py
```

Commit:

```bash
git add affectiveart/track2_v16_official_style_rag.py scripts/track2_v16_official_style_rag.py tests/test_track2_v16_official_style_rag.py experiments/track2_v16_official_style_rag_20260607/rag_queue
git commit -m "feat: build track2 v16 rag teacher queue"
```

## Task 3: Run Gemini RAG Teacher And Apply Decisions

**Files:**
- Modify: `affectiveart/track2_v16_official_style_rag.py`
- Modify: `scripts/track2_v16_official_style_rag.py`
- Test: `tests/test_track2_v16_official_style_rag.py`

- [ ] **Step 1: Add decision acceptance tests**

```python
from affectiveart.track2_v16_official_style_rag import should_accept_v16_decision


def test_should_accept_v16_decision_requires_strong_evidence_for_label_change():
    decision = {
        "sample_id": "track2_0001",
        "current_emotion": "calm",
        "proposed_emotion": "content",
        "confidence": 0.95,
        "evidence_sources": ["same_quadrant_batch"],
        "risk_gate": "blocked_by_failed_official_batch",
    }
    assert not should_accept_v16_decision(decision)


def test_should_accept_v16_decision_allows_rag_and_backbone_consensus():
    decision = {
        "sample_id": "track2_0002",
        "current_emotion": "content",
        "proposed_emotion": "glad",
        "confidence": 0.88,
        "evidence_sources": ["rag_teacher", "multibackbone_consensus"],
        "risk_gate": "allow_strong_consensus",
    }
    assert should_accept_v16_decision(decision)
```

- [ ] **Step 2: Add CLI `run-rag-teacher`**

The command should call Gemini 3.5 Flash with:

- test image
- current row
- public neighbor label summary
- model disagreement summary
- official failed-batch warning
- strict 12-emotion JSON schema

Run:

```bash
.venv/bin/python scripts/track2_v16_official_style_rag.py run-rag-teacher \
  --base-json submissions/track2_submission_v15_desc_expand300_candidate.json \
  --image-dir data/raw/track2_testset_20260507/track2_testset/images \
  --queue-json experiments/track2_v16_official_style_rag_20260607/rag_queue/rag_teacher_queue_top160.json \
  --decisions-jsonl experiments/track2_v16_official_style_rag_20260607/rag_teacher/decisions_top160_gemini35.jsonl \
  --model gemini-3.5-flash \
  --limit 160
```

Expected behavior:

- Resume-safe: skip sample IDs already present in decisions JSONL.
- No formal submission files are touched.
- If quota fails, preserve partial decisions and report the retry reason.

- [ ] **Step 3: Add CLI `apply`**

Run:

```bash
python3 scripts/track2_v16_official_style_rag.py apply \
  --base-json submissions/track2_submission_v15_desc_expand300_candidate.json \
  --decisions-jsonl experiments/track2_v16_official_style_rag_20260607/rag_teacher/decisions_top160_gemini35.jsonl \
  --out-json submissions/track2_submission_v16_rag_consensus_candidate.json \
  --out-zip submissions/track2_submission_v16_rag_consensus_candidate.zip \
  --report-json experiments/track2_v16_official_style_rag_20260607/candidates/v16_rag_consensus_report.json \
  --report-md experiments/track2_v16_official_style_rag_20260607/candidates/v16_rag_consensus_report.md \
  --max-changes 40
```

Expected outputs:

- Candidate JSON/ZIP under `submissions/`.
- Report with transition counts, accepted/blocked counts, and distribution.
- No description fields rewritten in this step.

- [ ] **Step 4: Verify and commit reports only**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v16_rag_consensus_candidate.json
python3 -m unittest tests/test_track2_v16_official_style_rag.py -v
git diff --check -- affectiveart/track2_v16_official_style_rag.py scripts/track2_v16_official_style_rag.py tests/test_track2_v16_official_style_rag.py
```

Commit code and reports, not raw AI decisions:

```bash
git add affectiveart/track2_v16_official_style_rag.py scripts/track2_v16_official_style_rag.py tests/test_track2_v16_official_style_rag.py experiments/track2_v16_official_style_rag_20260607/candidates
git commit -m "feat: build track2 v16 rag consensus candidate"
```

## Task 4: Ablation Score And Submission Gate

**Files:**
- Modify: `affectiveart/track2_v16_official_style_rag.py`
- Modify: `scripts/track2_v16_official_style_rag.py`
- Test: `tests/test_track2_v16_official_style_rag.py`

- [ ] **Step 1: Generate nested ablation candidates**

Create nested candidates from the accepted v16 decisions:

- top 5 label changes
- top 10 label changes
- top 20 label changes
- top 40 label changes

Run:

```bash
python3 scripts/track2_v16_official_style_rag.py ablation-score \
  --base-json submissions/track2_submission_v15_desc_expand300_candidate.json \
  --candidate-json submissions/track2_submission_v16_rag_consensus_candidate.json \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --out-dir experiments/track2_v16_official_style_rag_20260607/ablation_score
```

Expected outputs:

- `v16_ablation_candidates.json`
- `v16_ablation_summary.csv`
- `v16_ablation_summary.md`
- `fused_shadow_compare/fused_shadow_score_report.md`

- [ ] **Step 2: Enforce final gate**

The final gate must write one of:

- `recommend_submit_v16`
- `hold_v16_keep_v15`
- `submit_only_as_official_probe`

Gate rule:

```text
recommend_submit_v16 only if:
  validate-track2 OK
  accepted label changes <= 40
  cross-quadrant changes <= 3
  no failed-batch-only transitions
  fused overall expected > v15_desc_expand300
  fused overall lower >= v15_desc_expand300 lower OR explicit probe rationale exists
```

- [ ] **Step 3: Verify**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v16_rag_consensus_candidate.json
python3 scripts/track2_fused_shadow_evaluator.py score \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate official_779605_moe_v2_anchor=submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate v12_stable_probe=submissions/track2_submission_v12_stable_probe_candidate.json \
  --candidate v15_desc_expand300=submissions/track2_submission_v15_desc_expand300_candidate.json \
  --candidate v16_rag_consensus=submissions/track2_submission_v16_rag_consensus_candidate.json \
  --out-dir experiments/track2_v16_official_style_rag_20260607/final_fused_compare
python3 -m unittest tests/test_track2_v16_official_style_rag.py -v
git diff --check -- affectiveart/track2_v16_official_style_rag.py scripts/track2_v16_official_style_rag.py tests/test_track2_v16_official_style_rag.py
```

- [ ] **Step 4: Commit final gate report**

Commit:

```bash
git add experiments/track2_v16_official_style_rag_20260607/ablation_score experiments/track2_v16_official_style_rag_20260607/final_fused_compare
git commit -m "docs: record track2 v16 ablation gate"
```

## Submission Decision

Do not submit automatically.

After Task 4, report:

- Best local candidate ZIP path.
- Whether v16 beats `v15_desc_expand300`.
- Whether the result is a real candidate or only an official-score probe.
- Exact remaining risk, especially if local lower bound is still below the current official best.

## Self-Review

- Spec coverage: The plan covers official-score calibration, RAG teacher queue construction, candidate application, ablation, validation, and submission gate.
- Placeholder scan: No `TBD`, `TODO`, or unspecified tests remain.
- Type consistency: Function names used in tests match the implementation function names defined in tasks.
- Track2-only boundary: The plan does not touch Track1 files or formal Track2 submission paths.
