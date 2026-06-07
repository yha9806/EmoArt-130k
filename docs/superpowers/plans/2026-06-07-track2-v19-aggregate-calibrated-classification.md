# Track2 v19 Aggregate-Calibrated Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track2-only v19 local experiment that uses aggregate-aware classification calibration to produce and gate precision/balanced/probe candidates without uploading to Codabench.

**Architecture:** v19 reads official aggregate scores, local distributions, v17 evidence, and model prediction files; derives aggregate target bands; scores candidate label changes with row-level evidence plus aggregate constraints; writes side-path candidate JSON/ZIP files; and produces a final no-auto-submit gate. It preserves v15 description text and focuses on classification lift.

**Tech Stack:** Python standard library, existing Track2 JSON schema conventions, existing local/fused shadow scorer modules, `unittest`.

---

## File Structure

- Create `affectiveart/track2_v19_aggregate_calibration.py`
  - Owns leaderboard aggregate loading, target-band derivation, evidence scoring, candidate search, candidate writing, final gate reports, and CLI entrypoint.
- Create `scripts/track2_v19_aggregate_calibration.py`
  - Thin wrapper around the v19 module.
- Create `tests/test_track2_v19_aggregate_calibration.py`
  - Unit tests for aggregate loading, historical ordering constraints, row evidence gating, candidate writer safety, and CLI smoke behavior.
- Output under `experiments/track2_v19_aggregate_calibration_20260607/`
  - `aggregate_targets.json/.md`
  - `calibrated_evidence.json/.csv`
  - `candidate_reports/*.json/.md`
  - `fused_shadow_compare/*`
  - `final_gate_report.json/.md`
- Candidate files under `submissions/`
  - `track2_submission_v19_precision_candidate.json/.zip`
  - `track2_submission_v19_balanced_candidate.json/.zip`
  - `track2_submission_v19_probe_candidate.json/.zip`

## Defaults

Use these default inputs:

```text
base_json=submissions/track2_submission_v15_desc_expand300_candidate.json
anchor_json=submissions/track2_submission_moe_v2_accept5_candidate.json
leaderboard_csv=experiments/track2_official_results_20260606/track2_public_leaderboard_20260606.csv
official_scores=experiments/track2_official_results_20260606/track2_known_official_exact_scores_from_ledger_20260606.csv
local_distributions=experiments/track2_official_results_20260606/track2_my_submission_local_distributions_20260606.csv
pairwise_diffs=experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv
v17_evidence=experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json
v15_json=submissions/track2_submission_v15_desc_expand300_candidate.json
v12_json=submissions/track2_submission_v12_stable_probe_candidate.json
experiment_dir=experiments/track2_v19_aggregate_calibration_20260607
```

---

## Task 1: Aggregate Targets

**Files:**
- Create: `affectiveart/track2_v19_aggregate_calibration.py`
- Create: `scripts/track2_v19_aggregate_calibration.py`
- Create: `tests/test_track2_v19_aggregate_calibration.py`

- [ ] **Step 1: Write failing tests for leaderboard aggregate parsing**

Add tests that define a tiny leaderboard CSV and assert:

```python
def test_load_leaderboard_aggregates_extracts_our_gap_and_frontier():
    rows = load_leaderboard_aggregates(path)
    target = derive_aggregate_targets(rows, participant="vulcaart")
    self.assertEqual(target["our"]["emotion_accuracy"], 0.57)
    self.assertEqual(target["frontier"]["emotion_accuracy"], 0.80)
    self.assertGreater(target["gaps"]["emotion_accuracy"], 0.20)
    self.assertLess(abs(target["gaps"]["emotion_macro_f1"]), 0.02)
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
python3 -m unittest tests/test_track2_v19_aggregate_calibration.py -v
```

Expected: import error for missing v19 module.

- [ ] **Step 3: Implement aggregate loading**

Add these public functions:

```python
load_leaderboard_aggregates(path: str | Path) -> list[dict[str, Any]]
derive_aggregate_targets(rows: list[dict[str, Any]], participant: str = "vulcaart") -> dict[str, Any]
```

Requirements:

- Parse floats from public leaderboard columns.
- Identify our row by participant.
- Identify frontier row by highest overall score.
- Return gaps for classification, description, emotion accuracy, emotion macro-F1, valence accuracy/F1, and arousal accuracy/F1.
- Do not infer hidden gold labels.

- [ ] **Step 4: Add CLI wrapper**

Create `scripts/track2_v19_aggregate_calibration.py` matching existing script wrappers:

```python
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_v19_aggregate_calibration import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python3 -m unittest tests/test_track2_v19_aggregate_calibration.py -v
git diff --check
```

Commit:

```bash
git add affectiveart/track2_v19_aggregate_calibration.py scripts/track2_v19_aggregate_calibration.py tests/test_track2_v19_aggregate_calibration.py
git commit -m "feat: add track2 v19 aggregate targets"
```

---

## Task 2: Calibrated Evidence Scoring

**Files:**
- Modify: `affectiveart/track2_v19_aggregate_calibration.py`
- Modify: `tests/test_track2_v19_aggregate_calibration.py`

- [ ] **Step 1: Write tests for v17 evidence scoring**

Add tests for:

```python
def test_score_evidence_penalizes_known_failed_transition():
    row = {"transition": "calm->content", "support_score": 3.0, "model_vote_count": 3, "failed_transition_count": 43}
    scored = score_v19_evidence_row(row, failed_transition_penalty_weight=0.04)
    self.assertEqual(scored["decision"], "hold")
    self.assertIn("official_failed_transition_penalty", scored["reasons"])

def test_score_evidence_requires_row_level_support_even_when_aggregate_pressure_exists():
    row = {"transition": "calm->happy", "support_score": 0.0, "model_vote_count": 0}
    scored = score_v19_evidence_row(row, aggregate_pressure=1.0)
    self.assertEqual(scored["decision"], "hold")
    self.assertIn("insufficient_row_support", scored["reasons"])

def test_781601_failed_same_quadrant_batch_is_negative_regression_case():
    rows = [
        {"transition": "calm->content", "support_score": 3.0, "model_vote_count": 3, "failed_transition_count": 43},
        {"transition": "content->calm", "support_score": 3.0, "model_vote_count": 3, "failed_transition_count": 20},
    ]
    report = summarize_781601_regression_guard(rows)
    self.assertEqual(report["decision"], "block_bulk_same_quadrant_repeat")
    self.assertGreaterEqual(report["failed_same_quadrant_count"], 63)
```

- [ ] **Step 2: Implement evidence scoring**

Add this public function:

```python
score_v19_evidence_row(
    row: dict[str, Any],
    *,
    aggregate_pressure: float = 0.0,
    failed_transition_penalty_weight: float = 0.04,
) -> dict[str, Any]
```

Requirements:

- Aggregate pressure may increase rank but cannot accept a row alone.
- Block rows with invalid current/proposed emotion.
- Penalize known failed transitions from official feedback.
- Reward exact/near public duplicate evidence only when confidence is high.
- Keep a `decision` field: `accept_candidate`, `hold`, or `block`.
- Provide `summarize_781601_regression_guard(rows: list[dict[str, Any]]) -> dict[str, Any]` so final reports explicitly show that the previous 85-row same-quadrant failure is blocked.

- [ ] **Step 3: Build calibrated evidence table**

Add this public function:

```python
build_v19_calibrated_evidence(
    v17_evidence_rows: list[dict[str, Any]],
    aggregate_targets: dict[str, Any],
) -> list[dict[str, Any]]
```

Requirements:

- Preserve all original v17 evidence fields.
- Add `v19_score`, `v19_decision`, and `v19_reasons`.
- Sort by decision, descending score, sample ID.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
python3 -m unittest tests/test_track2_v19_aggregate_calibration.py -v
git diff --check
```

Commit:

```bash
git add affectiveart/track2_v19_aggregate_calibration.py tests/test_track2_v19_aggregate_calibration.py
git commit -m "feat: score track2 v19 calibrated evidence"
```

---

## Task 3: Candidate Ladder Writer

**Files:**
- Modify: `affectiveart/track2_v19_aggregate_calibration.py`
- Modify: `tests/test_track2_v19_aggregate_calibration.py`

- [ ] **Step 1: Write tests for candidate writing**

Add tests that assert:

```python
def test_write_candidates_rejects_formal_submission_names():
    base_rows = [
        {
            "sample_id": "track2_0001",
            "emotion": "tired",
            "emotional_valence": "Negative",
            "emotional_arousal_level": "Low",
            "caption": "A subdued figure in a quiet interior.",
            "brushstroke": "Soft brushwork.",
            "composition": "Centered composition.",
            "color": "Muted colors.",
            "line": "Restrained lines.",
            "light": "Dim light.",
        }
    ]
    with self.assertRaises(ValueError):
        write_v19_candidate_outputs(
            base_rows=base_rows,
            selected_changes=[],
            out_json=Path("submissions/track2_submission.json"),
            out_zip=Path("submissions/track2_submission_v19_precision_candidate.zip"),
            report_json=Path("experiments/v19_report.json"),
            report_md=Path("experiments/v19_report.md"),
            profile="precision",
        )

def test_build_candidate_ladder_applies_only_accepted_unique_rows():
    base_rows = [
        {
            "sample_id": "track2_0001",
            "emotion": "tired",
            "emotional_valence": "Negative",
            "emotional_arousal_level": "Low",
            "caption": "A subdued figure in a quiet interior.",
            "brushstroke": "Soft brushwork.",
            "composition": "Centered composition.",
            "color": "Muted colors.",
            "line": "Restrained lines.",
            "light": "Dim light.",
        }
    ]
    evidence_rows = [
        {
            "sample_id": "track2_0001",
            "current_emotion": "tired",
            "proposed_emotion": "sad",
            "transition": "tired->sad",
            "v19_score": 2.5,
            "v19_decision": "accept_candidate",
            "same_arousal": True,
            "same_valence": True,
        },
        {
            "sample_id": "track2_0001",
            "current_emotion": "tired",
            "proposed_emotion": "calm",
            "transition": "tired->calm",
            "v19_score": 2.4,
            "v19_decision": "accept_candidate",
            "same_arousal": True,
            "same_valence": False,
        },
    ]
    selected = select_v19_changes(evidence_rows, profile="precision", base_distribution={"tired": 1})
    report = summarize_v19_candidate(base_rows=base_rows, selected_changes=selected, profile="precision")
    self.assertEqual(report["candidates"]["precision"]["accepted_label_changes"], 1)
    self.assertEqual(report["candidates"]["precision"]["transition_counts"], {"tired->sad": 1})
```

- [ ] **Step 2: Implement candidate selection profiles**

Implement:

```python
PROFILE_CONFIG = {
    "precision": {"total_cap": 24, "min_score": 2.20, "cross_quadrant_cap": 4, "same_quadrant_cap": 12},
    "balanced": {"total_cap": 64, "min_score": 1.70, "cross_quadrant_cap": 10, "same_quadrant_cap": 32},
    "probe": {"total_cap": 120, "min_score": 1.20, "cross_quadrant_cap": 20, "same_quadrant_cap": 80},
}
```

Selection must:

- enforce one proposed label per sample,
- keep rare current classes above floor,
- cap transition families,
- reject no-op transitions,
- mechanically repair valence/arousal from emotion.

- [ ] **Step 3: Implement JSON/ZIP writer**

Write ZIP files with exactly `submission.json`, fixed timestamp, and no formal submission overwrite.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
python3 -m unittest tests/test_track2_v19_aggregate_calibration.py -v
git diff --check
```

Commit:

```bash
git add affectiveart/track2_v19_aggregate_calibration.py tests/test_track2_v19_aggregate_calibration.py
git commit -m "feat: write track2 v19 candidate ladder"
```

---

## Task 4: Local Scoring And Final Gate

**Files:**
- Modify: `affectiveart/track2_v19_aggregate_calibration.py`
- Modify: `tests/test_track2_v19_aggregate_calibration.py`

- [ ] **Step 1: Write final gate tests**

Add tests:

```python
def test_final_gate_holds_when_classification_lower_does_not_beat_anchor():
    report = choose_v19_final_gate(
        candidate_name="v19_precision",
        overall_lower=0.840,
        v15_overall_lower=0.835,
        classification_lower=0.722,
        description_lower=0.949,
        v15_description_lower=0.949,
        label_consistency_issue_count=0,
        missing_emotions=[],
        emotion_accuracy_proxy_delta=0.06,
        emotion_macro_f1_proxy_delta=0.0,
        historical_risk=0.05,
    )
    self.assertEqual(report["decision"], "hold")

def test_final_gate_recommends_when_candidate_clears_strict_thresholds():
    report = choose_v19_final_gate(
        candidate_name="v19_balanced",
        overall_lower=0.842,
        v15_overall_lower=0.835,
        classification_lower=0.742,
        description_lower=0.949,
        v15_description_lower=0.949,
        label_consistency_issue_count=0,
        missing_emotions=[],
        emotion_accuracy_proxy_delta=0.06,
        emotion_macro_f1_proxy_delta=0.0,
        historical_risk=0.02,
    )
    self.assertEqual(report["decision"], "recommend_submit")

def test_final_gate_holds_without_material_emotion_accuracy_lift():
    report = choose_v19_final_gate(
        candidate_name="v19_balanced",
        overall_lower=0.842,
        v15_overall_lower=0.835,
        classification_lower=0.742,
        description_lower=0.949,
        v15_description_lower=0.949,
        label_consistency_issue_count=0,
        missing_emotions=[],
        emotion_accuracy_proxy_delta=0.02,
        emotion_macro_f1_proxy_delta=0.0,
        historical_risk=0.02,
    )
    self.assertEqual(report["decision"], "hold")
    self.assertIn("emotion_accuracy_lift_below_0.05", report["reasons"])
```

- [ ] **Step 2: Implement final gate**

Add this public function:

```python
choose_v19_final_gate(
    *,
    candidate_name: str,
    overall_lower: float,
    v15_overall_lower: float,
    classification_lower: float,
    description_lower: float,
    v15_description_lower: float,
    label_consistency_issue_count: int,
    missing_emotions: list[str],
    emotion_accuracy_proxy_delta: float,
    emotion_macro_f1_proxy_delta: float,
    historical_risk: float,
) -> dict[str, Any]
```

Gate conditions:

- validator OK is required,
- label consistency issue count must be `0`,
- missing emotions must be empty,
- description lower must not drop below v15 by more than `0.003`,
- classification lower must exceed `0.723150`,
- local emotion accuracy proxy delta must be at least `0.05`,
- local emotion macro-F1 proxy delta must be at least `-0.005`,
- overall lower must exceed `v15_desc_expand300` lower,
- historical failed-transition risk must be under the configured threshold,
- no auto-submit.

- [ ] **Step 3: Integrate fused shadow evaluator**

Call existing `track2_fused_shadow_evaluator` with candidates:

- `official_779605_moe_v2_anchor`,
- `official_782683_v12_stable_probe`,
- `v15_desc_expand300`,
- `v19_precision`,
- `v19_balanced`,
- `v19_probe`.

- [ ] **Step 4: Write aggregate calibration ablation report**

Generate `experiments/track2_v19_aggregate_calibration_20260607/ablation_report.json/.md` with rows for:

- `no_aggregate_calibration`,
- `global_aggregate_calibration`,
- `transition_family_constrained_calibration`.

Each row must include changed rows, same-quadrant changes, cross-quadrant changes, emotion accuracy proxy delta, emotion macro-F1 proxy delta, classification lower, overall lower, and whether it repeats the `781601` failure family.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python3 -m unittest tests/test_track2_v19_aggregate_calibration.py -v
git diff --check
```

Commit:

```bash
git add affectiveart/track2_v19_aggregate_calibration.py tests/test_track2_v19_aggregate_calibration.py
git commit -m "feat: gate track2 v19 candidates"
```

---

## Task 5: Real v19 Run And Decision Report

**Files:**
- Modify: experiment outputs under `experiments/track2_v19_aggregate_calibration_20260607/`
- Candidate files under `submissions/` are generated but ignored unless project policy later changes.

- [ ] **Step 1: Run v19 build**

Run:

```bash
python3 scripts/track2_v19_aggregate_calibration.py build \
  --base-json submissions/track2_submission_v15_desc_expand300_candidate.json \
  --v17-evidence experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json \
  --leaderboard-csv experiments/track2_official_results_20260606/track2_public_leaderboard_20260606.csv \
  --experiment-dir experiments/track2_v19_aggregate_calibration_20260607
```

- [ ] **Step 2: Validate generated candidates**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v19_precision_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v19_balanced_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v19_probe_candidate.json
```

- [ ] **Step 3: Run full Track2 tests and diff check**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
git diff --check
```

- [ ] **Step 4: Commit reports if verified**

Commit only tracked code/tests/docs and non-ignored experiment reports:

```bash
git add experiments/track2_v19_aggregate_calibration_20260607
git commit -m "docs: record track2 v19 aggregate calibration run"
```

- [ ] **Step 5: Report decision**

Tell the user:

- exact recommended ZIP path if final gate says `recommend_submit`,
- exact hold reasons if final gate says `hold`,
- comparison against `779605`, `782683`, and `v15_desc_expand300`,
- whether this is worth one of the two remaining Codabench submissions.
