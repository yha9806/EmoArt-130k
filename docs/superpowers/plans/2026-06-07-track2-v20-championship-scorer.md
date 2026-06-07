# Track2 v20 Championship Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track2-only v20 championship scorer and candidate ladder that can justify one high-variance remaining submission aimed at first place.

**Architecture:** v20 layers a directional failed-batch interpreter and championship evidence scorer over the existing v17/v18/v19 artifacts. It keeps `v15_desc_expand300` as the text anchor, writes only side-path candidates, and produces an explicit final gate that distinguishes safe, probe, and last-shot roles.

**Tech Stack:** Python standard library, existing Track2 JSON schema conventions, existing `unittest` test suite, existing `affectiveart.challenge validate-track2` validator.

---

## File Structure

- Create `affectiveart/track2_v20_championship_scorer.py`
  - Owns frontier math, directional transition risk, v20 evidence scoring, candidate selection/writing, and final gate reports.
- Create `scripts/track2_v20_championship_scorer.py`
  - Thin CLI wrapper around the v20 module.
- Create `tests/test_track2_v20_championship_scorer.py`
  - Tests frontier requirement math, directional risk interpretation, evidence scoring thresholds, candidate path safety, and final gate behavior.
- Output under `experiments/track2_v20_championship_scorer_20260607/`
  - `frontier_requirements.json/.md`
  - `directional_risk_map.json/.md`
  - `championship_evidence.json/.csv`
  - `candidate_reports/*.json/.md`
  - `final_gate_report.json/.md`
- Candidate files under `submissions/`
  - `track2_submission_v20_precision_candidate.json/.zip`
  - `track2_submission_v20_champion_probe_candidate.json/.zip`
  - `track2_submission_v20_last_shot_candidate.json/.zip`

## Defaults

```text
base_json=submissions/track2_submission_v15_desc_expand300_candidate.json
leaderboard_csv=experiments/track2_official_results_20260606/track2_public_leaderboard_20260606.csv
pairwise_diffs=experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv
v17_evidence=experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json
experiment_dir=experiments/track2_v20_championship_scorer_20260607
submissions_dir=submissions
```

## Task 1: Frontier Requirement Model

**Files:**
- Create: `affectiveart/track2_v20_championship_scorer.py`
- Create: `scripts/track2_v20_championship_scorer.py`
- Create: `tests/test_track2_v20_championship_scorer.py`

- [ ] **Step 1: Write failing frontier tests**

Add tests for:

```python
def test_required_classification_for_target_overall_uses_official_weighting():
    self.assertAlmostEqual(required_classification_for_overall(0.89, 1.0), 0.78)
    self.assertAlmostEqual(required_classification_for_overall(0.89, 0.98), 0.80)

def test_frontier_report_identifies_classification_and_description_gap():
    rows = [
        {"Participant": "N&T", "Overall Score": "0.89", "Classification Score": "0.78", "Description Score": "1.0", "Emotion Accuracy": "0.80", "Emotion Macro F1": "0.31"},
        {"Participant": "vulcaart", "Overall Score": "0.84", "Classification Score": "0.72", "Description Score": "0.95", "Emotion Accuracy": "0.57", "Emotion Macro F1": "0.31"},
    ]
    report = build_frontier_requirement_report(rows, participant="vulcaart", target_overall=0.89)
    self.assertGreaterEqual(report["required_classification_if_description_1.00"], 0.78)
    self.assertGreater(report["emotion_accuracy_gap"], 0.20)
    self.assertEqual(report["macro_f1_gap"], 0.0)
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
python3 -m unittest tests/test_track2_v20_championship_scorer.py -v
```

Expected: import error for missing v20 module.

- [ ] **Step 3: Implement frontier functions**

Public functions:

```python
required_classification_for_overall(target_overall: float, description_score: float) -> float
build_frontier_requirement_report(rows: list[dict[str, Any]], participant: str = "vulcaart", target_overall: float = 0.89) -> dict[str, Any]
```

Implementation rules:

- Use `overall = 0.5 * classification + 0.5 * description`.
- Select the frontier row by highest visible overall.
- Do not infer hidden labels.

- [ ] **Step 4: Add CLI wrapper**

Create the script wrapper importing `main()` from the v20 module.

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m unittest tests/test_track2_v20_championship_scorer.py -v
git diff --check
```

## Task 2: Directional Failed-Batch Risk

**Files:**
- Modify: `affectiveart/track2_v20_championship_scorer.py`
- Modify: `tests/test_track2_v20_championship_scorer.py`

- [ ] **Step 1: Write failing directional risk tests**

Add tests for:

```python
def test_directional_risk_marks_dominant_failed_direction_dangerous():
    rows = [{"top_emotion_transitions": "calm->content:43; content->calm:20"}]
    risk = build_directional_risk_map(rows)
    self.assertEqual(risk["calm->content"]["risk"], "danger")
    self.assertEqual(risk["content->calm"]["risk"], "caution")

def test_directional_risk_blocks_weak_calm_to_content_but_allows_strong_content_to_calm():
    risk = build_directional_risk_map([{"top_emotion_transitions": "calm->content:43; content->calm:20"}])
    weak = score_v20_evidence_row(_row("calm", "content", support=3.0, votes=3), directional_risk=risk)
    strong = score_v20_evidence_row(_row("content", "calm", support=4.0, votes=3, near=True, duplicate=0.96, confidence=0.96), directional_risk=risk)
    self.assertEqual(weak["decision"], "block")
    self.assertEqual(strong["decision"], "accept_candidate")
```

- [ ] **Step 2: Implement directional risk**

Public function:

```python
build_directional_risk_map(pairwise_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]
```

Rules:

- Parse `top_emotion_transitions`.
- If one direction dominates its reverse by at least `2x` and count is at least `20`, mark the dominant direction `danger`.
- Mark the reverse `caution`, not `danger`.
- Mark smaller same-quadrant failed transitions `caution`.

- [ ] **Step 3: Implement v20 row scorer**

Public function:

```python
score_v20_evidence_row(row: dict[str, Any], *, directional_risk: dict[str, dict[str, Any]]) -> dict[str, Any]
```

Rules:

- Aggregate pressure cannot accept rows alone.
- `danger` transitions require exact duplicate override; otherwise `block`.
- `caution` transitions can be accepted only with `model_vote_count >= 3`, `support_score >= 3.2`, and either near/exact duplicate support `>= 0.95` or `max_confidence >= 0.94`.
- Cross-quadrant transitions are blocked unless exact duplicate support `>= 0.98`.
- Return `decision`, `score`, `role`, and `reasons`.

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m unittest tests/test_track2_v20_championship_scorer.py -v
git diff --check
```

## Task 3: Candidate Ladder And Final Gate

**Files:**
- Modify: `affectiveart/track2_v20_championship_scorer.py`
- Modify: `tests/test_track2_v20_championship_scorer.py`

- [ ] **Step 1: Write candidate safety and gate tests**

Add tests for:

```python
def test_write_v20_candidate_rejects_formal_submission_name():
    with self.assertRaises(ValueError):
        write_v20_candidate_outputs(base_rows=[_base_row()], selected_changes=[], out_json=Path("submissions/track2_submission.json"), out_zip=Path("submissions/x.zip"), report_json=Path("experiments/r.json"), report_md=Path("experiments/r.md"), profile="precision")

def test_choose_v20_final_gate_requires_championship_path():
    report = choose_v20_final_gate(candidate_name="v20_precision", accepted_label_changes=2, description_anchor_preserved=True, label_consistency_issue_count=0, missing_emotions=[], frontier_classification_gap=0.06)
    self.assertEqual(report["decision"], "hold")
    self.assertIn("insufficient_championship_classification_lift", report["reasons"])
```

- [ ] **Step 2: Implement candidate selection**

Public functions:

```python
build_v20_championship_evidence(v17_rows: list[dict[str, Any]], directional_risk: dict[str, dict[str, Any]]) -> list[dict[str, Any]]
select_v20_changes(evidence_rows: list[dict[str, Any]], profile: str, base_distribution: Counter[str]) -> list[dict[str, Any]]
```

Profiles:

- `precision`: cap `18`, caution transition cap `12`.
- `champion_probe`: cap `56`, caution transition cap `48`.
- `last_shot`: cap `96`, caution transition cap `80`.

- [ ] **Step 3: Implement candidate writer and final gate**

Public functions:

```python
write_v20_candidate_outputs(...)
write_v20_candidate_ladder_outputs(...)
choose_v20_final_gate(...)
write_v20_final_gate_report(...)
```

Rules:

- Write deterministic ZIP payload with `submission.json`.
- Repair valence/arousal from emotion.
- Reject formal submission names.
- Gate `precision` as safe probe only; gate `champion_probe` or `last_shot` as high-variance candidates only if they have enough accepted changes and no validation issues.

- [ ] **Step 4: Run tests and generate artifacts**

Run:

```bash
python3 -m unittest tests/test_track2_v20_championship_scorer.py -v
python3 scripts/track2_v20_championship_scorer.py run
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v20_precision_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v20_champion_probe_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v20_last_shot_candidate.json
git diff --check
```

## Task 4: Track2 Verification And Decision

**Files:**
- Read generated reports only.

- [ ] **Step 1: Run Track2 tests**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
```

- [ ] **Step 2: Summarize decision**

Report:

- best v20 candidate path,
- accepted label changes,
- dominant transitions,
- whether it is a precision probe, championship probe, or last-shot candidate,
- whether it is worth one of the final two Codabench submissions.

