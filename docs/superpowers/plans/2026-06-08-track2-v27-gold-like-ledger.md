# Track2 v27 Gold-Like Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and score a final-shot Track2 v27 candidate from strict public/author evidence without overwriting formal submission files.

**Architecture:** Add a focused `track2_v27_gold_like_ledger` module that loads existing v17/v26/deep-duplicate evidence, classifies each proposal into evidence tiers, writes a ledger, assembles strict/frontier sensitivity candidates, and gates them with the v23 official-anchor proxy. Keep CLI orchestration in `scripts/track2_v27_gold_like_ledger.py`; keep tests in one focused unittest file.

**Tech Stack:** Python standard library, existing Track2 JSON schema, existing `affectiveart.challenge validate-track2`, existing `affectiveart.track2_official_anchor_calibration.score_submission`.

---

### Task 1: Evidence Tier And Acceptance Rules

**Files:**
- Create: `tests/test_track2_v27_gold_like_ledger.py`
- Create: `affectiveart/track2_v27_gold_like_ledger.py`

- [ ] **Step 1: Write failing tests for strict duplicate acceptance and broad kNN rejection**

Add tests that import:

```python
from affectiveart.track2_v27_gold_like_ledger import (
    classify_v27_evidence_row,
    select_v27_changes,
)
```

Test behaviors:

```python
def test_classifies_visual_duplicate_as_tier1_accept() -> None:
    row = _evidence_row(near_duplicate=True, public_duplicate_support_score=0.97, model_vote_count=2)
    scored = classify_v27_evidence_row(row)
    assert scored["decision"] == "accept_candidate"
    assert scored["tier"] == "tier1_visual_duplicate"

def test_rejects_weak_knn_copy_without_duplicate_evidence() -> None:
    row = _evidence_row(near_duplicate=False, public_duplicate_support_score=0.0, model_vote_count=1)
    scored = classify_v27_evidence_row(row)
    assert scored["decision"] == "hold"
    assert "weak_model_or_style_only" in scored["reasons"]

def test_select_caps_same_series_and_avoids_duplicate_samples() -> None:
    rows = [
        {**_accepted_row("track2_0001", "content", "calm", 4.0), "v27_tier": "tier2_same_series"},
        {**_accepted_row("track2_0001", "content", "excited", 3.9), "v27_tier": "tier1_visual_duplicate"},
        {**_accepted_row("track2_0002", "calm", "content", 3.8), "v27_tier": "tier2_same_series"},
    ]
    selected = select_v27_changes(rows, base_distribution=Counter({"content": 100, "calm": 100}), tier2_cap=1)
    assert [row["sample_id"] for row in selected] == ["track2_0001"]
```

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m unittest tests/test_track2_v27_gold_like_ledger.py -v
```

Expected: import failure because `affectiveart.track2_v27_gold_like_ledger` does not exist yet.

- [ ] **Step 3: Implement minimal tiering functions**

Create `affectiveart/track2_v27_gold_like_ledger.py` with:

- constants for Track2 emotions, negative emotions, high-arousal emotions, text fields, and formal submission filenames;
- `_canonical_emotion`, `_safe_float`, `_safe_int`, `_safe_bool`, `_valence`, `_arousal`, `_same_quadrant`;
- `classify_v27_evidence_row(row)` returning `decision`, `tier`, `score`, `reasons`;
- `select_v27_changes(rows, base_distribution, total_cap=72, tier2_cap=24, top_emotion_cap=0.58, class_floor=4)`.

Acceptance rules:

- Tier 1 accepts exact/near duplicate when duplicate support is at least `0.95`, max confidence at least `0.90`, and model vote count is at least `2`.
- Tier 2 accepts same-series evidence only when support score is at least `2.25`, public style support at least `0.50`, model vote count at least `3`, and transition is same valence/arousal quadrant.
- Tier 3 never accepts; it returns `hold` with `weak_model_or_style_only`.
- Invalid labels and no-op transitions return `block`.

- [ ] **Step 4: Run GREEN**

Run:

```bash
python3 -m unittest tests/test_track2_v27_gold_like_ledger.py -v
```

Expected: tests pass for Task 1 behaviors.

### Task 2: Candidate Writer And Safety Report

**Files:**
- Modify: `tests/test_track2_v27_gold_like_ledger.py`
- Modify: `affectiveart/track2_v27_gold_like_ledger.py`

- [ ] **Step 1: Write failing tests for output safety and label repair**

Add tests for:

```python
from affectiveart.track2_v27_gold_like_ledger import write_v27_candidate_outputs
```

Behaviors:

- writing to `track2_submission.json` in a `submissions` directory raises `ValueError`;
- accepted `content->alarmed` repairs valence to `Negative` and arousal to `High`;
- ZIP contains exactly `submission.json`;
- report contains `formal_submission_overwritten=False`, `label_consistency_issue_count=0`, `accepted_label_changes=1`.

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m unittest tests/test_track2_v27_gold_like_ledger.py -v
```

Expected: failure because `write_v27_candidate_outputs` is missing.

- [ ] **Step 3: Implement writer helpers**

Add:

- `load_track2_rows(path)`;
- `_apply_changes(base_rows, selected_changes)`;
- `_label_issues(rows)`;
- `_write_json(path, payload)`;
- `_write_csv(path, rows)`;
- `_write_zip_payload(path, rows)`;
- `_render_candidate_md(report)`;
- `_reject_formal_submission_path(path)`;
- `write_v27_candidate_outputs(...)`.

Report fields:

- `method`;
- `profile`;
- `row_count`;
- `accepted_label_changes`;
- `transition_counts`;
- `tier_counts`;
- `cross_quadrant_change_count`;
- `distribution`;
- `missing_emotions`;
- `top_emotion`;
- `top_emotion_share`;
- `label_consistency_issue_count`;
- `label_consistency_issues`;
- `accepted_changes`;
- `formal_submission_overwritten`;
- `paths`.

- [ ] **Step 4: Run GREEN**

Run:

```bash
python3 -m unittest tests/test_track2_v27_gold_like_ledger.py -v
```

Expected: tests pass for Task 1 and Task 2 behaviors.

### Task 3: End-To-End Builder, Sensitivity Sweep, And Gate

**Files:**
- Modify: `tests/test_track2_v27_gold_like_ledger.py`
- Modify: `affectiveart/track2_v27_gold_like_ledger.py`
- Create: `scripts/track2_v27_gold_like_ledger.py`

- [ ] **Step 1: Write failing tests for final gate**

Add tests for:

```python
from affectiveart.track2_v27_gold_like_ledger import choose_v27_final_gate
```

Behaviors:

- if projected overall is below `0.89`, decision is `hold_no_submit`;
- if projected overall is at least `0.89`, label consistency issue count is `0`, missing emotions is empty, unsafe text count is `0`, and top emotion share is at most `0.58`, decision is `recommend_final_submit`;
- if cross-quadrant changes exist but the risk report count does not match them, decision is `hold_no_submit`.
- if the report's top emotion share exceeds `0.58`, decision is `hold_no_submit` with `top_emotion_collapse_risk`.

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m unittest tests/test_track2_v27_gold_like_ledger.py -v
```

Expected: failure because `choose_v27_final_gate` is missing.

- [ ] **Step 3: Implement builder and gate**

Add:

- `build_v27_ledger(v17_rows)` to score all evidence rows and produce sorted ledger rows;
- `choose_v27_final_gate(...)`;
- `build_v27_candidate_suite(...)`.

`build_v27_candidate_suite` writes two sensitivity profiles:

- `strict`: Tier 1 plus same-quadrant Tier 2 with `tier2_cap=12`;
- `frontier`: Tier 1 plus same-quadrant Tier 2 with `tier2_cap=36`.

For each profile, it records accepted changes, projected score, top emotion share, missing emotions, label consistency, cross-quadrant risk count, and decision. The summary chooses the highest-scoring profile only if it passes the final gate.

`build_v27_candidate_suite` writes:

- `experiments/track2_v27_gold_like_ledger_20260608/v27_gold_like_ledger.csv`;
- `experiments/track2_v27_gold_like_ledger_20260608/v27_gold_like_ledger.json`;
- `experiments/track2_v27_gold_like_ledger_20260608/v27_profile_scoreboard.csv`;
- `experiments/track2_v27_gold_like_ledger_20260608/v27_profile_scoreboard.json`;
- `experiments/track2_v27_gold_like_ledger_20260608/v27_candidate_report_zh.md`;
- `submissions/track2_submission_v27_gold_like_strict_candidate.json`;
- `submissions/track2_submission_v27_gold_like_strict_candidate.zip`;
- `submissions/track2_submission_v27_gold_like_frontier_candidate.json`;
- `submissions/track2_submission_v27_gold_like_frontier_candidate.zip`.

It must call `score_submission(candidate_json, candidate_name=<candidate_stem>)` for every profile and include the score fields in the report.

Create `scripts/track2_v27_gold_like_ledger.py` that calls `build_v27_candidate_suite` and prints:

```text
decision=<decision> best_profile=<profile> overall=<overall> class=<class> desc=<desc>
<report_path>
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
python3 -m unittest tests/test_track2_v27_gold_like_ledger.py -v
```

Expected: all v27 tests pass.

### Task 4: Run Real Candidate And Validate Score

**Files:**
- Generated: `experiments/track2_v27_gold_like_ledger_20260608/*`
- Generated: `submissions/track2_submission_v27_gold_like_strict_candidate.json`
- Generated: `submissions/track2_submission_v27_gold_like_strict_candidate.zip`
- Generated: `submissions/track2_submission_v27_gold_like_frontier_candidate.json`
- Generated: `submissions/track2_submission_v27_gold_like_frontier_candidate.zip`

- [ ] **Step 1: Run builder**

Run:

```bash
python3 scripts/track2_v27_gold_like_ledger.py
```

Expected: prints a decision and v23 proxy score.

- [ ] **Step 2: Validate candidate JSON**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v27_gold_like_frontier_candidate.json
```

Expected: validator OK.

- [ ] **Step 3: Run anchor calibration tests**

Run:

```bash
python3 -m unittest tests/test_track2_v27_gold_like_ledger.py tests/test_track2_official_anchor_calibration.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Check whitespace and review score**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

If the v27 report says `recommend_final_submit`, surface the exact ZIP path. If it says `hold_no_submit`, do not recommend using the final Codabench submission.
