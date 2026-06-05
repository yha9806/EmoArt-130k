# Track2 V7 Hard-Case Arbitration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Track2-only v7 hard-case label arbitration pass that writes side-path candidates and reports without touching formal submission files.

**Architecture:** A new focused module loads v3 baseline rows, v3 evidence rows, v5 Gemini arbitration rows, and a label-aligned text override candidate. It selects strict and diagnostic hard-case changes, writes candidate JSON/ZIP files, and runs the existing local shadow evaluator.

**Tech Stack:** Python standard library, existing `affectiveart.challenge`, `track2_audit`, `track2_local_shadow_evaluator`, and `track2_moe_specialist_ensemble` helpers.

---

### Task 1: V7 Selector Tests

**Files:**
- Create: `tests/test_track2_v7_hardcase_arbitration.py`

- [x] **Step 1: Write failing tests**

Cover strict accepted cross rows, missing text override holds, non-dangerous borderline diagnostic rows, dangerous Positive/Low shifts, side-path output writing, and CLI execution.

- [x] **Step 2: Run test to verify RED**

Run: `python3 -m unittest tests.test_track2_v7_hardcase_arbitration -v`

Expected: FAIL because `affectiveart.track2_v7_hardcase_arbitration` does not exist.

### Task 2: V7 Core Module

**Files:**
- Create: `affectiveart/track2_v7_hardcase_arbitration.py`

- [x] **Step 1: Implement minimal selector**

Add constants, threshold dataclass, decision dataclass, evidence/arbitration loaders, `select_v7_arbitrations`, and `build_candidate_rows`.

- [x] **Step 2: Implement writer**

Add `write_v7_outputs` to write side-path JSON/ZIP files, candidate reports, HTML review, summary files, and shadow evaluator output.

- [x] **Step 3: Run tests to verify GREEN**

Run: `python3 -m unittest tests.test_track2_v7_hardcase_arbitration -v`

Expected: OK.

### Task 3: V7 CLI

**Files:**
- Create: `scripts/track2_v7_hardcase_arbitration.py`

- [x] **Step 1: Add run command**

Expose `--baseline-json`, `--shadow-baseline-json`, `--evidence-matrix`, `--arbitration-json`, `--text-override-json`, `--out-dir`, and `--submission-dir`.

- [x] **Step 2: Verify CLI through test**

Run: `python3 -m unittest tests.test_track2_v7_hardcase_arbitration.Track2V7WriterTest.test_cli_run_writes_v7_summary -v`

Expected: OK.

### Task 4: Real V7 Run

**Files:**
- Create: `experiments/track2_v7_hardcase_arbitration_20260605/v1/`
- Create ignored side-path submission files under `submissions/track2_submission_v7_*_candidate.json/.zip`

- [x] **Step 1: Run v7 against frozen artifacts**

Run:

```bash
python3 scripts/track2_v7_hardcase_arbitration.py run \
  --baseline-json submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json \
  --shadow-baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --evidence-matrix experiments/track2_score_calibrated_champion_20260605/v1/evidence_matrix.csv \
  --arbitration-json experiments/track2_score_calibrated_champion_20260605/v1/v5_cross_arbitration/cross_score7_arbitration_summary.json \
  --text-override-json submissions/track2_submission_v5_cross1_gemini35_desc_192_candidate.json \
  --out-dir experiments/track2_v7_hardcase_arbitration_20260605/v1 \
  --submission-dir submissions
```

Expected: `strict_accepts=1`, `borderline_accepts=0`.

- [x] **Step 2: Validate outputs**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v7_cross1_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v7_cross3_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v7_hardcase_push_candidate.json
```

Expected: all `OK`.

### Task 5: Track2 Verification

**Files:**
- No new files.

- [x] **Step 1: Run Track2-only tests**

Run: `python3 -m unittest discover -s tests -p 'test_track2*.py' -v`

Expected: OK.

- [x] **Step 2: Check whitespace**

Run: `git diff --check`

Expected: no output.
