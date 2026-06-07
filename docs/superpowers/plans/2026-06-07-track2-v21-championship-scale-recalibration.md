# Track2 v21 Championship-Scale Recalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track2 v21 candidate ladder with 80/120/160-scale classification recalibration while preserving the v15 description anchor.

**Architecture:** v21 reads the v17 evidence matrix and v15 submission JSON, scores evidence rows with tiered reference/consensus/expansion rules, writes side-path candidate packages, and emits a final gate that selects the best high-variance submission candidate. It does not upload to Codabench and does not overwrite formal submission names. After initial mixed-profile evidence, v21 also writes a directional `calmshift90` package that only tests the official-feedback hypothesis that `content->calm` is safer than the prior failed `calm->content` batch.

**Tech Stack:** Python standard library, existing Track2 JSON schema, existing `affectiveart.challenge validate-track2`, `unittest`.

---

## Task 1: v21 Scoring API

**Files:**
- Create `affectiveart/track2_v21_championship_recalibration.py`
- Create `scripts/track2_v21_championship_recalibration.py`
- Create `tests/test_track2_v21_championship_recalibration.py`

- [x] Write failing tests for tier assignment:
  - exact/near duplicate becomes `reference`.
  - three model families plus public-style support becomes `consensus`.
  - two model families plus strong confidence becomes `expansion`.
  - weak one-source row is held.
- [x] Implement `score_v21_evidence_row(row)`.
- [x] Implement CLI wrapper.
- [x] Run `python3 -m unittest tests/test_track2_v21_championship_recalibration.py -v`.

## Task 2: Candidate Selection

**Files:**
- Modify `affectiveart/track2_v21_championship_recalibration.py`
- Modify `tests/test_track2_v21_championship_recalibration.py`

- [x] Write failing tests for profile caps:
  - `precision80` selects at most 80 rows.
  - top emotion share guard blocks projected collapse above `0.58`.
  - exact transition family cap is enforced.
  - current emotion cannot drop below 4 samples.
- [x] Implement `build_v21_championship_evidence`.
- [x] Implement `select_v21_changes`.
- [x] Run v21 unit tests.
- [x] Add and verify `calmshift90`, a high-variance directional profile limited to `content->calm`.

## Task 3: Candidate Writer And Gate

**Files:**
- Modify `affectiveart/track2_v21_championship_recalibration.py`
- Modify `tests/test_track2_v21_championship_recalibration.py`

- [x] Write failing tests for side-path safety and final gate.
- [x] Implement deterministic JSON/ZIP writing.
- [x] Implement `choose_v21_final_gate`.
- [x] Generate artifacts with:

```bash
python3 scripts/track2_v21_championship_recalibration.py run
```

- [x] Validate:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v21_precision80_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v21_champion120_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v21_lastshot160_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v21_calmshift90_candidate.json
```

## Task 4: Verification And Decision

- [ ] Run:

```bash
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
python3 scripts/track2_fused_shadow_evaluator.py score \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate v15_desc_expand300=submissions/track2_submission_v15_desc_expand300_candidate.json \
  --candidate v21_precision80=submissions/track2_submission_v21_precision80_candidate.json \
  --candidate v21_champion120=submissions/track2_submission_v21_champion120_candidate.json \
  --candidate v21_lastshot160=submissions/track2_submission_v21_lastshot160_candidate.json \
  --candidate v21_calmshift90=submissions/track2_submission_v21_calmshift90_candidate.json \
  --out-dir experiments/track2_v21_championship_recalibration_20260607/fused_shadow_compare
git diff --check
```

- [x] Commit verified v21 code, tests, docs, and reports.
- [x] Report whether v21 is worth the second-to-last submission and provide the exact ZIP path.
