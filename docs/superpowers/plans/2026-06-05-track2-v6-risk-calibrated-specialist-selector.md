# Track2 V6 Risk-Calibrated Specialist Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track2-only v6 selector that accepts only locally defensible label deltas, writes side-path candidates, and produces reports strong enough to decide whether an online submission is worth spending.

**Architecture:** Add a focused selector layer on top of the existing v3 evidence matrix and local shadow evaluator. The selector normalizes delta evidence, applies deterministic risk rules, writes v6 candidate ladders, and then scores those ladders locally without overwriting formal submission files.

**Tech Stack:** Python standard library, `unittest`, existing `affectiveart.track2_audit`, `affectiveart.track2_moe_specialist_ensemble`, `affectiveart.track2_local_shadow_evaluator`, and `python3 -m affectiveart.challenge validate-track2`.

---

## File Structure

- Create `affectiveart/track2_v6_specialist_selector.py`
  - Owns v6 evidence normalization, deterministic selector rules, proposal conflict handling, candidate row construction, side-path JSON/ZIP writing, and Markdown/HTML reports.
- Create `scripts/track2_v6_specialist_selector.py`
  - Thin CLI wrapper around the module. It loads the approved v3 description baseline, the v3 evidence matrix, writes v6 outputs, and runs local shadow scoring.
- Create `tests/test_track2_v6_specialist_selector.py`
  - Unit and CLI coverage for malformed evidence, same-quadrant accept, cross-quadrant hold/accept, proposal conflict, candidate writing, and formal path protection.
- Use existing `affectiveart/track2_local_shadow_evaluator.py`
  - No modification in this plan. The v6 CLI calls `write_shadow_evaluator_outputs`.
- Use existing `affectiveart/track2_score_calibrated_champion.py`
  - No modification in this plan. The v6 loader consumes its `evidence_matrix.csv`.
- Write generated run artifacts under `experiments/track2_v6_specialist_selector_20260605/v1/`.
- Write generated candidate artifacts only under side-path names:
  - `submissions/track2_submission_v6_safe_sameq_candidate.json`
  - `submissions/track2_submission_v6_safe_sameq_candidate.zip`
  - `submissions/track2_submission_v6_cross_micro_candidate.json`
  - `submissions/track2_submission_v6_cross_micro_candidate.zip`
  - `submissions/track2_submission_v6_desc_plus_candidate.json`
  - `submissions/track2_submission_v6_desc_plus_candidate.zip`

## Baseline Inputs

- Baseline JSON: `submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json`
- Evidence matrix: `experiments/track2_score_calibrated_champion_20260605/v1/evidence_matrix.csv`
- Primary output dir: `experiments/track2_v6_specialist_selector_20260605/v1`
- Submission dir: `submissions`
- Expected row count: `1000`

## Task 1: Add Failing Tests For V6 Selector Core

**Files:**
- Create: `tests/test_track2_v6_specialist_selector.py`
- Create in Task 2: `affectiveart/track2_v6_specialist_selector.py`

- [ ] **Step 1: Write the failing selector tests**

Create `tests/test_track2_v6_specialist_selector.py` with this content:

```python
import csv
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_v6_specialist_selector import (
    ACCEPT_CROSS_MICRO,
    ACCEPT_SAFE,
    HOLD_REVIEW,
    SelectorThresholds,
    build_candidate_rows,
    load_evidence_matrix,
    select_v6_deltas,
    write_v6_outputs,
)


def row(sample_id, emotion, valence=None, arousal=None):
    labels = {
        "alarmed": ("Negative", "High"),
        "annoyed": ("Negative", "High"),
        "aroused": ("Positive", "High"),
        "bored": ("Negative", "Low"),
        "calm": ("Positive", "Low"),
        "content": ("Positive", "Low"),
        "excited": ("Positive", "High"),
        "frustrated": ("Negative", "High"),
        "glad": ("Positive", "Low"),
        "happy": ("Positive", "High"),
        "sad": ("Negative", "Low"),
        "tired": ("Negative", "Low"),
    }
    if valence is None or arousal is None:
        valence, arousal = labels[emotion]
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": f"A {emotion} artwork with visible subject matter and emotional atmosphere.",
        "brushstroke": "Layered brushwork describes the visible forms.",
        "composition": "Balanced composition organizes the subject clearly.",
        "color": "Specific colors shape the emotional atmosphere.",
        "line": "Line quality defines the figure and spatial rhythm.",
        "light": "Light and shadow clarify depth and focus.",
    }


def write_matrix(path, rows):
    fieldnames = sorted({key for item in rows for key in item})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class Track2V6SelectorCoreTest(unittest.TestCase):
    def test_missing_required_delta_fields_default_to_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0001",
                        "current_emotion": "content",
                        "proposed_emotion": "",
                        "supporting_source_count": "4",
                        "supporting_family_count": "3",
                        "evidence_score": "8",
                    }
                ],
            )

            deltas = load_evidence_matrix(matrix_path)
            decisions = select_v6_deltas(deltas)

        self.assertEqual(len(deltas), 1)
        self.assertTrue(deltas[0].malformed)
        self.assertEqual(decisions[0].decision, HOLD_REVIEW)
        self.assertIn("missing_required_delta_field", decisions[0].reason_codes)

    def test_same_quadrant_multi_source_delta_accepts_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0002",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "supporting_sources": "public_style;teacher;siglip2",
                        "supporting_families": "public_style;teacher",
                        "same_quadrant": "true",
                        "public_reference_support": "true",
                        "public_reference_contradiction": "false",
                        "gemini35_objection": "false",
                        "vulca_objection": "false",
                        "evidence_score": "8",
                        "hard96_net_gain": "2",
                        "hard96_net_loss": "0",
                    }
                ],
            )

            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        self.assertEqual(decisions[0].decision, ACCEPT_SAFE)
        self.assertEqual(decisions[0].ladder, "v6_safe_sameq")

    def test_public_reference_contradiction_overrides_model_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0003",
                        "current_emotion": "calm",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "content",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "5",
                        "supporting_family_count": "4",
                        "same_quadrant": "true",
                        "public_reference_support": "false",
                        "public_reference_contradiction": "true",
                        "gemini35_objection": "false",
                        "vulca_objection": "false",
                        "evidence_score": "10",
                    }
                ],
            )

            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        self.assertEqual(decisions[0].decision, HOLD_REVIEW)
        self.assertIn("public_reference_contradiction", decisions[0].reason_codes)

    def test_cross_quadrant_delta_requires_arbitration_and_hard96_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0004",
                        "current_emotion": "annoyed",
                        "current_valence": "Negative",
                        "current_arousal": "High",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "4",
                        "supporting_family_count": "3",
                        "same_quadrant": "false",
                        "gemini35_prefers_proposed": "true",
                        "gemini35_fit_margin": "0.41",
                        "public_reference_contradiction": "false",
                        "vulca_objection": "false",
                        "evidence_score": "9",
                        "hard96_net_gain": "2",
                        "hard96_net_loss": "0",
                    }
                ],
            )

            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        self.assertEqual(decisions[0].decision, ACCEPT_CROSS_MICRO)
        self.assertEqual(decisions[0].ladder, "v6_cross_micro")

    def test_cross_quadrant_without_arbitration_stays_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0005",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "frustrated",
                        "proposed_valence": "Negative",
                        "proposed_arousal": "High",
                        "supporting_source_count": "5",
                        "supporting_family_count": "4",
                        "same_quadrant": "false",
                        "gemini35_prefers_proposed": "false",
                        "gemini35_fit_margin": "0.10",
                        "evidence_score": "10",
                        "hard96_net_gain": "2",
                        "hard96_net_loss": "0",
                    }
                ],
            )

            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        self.assertEqual(decisions[0].decision, HOLD_REVIEW)
        self.assertIn("missing_cross_arbitration", decisions[0].reason_codes)

    def test_multiple_accepted_labels_for_one_sample_become_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0622",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "same_quadrant": "true",
                        "evidence_score": "8",
                    },
                    {
                        "sample_id": "track2_0622",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "glad",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "same_quadrant": "true",
                        "evidence_score": "8",
                    },
                ],
            )

            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        self.assertEqual([item.decision for item in decisions], [HOLD_REVIEW, HOLD_REVIEW])
        self.assertTrue(all("proposal_conflict" in item.reason_codes for item in decisions))

    def test_candidate_rows_apply_only_selected_ladder_changes(self):
        baseline = [row("track2_0006", "content"), row("track2_0007", "annoyed")]
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0006",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "same_quadrant": "true",
                        "evidence_score": "8",
                    },
                    {
                        "sample_id": "track2_0007",
                        "current_emotion": "annoyed",
                        "current_valence": "Negative",
                        "current_arousal": "High",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "4",
                        "supporting_family_count": "3",
                        "same_quadrant": "false",
                        "gemini35_prefers_proposed": "true",
                        "gemini35_fit_margin": "0.41",
                        "evidence_score": "9",
                        "hard96_net_gain": "2",
                        "hard96_net_loss": "0",
                    },
                ],
            )
            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        safe_rows = build_candidate_rows(baseline, decisions, ladder="v6_safe_sameq")
        cross_rows = build_candidate_rows(baseline, decisions, ladder="v6_cross_micro")
        safe_by_id = {item["sample_id"]: item for item in safe_rows}
        cross_by_id = {item["sample_id"]: item for item in cross_rows}

        self.assertEqual(safe_by_id["track2_0006"]["emotion"], "calm")
        self.assertEqual(safe_by_id["track2_0007"]["emotion"], "annoyed")
        self.assertEqual(cross_by_id["track2_0007"]["emotion"], "calm")
        self.assertEqual(cross_by_id["track2_0007"]["emotional_valence"], "Positive")
        self.assertEqual(cross_by_id["track2_0007"]["emotional_arousal_level"], "Low")
```

- [ ] **Step 2: Run the selector tests to verify they fail**

Run:

```bash
python3 -m unittest tests.test_track2_v6_specialist_selector -v
```

Expected result:

```text
ModuleNotFoundError: No module named 'affectiveart.track2_v6_specialist_selector'
```

## Task 2: Implement V6 Evidence Normalization And Deterministic Selector

**Files:**
- Create: `affectiveart/track2_v6_specialist_selector.py`
- Test: `tests/test_track2_v6_specialist_selector.py`

- [ ] **Step 1: Create the selector module core**

Create `affectiveart/track2_v6_specialist_selector.py` with this content:

```python
from __future__ import annotations

import csv
import html
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_audit import compute_track2_distribution, strict_track2_label_issues
from affectiveart.track2_local_shadow_evaluator import write_shadow_evaluator_outputs
from affectiveart.track2_moe_specialist_ensemble import expected_label_for_emotion


ACCEPT_SAFE = "accept_safe"
ACCEPT_CROSS_MICRO = "accept_cross_micro"
HOLD_REVIEW = "hold_review"
BLOCK = "block"

V6_LADDERS = ("v6_safe_sameq", "v6_cross_micro", "v6_desc_plus")
FORMAL_SUBMISSION_PATHS = {
    Path("submissions/track2_submission.json"),
    Path("submissions/track2_submission.zip"),
}
HIGH_RISK_NEGATIVE_TARGETS = {"alarmed", "annoyed", "bored", "frustrated", "sad", "tired"}
CALM_CONTENT_FAMILY = {"calm", "content", "glad"}


@dataclass(frozen=True)
class V6Delta:
    sample_id: str
    current_emotion: str
    current_valence: str
    current_arousal: str
    proposed_emotion: str
    proposed_valence: str
    proposed_arousal: str
    transition: str
    same_quadrant: bool
    cross_quadrant_risk: bool
    supporting_family_count: int
    supporting_source_count: int
    supporting_sources: tuple[str, ...]
    supporting_families: tuple[str, ...]
    evidence_score: int
    gemini35_objection: bool
    vulca_objection: bool
    gemini35_prefers_proposed: bool
    gemini35_fit_margin: float
    public_reference_support: bool
    public_reference_contradiction: bool
    human_accept_support: bool
    hard96_net_gain: int
    hard96_net_loss: int
    hard96_support_count: int
    malformed: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class SelectorThresholds:
    min_sameq_score: int = 6
    min_sameq_sources: int = 2
    min_sameq_families: int = 2
    min_sameq_human_score: int = 4
    min_cross_score: int = 8
    min_cross_sources: int = 3
    min_cross_families: int = 3
    min_cross_fit_margin: float = 0.30
    max_cross_micro: int = 3
    max_hard96_net_loss: int = 0


@dataclass(frozen=True)
class SelectorDecision:
    sample_id: str
    proposed_emotion: str
    transition: str
    decision: str
    ladder: str
    reason_codes: tuple[str, ...]
    evidence_score: int
    same_quadrant: bool
    cross_quadrant_risk: bool
    proposed_valence: str
    proposed_arousal: str
    source_count: int
    family_count: int


def load_evidence_matrix(path: str | Path) -> list[V6Delta]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return [_delta_from_row(dict(row)) for row in csv.DictReader(handle)]


def select_v6_deltas(
    deltas: list[V6Delta],
    thresholds: SelectorThresholds | None = None,
) -> list[SelectorDecision]:
    thresholds = thresholds or SelectorThresholds()
    initial = [_select_one_delta(delta, thresholds) for delta in deltas]
    return _demote_conflicting_accepts(initial)


def build_candidate_rows(
    baseline_rows: list[dict[str, Any]],
    decisions: list[SelectorDecision],
    *,
    ladder: str,
) -> list[dict[str, Any]]:
    if ladder not in V6_LADDERS:
        raise ValueError(f"unknown v6 ladder: {ladder}")
    selected = _selected_decisions_for_ladder(decisions, ladder)
    selected_by_id = {item.sample_id: item for item in selected}
    output: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        row = dict(baseline)
        selected_decision = selected_by_id.get(str(row.get("sample_id", "")))
        if selected_decision:
            row["emotion"] = selected_decision.proposed_emotion
            row["emotional_valence"] = selected_decision.proposed_valence
            row["emotional_arousal_level"] = selected_decision.proposed_arousal
        output.append({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})
    return output


def _delta_from_row(row: dict[str, Any]) -> V6Delta:
    sample_id = _clean(row.get("sample_id"))
    current_emotion = _clean(row.get("current_emotion"))
    proposed_emotion = _clean(row.get("proposed_emotion"))
    current_valence = _clean(row.get("current_valence")) or _expected_part(current_emotion, 0)
    current_arousal = _clean(row.get("current_arousal")) or _expected_part(current_emotion, 1)
    proposed_valence = _clean(row.get("proposed_valence")) or _expected_part(proposed_emotion, 0)
    proposed_arousal = _clean(row.get("proposed_arousal")) or _expected_part(proposed_emotion, 1)
    missing_required = not sample_id or not current_emotion or not proposed_emotion
    same_quadrant = _bool_from_any(row.get("same_quadrant"))
    if not _has_value(row, "same_quadrant") and current_valence and proposed_valence:
        same_quadrant = current_valence == proposed_valence and current_arousal == proposed_arousal
    transition = _clean(row.get("transition")) or f"{current_emotion}->{proposed_emotion}"
    sources = _split_tokens(row.get("supporting_sources"))
    families = _split_tokens(row.get("supporting_families"))
    supporting_source_count = _int_from_any(row.get("supporting_source_count"), len(sources))
    supporting_family_count = _int_from_any(row.get("supporting_family_count"), len(families))
    evidence_score = _int_from_any(row.get("evidence_score"), 0)
    return V6Delta(
        sample_id=sample_id,
        current_emotion=current_emotion,
        current_valence=current_valence,
        current_arousal=current_arousal,
        proposed_emotion=proposed_emotion,
        proposed_valence=proposed_valence,
        proposed_arousal=proposed_arousal,
        transition=transition,
        same_quadrant=same_quadrant,
        cross_quadrant_risk=not same_quadrant,
        supporting_family_count=supporting_family_count,
        supporting_source_count=supporting_source_count,
        supporting_sources=sources,
        supporting_families=families,
        evidence_score=evidence_score,
        gemini35_objection=_bool_from_any(row.get("gemini35_objection") or row.get("gemini_objection")),
        vulca_objection=_bool_from_any(row.get("vulca_objection")),
        gemini35_prefers_proposed=_bool_from_any(
            row.get("gemini35_prefers_proposed") or row.get("arbitration_support") or row.get("gemini_prefers_proposed")
        ),
        gemini35_fit_margin=_float_from_any(row.get("gemini35_fit_margin") or row.get("fit_margin"), 0.0),
        public_reference_support=_bool_from_any(
            row.get("public_reference_support") or row.get("public_style_agreement")
        ),
        public_reference_contradiction=_bool_from_any(row.get("public_reference_contradiction")),
        human_accept_support=_bool_from_any(row.get("human_accept_support")),
        hard96_net_gain=_int_from_any(row.get("hard96_net_gain"), 0),
        hard96_net_loss=_int_from_any(row.get("hard96_net_loss"), 0),
        hard96_support_count=_int_from_any(row.get("hard96_support_count"), 0),
        malformed=missing_required,
        raw=row,
    )


def _select_one_delta(delta: V6Delta, thresholds: SelectorThresholds) -> SelectorDecision:
    reasons: list[str] = []
    if delta.malformed:
        reasons.append("missing_required_delta_field")
        return _decision(delta, HOLD_REVIEW, "", reasons)
    if delta.public_reference_contradiction:
        reasons.append("public_reference_contradiction")
        return _decision(delta, HOLD_REVIEW, "", reasons)
    if delta.gemini35_objection:
        reasons.append("gemini35_objection")
        return _decision(delta, HOLD_REVIEW, "", reasons)
    if delta.vulca_objection:
        reasons.append("vulca_objection")
        return _decision(delta, HOLD_REVIEW, "", reasons)
    if delta.hard96_net_loss > thresholds.max_hard96_net_loss:
        reasons.append("hard96_transition_loss")
        return _decision(delta, HOLD_REVIEW, "", reasons)

    if delta.same_quadrant:
        if delta.human_accept_support and delta.evidence_score >= thresholds.min_sameq_human_score:
            return _decision(delta, ACCEPT_SAFE, "v6_safe_sameq", ["human_accept_support"])
        if (
            delta.evidence_score >= thresholds.min_sameq_score
            and delta.supporting_source_count >= thresholds.min_sameq_sources
            and delta.supporting_family_count >= thresholds.min_sameq_families
        ):
            return _decision(delta, ACCEPT_SAFE, "v6_safe_sameq", ["same_quadrant_multi_source"])
        reasons.append("insufficient_same_quadrant_support")
        return _decision(delta, HOLD_REVIEW, "", reasons)

    if delta.current_emotion in CALM_CONTENT_FAMILY and delta.proposed_emotion in HIGH_RISK_NEGATIVE_TARGETS:
        reasons.append("dangerous_calm_content_to_negative")
        return _decision(delta, HOLD_REVIEW, "", reasons)
    if not delta.gemini35_prefers_proposed or delta.gemini35_fit_margin < thresholds.min_cross_fit_margin:
        reasons.append("missing_cross_arbitration")
        return _decision(delta, HOLD_REVIEW, "", reasons)
    if delta.hard96_net_gain <= 0:
        reasons.append("missing_hard96_cross_support")
        return _decision(delta, HOLD_REVIEW, "", reasons)
    if (
        delta.evidence_score >= thresholds.min_cross_score
        and delta.supporting_source_count >= thresholds.min_cross_sources
        and delta.supporting_family_count >= thresholds.min_cross_families
    ):
        return _decision(delta, ACCEPT_CROSS_MICRO, "v6_cross_micro", ["cross_micro_arbitrated"])
    reasons.append("insufficient_cross_support")
    return _decision(delta, HOLD_REVIEW, "", reasons)


def _demote_conflicting_accepts(decisions: list[SelectorDecision]) -> list[SelectorDecision]:
    accepted_by_sample: dict[str, list[SelectorDecision]] = defaultdict(list)
    for decision in decisions:
        if decision.decision in {ACCEPT_SAFE, ACCEPT_CROSS_MICRO}:
            accepted_by_sample[decision.sample_id].append(decision)
    conflict_ids = {sample_id for sample_id, items in accepted_by_sample.items() if len(items) > 1}
    if not conflict_ids:
        return decisions
    output: list[SelectorDecision] = []
    for decision in decisions:
        if decision.sample_id not in conflict_ids:
            output.append(decision)
            continue
        output.append(
            SelectorDecision(
                sample_id=decision.sample_id,
                proposed_emotion=decision.proposed_emotion,
                transition=decision.transition,
                decision=HOLD_REVIEW,
                ladder="",
                reason_codes=tuple(sorted(set(decision.reason_codes + ("proposal_conflict",)))),
                evidence_score=decision.evidence_score,
                same_quadrant=decision.same_quadrant,
                cross_quadrant_risk=decision.cross_quadrant_risk,
                proposed_valence=decision.proposed_valence,
                proposed_arousal=decision.proposed_arousal,
                source_count=decision.source_count,
                family_count=decision.family_count,
            )
        )
    return output


def _selected_decisions_for_ladder(decisions: list[SelectorDecision], ladder: str) -> list[SelectorDecision]:
    safe = [item for item in decisions if item.decision == ACCEPT_SAFE]
    cross = [
        item
        for item in decisions
        if item.decision == ACCEPT_CROSS_MICRO
    ]
    cross = sorted(cross, key=lambda item: (-item.evidence_score, item.sample_id, item.proposed_emotion))[:3]
    if ladder == "v6_safe_sameq":
        return safe
    if ladder in {"v6_cross_micro", "v6_desc_plus"}:
        return safe + cross
    raise ValueError(f"unknown v6 ladder: {ladder}")


def _decision(delta: V6Delta, decision: str, ladder: str, reasons: list[str]) -> SelectorDecision:
    return SelectorDecision(
        sample_id=delta.sample_id,
        proposed_emotion=delta.proposed_emotion,
        transition=delta.transition,
        decision=decision,
        ladder=ladder,
        reason_codes=tuple(reasons),
        evidence_score=delta.evidence_score,
        same_quadrant=delta.same_quadrant,
        cross_quadrant_risk=delta.cross_quadrant_risk,
        proposed_valence=delta.proposed_valence,
        proposed_arousal=delta.proposed_arousal,
        source_count=delta.supporting_source_count,
        family_count=delta.supporting_family_count,
    )


def _expected_part(emotion: str, index: int) -> str:
    if not emotion:
        return ""
    try:
        return expected_label_for_emotion(emotion)[index]
    except KeyError:
        return ""


def _clean(value: Any) -> str:
    return str(value or "").strip().lower()


def _has_value(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).strip() != ""


def _bool_from_any(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "allow", "accept", "accepted"}


def _int_from_any(value: Any, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _float_from_any(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _split_tokens(value: Any) -> tuple[str, ...]:
    raw = str(value or "")
    tokens = [item.strip() for chunk in raw.split(";") for item in chunk.split(",")]
    return tuple(sorted({item for item in tokens if item}))
```

- [ ] **Step 2: Run the selector tests**

Run:

```bash
python3 -m unittest tests.test_track2_v6_specialist_selector -v
```

Expected result:

```text
Ran 7 tests
OK
```

- [ ] **Step 3: Commit the selector core**

Run:

```bash
git add affectiveart/track2_v6_specialist_selector.py tests/test_track2_v6_specialist_selector.py
git commit -m "feat: add track2 v6 specialist selector core"
```

Expected result:

```text
[codex/track2-human-review-final ...] feat: add track2 v6 specialist selector core
```

## Task 3: Add Candidate Writer, Reports, And Shadow Evaluator Integration

**Files:**
- Modify: `affectiveart/track2_v6_specialist_selector.py`
- Test: `tests/test_track2_v6_specialist_selector.py`

- [ ] **Step 1: Add writer tests to the same test file**

Append this test class to `tests/test_track2_v6_specialist_selector.py`, before the `if __name__ == "__main__":` block added in Task 4:

```python
class Track2V6WriterTest(unittest.TestCase):
    def test_writer_protects_formal_paths_and_writes_v6_candidates(self):
        baseline = [
            row("track2_0001", "content"),
            row("track2_0002", "annoyed"),
            row("track2_0003", "calm"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_json = tmp_path / "baseline.json"
            matrix_path = tmp_path / "evidence.csv"
            out_dir = tmp_path / "out"
            submission_dir = tmp_path / "submissions"
            baseline_json.write_text(json.dumps(baseline), encoding="utf-8")
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0001",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "same_quadrant": "true",
                        "evidence_score": "8",
                    },
                    {
                        "sample_id": "track2_0002",
                        "current_emotion": "annoyed",
                        "current_valence": "Negative",
                        "current_arousal": "High",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "4",
                        "supporting_family_count": "3",
                        "same_quadrant": "false",
                        "gemini35_prefers_proposed": "true",
                        "gemini35_fit_margin": "0.41",
                        "evidence_score": "9",
                        "hard96_net_gain": "2",
                        "hard96_net_loss": "0",
                    },
                ],
            )

            with self.assertRaises(ValueError):
                write_v6_outputs(
                    baseline_json=baseline_json,
                    evidence_matrix=matrix_path,
                    out_dir=out_dir,
                    submission_dir=submission_dir,
                    output_names={"v6_safe_sameq": "track2_submission"},
                    expected_row_count=len(baseline),
                    require_all_emotions=False,
                )

            report = write_v6_outputs(
                baseline_json=baseline_json,
                evidence_matrix=matrix_path,
                out_dir=out_dir,
                submission_dir=submission_dir,
                expected_row_count=len(baseline),
                require_all_emotions=False,
            )

        self.assertEqual(report["decision"], "recommend_hold")
        self.assertEqual(report["evidence_row_count"], 2)
        self.assertTrue(Path(report["normalized_evidence_csv"]).exists())
        self.assertTrue(Path(report["selector_decisions_csv"]).exists())
        self.assertTrue(Path(report["stability_report_json"]).exists())
        safe_json = Path(report["candidates"]["v6_safe_sameq"]["json"])
        cross_zip = Path(report["candidates"]["v6_cross_micro"]["zip"])
        self.assertTrue(safe_json.exists())
        self.assertTrue(cross_zip.exists())
        with zipfile.ZipFile(cross_zip) as archive:
            self.assertEqual(archive.namelist(), ["submission.json"])
            rows = json.loads(archive.read("submission.json").decode("utf-8"))
        by_id = {item["sample_id"]: item for item in rows}
        self.assertEqual(by_id["track2_0001"]["emotion"], "calm")
        self.assertEqual(by_id["track2_0002"]["emotion"], "calm")
        self.assertTrue((Path(report["out_dir"]) / "html_review" / "track2_v6_specialist_selector_review.html").exists())
        self.assertTrue((Path(report["out_dir"]) / "shadow_eval" / "shadow_score_report.json").exists())
```

- [ ] **Step 2: Run the tests to verify the writer API is missing**

Run:

```bash
python3 -m unittest tests.test_track2_v6_specialist_selector.Track2V6WriterTest -v
```

Expected result:

```text
ImportError: cannot import name 'write_v6_outputs'
```

- [ ] **Step 3: Add writer and report functions to the module**

Append this code to `affectiveart/track2_v6_specialist_selector.py`:

```python
DEFAULT_OUTPUT_NAMES = {
    "v6_safe_sameq": "track2_submission_v6_safe_sameq_candidate",
    "v6_cross_micro": "track2_submission_v6_cross_micro_candidate",
    "v6_desc_plus": "track2_submission_v6_desc_plus_candidate",
}


def write_v6_outputs(
    *,
    baseline_json: str | Path,
    evidence_matrix: str | Path,
    out_dir: str | Path,
    submission_dir: str | Path,
    output_names: dict[str, str] | None = None,
    expected_row_count: int = 1000,
    require_all_emotions: bool | None = None,
    formal_submission_paths: set[Path] | None = None,
) -> dict[str, Any]:
    baseline_json = Path(baseline_json)
    out_dir = Path(out_dir)
    submission_dir = Path(submission_dir)
    formal_paths = formal_submission_paths or FORMAL_SUBMISSION_PATHS
    output_names = {**DEFAULT_OUTPUT_NAMES, **(output_names or {})}
    baseline_rows = _load_json_rows(baseline_json)
    deltas = load_evidence_matrix(evidence_matrix)
    decisions = select_v6_deltas(deltas)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "html_review").mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    normalized_rows = [_delta_to_dict(item) for item in deltas]
    decision_rows = [_decision_to_dict(item) for item in decisions]
    _write_json(out_dir / "normalized_evidence.json", normalized_rows)
    _write_csv(out_dir / "normalized_evidence.csv", normalized_rows)
    _write_json(out_dir / "selector_decisions.json", decision_rows)
    _write_csv(out_dir / "selector_decisions.csv", decision_rows)

    candidates: dict[str, Any] = {}
    shadow_candidates: list[dict[str, Any]] = []
    for ladder in V6_LADDERS:
        rows = build_candidate_rows(baseline_rows, decisions, ladder=ladder)
        json_path = submission_dir / f"{output_names[ladder]}.json"
        zip_path = submission_dir / f"{output_names[ladder]}.zip"
        _assert_safe_v6_candidate_path(json_path, formal_paths)
        _assert_safe_v6_candidate_path(zip_path, formal_paths)
        _write_json(json_path, rows)
        _write_zip(zip_path, json_path)
        candidate_report = _candidate_report(ladder, json_path, zip_path, rows, baseline_rows, decisions)
        _write_json(out_dir / f"candidate_report_{ladder}.json", candidate_report)
        (out_dir / f"candidate_report_{ladder}.md").write_text(
            _render_candidate_report_markdown(candidate_report),
            encoding="utf-8",
        )
        candidates[ladder] = candidate_report
        shadow_candidates.append({"name": ladder, "json": json_path})

    stability = _build_stability_report(decisions)
    _write_json(out_dir / "hard96_stability_report.json", stability)
    (out_dir / "hard96_stability_report.md").write_text(_render_stability_markdown(stability), encoding="utf-8")
    shadow_report = write_shadow_evaluator_outputs(
        baseline_json=baseline_json,
        candidates=shadow_candidates,
        out_dir=out_dir / "shadow_eval",
        expected_row_count=expected_row_count,
        require_all_emotions=require_all_emotions,
    )
    decision = _top_level_decision(stability, shadow_report)
    summary = {
        "method": "track2_v6_risk_calibrated_specialist_selector",
        "decision": decision,
        "baseline_json": str(baseline_json),
        "evidence_matrix": str(evidence_matrix),
        "out_dir": str(out_dir),
        "evidence_row_count": len(deltas),
        "selector_decision_count": len(decisions),
        "candidates": candidates,
        "normalized_evidence_csv": str(out_dir / "normalized_evidence.csv"),
        "selector_decisions_csv": str(out_dir / "selector_decisions.csv"),
        "stability_report_json": str(out_dir / "hard96_stability_report.json"),
        "shadow_report_json": str(out_dir / "shadow_eval" / "shadow_score_report.json"),
        "formal_submission_overwritten": False,
    }
    _write_json(out_dir / "v6_summary.json", summary)
    (out_dir / "v6_summary.md").write_text(_render_summary_markdown(summary, shadow_report), encoding="utf-8")
    (out_dir / "html_review" / "track2_v6_specialist_selector_review.html").write_text(
        _render_html_review(summary, decision_rows),
        encoding="utf-8",
    )
    return summary


def _candidate_report(
    ladder: str,
    json_path: Path,
    zip_path: Path,
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    decisions: list[SelectorDecision],
) -> dict[str, Any]:
    selected_ids = {item.sample_id for item in _selected_decisions_for_ladder(decisions, ladder)}
    changed = [
        {
            "sample_id": decision.sample_id,
            "transition": decision.transition,
            "decision": decision.decision,
            "evidence_score": decision.evidence_score,
            "reason_codes": list(decision.reason_codes),
        }
        for decision in decisions
        if decision.sample_id in selected_ids
    ]
    distribution = compute_track2_distribution(rows)
    label_issue_count = sum(len(strict_track2_label_issues(row)) for row in rows)
    return {
        "ladder": ladder,
        "json": str(json_path),
        "zip": str(zip_path),
        "changed_rows": len(_row_risks(baseline_rows, rows)),
        "selected_changes": changed,
        "distribution": distribution,
        "label_consistency_issue_count": label_issue_count,
        "missing_emotions": distribution.get("missing_emotions", []),
        "formal_submission_overwritten": False,
    }


def _build_stability_report(decisions: list[SelectorDecision]) -> dict[str, Any]:
    accepted = [item for item in decisions if item.decision in {ACCEPT_SAFE, ACCEPT_CROSS_MICRO}]
    cross = [item for item in accepted if item.decision == ACCEPT_CROSS_MICRO]
    transition_counts = Counter(item.transition for item in accepted)
    issues: list[dict[str, Any]] = []
    if len(cross) > 3:
        issues.append({"code": "too_many_cross_micro", "count": len(cross)})
    if accepted:
        top_transition, top_count = transition_counts.most_common(1)[0]
        if top_count >= 5 and top_count / len(accepted) > 0.55:
            issues.append({"code": "transition_concentration", "transition": top_transition, "count": top_count})
    return {
        "method": "track2_v6_rule_stability_summary",
        "passed": not issues,
        "accepted_count": len(accepted),
        "safe_sameq_count": sum(1 for item in accepted if item.decision == ACCEPT_SAFE),
        "cross_micro_count": len(cross),
        "hold_count": sum(1 for item in decisions if item.decision == HOLD_REVIEW),
        "block_count": sum(1 for item in decisions if item.decision == BLOCK),
        "transition_counts": dict(sorted(transition_counts.items())),
        "issues": issues,
    }


def _top_level_decision(stability: dict[str, Any], shadow_report: dict[str, Any]) -> str:
    if not stability.get("passed", False):
        return "recommend_hold"
    ranking = shadow_report.get("ranking", [])
    if not ranking:
        return "invalid"
    top = ranking[0]
    if top.get("decision") != "recommend_submit":
        return "recommend_hold"
    if float(top.get("overall_expected", 0.0)) >= 0.8639085 and float(top.get("overall_lower", 0.0)) >= 0.796000:
        return "recommend_submit"
    return "recommend_hold"


def _row_risks(baseline_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_id = {str(row.get("sample_id", "")): row for row in baseline_rows}
    risks: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        before = baseline_by_id.get(sample_id)
        if before and (
            before.get("emotion"),
            before.get("emotional_valence"),
            before.get("emotional_arousal_level"),
        ) != (
            row.get("emotion"),
            row.get("emotional_valence"),
            row.get("emotional_arousal_level"),
        ):
            risks.append({"sample_id": sample_id, "before": before, "after": row})
    return risks


def _delta_to_dict(delta: V6Delta) -> dict[str, Any]:
    return {
        "sample_id": delta.sample_id,
        "current_emotion": delta.current_emotion,
        "current_valence": delta.current_valence,
        "current_arousal": delta.current_arousal,
        "proposed_emotion": delta.proposed_emotion,
        "proposed_valence": delta.proposed_valence,
        "proposed_arousal": delta.proposed_arousal,
        "transition": delta.transition,
        "same_quadrant": delta.same_quadrant,
        "cross_quadrant_risk": delta.cross_quadrant_risk,
        "supporting_family_count": delta.supporting_family_count,
        "supporting_source_count": delta.supporting_source_count,
        "supporting_sources": ";".join(delta.supporting_sources),
        "supporting_families": ";".join(delta.supporting_families),
        "evidence_score": delta.evidence_score,
        "gemini35_objection": delta.gemini35_objection,
        "vulca_objection": delta.vulca_objection,
        "gemini35_prefers_proposed": delta.gemini35_prefers_proposed,
        "gemini35_fit_margin": delta.gemini35_fit_margin,
        "public_reference_support": delta.public_reference_support,
        "public_reference_contradiction": delta.public_reference_contradiction,
        "human_accept_support": delta.human_accept_support,
        "hard96_net_gain": delta.hard96_net_gain,
        "hard96_net_loss": delta.hard96_net_loss,
        "hard96_support_count": delta.hard96_support_count,
        "malformed": delta.malformed,
    }


def _decision_to_dict(decision: SelectorDecision) -> dict[str, Any]:
    return {
        "sample_id": decision.sample_id,
        "proposed_emotion": decision.proposed_emotion,
        "transition": decision.transition,
        "decision": decision.decision,
        "ladder": decision.ladder,
        "reason_codes": ";".join(decision.reason_codes),
        "evidence_score": decision.evidence_score,
        "same_quadrant": decision.same_quadrant,
        "cross_quadrant_risk": decision.cross_quadrant_risk,
        "proposed_valence": decision.proposed_valence,
        "proposed_arousal": decision.proposed_arousal,
        "source_count": decision.source_count,
        "family_count": decision.family_count,
    }


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected Track2 JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _assert_safe_v6_candidate_path(path: Path, formal_paths: set[Path]) -> None:
    normalized = Path(path)
    formal_names = {Path(item).name for item in formal_paths}
    if normalized.name in formal_names:
        raise ValueError(f"refusing to write formal Track2 submission path: {path}")
    if not normalized.name.startswith("track2_submission_v6_") or "_candidate." not in normalized.name:
        raise ValueError(f"candidate output must be a v6 side-path candidate: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_zip(zip_path: Path, json_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, "submission.json")


def _render_candidate_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Track2 {report['ladder']} Candidate Report",
        "",
        f"- JSON: `{report['json']}`",
        f"- ZIP: `{report['zip']}`",
        f"- Changed rows: {report['changed_rows']}",
        f"- Label consistency issues: {report['label_consistency_issue_count']}",
        f"- Missing emotions: {', '.join(report.get('missing_emotions') or []) or 'none'}",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        "",
        "## Selected Changes",
        "",
    ]
    for item in report.get("selected_changes", []):
        lines.append(
            f"- {item['sample_id']}: {item['transition']}; score={item['evidence_score']}; reasons={','.join(item['reason_codes'])}"
        )
    if not report.get("selected_changes"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _render_stability_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 V6 Stability Report",
        "",
        f"- Passed: {report['passed']}",
        f"- Accepted count: {report['accepted_count']}",
        f"- Safe same-quadrant count: {report['safe_sameq_count']}",
        f"- Cross micro count: {report['cross_micro_count']}",
        f"- Hold count: {report['hold_count']}",
        "",
        "## Issues",
        "",
    ]
    for issue in report.get("issues", []):
        lines.append(f"- {issue['code']}: `{json.dumps(issue, ensure_ascii=False, sort_keys=True)}`")
    if not report.get("issues"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _render_summary_markdown(summary: dict[str, Any], shadow_report: dict[str, Any]) -> str:
    lines = [
        "# Track2 V6 Specialist Selector Summary",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Baseline JSON: `{summary['baseline_json']}`",
        f"- Evidence rows: {summary['evidence_row_count']}",
        f"- Selector decisions: {summary['selector_decision_count']}",
        f"- Formal submission overwritten: {summary['formal_submission_overwritten']}",
        "",
        "## Shadow Ranking",
        "",
        "| rank | candidate | decision | overall lower | overall expected | changes | cross |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(shadow_report.get("ranking", []), start=1):
        lines.append(
            f"| {index} | {item['candidate_name']} | {item['decision']} | "
            f"{float(item['overall_lower']):.6f} | {float(item['overall_expected']):.6f} | "
            f"{int(item['changed_rows'])} | {int(item['cross_quadrant_changes'])} |"
        )
    return "\n".join(lines) + "\n"


def _render_html_review(summary: dict[str, Any], decisions: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item['sample_id']))}</td>"
        f"<td>{html.escape(str(item['transition']))}</td>"
        f"<td>{html.escape(str(item['decision']))}</td>"
        f"<td>{html.escape(str(item['evidence_score']))}</td>"
        f"<td>{html.escape(str(item['reason_codes']))}</td>"
        "</tr>"
        for item in decisions[:500]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Track2 V6 Specialist Selector</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 7px; text-align: left; }}
    th {{ background: #eef2f6; }}
    .decision {{ padding: 12px; background: #eef7ee; border: 1px solid #b9d8b9; }}
  </style>
</head>
<body>
  <h1>Track2 V6 Specialist Selector</h1>
  <p class="decision">Decision: <code>{html.escape(str(summary['decision']))}</code></p>
  <p>Baseline: <code>{html.escape(str(summary['baseline_json']))}</code></p>
  <h2>Selector Decisions</h2>
  <table>
    <thead><tr><th>sample</th><th>transition</th><th>decision</th><th>score</th><th>reasons</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
```

- [ ] **Step 4: Run the writer tests**

Run:

```bash
python3 -m unittest tests.test_track2_v6_specialist_selector.Track2V6WriterTest -v
```

Expected result:

```text
Ran 1 test
OK
```

- [ ] **Step 5: Run the full v6 test file**

Run:

```bash
python3 -m unittest tests.test_track2_v6_specialist_selector -v
```

Expected result:

```text
Ran 8 tests
OK
```

- [ ] **Step 6: Commit writer integration**

Run:

```bash
git add affectiveart/track2_v6_specialist_selector.py tests/test_track2_v6_specialist_selector.py
git commit -m "feat: add track2 v6 candidate writer"
```

Expected result:

```text
[codex/track2-human-review-final ...] feat: add track2 v6 candidate writer
```

## Task 4: Add The V6 CLI And CLI Test

**Files:**
- Create: `scripts/track2_v6_specialist_selector.py`
- Modify: `tests/test_track2_v6_specialist_selector.py`

- [ ] **Step 1: Add CLI tests**

Append this test class to `tests/test_track2_v6_specialist_selector.py`:

```python
class Track2V6CliTest(unittest.TestCase):
    def test_cli_writes_v6_summary_and_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_json = tmp_path / "baseline.json"
            matrix_path = tmp_path / "evidence.csv"
            out_dir = tmp_path / "out"
            submission_dir = tmp_path / "submissions"
            baseline_json.write_text(
                json.dumps([row("track2_0001", "content"), row("track2_0002", "calm")]),
                encoding="utf-8",
            )
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0001",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "same_quadrant": "true",
                        "evidence_score": "8",
                    }
                ],
            )
            repo_root = Path(__file__).resolve().parents[1]

            result = subprocess.run(
                [
                    "python3",
                    str(repo_root / "scripts" / "track2_v6_specialist_selector.py"),
                    "run",
                    "--baseline-json",
                    str(baseline_json),
                    "--evidence-matrix",
                    str(matrix_path),
                    "--out-dir",
                    str(out_dir),
                    "--submission-dir",
                    str(submission_dir),
                    "--expected-row-count",
                    "2",
                    "--allow-missing-emotions-for-smoke",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("track2 v6 selector", result.stdout)
        self.assertTrue((out_dir / "v6_summary.json").exists())
        self.assertTrue((submission_dir / "track2_submission_v6_safe_sameq_candidate.zip").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the CLI test to verify the script is missing**

Run:

```bash
python3 -m unittest tests.test_track2_v6_specialist_selector.Track2V6CliTest -v
```

Expected result:

```text
can't open file
```

- [ ] **Step 3: Create the CLI script**

Create `scripts/track2_v6_specialist_selector.py` with this content:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from affectiveart.track2_v6_specialist_selector import write_v6_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Track2 v6 risk-calibrated specialist selector")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="build v6 candidates and local reports")
    run.add_argument("--baseline-json", required=True)
    run.add_argument("--evidence-matrix", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--submission-dir", default="submissions")
    run.add_argument("--expected-row-count", type=int, default=1000)
    run.add_argument(
        "--allow-missing-emotions-for-smoke",
        action="store_true",
        help="disable all-emotion enforcement for tiny unit-test fixtures",
    )
    args = parser.parse_args()

    if args.command == "run":
        summary = write_v6_outputs(
            baseline_json=Path(args.baseline_json),
            evidence_matrix=Path(args.evidence_matrix),
            out_dir=Path(args.out_dir),
            submission_dir=Path(args.submission_dir),
            expected_row_count=args.expected_row_count,
            require_all_emotions=False if args.allow_missing_emotions_for_smoke else None,
        )
        top_shadow = json.loads(Path(summary["shadow_report_json"]).read_text(encoding="utf-8"))["ranking"][0]
        print(
            "track2 v6 selector "
            f"decision={summary['decision']} "
            f"top_candidate={top_shadow['candidate_name']} "
            f"overall_expected={float(top_shadow['overall_expected']):.6f} "
            f"overall_lower={float(top_shadow['overall_lower']):.6f}"
        )
        return 0
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the CLI test**

Run:

```bash
python3 -m unittest tests.test_track2_v6_specialist_selector.Track2V6CliTest -v
```

Expected result:

```text
Ran 1 test
OK
```

- [ ] **Step 5: Run the full v6 test file**

Run:

```bash
python3 -m unittest tests.test_track2_v6_specialist_selector -v
```

Expected result:

```text
Ran 9 tests
OK
```

- [ ] **Step 6: Commit the CLI**

Run:

```bash
git add scripts/track2_v6_specialist_selector.py tests/test_track2_v6_specialist_selector.py
git commit -m "feat: add track2 v6 selector cli"
```

Expected result:

```text
[codex/track2-human-review-final ...] feat: add track2 v6 selector cli
```

## Task 5: Run V6 On Real Track2 Artifacts

**Files:**
- Generated: `experiments/track2_v6_specialist_selector_20260605/v1/`
- Generated: `submissions/track2_submission_v6_safe_sameq_candidate.json`
- Generated: `submissions/track2_submission_v6_safe_sameq_candidate.zip`
- Generated: `submissions/track2_submission_v6_cross_micro_candidate.json`
- Generated: `submissions/track2_submission_v6_cross_micro_candidate.zip`
- Generated: `submissions/track2_submission_v6_desc_plus_candidate.json`
- Generated: `submissions/track2_submission_v6_desc_plus_candidate.zip`

- [ ] **Step 1: Run the real v6 selector**

Run:

```bash
python3 scripts/track2_v6_specialist_selector.py run \
  --baseline-json submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json \
  --evidence-matrix experiments/track2_score_calibrated_champion_20260605/v1/evidence_matrix.csv \
  --out-dir experiments/track2_v6_specialist_selector_20260605/v1 \
  --submission-dir submissions \
  --expected-row-count 1000
```

Expected result:

```text
track2 v6 selector decision=...
```

- [ ] **Step 2: Inspect generated summary paths**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("experiments/track2_v6_specialist_selector_20260605/v1/v6_summary.json").read_text())
print(summary["decision"])
for name, payload in summary["candidates"].items():
    print(name, payload["changed_rows"], payload["json"], payload["zip"])
PY
```

Expected result:

```text
recommend_submit
v6_safe_sameq ...
v6_cross_micro ...
v6_desc_plus ...
```

If the first line is `recommend_hold`, continue verification and report that v6 did not earn an online submission.

- [ ] **Step 3: Validate all generated candidate JSON files**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v6_safe_sameq_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v6_cross_micro_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v6_desc_plus_candidate.json
```

Expected result for each command:

```text
Track2 validation OK
```

- [ ] **Step 4: Inspect local shadow ranking**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
ranking = json.loads(Path("experiments/track2_v6_specialist_selector_20260605/v1/shadow_eval/candidate_ranking.json").read_text())
for item in ranking:
    print(
        item["candidate_name"],
        item["decision"],
        f"overall_expected={item['overall_expected']:.6f}",
        f"overall_lower={item['overall_lower']:.6f}",
        f"changed={item['changed_rows']}",
        f"cross={item['cross_quadrant_changes']}",
    )
PY
```

Expected result:

```text
v6_... recommend_submit overall_expected=...
```

The v6 candidate may be proposed for online submission only if its `overall_expected` beats `0.8639085` and its `overall_lower` does not fall below the configured tolerance.

- [ ] **Step 5: Open the HTML review when the browser is useful**

Use the in-app browser to open:

```text
file:///Users/yhryzy/dev/emoart-130k/experiments/track2_v6_specialist_selector_20260605/v1/html_review/track2_v6_specialist_selector_review.html
```

Expected result:

```text
The page shows the v6 top-level decision and selector decision table.
```

- [ ] **Step 6: Commit real v6 run artifacts only if generated checks pass**

Run:

```bash
git add affectiveart/track2_v6_specialist_selector.py \
  scripts/track2_v6_specialist_selector.py \
  tests/test_track2_v6_specialist_selector.py \
  experiments/track2_v6_specialist_selector_20260605/v1 \
  submissions/track2_submission_v6_safe_sameq_candidate.json \
  submissions/track2_submission_v6_safe_sameq_candidate.zip \
  submissions/track2_submission_v6_cross_micro_candidate.json \
  submissions/track2_submission_v6_cross_micro_candidate.zip \
  submissions/track2_submission_v6_desc_plus_candidate.json \
  submissions/track2_submission_v6_desc_plus_candidate.zip
git commit -m "feat: run track2 v6 specialist selector"
```

Expected result:

```text
[codex/track2-human-review-final ...] feat: run track2 v6 specialist selector
```

If validation fails, do not commit generated candidate files. Commit only code and tests after fixing the failing condition.

## Task 6: Track2 Regression Verification And Final Report

**Files:**
- Read: `experiments/track2_v6_specialist_selector_20260605/v1/v6_summary.md`
- Read: `experiments/track2_v6_specialist_selector_20260605/v1/shadow_eval/shadow_score_report.md`
- Read: `experiments/track2_v6_specialist_selector_20260605/v1/hard96_stability_report.md`

- [ ] **Step 1: Run Track2-only unit tests**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
```

Expected result:

```text
OK
```

If unrelated Track2 tests fail because of existing dirty worktree state, record the failing test names and exact error lines in the final report.

- [ ] **Step 2: Run diff whitespace check**

Run:

```bash
git diff --check
```

Expected result:

```text

```

- [ ] **Step 3: Produce the final Track2 v6 decision summary**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path("experiments/track2_v6_specialist_selector_20260605/v1")
summary = json.loads((root / "v6_summary.json").read_text())
ranking = json.loads((root / "shadow_eval" / "candidate_ranking.json").read_text())
print("decision:", summary["decision"])
print("html:", root / "html_review" / "track2_v6_specialist_selector_review.html")
for item in ranking:
    print(item["candidate_name"], item["decision"], item["overall_expected"], item["overall_lower"])
PY
```

Expected result:

```text
decision: recommend_submit
html: experiments/track2_v6_specialist_selector_20260605/v1/html_review/track2_v6_specialist_selector_review.html
...
```

If the first line is `decision: recommend_hold`, the final answer must say v6 is a diagnostic run and not a Codabench-spend candidate.

- [ ] **Step 4: Final commit if verification is clean**

Run:

```bash
git status --short
git log --oneline -5
```

Expected result:

```text
Recent commits include the v6 plan, selector core, CLI, and real run commit.
```

Do not stage unrelated Track1 files or unrelated pre-existing untracked files.

## Self-Review

- Spec coverage: Evidence normalization, deterministic selector, malformed evidence default-hold, high-authority contradiction hold, cross-quadrant micro gate, side-path candidates, shadow scoring, reports, Track2-only tests, and formal submission protection are covered.
- Type consistency: `V6Delta`, `SelectorThresholds`, `SelectorDecision`, `load_evidence_matrix`, `select_v6_deltas`, `build_candidate_rows`, and `write_v6_outputs` are defined before they are referenced by tests and CLI.
- Submission safety: The plan never writes `submissions/track2_submission.json` or `submissions/track2_submission.zip`.
- Online safety: The plan does not submit to Codabench.
