# Track2 MoE Specialist Dry-Run Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track2-only MoE-like specialist dry-run gate that explains proposed label changes without writing any submission JSON or ZIP.

**Architecture:** Add a focused `affectiveart.track2_moe_specialist_ensemble` module that normalizes expert evidence, builds a hard-case queue, applies deterministic acceptance/rejection rules, and writes JSON/Markdown/HTML review artifacts. The first executable milestone is dry-run only; candidate generation is deliberately excluded from this plan.

**Tech Stack:** Python standard library, existing `affectiveart.track2_audit`, existing `affectiveart.track2_visual_audit`, existing Track2 JSON prediction payloads, `unittest`, local HTML review files.

---

## File Structure

- Create `affectiveart/track2_moe_specialist_ensemble.py`
  Owns normalized expert evidence, gate thresholds, dry-run decision logic, report rendering, and artifact writing.

- Create `scripts/track2_moe_specialist_ensemble.py`
  Exposes the dry-run command. It must not expose candidate generation in this plan.

- Create `tests/test_track2_moe_specialist_ensemble.py`
  Covers normalization, acceptance/rejection gates, dry-run report output, CLI behavior, and no submission writes.

- Output directory for real runs:
  `experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/`

## Non-Goals For This Plan

- Do not write `submissions/final_track2_20260603_moe_specialist_candidate.json`.
- Do not write any ZIP file.
- Do not modify `submissions/track2_submission.json`.
- Do not modify `submissions/track2_submission.zip`.
- Do not run Track1 tests.
- Do not call Gemini or Vertex from this dry-run implementation.

## Task 1: Gate Data Model And Label Safety

**Files:**
- Create: `affectiveart/track2_moe_specialist_ensemble.py`
- Create: `tests/test_track2_moe_specialist_ensemble.py`

- [ ] **Step 1: Write failing unit tests for thresholds, expected labels, and expert normalization**

Add this test file:

```python
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_moe_specialist_ensemble import (
    GateThresholds,
    expected_label_for_emotion,
    normalize_expert_entries,
)


class Track2MoeSpecialistEnsembleTest(unittest.TestCase):
    def test_expected_label_for_emotion_maps_track2_quadrants(self):
        self.assertEqual(expected_label_for_emotion("calm"), ("Positive", "Low"))
        self.assertEqual(expected_label_for_emotion("aroused"), ("Positive", "High"))
        self.assertEqual(expected_label_for_emotion("frustrated"), ("Negative", "High"))
        self.assertEqual(expected_label_for_emotion("tired"), ("Negative", "Low"))

    def test_default_thresholds_are_conservative(self):
        thresholds = GateThresholds()
        self.assertEqual(thresholds.min_supporting_sources, 2)
        self.assertEqual(thresholds.high_confidence, 0.86)
        self.assertAlmostEqual(thresholds.min_macro_f1_gain, 0.015)
        self.assertAlmostEqual(thresholds.max_accuracy_drop, 0.010)
        self.assertAlmostEqual(thresholds.max_top_emotion_share_delta, 0.02)

    def test_normalize_expert_entries_accepts_list_and_payload_entries(self):
        rows = normalize_expert_entries(
            {
                "entries": [
                    {
                        "sample_id": "track2_0001",
                        "emotion": "calm",
                        "confidence": "0.91",
                        "margin": "0.31",
                        "top3": [{"emotion": "calm", "probability": 0.91}],
                    }
                ]
            },
            source="siglip2_clean",
            role="global",
        )

        self.assertEqual(rows[0]["sample_id"], "track2_0001")
        self.assertEqual(rows[0]["source"], "siglip2_clean")
        self.assertEqual(rows[0]["role"], "global")
        self.assertEqual(rows[0]["emotion"], "calm")
        self.assertAlmostEqual(rows[0]["confidence"], 0.91)
        self.assertAlmostEqual(rows[0]["margin"], 0.31)
        self.assertEqual(rows[0]["top3"][0]["emotion"], "calm")
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python3 -m unittest tests.test_track2_moe_specialist_ensemble -v
```

Expected:

```text
ModuleNotFoundError: No module named 'affectiveart.track2_moe_specialist_ensemble'
```

- [ ] **Step 3: Implement the minimal data model and normalization code**

Create `affectiveart/track2_moe_specialist_ensemble.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from affectiveart.challenge import TRACK2_JSON_EMOTIONS
from affectiveart.track2_audit import (
    HIGH_AROUSAL_EMOTIONS,
    LOW_AROUSAL_EMOTIONS,
    NEGATIVE_EMOTIONS,
    POSITIVE_EMOTIONS,
)


@dataclass(frozen=True)
class GateThresholds:
    min_supporting_sources: int = 2
    high_confidence: float = 0.86
    min_margin: float = 0.12
    min_macro_f1_gain: float = 0.015
    min_hardcase_macro_f1_gain: float = 0.030
    max_accuracy_drop: float = 0.010
    max_valence_accuracy_drop: float = 0.005
    max_arousal_accuracy_drop: float = 0.005
    max_top_emotion_share_delta: float = 0.020


def expected_label_for_emotion(emotion: str) -> tuple[str, str]:
    normalized = str(emotion).strip().lower()
    if normalized in POSITIVE_EMOTIONS:
        valence = "Positive"
    elif normalized in NEGATIVE_EMOTIONS:
        valence = "Negative"
    else:
        raise ValueError(f"unknown Track2 emotion: {emotion}")

    if normalized in HIGH_AROUSAL_EMOTIONS:
        arousal = "High"
    elif normalized in LOW_AROUSAL_EMOTIONS:
        arousal = "Low"
    else:
        raise ValueError(f"unknown Track2 emotion: {emotion}")
    return valence, arousal


def normalize_expert_entries(
    payload: dict[str, Any] | list[Any],
    *,
    source: str,
    role: str,
) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw_entries = payload.get("entries") or payload.get("predictions") or payload.get("rows") or []
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raise ValueError("expert payload must be a JSON object or list")

    entries: list[dict[str, Any]] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        sample_id = str(raw.get("sample_id") or raw.get("request_id") or "").strip()
        emotion = str(raw.get("emotion", "")).strip().lower()
        if not sample_id or emotion not in TRACK2_JSON_EMOTIONS:
            continue
        entries.append(
            {
                "sample_id": sample_id,
                "source": source,
                "role": role,
                "emotion": emotion,
                "confidence": _safe_float(raw.get("confidence"), 0.0),
                "margin": _safe_float(raw.get("margin"), 0.0),
                "top3": _normalize_top3(raw.get("top3") or raw.get("top_k") or []),
            }
        )
    return entries


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_top3(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    top3: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        emotion = str(item.get("emotion", "")).strip().lower()
        if emotion not in TRACK2_JSON_EMOTIONS:
            continue
        probability = _safe_float(item.get("probability", item.get("score", 0.0)), 0.0)
        top3.append({"emotion": emotion, "probability": probability})
    return top3
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
python3 -m unittest tests.test_track2_moe_specialist_ensemble -v
```

Expected:

```text
Ran 3 tests
OK
```

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add affectiveart/track2_moe_specialist_ensemble.py tests/test_track2_moe_specialist_ensemble.py
git commit -m "feat: add track2 moe gate primitives"
```

## Task 2: Deterministic Gate Decisions

**Files:**
- Modify: `affectiveart/track2_moe_specialist_ensemble.py`
- Modify: `tests/test_track2_moe_specialist_ensemble.py`

- [ ] **Step 1: Add failing tests for accept, hold, and rejection behavior**

Append these tests to `tests/test_track2_moe_specialist_ensemble.py`:

```python
from affectiveart.track2_moe_specialist_ensemble import build_gate_decision


def current_row(sample_id="track2_0001", emotion="content", valence="Positive", arousal="Low"):
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": "A quiet landscape with balanced color and soft light.",
        "brushstroke": "soft layered brushwork",
        "composition": "stable horizontal composition",
        "color": "muted greens and blues",
        "line": "gentle contour lines",
        "light": "diffuse low-contrast light",
    }


def expert(sample_id, source, emotion, confidence=0.9, margin=0.2, role="global"):
    return {
        "sample_id": sample_id,
        "source": source,
        "role": role,
        "emotion": emotion,
        "confidence": confidence,
        "margin": margin,
        "top3": [{"emotion": emotion, "probability": confidence}],
    }


class Track2MoeSpecialistEnsembleTest(Track2MoeSpecialistEnsembleTest):
    def test_gate_accepts_two_source_supported_va_consistent_change(self):
        decision = build_gate_decision(
            current_row("track2_0001", "content"),
            [
                expert("track2_0001", "siglip2_clean", "calm"),
                expert("track2_0001", "boundary_head", "calm", role="boundary"),
            ],
        )

        self.assertEqual(decision["decision"], "accept_change")
        self.assertEqual(decision["proposed_emotion"], "calm")
        self.assertEqual(decision["proposed_valence"], "Positive")
        self.assertEqual(decision["proposed_arousal"], "Low")
        self.assertIn("supported_by_2_sources", decision["reasons"])

    def test_gate_holds_single_source_change(self):
        decision = build_gate_decision(
            current_row("track2_0001", "content"),
            [expert("track2_0001", "siglip2_clean", "calm")],
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertEqual(decision["proposed_emotion"], "calm")
        self.assertIn("insufficient_independent_support", decision["reasons"])

    def test_gate_rejects_text_contradiction(self):
        decision = build_gate_decision(
            current_row("track2_0001", "content"),
            [
                expert("track2_0001", "siglip2_clean", "sad"),
                expert("track2_0001", "boundary_head", "sad", role="boundary"),
            ],
            description_audit={"verdict": "contradiction", "reason": "caption reads peaceful"},
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertIn("description_contradiction", decision["reasons"])

    def test_gate_holds_high_similarity_change_without_human_approval(self):
        decision = build_gate_decision(
            current_row("track2_0001", "content"),
            [
                expert("track2_0001", "siglip2_clean", "calm"),
                expert("track2_0001", "boundary_head", "calm", role="boundary"),
            ],
            high_similarity_public_reference=True,
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertIn("high_similarity_requires_explicit_review", decision["reasons"])
```

- [ ] **Step 2: Run the focused test and verify the new tests fail**

Run:

```bash
python3 -m unittest tests.test_track2_moe_specialist_ensemble -v
```

Expected:

```text
ImportError: cannot import name 'build_gate_decision'
```

- [ ] **Step 3: Implement deterministic gate decisions**

Add these functions to `affectiveart/track2_moe_specialist_ensemble.py`:

```python
def build_gate_decision(
    current_row: dict[str, Any],
    expert_rows: list[dict[str, Any]],
    *,
    thresholds: GateThresholds | None = None,
    description_audit: dict[str, Any] | None = None,
    high_similarity_public_reference: bool = False,
) -> dict[str, Any]:
    thresholds = thresholds or GateThresholds()
    sample_id = str(current_row.get("sample_id", ""))
    current_emotion = str(current_row.get("emotion", "")).strip().lower()
    relevant = [row for row in expert_rows if str(row.get("sample_id", "")) == sample_id]
    evidence = _summarize_evidence(relevant)
    proposed_emotion = _choose_proposal(current_emotion, evidence)
    reasons: list[str] = []

    if not proposed_emotion or proposed_emotion == current_emotion:
        return _decision_row(
            current_row,
            relevant,
            "keep_current",
            current_emotion,
            ["no_supported_change"],
        )

    support = evidence.get(proposed_emotion, [])
    high_conf_support = [
        row for row in support
        if float(row.get("confidence", 0.0)) >= thresholds.high_confidence
        and float(row.get("margin", 0.0)) >= thresholds.min_margin
    ]
    if len(support) >= thresholds.min_supporting_sources:
        reasons.append(f"supported_by_{len(support)}_sources")
    elif len(high_conf_support) == 1 and not _strong_opposition(current_emotion, proposed_emotion, evidence, thresholds):
        reasons.append("single_high_confidence_source_without_strong_opposition")
    else:
        reasons.append("insufficient_independent_support")

    proposed_valence, proposed_arousal = expected_label_for_emotion(proposed_emotion)
    if description_audit and str(description_audit.get("verdict", "")).lower() == "contradiction":
        reasons.append("description_contradiction")
    if high_similarity_public_reference:
        reasons.append("high_similarity_requires_explicit_review")

    blocking = {
        "insufficient_independent_support",
        "description_contradiction",
        "high_similarity_requires_explicit_review",
    }
    decision = "hold" if blocking.intersection(reasons) else "accept_change"
    return _decision_row(
        current_row,
        relevant,
        decision,
        proposed_emotion,
        reasons,
        proposed_valence=proposed_valence,
        proposed_arousal=proposed_arousal,
    )


def _summarize_evidence(expert_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {}
    for row in expert_rows:
        emotion = str(row.get("emotion", "")).strip().lower()
        if emotion not in TRACK2_JSON_EMOTIONS:
            continue
        evidence.setdefault(emotion, []).append(row)
    return evidence


def _choose_proposal(current_emotion: str, evidence: dict[str, list[dict[str, Any]]]) -> str:
    candidates = [emotion for emotion in evidence if emotion != current_emotion]
    if not candidates:
        return current_emotion
    return sorted(
        candidates,
        key=lambda emotion: (
            -len(evidence[emotion]),
            -sum(float(row.get("confidence", 0.0)) for row in evidence[emotion]),
            emotion,
        ),
    )[0]


def _strong_opposition(
    current_emotion: str,
    proposed_emotion: str,
    evidence: dict[str, list[dict[str, Any]]],
    thresholds: GateThresholds,
) -> bool:
    for emotion, rows in evidence.items():
        if emotion == proposed_emotion:
            continue
        for row in rows:
            if float(row.get("confidence", 0.0)) >= thresholds.high_confidence and float(row.get("margin", 0.0)) >= thresholds.min_margin:
                return True
    return False


def _decision_row(
    current_row: dict[str, Any],
    expert_rows: list[dict[str, Any]],
    decision: str,
    proposed_emotion: str,
    reasons: list[str],
    *,
    proposed_valence: str | None = None,
    proposed_arousal: str | None = None,
) -> dict[str, Any]:
    if proposed_valence is None or proposed_arousal is None:
        proposed_valence, proposed_arousal = expected_label_for_emotion(proposed_emotion)
    return {
        "sample_id": str(current_row.get("sample_id", "")),
        "decision": decision,
        "current_emotion": str(current_row.get("emotion", "")),
        "current_valence": str(current_row.get("emotional_valence", "")),
        "current_arousal": str(current_row.get("emotional_arousal_level", "")),
        "proposed_emotion": proposed_emotion,
        "proposed_valence": proposed_valence,
        "proposed_arousal": proposed_arousal,
        "reasons": reasons,
        "expert_evidence": sorted(
            expert_rows,
            key=lambda row: (str(row.get("source", "")), str(row.get("emotion", ""))),
        ),
    }
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
python3 -m unittest tests.test_track2_moe_specialist_ensemble -v
```

Expected:

```text
Ran 7 tests
OK
```

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add affectiveart/track2_moe_specialist_ensemble.py tests/test_track2_moe_specialist_ensemble.py
git commit -m "feat: add track2 moe dry-run gate decisions"
```

## Task 3: Dry-Run Report Builder

**Files:**
- Modify: `affectiveart/track2_moe_specialist_ensemble.py`
- Modify: `tests/test_track2_moe_specialist_ensemble.py`

- [ ] **Step 1: Add failing tests for dry-run report summary and no candidate outputs**

Append this test:

```python
from affectiveart.track2_moe_specialist_ensemble import build_dry_run_report


class Track2MoeSpecialistEnsembleTest(Track2MoeSpecialistEnsembleTest):
    def test_build_dry_run_report_summarizes_gate_decisions(self):
        current_rows = [
            current_row("track2_0001", "content"),
            current_row("track2_0002", "content"),
        ]
        expert_rows = [
            expert("track2_0001", "siglip2_clean", "calm"),
            expert("track2_0001", "boundary_head", "calm", role="boundary"),
            expert("track2_0002", "siglip2_clean", "sad"),
        ]

        report = build_dry_run_report(
            current_rows,
            expert_rows,
            queue_sample_ids=["track2_0001", "track2_0002"],
            high_similarity_sample_ids=set(),
        )

        self.assertEqual(report["method"], "track2_moe_specialist_dry_run_v1")
        self.assertEqual(report["row_count"], 2)
        self.assertEqual(report["decision_counts"]["accept_change"], 1)
        self.assertEqual(report["decision_counts"]["hold"], 1)
        self.assertEqual(report["formal_submission_overwritten"], False)
        self.assertEqual(report["candidate_json_written"], False)
        self.assertEqual(report["candidate_zip_written"], False)
        by_id = {row["sample_id"]: row for row in report["rows"]}
        self.assertEqual(by_id["track2_0001"]["proposed_emotion"], "calm")
        self.assertEqual(by_id["track2_0002"]["decision"], "hold")
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python3 -m unittest tests.test_track2_moe_specialist_ensemble -v
```

Expected:

```text
ImportError: cannot import name 'build_dry_run_report'
```

- [ ] **Step 3: Implement dry-run report construction**

Add this code:

```python
import json
from collections import Counter
from pathlib import Path

from affectiveart.track2_audit import compute_track2_distribution


def build_dry_run_report(
    current_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    *,
    queue_sample_ids: list[str],
    high_similarity_sample_ids: set[str] | None = None,
    description_audit_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    high_similarity_sample_ids = high_similarity_sample_ids or set()
    description_audit_by_id = description_audit_by_id or {}
    current_by_id = {
        str(row.get("sample_id", "")): row
        for row in current_rows
        if row.get("sample_id")
    }
    decisions: list[dict[str, Any]] = []
    for sample_id in sorted(dict.fromkeys(queue_sample_ids)):
        current = current_by_id.get(sample_id)
        if not current:
            continue
        decisions.append(
            build_gate_decision(
                current,
                expert_rows,
                description_audit=description_audit_by_id.get(sample_id),
                high_similarity_public_reference=sample_id in high_similarity_sample_ids,
            )
        )

    decision_counts = Counter(row["decision"] for row in decisions)
    proposed_transitions = Counter(
        f"{row['current_emotion']}->{row['proposed_emotion']}"
        for row in decisions
        if row["decision"] == "accept_change"
    )
    return {
        "method": "track2_moe_specialist_dry_run_v1",
        "row_count": len(decisions),
        "decision_counts": dict(decision_counts),
        "accepted_transition_counts": dict(proposed_transitions),
        "baseline_distribution": compute_track2_distribution(current_rows),
        "formal_submission_overwritten": False,
        "candidate_json_written": False,
        "candidate_zip_written": False,
        "rows": decisions,
    }
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
python3 -m unittest tests.test_track2_moe_specialist_ensemble -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add affectiveart/track2_moe_specialist_ensemble.py tests/test_track2_moe_specialist_ensemble.py
git commit -m "feat: build track2 moe dry-run reports"
```

## Task 4: Dry-Run Artifact Writer And HTML Review

**Files:**
- Modify: `affectiveart/track2_moe_specialist_ensemble.py`
- Modify: `tests/test_track2_moe_specialist_ensemble.py`

- [ ] **Step 1: Add failing tests for JSON, Markdown, HTML, and asset output**

Append this test:

```python
import zipfile
from PIL import Image

from affectiveart.track2_moe_specialist_ensemble import write_dry_run_outputs


class Track2MoeSpecialistEnsembleTest(Track2MoeSpecialistEnsembleTest):
    def test_write_dry_run_outputs_writes_review_artifacts_without_submission_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_zip = tmp_path / "track2.zip"
            source_image = tmp_path / "track2_0001.jpg"
            Image.new("RGB", (64, 48), (80, 120, 160)).save(source_image)
            with zipfile.ZipFile(image_zip, "w") as zf:
                zf.write(source_image, "track2_testset/images/track2_0001.jpg")

            out_dir = tmp_path / "dry_run"
            report = build_dry_run_report(
                [current_row("track2_0001", "content")],
                [
                    expert("track2_0001", "siglip2_clean", "calm"),
                    expert("track2_0001", "boundary_head", "calm", role="boundary"),
                ],
                queue_sample_ids=["track2_0001"],
            )
            outputs = write_dry_run_outputs(report, image_zip=image_zip, out_dir=out_dir)

            self.assertTrue(Path(outputs["json"]).exists())
            self.assertTrue(Path(outputs["markdown"]).exists())
            self.assertTrue(Path(outputs["html"]).exists())
            html_text = Path(outputs["html"]).read_text(encoding="utf-8")
            self.assertIn("Track2 MoE Specialist Dry-Run", html_text)
            self.assertIn("track2_0001", html_text)
            self.assertIn("accept_change", html_text)
            self.assertTrue((out_dir / "html_review" / "assets" / "track2_0001.jpg").exists())
            self.assertFalse((out_dir / "submission.json").exists())
            self.assertFalse((out_dir / "submission.zip").exists())
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python3 -m unittest tests.test_track2_moe_specialist_ensemble -v
```

Expected:

```text
ImportError: cannot import name 'write_dry_run_outputs'
```

- [ ] **Step 3: Implement artifact writer and compact HTML renderer**

Add these functions:

```python
import html
import shutil
import zipfile

from affectiveart.track2_visual_audit import find_track2_image_member


DRY_RUN_OUTPUT_FILENAMES = {
    "json": "track2_moe_specialist_dry_run_report.json",
    "markdown": "track2_moe_specialist_dry_run_report.md",
    "html": "html_review/track2_moe_specialist_dry_run_review.html",
}


def write_dry_run_outputs(
    report: dict[str, Any],
    *,
    image_zip: str | Path,
    out_dir: str | Path,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    html_dir = out_dir / "html_review"
    assets_dir = html_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    _extract_review_assets(report["rows"], Path(image_zip), assets_dir)

    outputs = {key: str(out_dir / value) for key, value in DRY_RUN_OUTPUT_FILENAMES.items()}
    Path(outputs["json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(outputs["json"]).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(outputs["markdown"]).write_text(render_dry_run_markdown(report, outputs), encoding="utf-8")
    Path(outputs["html"]).write_text(render_dry_run_html(report), encoding="utf-8")
    return outputs


def render_dry_run_markdown(report: dict[str, Any], outputs: dict[str, str]) -> str:
    lines = [
        "# Track2 MoE Specialist Dry-Run",
        "",
        f"- Rows: {report['row_count']}",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        f"- Candidate JSON written: {report['candidate_json_written']}",
        f"- Candidate ZIP written: {report['candidate_zip_written']}",
        f"- HTML: `{outputs['html']}`",
        "",
        "## Decision Counts",
        "",
    ]
    for decision, count in sorted(report["decision_counts"].items()):
        lines.append(f"- {decision}: {count}")
    lines.extend(["", "## Accepted Transitions", ""])
    if report["accepted_transition_counts"]:
        for transition, count in sorted(report["accepted_transition_counts"].items()):
            lines.append(f"- {transition}: {count}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def render_dry_run_html(report: dict[str, Any]) -> str:
    cards = "\n".join(_render_dry_run_card(row) for row in report["rows"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track2 MoE Specialist Dry-Run</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f7fb; color: #1f2933; }}
    header {{ position: sticky; top: 0; padding: 16px 24px; background: #fff; border-bottom: 1px solid #d8dee7; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 20px 24px 48px; }}
    .card {{ display: grid; grid-template-columns: minmax(220px, 330px) 1fr; gap: 16px; margin-bottom: 16px; padding: 16px; background: #fff; border: 1px solid #d8dee7; border-radius: 8px; }}
    img {{ width: 100%; aspect-ratio: 4 / 3; object-fit: contain; border: 1px solid #d8dee7; border-radius: 6px; background: #eef2f7; }}
    .pill {{ display: inline-block; margin: 0 6px 6px 0; padding: 4px 7px; border-radius: 6px; background: #eaf7f5; color: #0f615b; font-size: 12px; font-weight: 650; }}
    .hold {{ background: #fff3dc; color: #9f580a; }}
    .reject {{ background: #fee2e2; color: #991b1b; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border-bottom: 1px solid #d8dee7; padding: 6px; text-align: left; font-size: 13px; }}
    @media (max-width: 820px) {{ .card {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Track2 MoE Specialist Dry-Run</h1>
    <div>Rows: {report['row_count']} | No submission JSON/ZIP written</div>
  </header>
  <main>{cards}</main>
</body>
</html>
"""


def _render_dry_run_card(row: dict[str, Any]) -> str:
    sample_id = html.escape(row["sample_id"])
    decision = html.escape(row["decision"])
    badge_class = "hold" if row["decision"] == "hold" else ""
    reasons = " ".join(f'<span class="pill">{html.escape(reason)}</span>' for reason in row["reasons"])
    evidence_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('source', '')))}</td>"
        f"<td>{html.escape(str(item.get('role', '')))}</td>"
        f"<td>{html.escape(str(item.get('emotion', '')))}</td>"
        f"<td>{float(item.get('confidence', 0.0)):.3f}</td>"
        f"<td>{float(item.get('margin', 0.0)):.3f}</td>"
        "</tr>"
        for item in row["expert_evidence"]
    )
    return f"""<section class="card" id="{sample_id}">
  <div><img src="assets/{sample_id}.jpg" alt="{sample_id}"></div>
  <div>
    <h2>{sample_id}</h2>
    <span class="pill {badge_class}">{decision}</span>
    <p>Current: {html.escape(row['current_emotion'])} / {html.escape(row['current_valence'])} / {html.escape(row['current_arousal'])}</p>
    <p>Proposed: {html.escape(row['proposed_emotion'])} / {html.escape(row['proposed_valence'])} / {html.escape(row['proposed_arousal'])}</p>
    <div>{reasons}</div>
    <table><thead><tr><th>Source</th><th>Role</th><th>Emotion</th><th>Conf</th><th>Margin</th></tr></thead><tbody>{evidence_rows}</tbody></table>
  </div>
</section>"""


def _extract_review_assets(rows: list[dict[str, Any]], image_zip: Path, assets_dir: Path) -> None:
    with zipfile.ZipFile(image_zip) as zf:
        for row in rows:
            sample_id = str(row["sample_id"])
            member = find_track2_image_member(zf, sample_id)
            if not member:
                continue
            target = assets_dir / f"{_safe_asset_filename(sample_id)}.jpg"
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _safe_asset_filename(sample_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in sample_id)
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
python3 -m unittest tests.test_track2_moe_specialist_ensemble -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add affectiveart/track2_moe_specialist_ensemble.py tests/test_track2_moe_specialist_ensemble.py
git commit -m "feat: render track2 moe dry-run review"
```

## Task 5: Dry-Run CLI

**Files:**
- Create: `scripts/track2_moe_specialist_ensemble.py`
- Modify: `tests/test_track2_moe_specialist_ensemble.py`

- [ ] **Step 1: Add a failing CLI test**

Append this test:

```python
import subprocess


class Track2MoeSpecialistEnsembleTest(Track2MoeSpecialistEnsembleTest):
    def test_cli_dry_run_writes_artifacts_and_no_submission_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(__file__).resolve().parents[1]
            tmp_path = Path(tmp)
            current_json = tmp_path / "current.json"
            source_json = tmp_path / "source.json"
            image_zip = tmp_path / "track2.zip"
            out_dir = tmp_path / "out"
            current_json.write_text(json.dumps([current_row("track2_0001", "content")]), encoding="utf-8")
            source_json.write_text(
                json.dumps(
                    {
                        "entries": [
                            expert("track2_0001", "siglip2_clean", "calm"),
                            expert("track2_0001", "boundary_head", "calm", role="boundary"),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            image = tmp_path / "track2_0001.jpg"
            Image.new("RGB", (64, 48), (120, 90, 70)).save(image)
            with zipfile.ZipFile(image_zip, "w") as zf:
                zf.write(image, "track2_testset/images/track2_0001.jpg")

            result = subprocess.run(
                [
                    "python3",
                    "scripts/track2_moe_specialist_ensemble.py",
                    "dry-run",
                    "--current-json",
                    str(current_json),
                    "--expert-source",
                    f"combined=global={source_json}",
                    "--queue-sample-id",
                    "track2_0001",
                    "--image-zip",
                    str(image_zip),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out_dir / "track2_moe_specialist_dry_run_report.json").exists())
            self.assertTrue((out_dir / "html_review" / "track2_moe_specialist_dry_run_review.html").exists())
            self.assertFalse((out_dir / "submission.json").exists())
            self.assertFalse((out_dir / "submission.zip").exists())

    def test_cli_rejects_output_under_submissions_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(__file__).resolve().parents[1]
            tmp_path = Path(tmp)
            current_json = tmp_path / "current.json"
            source_json = tmp_path / "source.json"
            image_zip = tmp_path / "track2.zip"
            blocked_out = repo_root / "submissions" / "moe_dry_run_blocked"
            current_json.write_text(json.dumps([current_row("track2_0001", "content")]), encoding="utf-8")
            source_json.write_text(json.dumps({"entries": [expert("track2_0001", "siglip2_clean", "calm")]}), encoding="utf-8")
            image = tmp_path / "track2_0001.jpg"
            Image.new("RGB", (64, 48), (120, 90, 70)).save(image)
            with zipfile.ZipFile(image_zip, "w") as zf:
                zf.write(image, "track2_testset/images/track2_0001.jpg")

            result = subprocess.run(
                [
                    "python3",
                    "scripts/track2_moe_specialist_ensemble.py",
                    "dry-run",
                    "--current-json",
                    str(current_json),
                    "--expert-source",
                    f"combined=global={source_json}",
                    "--queue-sample-id",
                    "track2_0001",
                    "--image-zip",
                    str(image_zip),
                    "--out-dir",
                    str(blocked_out),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dry-run output must not be under submissions", result.stderr)
            self.assertFalse(blocked_out.exists())
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python3 -m unittest tests.test_track2_moe_specialist_ensemble -v
```

Expected:

```text
can't open file
```

- [ ] **Step 3: Implement the CLI script**

Create `scripts/track2_moe_specialist_ensemble.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_moe_specialist_ensemble import (
    build_dry_run_report,
    normalize_expert_entries,
    write_dry_run_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track2 MoE-like specialist dry-run gate.")
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run")
    dry.add_argument("--current-json", type=Path, required=True)
    dry.add_argument("--expert-source", action="append", default=[], help="name=role=path")
    dry.add_argument("--queue-sample-id", action="append", default=[])
    dry.add_argument("--queue-csv", type=Path)
    dry.add_argument("--image-zip", type=Path, required=True)
    dry.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "dry-run":
        _validate_dry_run_out_dir(args.out_dir)
        current_rows = _read_json_list(args.current_json)
        expert_rows = []
        for spec in args.expert_source:
            source, role, path = _parse_source_spec(spec)
            payload = json.loads(path.read_text(encoding="utf-8"))
            expert_rows.extend(normalize_expert_entries(payload, source=source, role=role))
        queue_sample_ids = _queue_sample_ids(args.queue_sample_id, args.queue_csv, expert_rows)
        report = build_dry_run_report(
            current_rows,
            expert_rows,
            queue_sample_ids=queue_sample_ids,
        )
        outputs = write_dry_run_outputs(report, image_zip=args.image_zip, out_dir=args.out_dir)
        print(json.dumps({"row_count": report["row_count"], "outputs": outputs}, indent=2, ensure_ascii=False))


def _read_json_list(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _parse_source_spec(spec: str) -> tuple[str, str, Path]:
    parts = spec.split("=", 2)
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
        raise ValueError(f"expected name=role=path: {spec}")
    return parts[0], parts[1], Path(parts[2])


def _queue_sample_ids(cli_ids: list[str], queue_csv: Path | None, expert_rows: list[dict]) -> list[str]:
    if cli_ids:
        return cli_ids
    if queue_csv:
        import csv

        with queue_csv.open(newline="", encoding="utf-8") as fh:
            return [
                str(row.get("sample_id", "")).strip()
                for row in csv.DictReader(fh)
                if str(row.get("sample_id", "")).strip()
            ]
    return sorted({row["sample_id"] for row in expert_rows})


def _validate_dry_run_out_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    submissions_dir = (Path.cwd() / "submissions").resolve()
    if resolved == submissions_dir or submissions_dir in resolved.parents:
        raise SystemExit("dry-run output must not be under submissions/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
python3 -m unittest tests.test_track2_moe_specialist_ensemble -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add scripts/track2_moe_specialist_ensemble.py tests/test_track2_moe_specialist_ensemble.py
git commit -m "feat: add track2 moe dry-run cli"
```

## Task 6: Real Track2 Dry-Run Execution

**Files:**
- Output: `experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/`

- [ ] **Step 1: Run the dry-run gate on existing clean/inclusive evidence**

Run:

```bash
python3 scripts/track2_moe_specialist_ensemble.py dry-run \
  --current-json submissions/final_track2_20260602_scaled_human_gate_gemini35_guarded_v1.json \
  --expert-source clean=global=experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/clean_predictions.json \
  --expert-source inclusive=global=experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/inclusive_predictions.json \
  --queue-csv experiments/track2_final_push_20260603/clean_inclusive_disagreement/track2_clean_inclusive_disagreement_decisions.csv \
  --image-zip data/raw/Track2_testset.zip \
  --out-dir experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1
```

Expected:

```text
"row_count": 56
```

This first run intentionally queues the existing clean/inclusive disagreement review rows and still only writes dry-run artifacts.

- [ ] **Step 2: Verify no submission files were created in the output directory**

Run:

```bash
test ! -e experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/submission.json
test ! -e experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/submission.zip
```

Expected: both commands exit with status `0`.

- [ ] **Step 3: Open the HTML review in the in-app browser**

Open:

```text
http://127.0.0.1:8772/track2_moe_specialist_dry_run_review.html
```

If the current `8772` server is still rooted at the clean/inclusive review directory, start a separate static server:

```bash
cd experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/html_review
python3 -m http.server 8773 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8773/track2_moe_specialist_dry_run_review.html
```

- [ ] **Step 4: Run Track2-only tests**

Run:

```bash
python3 -m unittest tests.test_track2_moe_specialist_ensemble tests.test_track2_disagreement_review tests.test_track2_final_gate -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Run whitespace validation**

Run:

```bash
git diff --check
```

Expected: no output and exit status `0`.

- [ ] **Step 6: Commit Task 6 artifacts if they are small and useful**

If the dry-run JSON/Markdown/HTML are small enough for review, commit the script outputs. If copied image assets are large, commit only JSON/Markdown and leave assets untracked.

Run one of these:

```bash
git add experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/track2_moe_specialist_dry_run_report.json \
  experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/track2_moe_specialist_dry_run_report.md \
  experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/html_review/track2_moe_specialist_dry_run_review.html
git commit -m "test: record track2 moe dry-run review"
```

or, if the output is too large:

```bash
git status --short experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1
```

Expected: report the uncommitted output paths in the handoff.

## Task 7: Handoff Decision

**Files:**
- Read: `experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/track2_moe_specialist_dry_run_report.md`
- Read: `experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/track2_moe_specialist_dry_run_report.json`

- [ ] **Step 1: Summarize dry-run decision quality**

Report:

```text
accept_change count
hold count
keep_current count
top accepted transitions
high-similarity holds
whether any proposed change would need description rewrite
whether this is safe enough to create a side-path candidate in a new plan
```

- [ ] **Step 2: Recommend next plan boundary**

Use this decision rule:

```text
If accept_change rows are few, visually coherent, and not dominated by calm/content overcorrection, write a separate candidate-generation plan.
If accept_change rows are numerous or mostly low-confidence boundary cases, tune dry-run thresholds and do not generate a candidate.
If dry-run reveals text-emotion contradiction risk, run Description Score audit/rewrite before candidate generation.
```

Expected: no candidate path is generated in this plan.
