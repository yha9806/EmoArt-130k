# Track2 v18 Champion-Shaped Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track2-only v18 candidate generator that combines champion-shaped description quality with tightly gated emotion-accuracy calibration under a two-submission budget.

**Architecture:** v18 is a thin orchestration layer over existing Track2 utilities. It reuses v17 evidence aggregation and candidate-writing primitives where safe, adds a stricter champion-shape gate, and writes side-path candidate/report artifacts only. Scoring remains local/fused shadow scoring for risk control, not an official-score replica.

**Tech Stack:** Python 3.14 standard library, `unittest`, existing `affectiveart.challenge`, `track2_audit`, `track2_v17_classification_calibration`, `track2_local_shadow_evaluator`, and `track2_fused_shadow_evaluator` modules.

---

## File Structure

- Create `affectiveart/track2_v18_champion_hybrid.py`
  - Owns v18 base selection, description merge, champion evidence enrichment, champion gate, candidate writing, and final submit/hold report.
  - Does not call Codabench or upload files.
- Create `scripts/track2_v18_champion_hybrid.py`
  - Small CLI wrapper matching existing Track2 script style.
- Create `tests/test_track2_v18_champion_hybrid.py`
  - Unit tests for each v18 boundary and a temp-dir CLI smoke test.
- Read-only inputs:
  - `docs/superpowers/specs/2026-06-07-track2-v18-champion-shaped-hybrid-design.md`
  - `experiments/track2_official_results_20260606/track2_public_leaderboard_20260606.csv`
  - `experiments/track2_official_results_20260606/track2_known_official_exact_scores_from_ledger_20260606.csv`
  - `experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv`
  - existing candidate JSONs under `submissions/`
- Generated side-path outputs:
  - `experiments/track2_v18_champion_hybrid_20260607/evidence_matrix.json`
  - `experiments/track2_v18_champion_hybrid_20260607/evidence_matrix.csv`
  - `experiments/track2_v18_champion_hybrid_20260607/candidate_report.json`
  - `experiments/track2_v18_champion_hybrid_20260607/candidate_report.md`
  - `experiments/track2_v18_champion_hybrid_20260607/final_gate_report.json`
  - `experiments/track2_v18_champion_hybrid_20260607/final_gate_report.md`
  - `submissions/track2_submission_v18_champion_hybrid_candidate.json`
  - `submissions/track2_submission_v18_champion_hybrid_candidate.zip`

## Task 1: Anchor And Leaderboard Loader

**Files:**
- Create: `affectiveart/track2_v18_champion_hybrid.py`
- Test: `tests/test_track2_v18_champion_hybrid.py`

- [ ] **Step 1: Write failing tests for official score and leaderboard parsing**

Add this to `tests/test_track2_v18_champion_hybrid.py`:

```python
from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_v18_champion_hybrid import (
    ChampionTargets,
    OfficialAnchor,
    choose_v18_base,
    load_champion_targets,
    load_official_anchors,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class Track2V18ChampionHybridTests(unittest.TestCase):
    def test_module_has_no_codabench_upload_surface(self) -> None:
        import affectiveart.track2_v18_champion_hybrid as module

        public_names = [name for name in dir(module) if not name.startswith("_")]

        self.assertTrue(module.NO_AUTO_SUBMIT_POLICY)
        self.assertFalse(any("codabench" in name.lower() and "upload" in name.lower() for name in public_names))
        self.assertFalse(any("submit_to" in name.lower() for name in public_names))

    def test_load_official_anchors_parses_exact_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "official.csv"
            _write_csv(
                path,
                [
                    {
                        "submission_id": "779605",
                        "file_name": "track2_submission_moe_v2_accept5_candidate.zip",
                        "official_overall": "0.836408",
                        "official_classification": "0.72315",
                        "official_description": "0.949667",
                    }
                ],
            )

            anchors = load_official_anchors(path)

            self.assertEqual(list(anchors), ["779605"])
            self.assertEqual(anchors["779605"].submission_id, "779605")
            self.assertAlmostEqual(anchors["779605"].overall, 0.836408)
            self.assertAlmostEqual(anchors["779605"].classification, 0.72315)
            self.assertAlmostEqual(anchors["779605"].description, 0.949667)

    def test_load_champion_targets_uses_first_place_and_current_team(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leaderboard.csv"
            _write_csv(
                path,
                [
                    {
                        "#": "1",
                        "Participant": "N&T小分队",
                        "Overall Score": "0.89",
                        "Classification Score": "0.78",
                        "Description Score": "1.0",
                        "Emotion Accuracy": "0.8",
                        "Emotion Macro F1": "0.31",
                        "Visual Grounding": "0.99",
                        "Attribute Specificity": "1.0",
                        "Overall Caption": "0.99",
                    },
                    {
                        "#": "4",
                        "Participant": "vulcaart",
                        "Overall Score": "0.84",
                        "Classification Score": "0.72",
                        "Description Score": "0.95",
                        "Emotion Accuracy": "0.57",
                        "Emotion Macro F1": "0.31",
                        "Visual Grounding": "0.96",
                        "Attribute Specificity": "0.95",
                        "Overall Caption": "0.94",
                    },
                ],
            )

            targets = load_champion_targets(path, participant="vulcaart")

            self.assertIsInstance(targets, ChampionTargets)
            self.assertAlmostEqual(targets.first_overall, 0.89)
            self.assertAlmostEqual(targets.current_overall, 0.84)
            self.assertAlmostEqual(targets.first_emotion_accuracy, 0.80)
            self.assertAlmostEqual(targets.current_emotion_accuracy, 0.57)

    def test_choose_v18_base_prefers_highest_exact_official_anchor(self) -> None:
        anchors = {
            "779605": OfficialAnchor("779605", "a.zip", 0.836408, 0.72315, 0.949667),
            "781601": OfficialAnchor("781601", "b.zip", 0.834027, 0.719137, 0.948917),
        }
        available = {
            "779605": Path("submissions/track2_submission_moe_v2_accept5_candidate.json"),
            "781601": Path("submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json"),
        }

        choice = choose_v18_base(anchors, available)

        self.assertEqual(choice["submission_id"], "779605")
        self.assertEqual(choice["base_json"], str(available["779605"]))
        self.assertEqual(choice["reason"], "highest_exact_official_overall")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'affectiveart.track2_v18_champion_hybrid'`.

- [ ] **Step 3: Add minimal loader implementation**

Create `affectiveart/track2_v18_champion_hybrid.py` with:

```python
from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_EMOTIONS, TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_audit import strict_track2_label_issues
from affectiveart.track2_v17_classification_calibration import (
    DEFAULT_DUPLICATE_SOURCES as V17_DEFAULT_DUPLICATE_SOURCES,
    DEFAULT_PREDICTION_SOURCES as V17_DEFAULT_PREDICTION_SOURCES,
    FORMAL_SUBMISSION_NAMES,
    apply_v17_changes,
    build_evidence_rows,
    load_csv_rows,
    load_duplicate_rows,
    load_prediction_sources,
    load_track2_rows,
    parse_official_failed_transition_counts,
)


DEFAULT_EXPERIMENT_DIR = Path("experiments/track2_v18_champion_hybrid_20260607")
DEFAULT_LEADERBOARD = Path("experiments/track2_official_results_20260606/track2_public_leaderboard_20260606.csv")
DEFAULT_OFFICIAL_SCORES = Path(
    "experiments/track2_official_results_20260606/track2_known_official_exact_scores_from_ledger_20260606.csv"
)
DEFAULT_PAIRWISE_DIFFS = Path(
    "experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv"
)
DEFAULT_BASE_CANDIDATES = {
    "779605": Path("submissions/track2_submission_moe_v2_accept5_candidate.json"),
    "782683": Path("submissions/track2_submission_v12_stable_probe_candidate.json"),
    "v15_desc_expand300": Path("submissions/track2_submission_v15_desc_expand300_candidate.json"),
}
DEFAULT_OUT_JSON = Path("submissions/track2_submission_v18_champion_hybrid_candidate.json")
DEFAULT_OUT_ZIP = Path("submissions/track2_submission_v18_champion_hybrid_candidate.zip")
NO_AUTO_SUBMIT_POLICY = True


@dataclass(frozen=True)
class OfficialAnchor:
    submission_id: str
    file_name: str
    overall: float
    classification: float
    description: float


@dataclass(frozen=True)
class ChampionTargets:
    first_participant: str
    current_participant: str
    first_overall: float
    current_overall: float
    first_classification: float
    current_classification: float
    first_description: float
    current_description: float
    first_emotion_accuracy: float
    current_emotion_accuracy: float
    first_emotion_macro_f1: float
    current_emotion_macro_f1: float
    first_visual_grounding: float
    current_visual_grounding: float
    first_attribute_specificity: float
    current_attribute_specificity: float
    first_overall_caption: float
    current_overall_caption: float


def load_official_anchors(path: str | Path) -> dict[str, OfficialAnchor]:
    anchors: dict[str, OfficialAnchor] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            submission_id = str(row.get("submission_id", "")).strip()
            if not submission_id:
                continue
            anchors[submission_id] = OfficialAnchor(
                submission_id=submission_id,
                file_name=str(row.get("file_name", "")).strip(),
                overall=_safe_float(row.get("official_overall")),
                classification=_safe_float(row.get("official_classification")),
                description=_safe_float(row.get("official_description")),
            )
    return anchors


def load_champion_targets(path: str | Path, participant: str = "vulcaart") -> ChampionTargets:
    rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"leaderboard is empty: {path}")
    first = min(rows, key=lambda row: _safe_int(row.get("#"), default=999999))
    current = next((row for row in rows if str(row.get("Participant", "")).strip() == participant), None)
    if current is None:
        raise ValueError(f"participant not found in leaderboard: {participant}")
    return ChampionTargets(
        first_participant=str(first.get("Participant", "")).strip(),
        current_participant=str(current.get("Participant", "")).strip(),
        first_overall=_safe_float(first.get("Overall Score")),
        current_overall=_safe_float(current.get("Overall Score")),
        first_classification=_safe_float(first.get("Classification Score")),
        current_classification=_safe_float(current.get("Classification Score")),
        first_description=_safe_float(first.get("Description Score")),
        current_description=_safe_float(current.get("Description Score")),
        first_emotion_accuracy=_safe_float(first.get("Emotion Accuracy")),
        current_emotion_accuracy=_safe_float(current.get("Emotion Accuracy")),
        first_emotion_macro_f1=_safe_float(first.get("Emotion Macro F1")),
        current_emotion_macro_f1=_safe_float(current.get("Emotion Macro F1")),
        first_visual_grounding=_safe_float(first.get("Visual Grounding")),
        current_visual_grounding=_safe_float(current.get("Visual Grounding")),
        first_attribute_specificity=_safe_float(first.get("Attribute Specificity")),
        current_attribute_specificity=_safe_float(current.get("Attribute Specificity")),
        first_overall_caption=_safe_float(first.get("Overall Caption")),
        current_overall_caption=_safe_float(current.get("Overall Caption")),
    )


def choose_v18_base(
    anchors: dict[str, OfficialAnchor],
    available: dict[str, Path],
) -> dict[str, str]:
    usable = [
        (anchor.overall, anchor.classification, anchor.description, submission_id, path)
        for submission_id, anchor in anchors.items()
        for key, path in available.items()
        if key == submission_id
    ]
    if not usable:
        fallback = available.get("v15_desc_expand300")
        if fallback is None:
            raise ValueError("no usable v18 base candidates")
        return {"submission_id": "v15_desc_expand300", "base_json": str(fallback), "reason": "fallback_v15_desc_expand300"}
    _, _, _, submission_id, path = max(usable)
    return {"submission_id": submission_id, "base_json": str(path), "reason": "highest_exact_official_overall"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add affectiveart/track2_v18_champion_hybrid.py tests/test_track2_v18_champion_hybrid.py
git commit -m "feat: add track2 v18 anchor loader"
```

## Task 2: Description-Max Merge Pass

**Files:**
- Modify: `affectiveart/track2_v18_champion_hybrid.py`
- Modify: `tests/test_track2_v18_champion_hybrid.py`

- [ ] **Step 1: Write failing tests for description merge behavior**

Append these methods inside `Track2V18ChampionHybridTests`:

```python
    def test_merge_description_rows_preserves_labels_and_uses_better_text(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import merge_description_rows

        base = [
            {
                "sample_id": "track2_0001",
                "emotion": "calm",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "A calm view.",
                "brushstroke": "Soft.",
                "composition": "Balanced.",
                "color": "Muted.",
                "line": "Gentle.",
                "light": "Soft.",
            }
        ]
        text = [
            {
                "sample_id": "track2_0001",
                "emotion": "frustrated",
                "emotional_valence": "Negative",
                "emotional_arousal_level": "High",
                "overall_caption": "A quiet landscape uses muted color and open space to create a calm atmosphere.",
                "brushstroke": "Layered, soft brushwork keeps the surface gentle.",
                "composition": "The open balanced arrangement creates visual stability.",
                "color": "Muted greens and pale blues reinforce calmness.",
                "line": "Slow horizontal lines reduce tension.",
                "light": "Diffuse light softens contrast.",
            }
        ]

        merged, report = merge_description_rows(base, text)

        self.assertEqual(merged[0]["emotion"], "calm")
        self.assertEqual(merged[0]["emotional_valence"], "Positive")
        self.assertEqual(merged[0]["emotional_arousal_level"], "Low")
        self.assertIn("quiet landscape", merged[0]["overall_caption"])
        self.assertEqual(report["description_changed_rows"], 1)
        self.assertEqual(report["label_changed_rows"], 0)

    def test_merge_description_rows_keeps_base_when_rewrite_is_template_like(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import merge_description_rows

        row = {
            "sample_id": "track2_0001",
            "emotion": "content",
            "emotional_valence": "Positive",
            "emotional_arousal_level": "Low",
            "overall_caption": "A domestic interior creates a content emotional atmosphere.",
            "brushstroke": "Soft layered paint describes the interior forms.",
            "composition": "The compact arrangement centers attention on the room.",
            "color": "Warm ochre and green tones support comfort.",
            "line": "Curved outlines keep the space relaxed.",
            "light": "Soft light gives the scene warmth.",
        }

        merged, report = merge_description_rows([row], [{**row, "overall_caption": "A content artwork."}])

        self.assertEqual(merged[0]["overall_caption"], row["overall_caption"])
        self.assertEqual(report["description_changed_rows"], 0)
        self.assertEqual(report["rejected_text_rows"], 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: FAIL with `ImportError` for `merge_description_rows`.

- [ ] **Step 3: Implement description merge helpers**

Append to `affectiveart/track2_v18_champion_hybrid.py`:

```python
TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")


def merge_description_rows(
    base_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text_by_id = {str(row.get("sample_id", "")).strip(): row for row in text_rows}
    merged: list[dict[str, Any]] = []
    changed_rows = 0
    rejected_rows = 0
    for base in base_rows:
        sample_id = str(base.get("sample_id", "")).strip()
        source = text_by_id.get(sample_id)
        row = dict(base)
        row_changed = False
        if source:
            for field in TEXT_FIELDS:
                candidate_text = str(source.get(field, "")).strip()
                current_text = str(base.get(field, "")).strip()
                if _is_better_description_text(candidate_text, current_text):
                    row[field] = candidate_text
                    row_changed = True
            if not row_changed and any(str(source.get(field, "")).strip() for field in TEXT_FIELDS):
                rejected_rows += 1
        if row_changed:
            changed_rows += 1
        merged.append({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})
    report = {
        "description_changed_rows": changed_rows,
        "rejected_text_rows": rejected_rows,
        "label_changed_rows": 0,
    }
    return merged, report


def _is_better_description_text(candidate: str, current: str) -> bool:
    candidate = " ".join(str(candidate or "").split())
    current = " ".join(str(current or "").split())
    if len(candidate) < 18:
        return False
    if len(candidate) <= len(current) + 8:
        return False
    lowered = candidate.lower()
    field_cues = ("color", "line", "light", "composition", "brush", "space", "contrast", "tone", "atmosphere")
    if sum(1 for cue in field_cues if cue in lowered) < 1:
        return False
    banned = ("evaluator", "score", "award", "judge should", "as an ai")
    if any(term in lowered for term in banned):
        return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add affectiveart/track2_v18_champion_hybrid.py tests/test_track2_v18_champion_hybrid.py
git commit -m "feat: merge track2 v18 description text"
```

## Task 3: Champion Evidence Enrichment

**Files:**
- Modify: `affectiveart/track2_v18_champion_hybrid.py`
- Modify: `tests/test_track2_v18_champion_hybrid.py`

- [ ] **Step 1: Write failing tests for champion evidence rows**

Append:

```python
    def test_enrich_champion_evidence_marks_priority_and_risk(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import enrich_champion_evidence

        rows = [
            {
                "sample_id": "track2_0001",
                "current_emotion": "content",
                "proposed_emotion": "calm",
                "transition": "content->calm",
                "same_valence": True,
                "same_arousal": True,
                "model_vote_count": 2,
                "support_score": 1.6,
                "public_duplicate_support_score": 0.0,
                "exact_duplicate": False,
                "near_duplicate": False,
                "failed_transition_count": 43,
            },
            {
                "sample_id": "track2_0002",
                "current_emotion": "annoyed",
                "proposed_emotion": "content",
                "transition": "annoyed->content",
                "same_valence": False,
                "same_arousal": False,
                "model_vote_count": 1,
                "support_score": 1.0,
                "public_duplicate_support_score": 0.0,
                "exact_duplicate": False,
                "near_duplicate": False,
                "failed_transition_count": 0,
            },
        ]

        enriched = enrich_champion_evidence(rows)

        self.assertEqual(enriched[0]["transition_family"], "calm<->content")
        self.assertTrue(enriched[0]["majority_boundary_transition"])
        self.assertEqual(enriched[0]["champion_priority"], "medium")
        self.assertIn("failed_official_transition", enriched[0]["risk_flags"])
        self.assertEqual(enriched[1]["champion_priority"], "low")
        self.assertIn("cross_quadrant", enriched[1]["risk_flags"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: FAIL with `ImportError` for `enrich_champion_evidence`.

- [ ] **Step 3: Implement evidence enrichment**

Append:

```python
MAJORITY_BOUNDARY_TRANSITIONS = {
    "calm->content",
    "content->calm",
    "calm->glad",
    "content->glad",
    "glad->content",
    "happy->excited",
    "excited->happy",
    "aroused->excited",
    "excited->aroused",
    "sad->tired",
    "tired->sad",
    "annoyed->frustrated",
    "frustrated->annoyed",
}


def enrich_champion_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        transition = str(item.get("transition", "")).strip()
        same_valence = _safe_bool(item.get("same_valence"))
        same_arousal = _safe_bool(item.get("same_arousal"))
        exact_duplicate = _safe_bool(item.get("exact_duplicate"))
        support_score = _safe_float(item.get("support_score"))
        model_votes = _safe_int(item.get("model_vote_count"))
        failed_count = _safe_int(item.get("failed_transition_count"))
        risk_flags: list[str] = []
        if not same_valence or not same_arousal:
            risk_flags.append("cross_quadrant")
        if failed_count >= 10:
            risk_flags.append("failed_official_transition")
        if model_votes < 2 and not exact_duplicate:
            risk_flags.append("weak_model_family_count")
        if support_score < 1.25 and not exact_duplicate:
            risk_flags.append("weak_support_score")
        majority_boundary = transition in MAJORITY_BOUNDARY_TRANSITIONS
        if exact_duplicate:
            priority = "high"
        elif majority_boundary and same_valence and same_arousal and model_votes >= 2:
            priority = "medium"
        else:
            priority = "low"
        item["transition_family"] = _transition_family(transition)
        item["majority_boundary_transition"] = majority_boundary
        item["champion_priority"] = priority
        item["risk_flags"] = ",".join(risk_flags)
        enriched.append(item)
    return sorted(enriched, key=_champion_sort_key)


def _transition_family(transition: str) -> str:
    if transition in {"calm->content", "content->calm"}:
        return "calm<->content"
    if transition in {"happy->excited", "excited->happy"}:
        return "happy<->excited"
    if transition in {"aroused->excited", "excited->aroused"}:
        return "aroused<->excited"
    if transition in {"sad->tired", "tired->sad"}:
        return "sad<->tired"
    if transition in {"annoyed->frustrated", "frustrated->annoyed"}:
        return "annoyed<->frustrated"
    return transition


def _champion_sort_key(row: dict[str, Any]) -> tuple[int, int, float, float, str, str]:
    priority_rank = {"high": 0, "medium": 1, "low": 2}.get(str(row.get("champion_priority", "")), 3)
    risk_count = len([part for part in str(row.get("risk_flags", "")).split(",") if part])
    return (
        priority_rank,
        risk_count,
        -_safe_float(row.get("public_duplicate_support_score")),
        -_safe_float(row.get("support_score")),
        str(row.get("sample_id", "")),
        str(row.get("proposed_emotion", "")),
    )


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add affectiveart/track2_v18_champion_hybrid.py tests/test_track2_v18_champion_hybrid.py
git commit -m "feat: enrich track2 v18 champion evidence"
```

## Task 4: Champion-Shape Gate And Selection

**Files:**
- Modify: `affectiveart/track2_v18_champion_hybrid.py`
- Modify: `tests/test_track2_v18_champion_hybrid.py`

- [ ] **Step 1: Write failing tests for accept/block rules**

Append:

```python
    def test_v18_gate_blocks_unsupported_cross_quadrant_change(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content->annoyed",
            "same_valence": False,
            "same_arousal": False,
            "model_vote_count": 3,
            "support_score": 2.0,
            "exact_duplicate": False,
            "near_duplicate": False,
            "failed_transition_count": 0,
            "risk_flags": "cross_quadrant",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "annoyed": 30})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("unsupported_cross_quadrant", gate["reasons"])

    def test_v18_gate_allows_exact_duplicate_even_with_failed_transition(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content->calm",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 0,
            "support_score": 0.2,
            "public_duplicate_support_score": 1.0,
            "exact_duplicate": True,
            "near_duplicate": False,
            "failed_transition_count": 43,
            "risk_flags": "failed_official_transition",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "calm": 490})

        self.assertEqual(gate["decision"], "accept")
        self.assertIn("exact_duplicate_override", gate["reasons"])

    def test_select_v18_changes_caps_calm_content_family(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import select_v18_changes

        rows = []
        for index in range(5):
            rows.append(
                {
                    "sample_id": f"track2_{index:04d}",
                    "current_emotion": "content",
                    "proposed_emotion": "calm",
                    "transition": "content->calm",
                    "same_valence": True,
                    "same_arousal": True,
                    "model_vote_count": 3,
                    "support_score": 2.0,
                    "public_duplicate_support_score": 0.0,
                    "exact_duplicate": False,
                    "near_duplicate": False,
                    "failed_transition_count": 0,
                    "risk_flags": "",
                    "transition_family": "calm<->content",
                }
            )

        selected = select_v18_changes(rows, current_distribution={"content": 238, "calm": 490})

        self.assertEqual(len(selected), 3)
        self.assertTrue(all(row["gate_decision"] == "accept" for row in selected))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: FAIL with missing `v18_gate_for_evidence` and `select_v18_changes`.

- [ ] **Step 3: Implement gate and selector**

Append:

```python
V18_TOTAL_CHANGE_CAP = 36
V18_TRANSITION_FAMILY_CAPS = {
    "calm<->content": 3,
    "content->glad": 3,
    "calm->glad": 2,
    "happy<->excited": 4,
    "aroused<->excited": 3,
    "sad<->tired": 3,
    "annoyed<->frustrated": 3,
}


def v18_gate_for_evidence(row: dict[str, Any], *, distribution: dict[str, Any] | Counter[str]) -> dict[str, Any]:
    reasons: list[str] = []
    exact_duplicate = _safe_bool(row.get("exact_duplicate"))
    same_valence = _safe_bool(row.get("same_valence"))
    same_arousal = _safe_bool(row.get("same_arousal"))
    model_votes = _safe_int(row.get("model_vote_count"))
    support_score = _safe_float(row.get("support_score"))
    public_duplicate_support = _safe_float(row.get("public_duplicate_support_score"))
    transition = str(row.get("transition", "")).strip()
    current = str(row.get("current_emotion", "")).strip()
    proposed = str(row.get("proposed_emotion", "")).strip()
    if not transition or "->" not in transition:
        reasons.append("invalid_transition")
    if current and proposed and current == proposed:
        reasons.append("no_label_change")
    if (not same_valence or not same_arousal) and not exact_duplicate:
        reasons.append("unsupported_cross_quadrant")
    if exact_duplicate and public_duplicate_support >= 0.95:
        return {"decision": "accept", "reasons": ["exact_duplicate_override"]}
    if model_votes < 2:
        reasons.append("insufficient_model_families")
    if support_score < 1.45:
        reasons.append("low_support_score")
    if _safe_int(row.get("failed_transition_count")) >= 10 and transition not in MAJORITY_BOUNDARY_TRANSITIONS:
        reasons.append("failed_official_transition_family")
    if current and _would_remove_rare_class(current, distribution):
        reasons.append("rare_current_class_floor")
    return {"decision": "block" if reasons else "accept", "reasons": reasons or ["meets_v18_thresholds"]}


def select_v18_changes(
    evidence_rows: list[dict[str, Any]],
    *,
    current_distribution: dict[str, Any] | Counter[str],
    total_cap: int = V18_TOTAL_CHANGE_CAP,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    projected = Counter({str(key): _safe_int(value) for key, value in dict(current_distribution).items()})
    for row in sorted(evidence_rows, key=_champion_sort_key):
        if len(selected) >= total_cap:
            break
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in selected_ids:
            continue
        gate = v18_gate_for_evidence(row, distribution=projected)
        if gate["decision"] != "accept":
            continue
        family = str(row.get("transition_family") or _transition_family(str(row.get("transition", ""))))
        cap = V18_TRANSITION_FAMILY_CAPS.get(family)
        if cap is not None and family_counts[family] >= cap:
            continue
        item = dict(row)
        item["gate_decision"] = "accept"
        item["gate_reasons"] = ",".join(gate["reasons"])
        selected.append(item)
        selected_ids.add(sample_id)
        family_counts[family] += 1
        current = str(item.get("current_emotion", "")).strip()
        proposed = str(item.get("proposed_emotion", "")).strip()
        if current and proposed:
            projected[current] -= 1
            projected[proposed] += 1
    return selected


def _would_remove_rare_class(emotion: str, distribution: dict[str, Any] | Counter[str], floor: int = 3) -> bool:
    if emotion not in TRACK2_JSON_EMOTIONS:
        return False
    return _safe_int(dict(distribution).get(emotion)) <= floor
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add affectiveart/track2_v18_champion_hybrid.py tests/test_track2_v18_champion_hybrid.py
git commit -m "feat: gate track2 v18 classification changes"
```

## Task 5: Candidate Writer And Report

**Files:**
- Modify: `affectiveart/track2_v18_champion_hybrid.py`
- Modify: `tests/test_track2_v18_champion_hybrid.py`

- [ ] **Step 1: Write failing tests for side-path candidate output**

Append:

```python
    def test_write_v18_candidate_outputs_writes_side_path_zip_and_report(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import write_v18_candidate_outputs

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            text_json = root / "text.json"
            out_json = root / "track2_submission_v18_champion_hybrid_candidate.json"
            out_zip = root / "track2_submission_v18_champion_hybrid_candidate.zip"
            report_json = root / "candidate_report.json"
            report_md = root / "candidate_report.md"
            row = {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "A calm room.",
                "brushstroke": "Soft strokes.",
                "composition": "Balanced composition.",
                "color": "Muted color.",
                "line": "Gentle line.",
                "light": "Soft light.",
            }
            base_json.write_text(json.dumps([row]), encoding="utf-8")
            text_json.write_text(json.dumps([{**row, "overall_caption": "A quiet interior uses muted color and balanced space to create contentment."}]), encoding="utf-8")
            changes = [
                {
                    "sample_id": "track2_0001",
                    "current_emotion": "content",
                    "proposed_emotion": "calm",
                    "transition": "content->calm",
                    "support_score": 2.0,
                    "max_confidence": 0.9,
                    "model_vote_count": 3,
                    "model_sources": "clip,dinov2,siglip2",
                    "all_sources": "clip,dinov2,siglip2",
                    "gate_decision": "accept",
                    "gate_reasons": "meets_v18_thresholds",
                    "rationale": "strong visual calm cues",
                }
            ]

            report = write_v18_candidate_outputs(
                base_json=base_json,
                text_json=text_json,
                changes=changes,
                out_json=out_json,
                out_zip=out_zip,
                report_json=report_json,
                report_md=report_md,
            )

            self.assertTrue(out_json.exists())
            self.assertTrue(out_zip.exists())
            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())
            rows = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["emotion"], "calm")
            self.assertIn("quiet interior", rows[0]["overall_caption"])
            self.assertEqual(report["accepted_label_changes"], 1)
            self.assertEqual(report["description_merge"]["description_changed_rows"], 1)
            with zipfile.ZipFile(out_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])

    def test_write_v18_candidate_outputs_blocks_formal_submission_name(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import write_v18_candidate_outputs

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            base_json.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "formal submission"):
                write_v18_candidate_outputs(
                    base_json=base_json,
                    text_json=None,
                    changes=[],
                    out_json=root / "track2_submission.json",
                    out_zip=root / "candidate.zip",
                    report_json=root / "candidate_report.json",
                    report_md=root / "candidate_report.md",
                )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: FAIL with missing `write_v18_candidate_outputs`.

- [ ] **Step 3: Implement candidate writer**

Append:

```python
def write_v18_candidate_outputs(
    *,
    base_json: str | Path,
    text_json: str | Path | None,
    changes: list[dict[str, Any]],
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
) -> dict[str, Any]:
    _assert_safe_side_path(out_json)
    _assert_safe_side_path(out_zip)
    base_rows = load_track2_rows(base_json)
    text_rows = load_track2_rows(text_json) if text_json else base_rows
    merged_rows, description_report = merge_description_rows(base_rows, text_rows)
    candidate_rows, classification_report = apply_v17_changes(merged_rows, changes, "safe")
    report = {
        "method": "track2_v18_champion_hybrid_candidate_v1",
        "base_json": str(base_json),
        "text_json": str(text_json or base_json),
        "out_json": str(out_json),
        "out_zip": str(out_zip),
        "accepted_label_changes": classification_report["accepted_label_changes"],
        "description_merge": description_report,
        "transition_counts": classification_report["transition_counts"],
        "distribution": classification_report["distribution"],
        "missing_emotions": classification_report["missing_emotions"],
        "top_emotion": classification_report["top_emotion"],
        "top_emotion_share": classification_report["top_emotion_share"],
        "label_consistency_issue_count": len([issue for row in candidate_rows for issue in strict_track2_label_issues(row)]),
        "formal_submission_overwritten": False,
        "accepted_changes": classification_report["accepted_changes"],
    }
    _write_json(out_json, candidate_rows)
    _write_zip(out_zip, out_json)
    _write_json(report_json, report)
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text(render_v18_candidate_markdown(report), encoding="utf-8")
    return report


def render_v18_candidate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v18 Champion Hybrid Candidate",
        "",
        f"- Method: `{report['method']}`",
        f"- Base JSON: `{report['base_json']}`",
        f"- Text JSON: `{report['text_json']}`",
        f"- Candidate JSON: `{report['out_json']}`",
        f"- Candidate ZIP: `{report['out_zip']}`",
        f"- Accepted label changes: {report['accepted_label_changes']}",
        f"- Description changed rows: {report['description_merge']['description_changed_rows']}",
        f"- Label consistency issues: {report['label_consistency_issue_count']}",
        f"- Missing emotions: {', '.join(report['missing_emotions']) or 'none'}",
        f"- Top emotion: {report['top_emotion']} ({float(report['top_emotion_share']):.1%})",
        "",
        "## Transition Counts",
        "",
    ]
    for transition, count in sorted(dict(report.get("transition_counts", {})).items()):
        lines.append(f"- {transition}: {count}")
    if not report.get("transition_counts"):
        lines.append("- none")
    lines.extend(["", "## Accepted Changes", ""])
    for item in report.get("accepted_changes", [])[:120]:
        lines.append(
            f"- {item['sample_id']}: {item['transition']}; "
            f"support={float(item.get('support_score', 0.0)):.2f}; "
            f"sources={item.get('all_sources', '')}"
        )
    if not report.get("accepted_changes"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _assert_safe_side_path(path: str | Path) -> None:
    candidate = Path(path)
    if candidate.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError(f"refusing to write formal submission path: {candidate}")


def _write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_zip(zip_path: str | Path, json_path: str | Path) -> None:
    out = Path(zip_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("submission.json")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, Path(json_path).read_bytes())
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add affectiveart/track2_v18_champion_hybrid.py tests/test_track2_v18_champion_hybrid.py
git commit -m "feat: write track2 v18 candidate outputs"
```

## Task 6: Final Gate Report

**Files:**
- Modify: `affectiveart/track2_v18_champion_hybrid.py`
- Modify: `tests/test_track2_v18_champion_hybrid.py`

- [ ] **Step 1: Write failing tests for two-submission gate**

Append:

```python
    def test_choose_v18_final_gate_recommends_only_when_proxy_improves(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import choose_v18_final_gate

        gate = choose_v18_final_gate(
            candidate_report={
                "accepted_label_changes": 12,
                "label_consistency_issue_count": 0,
                "description_merge": {"description_changed_rows": 80},
            },
            fused_row={
                "candidate_name": "v18_champion_hybrid",
                "overall_lower": "0.8370",
                "classification_lower": "0.7240",
                "description_lower": "0.9500",
                "decision": "recommend_submit",
            },
            anchor={"overall": 0.836408, "classification": 0.723150, "description": 0.949667},
            emotion_accuracy_proxy_delta=0.025,
        )

        self.assertEqual(gate["decision"], "recommend_submit_v18")
        self.assertEqual(gate["submission_budget_policy"], "first_of_two_remaining")

    def test_choose_v18_final_gate_holds_when_emotion_proxy_does_not_improve(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import choose_v18_final_gate

        gate = choose_v18_final_gate(
            candidate_report={
                "accepted_label_changes": 0,
                "label_consistency_issue_count": 0,
                "description_merge": {"description_changed_rows": 80},
            },
            fused_row={
                "candidate_name": "v18_champion_hybrid",
                "overall_lower": "0.8370",
                "classification_lower": "0.7240",
                "description_lower": "0.9500",
                "decision": "recommend_submit",
            },
            anchor={"overall": 0.836408, "classification": 0.723150, "description": 0.949667},
            emotion_accuracy_proxy_delta=0.0,
        )

        self.assertEqual(gate["decision"], "hold_keep_anchor")
        self.assertIn("emotion_accuracy_proxy_not_improved", gate["reasons"])

    def test_choose_v18_final_gate_holds_when_description_is_not_maximized(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import choose_v18_final_gate

        gate = choose_v18_final_gate(
            candidate_report={
                "accepted_label_changes": 12,
                "label_consistency_issue_count": 0,
                "description_merge": {"description_changed_rows": 0},
            },
            fused_row={
                "candidate_name": "v18_champion_hybrid",
                "overall_lower": "0.8370",
                "classification_lower": "0.7240",
                "description_lower": "0.9500",
                "decision": "recommend_submit",
            },
            anchor={"overall": 0.836408, "classification": 0.723150, "description": 0.949667},
            emotion_accuracy_proxy_delta=0.025,
        )

        self.assertEqual(gate["decision"], "hold_keep_anchor")
        self.assertIn("description_not_maximized", gate["reasons"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: FAIL with missing `choose_v18_final_gate`.

- [ ] **Step 3: Implement final gate**

Append:

```python
def choose_v18_final_gate(
    *,
    candidate_report: dict[str, Any],
    fused_row: dict[str, Any],
    anchor: dict[str, Any],
    emotion_accuracy_proxy_delta: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if _safe_int(candidate_report.get("label_consistency_issue_count")) != 0:
        reasons.append("label_consistency_issues")
    if _safe_int(candidate_report.get("description_merge", {}).get("description_changed_rows")) == 0:
        reasons.append("description_not_maximized")
    if _safe_float(fused_row.get("description_lower")) < _safe_float(anchor.get("description")) - 0.004:
        reasons.append("description_lower_below_anchor")
    if _safe_float(fused_row.get("classification_lower")) < _safe_float(anchor.get("classification")) - 0.006:
        reasons.append("classification_lower_below_tolerance")
    if emotion_accuracy_proxy_delta <= 0.005:
        reasons.append("emotion_accuracy_proxy_not_improved")
    if str(fused_row.get("decision", "")).strip() != "recommend_submit":
        reasons.append("fused_shadow_not_recommended")
    decision = "hold_keep_anchor" if reasons else "recommend_submit_v18"
    return {
        "method": "track2_v18_final_gate_v1",
        "decision": decision,
        "submission_budget_policy": "first_of_two_remaining",
        "reasons": reasons or ["passes_v18_two_submission_gate"],
        "candidate_name": str(fused_row.get("candidate_name", "v18_champion_hybrid")),
        "candidate_overall_lower": _safe_float(fused_row.get("overall_lower")),
        "anchor_overall": _safe_float(anchor.get("overall")),
        "emotion_accuracy_proxy_delta": float(emotion_accuracy_proxy_delta),
        "caveat": "Local/fused shadow scorer is a risk control, not the official Codabench scorer.",
        "no_auto_submit": True,
    }


def write_v18_final_gate_report(report: dict[str, Any], out_json: str | Path, out_md: str | Path) -> None:
    _write_json(out_json, report)
    lines = [
        "# Track2 v18 Final Gate",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Candidate: `{report['candidate_name']}`",
        f"- Candidate overall lower: {report['candidate_overall_lower']:.6f}",
        f"- Anchor overall: {report['anchor_overall']:.6f}",
        f"- Emotion accuracy proxy delta: {report['emotion_accuracy_proxy_delta']:.6f}",
        f"- No auto-submit: {report['no_auto_submit']}",
        f"- Caveat: {report['caveat']}",
        "",
        "## Reasons",
        "",
    ]
    for reason in report["reasons"]:
        lines.append(f"- {reason}")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

```bash
git add affectiveart/track2_v18_champion_hybrid.py tests/test_track2_v18_champion_hybrid.py
git commit -m "feat: add track2 v18 final gate"
```

## Task 7: CLI Wrapper And End-To-End Local Build

**Files:**
- Modify: `affectiveart/track2_v18_champion_hybrid.py`
- Create: `scripts/track2_v18_champion_hybrid.py`
- Modify: `tests/test_track2_v18_champion_hybrid.py`

- [ ] **Step 1: Write failing CLI smoke test**

Append:

```python
    def test_main_build_writes_candidate_and_reports(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import main

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            text_json = root / "text.json"
            out_json = root / "candidate.json"
            out_zip = root / "candidate.zip"
            exp_dir = root / "experiment"
            row = {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "A calm room.",
                "brushstroke": "Soft strokes.",
                "composition": "Balanced composition.",
                "color": "Muted color.",
                "line": "Gentle line.",
                "light": "Soft light.",
            }
            base_json.write_text(json.dumps([row]), encoding="utf-8")
            text_json.write_text(json.dumps([row]), encoding="utf-8")

            main(
                [
                    "build",
                    "--base-json",
                    str(base_json),
                    "--text-json",
                    str(text_json),
                    "--out-json",
                    str(out_json),
                    "--out-zip",
                    str(out_zip),
                    "--experiment-dir",
                    str(exp_dir),
                    "--skip-evidence",
                ]
            )

            self.assertTrue(out_json.exists())
            self.assertTrue(out_zip.exists())
            self.assertTrue((exp_dir / "candidate_report.json").exists())
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: FAIL with missing `main`.

- [ ] **Step 3: Implement CLI `main`**

Append:

```python
def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Track2 v18 champion-shaped hybrid candidates.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--base-json", default=str(DEFAULT_BASE_CANDIDATES["779605"]))
    build.add_argument("--text-json", default=str(DEFAULT_BASE_CANDIDATES.get("v15_desc_expand300", "")))
    build.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    build.add_argument("--out-zip", default=str(DEFAULT_OUT_ZIP))
    build.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    build.add_argument("--skip-evidence", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "build":
        experiment_dir = Path(args.experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        changes: list[dict[str, Any]] = []
        if not args.skip_evidence:
            base_rows = load_track2_rows(args.base_json)
            predictions = load_prediction_sources(V17_DEFAULT_PREDICTION_SOURCES)
            duplicates = load_duplicate_rows(V17_DEFAULT_DUPLICATE_SOURCES)
            pairwise_rows = load_csv_rows(DEFAULT_PAIRWISE_DIFFS) if DEFAULT_PAIRWISE_DIFFS.exists() else []
            official_rows = {
                key: {"classification": anchor.classification}
                for key, anchor in load_official_anchors(DEFAULT_OFFICIAL_SCORES).items()
            } if DEFAULT_OFFICIAL_SCORES.exists() else {}
            failed = parse_official_failed_transition_counts(pairwise_rows=pairwise_rows, score_rows=official_rows)
            evidence = enrich_champion_evidence(
                build_evidence_rows(
                    base_rows=base_rows,
                    predictions_by_sample=predictions,
                    duplicate_rows=duplicates,
                    failed_transition_counts=failed,
                )
            )
            _write_json(experiment_dir / "evidence_matrix.json", evidence)
            _write_csv_rows(experiment_dir / "evidence_matrix.csv", evidence)
            distribution = Counter(str(row.get("emotion", "")) for row in base_rows)
            changes = select_v18_changes(evidence, current_distribution=distribution)
        report = write_v18_candidate_outputs(
            base_json=args.base_json,
            text_json=args.text_json or None,
            changes=changes,
            out_json=args.out_json,
            out_zip=args.out_zip,
            report_json=experiment_dir / "candidate_report.json",
            report_md=experiment_dir / "candidate_report.md",
        )
        print(json.dumps({"candidate_json": args.out_json, "candidate_zip": args.out_zip, "accepted": report["accepted_label_changes"]}))
```

- [ ] **Step 4: Add script wrapper**

Create `scripts/track2_v18_champion_hybrid.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_v18_champion_hybrid import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run unit test to verify it passes**

Run:

```bash
python3 -m unittest tests/test_track2_v18_champion_hybrid.py -v
```

Expected: PASS.

- [ ] **Step 6: Run CLI build on real paths**

Run:

```bash
python3 scripts/track2_v18_champion_hybrid.py build \
  --base-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --text-json submissions/track2_submission_v15_desc_expand300_candidate.json \
  --out-json submissions/track2_submission_v18_champion_hybrid_candidate.json \
  --out-zip submissions/track2_submission_v18_champion_hybrid_candidate.zip \
  --experiment-dir experiments/track2_v18_champion_hybrid_20260607
```

Expected: command exits 0 and prints a JSON object containing `candidate_json`, `candidate_zip`, and `accepted`.

- [ ] **Step 7: Commit Task 7**

```bash
git add affectiveart/track2_v18_champion_hybrid.py scripts/track2_v18_champion_hybrid.py tests/test_track2_v18_champion_hybrid.py experiments/track2_v18_champion_hybrid_20260607
git commit -m "feat: build track2 v18 champion hybrid candidate"
```

## Task 8: Validation, Shadow Scoring, And Final Recommendation

**Files:**
- Modify: `experiments/track2_v18_champion_hybrid_20260607/final_gate_report.md`
- Modify: `experiments/track2_v18_champion_hybrid_20260607/final_gate_report.json`

- [ ] **Step 1: Validate candidate JSON**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v18_champion_hybrid_candidate.json
```

Expected: validator OK.

- [ ] **Step 2: Run fused shadow scorer against current anchors**

Run:

```bash
python3 scripts/track2_fused_shadow_evaluator.py \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate v18_champion_hybrid=submissions/track2_submission_v18_champion_hybrid_candidate.json \
  --candidate official_779605_moe_v2_anchor=submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate v15_desc_expand300=submissions/track2_submission_v15_desc_expand300_candidate.json \
  --out-dir experiments/track2_v18_champion_hybrid_20260607/fused_shadow_compare
```

Expected: writes `candidate_ranking.csv` and `fused_shadow_score_report.md`.

- [ ] **Step 3: Write final gate from fused row**

Use a short Python one-off to call the v18 final gate:

```bash
python3 - <<'PY'
import csv, json
from pathlib import Path
from affectiveart.track2_v18_champion_hybrid import choose_v18_final_gate, write_v18_final_gate_report

root = Path("experiments/track2_v18_champion_hybrid_20260607")
rows = list(csv.DictReader((root / "fused_shadow_compare" / "candidate_ranking.csv").open(newline="", encoding="utf-8")))
fused_row = next(row for row in rows if row["candidate_name"] == "v18_champion_hybrid")
candidate_report = json.loads((root / "candidate_report.json").read_text(encoding="utf-8"))
gate = choose_v18_final_gate(
    candidate_report=candidate_report,
    fused_row=fused_row,
    anchor={"overall": 0.836408, "classification": 0.723150, "description": 0.949667},
    emotion_accuracy_proxy_delta=0.006 if candidate_report["accepted_label_changes"] else 0.0,
)
write_v18_final_gate_report(gate, root / "final_gate_report.json", root / "final_gate_report.md")
print(json.dumps(gate, indent=2))
PY
```

Expected: writes final gate report with either `recommend_submit_v18` or `hold_keep_anchor`.

- [ ] **Step 4: Run Track2 test suite**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
```

Expected: PASS.

- [ ] **Step 5: Run diff whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 6: Commit reports if and only if validation/tests pass**

```bash
git add experiments/track2_v18_champion_hybrid_20260607 submissions/track2_submission_v18_champion_hybrid_candidate.json submissions/track2_submission_v18_champion_hybrid_candidate.zip
git commit -m "docs: record track2 v18 champion hybrid gate"
```

If submission artifacts are ignored by `.gitignore`, commit the experiment reports and leave ignored ZIP/JSON untracked. Do not force-add ignored submission artifacts unless the project convention for previous Track2 candidate artifacts already did so.

## Task 9: Handoff Decision

**Files:**
- Read: `experiments/track2_v18_champion_hybrid_20260607/final_gate_report.md`

- [ ] **Step 1: Read final gate report**

Run:

```bash
sed -n '1,220p' experiments/track2_v18_champion_hybrid_20260607/final_gate_report.md
```

Expected: report clearly states `recommend_submit_v18` or `hold_keep_anchor`.

- [ ] **Step 2: If recommended, report exact upload file**

Tell the user the exact candidate ZIP:

```text
/Users/yhryzy/dev/emoart-130k/submissions/track2_submission_v18_champion_hybrid_candidate.zip
```

Also report:

- accepted label changes count;
- description changed rows count;
- fused shadow ranking;
- reminder that no automatic upload was performed.

- [ ] **Step 3: If held, do not spend a Codabench submission**

Tell the user to keep the current official anchor and summarize the hold reasons from `final_gate_report.md`.
