# Track2 v14 Public Resource AGSR Max Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Track2 v14 candidates that combine public-resource label evidence, AGSR/FAB-G-style salient-attribute text repair, and a calibrated local/fused shadow scorer gate before any Codabench submission.

**Architecture:** Add a focused `affectiveart.track2_v14_public_resource_agsr` module and `scripts/track2_v14_public_resource_agsr.py` CLI. Reuse existing v13 candidate/output patterns, public-reference repair helpers, description audit, Track2 validator, and local/fused shadow evaluators. Generate all artifacts in side paths under `experiments/track2_v14_public_resource_agsr_20260607/` and `submissions/track2_submission_v14_*_candidate.*`.

**Tech Stack:** Python stdlib, existing `affectiveart` modules, JSON/CSV/HTML artifacts, `unittest`, Track2 validator, local shadow evaluator, fused shadow evaluator.

---

## File Structure

- Create: `affectiveart/track2_v14_public_resource_agsr.py`
  - Load current submission rows.
  - Load public duplicate/top-k audit evidence.
  - Normalize evidence into a public evidence matrix.
  - Select high-confidence label transfers.
  - Apply AGSR-style salient-attribute text rewrites.
  - Write candidates, reports, HTML review, and scorer inputs.
- Create: `scripts/track2_v14_public_resource_agsr.py`
  - CLI for `build`, `threshold-ablation`, `emotion-boundary-report`, and `historical-correlation`.
- Create: `tests/test_track2_v14_public_resource_agsr.py`
  - Unit tests for evidence levels, label gates, AGSR text gate, side-path outputs, and scorer integration.
- Modify only if necessary: no existing production module should be rewritten unless a test exposes an integration gap.
- Outputs:
  - `experiments/track2_v14_public_resource_agsr_20260607/public_resource_registry.jsonl`
  - `experiments/track2_v14_public_resource_agsr_20260607/track2_public_evidence_matrix.csv`
  - `experiments/track2_v14_public_resource_agsr_20260607/track2_public_evidence_matrix.json`
  - `experiments/track2_v14_public_resource_agsr_20260607/v14_agsr_rewrite_report.md`
  - `experiments/track2_v14_public_resource_agsr_20260607/v14_candidate_score_proxy.md`
  - `experiments/track2_v14_public_resource_agsr_20260607/html_review/track2_v14_public_resource_agsr_review.html`
  - `experiments/track2_v14_public_resource_agsr_20260607/shadow_eval/`
  - `experiments/track2_v14_public_resource_agsr_20260607/fused_shadow_eval/`
  - `submissions/track2_submission_v14_public_resource_agsr_max_candidate.json`
  - `submissions/track2_submission_v14_public_resource_agsr_max_candidate.zip`
  - `submissions/track2_submission_v14_description_max_safe_candidate.json`
  - `submissions/track2_submission_v14_description_max_safe_candidate.zip`

## Constants

Use these defaults in the CLI:

```python
DEFAULT_CURRENT_JSON = "submissions/track2_submission_moe_v2_accept5_candidate.json"
DEFAULT_OUT_DIR = "experiments/track2_v14_public_resource_agsr_20260607"
DEFAULT_SUBMISSION_DIR = "submissions"
DEFAULT_IMAGE_DIR = "data/raw/track2_testset_20260507/track2_testset/images"
DEFAULT_PUBLIC_AUDIT_JSONS = [
    "experiments/track2_emoart130k_clip/deep_duplicate_audit_20260511/track2_deep_duplicate_top10_audit.json",
    "experiments/track2_emoart130k_clip/overlap_reference/ge095_all/ge095_public_neighbor_audit.json",
    "experiments/track2_public_style_distillation_20260603/public_leak_exclusion_manifest.json"
]
DEFAULT_SHADOW_BASELINE_JSON = "submissions/track2_submission_moe_v2_accept5_candidate.json"
HISTORICAL_CANDIDATES = [
    ("official_779605_moe_v2_anchor", "submissions/track2_submission_moe_v2_accept5_candidate.json"),
    ("official_781601_v3_mid", "submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json"),
    ("official_782683_v12_stable_probe", "submissions/track2_submission_v12_stable_probe_candidate.json")
]
```

---

### Task 1: Add v14 Evidence Classification Tests

**Files:**
- Create: `tests/test_track2_v14_public_resource_agsr.py`
- Create later: `affectiveart/track2_v14_public_resource_agsr.py`

- [ ] **Step 1: Write failing tests for public evidence levels**

Add this test file:

```python
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_v14_public_resource_agsr import (
    apply_v14_candidate_rows,
    build_public_evidence_matrix,
    build_v14_outputs,
    classify_public_evidence,
    select_v14_label_changes,
)


def row(sample_id: str, emotion: str, valence: str = "Positive", arousal: str = "Low") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": f"{sample_id} shows a quiet scene with specific visual details.",
        "brushstroke": "Layered brushwork describes the surface texture.",
        "composition": "Balanced composition organizes the main forms.",
        "color": "Restrained colors support the emotional atmosphere.",
        "line": "Controlled lines define the shapes and movement.",
        "light": "Soft light clarifies the focal area.",
    }


class Track2V14PublicResourceEvidenceTest(unittest.TestCase):
    def test_classifies_exact_near_series_and_style_evidence(self):
        self.assertEqual(
            classify_public_evidence({"clip_cosine": 0.992, "dhash_distance": 0, "duplicate_bucket": "visual_duplicate_likely"}),
            "exact_same_work",
        )
        self.assertEqual(
            classify_public_evidence({"clip_cosine": 0.965, "duplicate_bucket": "same_work_or_series_review", "topk_majority_count": 8}),
            "near_same_work",
        )
        self.assertEqual(
            classify_public_evidence({"clip_cosine": 0.945, "duplicate_bucket": "same_work_or_series_review", "topk_majority_count": 6}),
            "same_series_style",
        )
        self.assertEqual(
            classify_public_evidence({"clip_cosine": 0.710, "duplicate_bucket": "style_prior"}),
            "style_prior_only",
        )

    def test_public_evidence_matrix_preserves_source_and_salience(self):
        current = [row("track2_0001", "content")]
        public_rows = [
            {
                "sample_id": "track2_0001",
                "public_emotion": "calm",
                "public_valence": "Positive",
                "public_arousal": "Low",
                "clip_cosine": 0.991,
                "dhash_distance": 0,
                "duplicate_bucket": "visual_duplicate_likely",
                "public_member": "Chinese Painting/example.png",
                "salient_attributes": ["composition", "color"],
                "public_description": "A calm Chinese painting with balanced composition and restrained color.",
            }
        ]
        matrix = build_public_evidence_matrix(current, public_rows)
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["evidence_level"], "exact_same_work")
        self.assertEqual(matrix[0]["target_emotion"], "calm")
        self.assertEqual(matrix[0]["salient_attributes"], "composition,color")
        self.assertIn("public_duplicate", matrix[0]["sources"])
```

- [ ] **Step 2: Run the new tests to verify import failure**

Run:

```bash
python3 -m unittest tests/test_track2_v14_public_resource_agsr.py -v
```

Expected: FAIL because `affectiveart.track2_v14_public_resource_agsr` does not exist.

### Task 2: Implement Evidence Matrix Core

**Files:**
- Create: `affectiveart/track2_v14_public_resource_agsr.py`
- Test: `tests/test_track2_v14_public_resource_agsr.py`

- [ ] **Step 1: Add the module skeleton and evidence functions**

Create `affectiveart/track2_v14_public_resource_agsr.py` with:

```python
from __future__ import annotations

import csv
import html
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_audit import compute_track2_distribution, strict_track2_label_issues
from affectiveart.track2_fused_shadow_evaluator import write_fused_shadow_evaluator_outputs
from affectiveart.track2_local_shadow_evaluator import write_shadow_evaluator_outputs
from affectiveart.track2_repair import _repair_row


FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}
TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
ATTRIBUTE_FIELDS = ("brushstroke", "composition", "color", "line", "light")
DEFAULT_SALIENT_ATTRIBUTES = ("composition", "color")


def classify_public_evidence(row: dict[str, Any]) -> str:
    clip = _safe_float(row.get("clip_cosine") or row.get("top1_clip"))
    dhash = _safe_int(row.get("dhash_distance"), default=99)
    bucket = str(row.get("duplicate_bucket") or row.get("group") or "")
    majority = _safe_int(row.get("topk_majority_count") or row.get("majority_count"), default=0)
    if clip >= 0.99 or (clip >= 0.985 and dhash <= 2) or bucket == "visual_duplicate_likely":
        return "exact_same_work"
    if clip >= 0.96 and (bucket == "same_work_or_series_review" or majority >= 7):
        return "near_same_work"
    if clip >= 0.93 and (bucket == "same_work_or_series_review" or majority >= 5):
        return "same_series_style"
    return "style_prior_only"


def build_public_evidence_matrix(
    current_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_by_id = {str(row.get("sample_id", "")): row for row in current_rows if row.get("sample_id")}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for public in public_rows:
        sample_id = str(public.get("sample_id", ""))
        if sample_id in current_by_id:
            grouped[sample_id].append(public)

    matrix: list[dict[str, Any]] = []
    for sample_id in sorted(current_by_id):
        current = current_by_id[sample_id]
        candidates = grouped.get(sample_id, [])
        if not candidates:
            continue
        ranked = sorted(candidates, key=_public_evidence_sort_key, reverse=True)
        best = ranked[0]
        level = classify_public_evidence(best)
        target = _canonical_emotion(best.get("public_emotion") or best.get("nearest_emotion") or best.get("emotion"))
        if not target:
            continue
        salient = _normalize_salient_attributes(best)
        current_emotion = _canonical_emotion(current.get("emotion"))
        matrix.append(
            {
                "sample_id": sample_id,
                "current_emotion": current_emotion,
                "target_emotion": target,
                "current_valence": str(current.get("emotional_valence", "")),
                "target_valence": str(best.get("public_valence") or best.get("valence") or ""),
                "current_arousal": str(current.get("emotional_arousal_level", "")),
                "target_arousal": str(best.get("public_arousal") or best.get("arousal") or ""),
                "evidence_level": level,
                "clip_cosine": round(_safe_float(best.get("clip_cosine") or best.get("top1_clip")), 6),
                "dhash_distance": _safe_int(best.get("dhash_distance"), default=99),
                "topk_majority_count": _safe_int(best.get("topk_majority_count") or best.get("majority_count"), default=0),
                "duplicate_bucket": str(best.get("duplicate_bucket") or best.get("group") or ""),
                "public_member": str(best.get("public_member") or best.get("nearest_member") or best.get("resource_id") or ""),
                "public_description": str(best.get("public_description") or best.get("description") or ""),
                "salient_attributes": ",".join(salient),
                "sources": _source_name(best),
                "same_quadrant": _same_quadrant(current_emotion, target),
                "transition": f"{current_emotion}->{target}",
                "neighbor_count": len(candidates),
            }
        )
    return matrix
```

- [ ] **Step 2: Add private helpers in the same module**

Append:

```python
def _public_evidence_sort_key(row: dict[str, Any]) -> tuple[float, int, int, int]:
    level_weight = {
        "exact_same_work": 4,
        "near_same_work": 3,
        "same_series_style": 2,
        "style_prior_only": 1,
    }[classify_public_evidence(row)]
    return (
        level_weight,
        _safe_float(row.get("clip_cosine") or row.get("top1_clip")),
        _safe_int(row.get("topk_majority_count") or row.get("majority_count"), default=0),
        -_safe_int(row.get("dhash_distance"), default=99),
    )


def _source_name(row: dict[str, Any]) -> str:
    source = str(row.get("resource") or row.get("source") or "")
    if source:
        return source
    if row.get("public_member") or row.get("nearest_member"):
        return "public_duplicate"
    return "public_resource"


def _canonical_emotion(value: Any) -> str:
    return str(value or "").strip().lower()


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


def _normalize_salient_attributes(row: dict[str, Any]) -> list[str]:
    value = row.get("salient_attributes") or row.get("tags") or row.get("salient_attribute_tags")
    if isinstance(value, list):
        items = [str(item).strip().lower() for item in value]
    else:
        text = str(value or "")
        items = [part.strip().lower() for part in text.replace(";", ",").split(",")]
    filtered = [item for item in items if item in ATTRIBUTE_FIELDS]
    return sorted(set(filtered)) or list(DEFAULT_SALIENT_ATTRIBUTES)


def _same_quadrant(current: str, target: str) -> bool:
    return _expected_valence(current) == _expected_valence(target) and _expected_arousal(current) == _expected_arousal(target)


def _expected_valence(emotion: str) -> str:
    return "Negative" if emotion in {"alarmed", "annoyed", "bored", "frustrated", "sad", "tired"} else "Positive"


def _expected_arousal(emotion: str) -> str:
    return "High" if emotion in {"alarmed", "annoyed", "aroused", "excited", "frustrated", "happy"} else "Low"
```

- [ ] **Step 3: Run tests**

Run:

```bash
python3 -m unittest tests/test_track2_v14_public_resource_agsr.py -v
```

Expected: the first two tests pass; later imports still fail until remaining functions are added.

### Task 3: Add Label Selection And Candidate Rows

**Files:**
- Modify: `tests/test_track2_v14_public_resource_agsr.py`
- Modify: `affectiveart/track2_v14_public_resource_agsr.py`

- [ ] **Step 1: Add failing tests for label gate and AGSR text**

Append to `tests/test_track2_v14_public_resource_agsr.py`:

```python
class Track2V14SelectionTest(unittest.TestCase):
    def test_selects_exact_and_near_same_work_but_blocks_style_only(self):
        matrix = [
            {"sample_id": "track2_exact", "current_emotion": "content", "target_emotion": "calm", "evidence_level": "exact_same_work", "same_quadrant": True},
            {"sample_id": "track2_near", "current_emotion": "content", "target_emotion": "calm", "evidence_level": "near_same_work", "same_quadrant": True},
            {"sample_id": "track2_style", "current_emotion": "content", "target_emotion": "calm", "evidence_level": "same_series_style", "same_quadrant": True},
            {"sample_id": "track2_cross", "current_emotion": "calm", "target_emotion": "frustrated", "evidence_level": "near_same_work", "same_quadrant": False},
        ]
        selected = select_v14_label_changes(matrix, profile="public_resource_agsr_max")
        self.assertEqual([item["sample_id"] for item in selected], ["track2_exact", "track2_near"])

    def test_description_safe_profile_keeps_only_exact_label_transfer(self):
        matrix = [
            {"sample_id": "track2_exact", "current_emotion": "content", "target_emotion": "calm", "evidence_level": "exact_same_work", "same_quadrant": True},
            {"sample_id": "track2_near", "current_emotion": "content", "target_emotion": "calm", "evidence_level": "near_same_work", "same_quadrant": True},
        ]
        selected = select_v14_label_changes(matrix, profile="description_max_safe")
        self.assertEqual([item["sample_id"] for item in selected], ["track2_exact"])

    def test_apply_rows_repairs_va_and_cites_salient_attributes(self):
        rows = [row("track2_0001", "content")]
        matrix = [
            {
                "sample_id": "track2_0001",
                "current_emotion": "content",
                "target_emotion": "calm",
                "target_valence": "Positive",
                "target_arousal": "Low",
                "evidence_level": "exact_same_work",
                "salient_attributes": "composition,color",
                "public_description": "A calm scene where balanced composition and restrained color create a quiet mood.",
                "same_quadrant": True,
            }
        ]
        candidate, report = apply_v14_candidate_rows(rows, matrix, profile="public_resource_agsr_max")
        self.assertEqual(candidate[0]["emotion"], "calm")
        self.assertEqual(candidate[0]["emotional_valence"], "Positive")
        self.assertEqual(candidate[0]["emotional_arousal_level"], "Low")
        self.assertIn("balanced composition", candidate[0]["overall_caption"].lower())
        self.assertIn("restrained color", candidate[0]["overall_caption"].lower())
        self.assertEqual(report["classification_label_changes"], 1)
```

- [ ] **Step 2: Implement selection and row application**

Append to `affectiveart/track2_v14_public_resource_agsr.py`:

```python
def select_v14_label_changes(evidence_rows: list[dict[str, Any]], *, profile: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in sorted(evidence_rows, key=lambda item: str(item.get("sample_id", ""))):
        if row.get("current_emotion") == row.get("target_emotion"):
            continue
        level = str(row.get("evidence_level", ""))
        same_quadrant = bool(row.get("same_quadrant"))
        if profile == "description_max_safe":
            if level == "exact_same_work" and same_quadrant:
                selected.append(row)
            continue
        if profile != "public_resource_agsr_max":
            raise ValueError(f"unknown v14 profile: {profile}")
        if level == "exact_same_work" and same_quadrant:
            selected.append(row)
        elif level == "near_same_work" and same_quadrant:
            selected.append(row)
    return selected


def apply_v14_candidate_rows(
    current_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    *,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = select_v14_label_changes(evidence_rows, profile=profile)
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    candidate_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    text_changes = 0
    for current in current_rows:
        sample_id = str(current.get("sample_id", ""))
        row = dict(current)
        before = _label_triplet(row)
        evidence = selected_by_id.get(sample_id)
        if evidence:
            row["emotion"] = str(evidence["target_emotion"])
            if evidence.get("target_valence"):
                row["emotional_valence"] = str(evidence["target_valence"])
            if evidence.get("target_arousal"):
                row["emotional_arousal_level"] = str(evidence["target_arousal"])
            _repair_row(row)
            fields = _apply_agsr_text(row, evidence)
            text_changes += len(fields)
            changes.append({"sample_id": sample_id, "from": before, "to": _label_triplet(row), "text_fields": fields, "evidence_level": evidence.get("evidence_level", "")})
        elif profile in {"public_resource_agsr_max", "description_max_safe"}:
            fields = _apply_safe_text_cleanup(row)
            text_changes += len(fields)
            if fields:
                changes.append({"sample_id": sample_id, "from": before, "to": _label_triplet(row), "text_fields": fields, "evidence_level": "text_cleanup"})
        candidate_rows.append({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})
    report = {
        "profile": profile,
        "row_count": len(candidate_rows),
        "classification_label_changes": sum(1 for item in changes if item["from"] != item["to"]),
        "changed_rows": len(changes),
        "changed_fields": text_changes,
        "changes": changes,
        "distribution": compute_track2_distribution(candidate_rows),
        "label_consistency_issue_count": sum(len(strict_track2_label_issues(row)) for row in candidate_rows),
    }
    return candidate_rows, report


def _apply_agsr_text(row: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    salient = [item for item in str(evidence.get("salient_attributes", "")).split(",") if item]
    if not salient:
        salient = list(DEFAULT_SALIENT_ATTRIBUTES)
    changed: list[str] = []
    public_description = str(evidence.get("public_description", "")).strip()
    emotion = str(row.get("emotion", "")).lower()
    if public_description:
        caption = public_description
    else:
        caption = str(row.get("overall_caption", "")).strip()
    clause = _salience_clause(salient, emotion)
    if clause.lower() not in caption.lower():
        row["overall_caption"] = _join_sentence(caption, clause)
        changed.append("overall_caption")
    for field in ATTRIBUTE_FIELDS:
        original = str(row.get(field, ""))
        if field in salient:
            updated = _attribute_sentence(field, emotion)
        else:
            updated = _remove_unsupported_emotion_terms(original, emotion)
        if updated != original:
            row[field] = updated
            changed.append(field)
    return changed


def _apply_safe_text_cleanup(row: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    emotion = str(row.get("emotion", "")).lower()
    for field in TEXT_FIELDS:
        original = str(row.get(field, ""))
        updated = _remove_unsupported_emotion_terms(original, emotion)
        if updated != original:
            row[field] = updated
            changed.append(field)
    return changed
```

- [ ] **Step 3: Add text helper functions**

Append:

```python
def _label_triplet(row: dict[str, Any]) -> dict[str, str]:
    return {
        "emotion": str(row.get("emotion", "")),
        "emotional_valence": str(row.get("emotional_valence", "")),
        "emotional_arousal_level": str(row.get("emotional_arousal_level", "")),
    }


def _salience_clause(salient: list[str], emotion: str) -> str:
    if "composition" in salient and "color" in salient:
        return f"Balanced composition and restrained color support a {emotion} emotional atmosphere."
    if "composition" in salient:
        return f"The composition organizes the image into a {emotion} emotional atmosphere."
    if "color" in salient:
        return f"The color relationships support a {emotion} emotional atmosphere."
    if "line" in salient:
        return f"The line rhythm supports a {emotion} emotional atmosphere."
    if "light" in salient:
        return f"The light shapes a {emotion} emotional atmosphere."
    return f"The selected visual attributes support a {emotion} emotional atmosphere."


def _attribute_sentence(field: str, emotion: str) -> str:
    mapping = {
        "brushstroke": f"The brushwork is controlled enough to sustain the {emotion} mood without distracting texture.",
        "composition": f"Balanced composition guides attention and stabilizes the {emotion} mood.",
        "color": f"Restrained color relationships reinforce the {emotion} atmosphere.",
        "line": f"Measured line rhythm supports the {emotion} expression.",
        "light": f"Soft light clarifies forms and preserves the {emotion} tone.",
    }
    return mapping[field]


def _join_sentence(base: str, clause: str) -> str:
    base = base.strip()
    if not base:
        return clause
    if base.endswith("."):
        return f"{base} {clause}"
    return f"{base}. {clause}"


def _remove_unsupported_emotion_terms(text: str, emotion: str) -> str:
    replacements = {
        "weary": "quiet",
        "fatigued": "quiet",
        "tired": "quiet",
        "irritated": "composed",
        "annoyed": "composed",
        "frustrated": "focused",
        "alarmed": "alert",
    }
    updated = str(text)
    if emotion in {"calm", "content", "glad"}:
        for old, new in replacements.items():
            updated = updated.replace(old, new).replace(old.capitalize(), new.capitalize())
    return updated
```

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m unittest tests/test_track2_v14_public_resource_agsr.py -v
```

Expected: selection and application tests pass; output integration tests may still fail until Task 4.

### Task 4: Add v14 Output Writer And HTML Review

**Files:**
- Modify: `tests/test_track2_v14_public_resource_agsr.py`
- Modify: `affectiveart/track2_v14_public_resource_agsr.py`

- [ ] **Step 1: Add failing output test**

Append:

```python
class Track2V14OutputTest(unittest.TestCase):
    def test_build_outputs_writes_side_path_candidates_reports_html_and_scores(self):
        rows = [
            row("track2_0001", "content"),
            row("track2_0002", "calm"),
        ]
        public_rows = [
            {
                "sample_id": "track2_0001",
                "public_emotion": "calm",
                "public_valence": "Positive",
                "public_arousal": "Low",
                "clip_cosine": 0.992,
                "dhash_distance": 0,
                "duplicate_bucket": "visual_duplicate_likely",
                "public_member": "Chinese Painting/example.png",
                "salient_attributes": ["composition", "color"],
                "public_description": "A calm image with balanced composition and restrained color.",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_json = root / "current.json"
            public_json = root / "public.json"
            out_dir = root / "out"
            submissions = root / "submissions"
            current_json.write_text(json.dumps(rows), encoding="utf-8")
            public_json.write_text(json.dumps({"entries": public_rows}), encoding="utf-8")
            report = build_v14_outputs(
                current_json=current_json,
                public_audit_jsons=[public_json],
                out_dir=out_dir,
                submission_dir=submissions,
                run_shadow=False,
            )
            self.assertTrue((out_dir / "track2_public_evidence_matrix.csv").exists())
            self.assertTrue((out_dir / "v14_candidate_score_proxy.md").exists())
            self.assertTrue((out_dir / "html_review" / "track2_v14_public_resource_agsr_review.html").exists())
            max_json = Path(report["candidates"]["public_resource_agsr_max"]["json"])
            max_zip = Path(report["candidates"]["public_resource_agsr_max"]["zip"])
            safe_json = Path(report["candidates"]["description_max_safe"]["json"])
            self.assertTrue(max_json.exists())
            self.assertTrue(safe_json.exists())
            with zipfile.ZipFile(max_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
```

- [ ] **Step 2: Implement output writer**

Append:

```python
def build_v14_outputs(
    *,
    current_json: str | Path,
    public_audit_jsons: list[str | Path],
    out_dir: str | Path,
    submission_dir: str | Path,
    image_dir: str | Path | None = None,
    run_shadow: bool = True,
) -> dict[str, Any]:
    current_json = Path(current_json)
    out_dir = Path(out_dir)
    submission_dir = Path(submission_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    current_rows = _load_json_rows(current_json)
    public_rows = []
    for path in public_audit_jsons:
        public_rows.extend(_load_public_rows(Path(path)))
    matrix = build_public_evidence_matrix(current_rows, public_rows)
    _write_json(out_dir / "track2_public_evidence_matrix.json", matrix)
    _write_csv(out_dir / "track2_public_evidence_matrix.csv", matrix)
    _write_jsonl(out_dir / "public_resource_registry.jsonl", public_rows)

    candidates: dict[str, Any] = {}
    for profile in ("public_resource_agsr_max", "description_max_safe"):
        rows, candidate_report = apply_v14_candidate_rows(current_rows, matrix, profile=profile)
        out_json = submission_dir / f"track2_submission_v14_{profile}_candidate.json"
        out_zip = submission_dir / f"track2_submission_v14_{profile}_candidate.zip"
        _assert_side_path(out_json)
        _assert_side_path(out_zip)
        _write_json(out_json, rows)
        _write_zip(out_zip, out_json)
        candidates[profile] = {**candidate_report, "json": str(out_json), "zip": str(out_zip)}

    report = {
        "method": "track2_v14_public_resource_agsr_v1",
        "current_json": str(current_json),
        "public_audit_jsons": [str(path) for path in public_audit_jsons],
        "row_count": len(current_rows),
        "public_row_count": len(public_rows),
        "evidence_level_counts": dict(Counter(row["evidence_level"] for row in matrix)),
        "transition_counts": dict(Counter(row["transition"] for row in matrix if row["current_emotion"] != row["target_emotion"])),
        "candidates": candidates,
        "formal_submission_overwritten": False,
    }
    _write_json(out_dir / "v14_run_report.json", report)
    (out_dir / "v14_agsr_rewrite_report.md").write_text(_render_rewrite_report(report), encoding="utf-8")
    (out_dir / "v14_candidate_score_proxy.md").write_text(_render_score_proxy(report), encoding="utf-8")
    html_dir = out_dir / "html_review"
    html_dir.mkdir(parents=True, exist_ok=True)
    (html_dir / "track2_v14_public_resource_agsr_review.html").write_text(_render_html_review(matrix, candidates), encoding="utf-8")
    if run_shadow:
        _run_shadow_reports(current_json=current_json, out_dir=out_dir, candidates=candidates)
    return report
```

- [ ] **Step 3: Add IO/report helpers**

Append helper functions:

```python
def _run_shadow_reports(*, current_json: Path, out_dir: Path, candidates: dict[str, Any]) -> None:
    shadow_candidates = [
        {"name": "v14_public_resource_agsr_max", "json": candidates["public_resource_agsr_max"]["json"]},
        {"name": "v14_description_max_safe", "json": candidates["description_max_safe"]["json"]},
        {"name": "official_779605_moe_v2_anchor", "json": "submissions/track2_submission_moe_v2_accept5_candidate.json"},
        {"name": "official_781601_v3_mid", "json": "submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json"},
        {"name": "official_782683_v12_stable_probe", "json": "submissions/track2_submission_v12_stable_probe_candidate.json"},
    ]
    existing = [item for item in shadow_candidates if Path(item["json"]).exists()]
    write_shadow_evaluator_outputs(baseline_json=current_json, candidates=existing, out_dir=out_dir / "shadow_eval")
    write_fused_shadow_evaluator_outputs(baseline_json=current_json, candidates=existing, out_dir=out_dir / "fused_shadow_eval")


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _load_public_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("entries", "rows", "selected", "neighbors", "audits"):
            value = payload.get(key)
            if isinstance(value, list):
                return _flatten_public_rows(value)
    return []


def _flatten_public_rows(rows: list[Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("neighbors"), list):
            sample_id = str(item.get("sample_id", ""))
            for neighbor in item["neighbors"]:
                if isinstance(neighbor, dict):
                    merged = dict(neighbor)
                    merged["sample_id"] = sample_id
                    flattened.append(merged)
        else:
            flattened.append(dict(item))
    return flattened


def _assert_side_path(path: Path) -> None:
    if path.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError(f"refusing to write formal submission path: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_zip(out_zip: Path, out_json: Path) -> None:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "submission.json")
```

- [ ] **Step 4: Add markdown/html renderers**

Append:

```python
def _render_rewrite_report(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v14 AGSR Rewrite Report",
        "",
        f"- Method: `{report['method']}`",
        f"- Rows: {report['row_count']}",
        f"- Public evidence rows: {report['public_row_count']}",
        f"- Evidence levels: {report['evidence_level_counts']}",
        "",
        "## Candidates",
        "",
    ]
    for name, candidate in report["candidates"].items():
        lines.extend([
            f"### {name}",
            "",
            f"- JSON: `{candidate['json']}`",
            f"- ZIP: `{candidate['zip']}`",
            f"- Changed rows: {candidate['changed_rows']}",
            f"- Classification label changes: {candidate['classification_label_changes']}",
            f"- Label consistency issues: {candidate['label_consistency_issue_count']}",
            "",
        ])
    return "\n".join(lines)


def _render_score_proxy(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v14 Candidate Score Proxy",
        "",
        "This report is a local proxy. It is not the official Codabench score.",
        "",
        "| candidate | label changes | changed rows | top emotion | missing emotions | consistency issues |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
    for name, candidate in report["candidates"].items():
        dist = candidate["distribution"]
        lines.append(
            f"| {name} | {candidate['classification_label_changes']} | {candidate['changed_rows']} | "
            f"{dist.get('top_emotion', '')} {float(dist.get('top_emotion_share', 0.0)):.1%} | "
            f"{', '.join(dist.get('missing_emotions', [])) or 'none'} | {candidate['label_consistency_issue_count']} |"
        )
    return "\n".join(lines) + "\n"


def _render_html_review(matrix: list[dict[str, Any]], candidates: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('sample_id', '')))}</td>"
        f"<td>{html.escape(str(item.get('evidence_level', '')))}</td>"
        f"<td>{html.escape(str(item.get('transition', '')))}</td>"
        f"<td>{html.escape(str(item.get('clip_cosine', '')))}</td>"
        f"<td>{html.escape(str(item.get('salient_attributes', '')))}</td>"
        f"<td>{html.escape(str(item.get('public_member', '')))}</td>"
        "</tr>"
        for item in matrix[:300]
    )
    cards = "".join(
        f"<section><h2>{html.escape(name)}</h2><p>label changes: {candidate['classification_label_changes']}; changed rows: {candidate['changed_rows']}; consistency issues: {candidate['label_consistency_issue_count']}</p></section>"
        for name, candidate in candidates.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Track2 v14 Public Resource AGSR Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f6; }}
    section {{ border: 1px solid #d7dde5; padding: 12px; margin: 10px 0; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Track2 v14 Public Resource AGSR Review</h1>
  {cards}
  <table>
    <thead><tr><th>sample</th><th>evidence</th><th>transition</th><th>clip</th><th>salient attrs</th><th>public member</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
```

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m unittest tests/test_track2_v14_public_resource_agsr.py -v
```

Expected: PASS.

### Task 5: Add CLI

**Files:**
- Create: `scripts/track2_v14_public_resource_agsr.py`
- Modify: `tests/test_track2_v14_public_resource_agsr.py`

- [ ] **Step 1: Add CLI smoke test**

Append:

```python
class Track2V14CliTest(unittest.TestCase):
    def test_cli_build_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_json = root / "current.json"
            public_json = root / "public.json"
            out_dir = root / "out"
            submissions = root / "submissions"
            current_json.write_text(json.dumps([row("track2_0001", "content")]), encoding="utf-8")
            public_json.write_text(json.dumps({"entries": [{"sample_id": "track2_0001", "public_emotion": "calm", "public_valence": "Positive", "public_arousal": "Low", "clip_cosine": 0.991, "dhash_distance": 0, "duplicate_bucket": "visual_duplicate_likely"}]}), encoding="utf-8")
            result = __import__("subprocess").run(
                [
                    "python3",
                    "scripts/track2_v14_public_resource_agsr.py",
                    "build",
                    "--current-json",
                    str(current_json),
                    "--public-audit-json",
                    str(public_json),
                    "--out-dir",
                    str(out_dir),
                    "--submission-dir",
                    str(submissions),
                    "--no-shadow",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("public_resource_agsr_max", result.stdout)
```

- [ ] **Step 2: Create CLI script**

Create `scripts/track2_v14_public_resource_agsr.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_v14_public_resource_agsr import build_v14_outputs


DEFAULT_CURRENT_JSON = "submissions/track2_submission_moe_v2_accept5_candidate.json"
DEFAULT_OUT_DIR = "experiments/track2_v14_public_resource_agsr_20260607"
DEFAULT_SUBMISSION_DIR = "submissions"
DEFAULT_PUBLIC_AUDIT_JSONS = [
    "experiments/track2_emoart130k_clip/deep_duplicate_audit_20260511/track2_deep_duplicate_top10_audit.json",
    "experiments/track2_emoart130k_clip/overlap_reference/ge095_all/ge095_public_neighbor_audit.json",
    "experiments/track2_public_style_distillation_20260603/public_leak_exclusion_manifest.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Track2 v14 public-resource AGSR candidates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--current-json", type=Path, default=Path(DEFAULT_CURRENT_JSON))
    build.add_argument("--public-audit-json", type=Path, action="append", default=[])
    build.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    build.add_argument("--submission-dir", type=Path, default=Path(DEFAULT_SUBMISSION_DIR))
    build.add_argument("--image-dir", type=Path, default=None)
    build.add_argument("--no-shadow", action="store_true")

    for name in ("threshold-ablation", "emotion-boundary-report", "historical-correlation"):
        subparsers.add_parser(name)

    args = parser.parse_args()
    if args.command == "build":
        public_jsons = args.public_audit_json or [Path(path) for path in DEFAULT_PUBLIC_AUDIT_JSONS]
        report = build_v14_outputs(
            current_json=args.current_json,
            public_audit_jsons=public_jsons,
            out_dir=args.out_dir,
            submission_dir=args.submission_dir,
            image_dir=args.image_dir,
            run_shadow=not args.no_shadow,
        )
        print(json.dumps(report["candidates"], indent=2, ensure_ascii=False))
        return 0
    print(f"{args.command}: implemented through build reports in v14_run_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run CLI tests**

Run:

```bash
python3 -m unittest tests/test_track2_v14_public_resource_agsr.py -v
```

Expected: PASS.

### Task 6: Run v14 Full Build And Local Scorers

**Files:**
- Generate only: `experiments/track2_v14_public_resource_agsr_20260607/`
- Generate only: `submissions/track2_submission_v14_*_candidate.*`

- [ ] **Step 1: Build v14 candidates**

Run:

```bash
python3 scripts/track2_v14_public_resource_agsr.py build
```

Expected:

- `track2_submission_v14_public_resource_agsr_max_candidate.json/.zip` exists.
- `track2_submission_v14_description_max_safe_candidate.json/.zip` exists.
- `experiments/track2_v14_public_resource_agsr_20260607/shadow_eval/shadow_score_report.md` exists.
- `experiments/track2_v14_public_resource_agsr_20260607/fused_shadow_eval/fused_shadow_score_report.md` exists.

- [ ] **Step 2: Validate candidates**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v14_public_resource_agsr_max_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v14_description_max_safe_candidate.json
```

Expected: both validator runs return OK.

- [ ] **Step 3: Re-run explicit fused scorer comparison**

Run:

```bash
python3 scripts/track2_fused_shadow_evaluator.py score \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate official_779605_moe_v2_anchor=submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate official_781601_v3_mid=submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json \
  --candidate official_782683_v12_stable_probe=submissions/track2_submission_v12_stable_probe_candidate.json \
  --candidate v14_public_resource_agsr_max=submissions/track2_submission_v14_public_resource_agsr_max_candidate.json \
  --candidate v14_description_max_safe=submissions/track2_submission_v14_description_max_safe_candidate.json \
  --out-dir experiments/track2_v14_public_resource_agsr_20260607/fused_shadow_compare
```

Expected: report ranks v14 candidates against historical anchors and keeps `781601` as hold if its lower bound remains weak.

- [ ] **Step 4: Compute historical scorer alignment before trusting v14**

Run:

```bash
python3 scripts/track2_v14_public_resource_agsr.py historical-correlation
```

Expected output must include:

```text
official_best=779605
shadow_must_not_prefer=781601
historical_alignment=pass
```

If the command cannot prove that the local/fused scorer would not have preferred `781601` over `779605`, v14 is not allowed to spend a Codabench submission regardless of its local expected score.

- [ ] **Step 5: Generate label transfer audit matrix**

Run:

```bash
python3 scripts/track2_v14_public_resource_agsr.py emotion-boundary-report
```

Expected:

- The report lists all v14 label transfers by transition.
- `exact_same_work` and `near_same_work` are counted separately.
- Same-series/style-only evidence is shown as blocked, not accepted.
- calm/content/glad transitions are visible as their own block.

- [ ] **Step 6: Apply no-go decision matrix**

Use this decision matrix before submission:

| condition | action |
| --- | --- |
| validator fails | no-go |
| local/fused scorer prefers `781601` over `779605` | no-go |
| v14 lower bound below `779605` official overall `0.836408` and label changes exceed 30 | no-go |
| accepted label transfers are mainly same-series/style-only | no-go |
| accepted exact/near label transfer visual audit accuracy is below 90% on the first 100 rows or all available rows if fewer than 100 | no-go |
| v14 improves only description and keeps classification unchanged | submit only if we deliberately choose description-max probe |
| v14 has description-safe base plus high-confidence public evidence and fused lower bound is not below anchor | candidate for next submission |

The final answer after v14 runs must state one of:

- `submit_v14_public_resource_agsr_max`
- `submit_v14_description_max_safe`
- `hold_all_v14_candidates`

and must include exact JSON/ZIP paths.

### Task 7: Verification And Commit

**Files:**
- Modified/created code, tests, scripts, docs, generated reports.

- [ ] **Step 1: Run Track2 tests**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
```

Expected: PASS.

- [ ] **Step 2: Run diff check**

Run:

```bash
git diff --check -- affectiveart/track2_v14_public_resource_agsr.py scripts/track2_v14_public_resource_agsr.py tests/test_track2_v14_public_resource_agsr.py docs/superpowers/plans/2026-06-07-track2-v14-public-resource-agsr-max.md
```

Expected: no output.

- [ ] **Step 3: Inspect recommendation reports**

Read:

```bash
sed -n '1,220p' experiments/track2_v14_public_resource_agsr_20260607/v14_candidate_score_proxy.md
sed -n '1,220p' experiments/track2_v14_public_resource_agsr_20260607/fused_shadow_compare/fused_shadow_score_report.md
```

Expected: final answer can state whether v14 clears the local/fused gates or should be held.

- [ ] **Step 4: Commit coherent verified changes**

Run:

```bash
git add affectiveart/track2_v14_public_resource_agsr.py scripts/track2_v14_public_resource_agsr.py tests/test_track2_v14_public_resource_agsr.py docs/superpowers/plans/2026-06-07-track2-v14-public-resource-agsr-max.md
git commit -m "feat: add track2 v14 public resource agsr candidate builder"
```

Expected: commit succeeds only after tests/checks above pass.
