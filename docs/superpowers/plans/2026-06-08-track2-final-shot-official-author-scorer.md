# Track2 Final-Shot Official/Author Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a final-shot Track2 scorer and candidate generator that uses official and EmoArt-author evidence as the only label-arbitration baseline, then produces one recommended v22 submission candidate for the last Codabench attempt.

**Architecture:** Add a focused `track2_v22_official_author_scorer` module that separates evidence ingestion, local EmoArt inventory, official-feedback calibration, candidate ladder generation, and recommendation. It reuses existing Track2 row loading/output patterns from v21, but changes the scoring logic to privilege official v21 feedback and author-data style priors over generic local shadow scores.

**Tech Stack:** Python stdlib, existing `affectiveart.challenge` validation constants, existing v21 candidate output style, `unittest`, local EmoArt-130k `Annotation.json`, JSON/CSV/Markdown artifacts.

---

## File Structure

- Create: `affectiveart/track2_v22_official_author_scorer.py`
  - Evidence boundary classification.
  - Local EmoArt-130k inventory loading.
  - Style/emotion prior extraction from `Annotation.json`.
  - Official-feedback calibrated scoring for `content->calm`.
  - v22 ladder selection and report generation.
- Create: `scripts/track2_v22_official_author_scorer.py`
  - CLI entrypoints: `inventory`, `score-ladders`, `build-candidates`.
- Create: `tests/test_track2_v22_official_author_scorer.py`
  - Unit tests for boundary rules, inventory parsing, calmshift calibration, candidate generation, and no formal overwrite.
- Output:
  - `experiments/track2_v22_official_author_scorer_20260608/local_emoart130k_inventory.json`
  - `experiments/track2_v22_official_author_scorer_20260608/style_emotion_prior.csv`
  - `experiments/track2_v22_official_author_scorer_20260608/v22_ladder_scoreboard.json`
  - `experiments/track2_v22_official_author_scorer_20260608/v22_ladder_scoreboard.md`
  - `experiments/track2_v22_official_author_scorer_20260608/v22_final_recommendation_zh.md`
  - `submissions/track2_submission_v22_official_author_calmshift120_candidate.json/.zip`
  - `submissions/track2_submission_v22_official_author_calmshift150_candidate.json/.zip`
  - `submissions/track2_submission_v22_official_author_calmshiftall_candidate.json/.zip`

## Constants

Use these defaults:

```python
DEFAULT_OUT_DIR = Path("experiments/track2_v22_official_author_scorer_20260608")
DEFAULT_EMOART_ROOT = Path("/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k")
DEFAULT_BASE_JSON = Path("submissions/track2_submission_v15_desc_expand300_candidate.json")
DEFAULT_V17_EVIDENCE = Path("experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json")
DEFAULT_V21_OFFICIAL_SCORE = {
    "submission_id": "785979",
    "overall": 0.842559,
    "classification": 0.740034,
    "description": 0.945083,
    "emotion_accuracy": 0.660000,
    "emotion_macro_f1": 0.320642,
}
DEFAULT_ANCHOR_OFFICIAL_SCORE = {
    "submission_id": "779605",
    "overall": 0.836408,
    "classification": 0.723150,
    "description": 0.949667,
    "emotion_accuracy": 0.570000,
    "emotion_macro_f1": 0.309338,
}
```

## Task 1: Evidence Boundary and Inventory Tests

**Files:**
- Create: `tests/test_track2_v22_official_author_scorer.py`
- Create later: `affectiveart/track2_v22_official_author_scorer.py`

- [ ] **Step 1: Write failing tests for evidence source boundaries**

Add:

```python
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_v22_official_author_scorer import (
    classify_source_scope,
    discover_local_emoart_inventory,
    source_can_arbitrate_label,
)


class Track2V22EvidenceBoundaryTests(unittest.TestCase):
    def test_classifies_official_and_author_sources_as_arbitration_sources(self):
        self.assertEqual(classify_source_scope("https://www.codabench.org/competitions/16304"), "official")
        self.assertEqual(classify_source_scope("https://openreview.net/forum?id=LbbHX8ofXZ"), "official")
        self.assertEqual(classify_source_scope("https://huggingface.co/datasets/printblue/EmoArt-130k"), "author")
        self.assertEqual(classify_source_scope("https://github.com/zhiliangzhang/FAB-G"), "author")
        self.assertEqual(classify_source_scope("/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k"), "local_author_data")

    def test_blocks_auxiliary_sources_from_label_arbitration(self):
        self.assertEqual(classify_source_scope("https://arxiv.org/abs/2101.07396"), "auxiliary")
        self.assertFalse(source_can_arbitrate_label("auxiliary"))
        self.assertTrue(source_can_arbitrate_label("official"))
        self.assertTrue(source_can_arbitrate_label("author"))
        self.assertTrue(source_can_arbitrate_label("local_author_data"))

    def test_discovers_local_emoart_inventory_from_minimal_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Annotation.json").write_text(
                json.dumps([
                    {"image_path": "Chinese Painting/example.jpg", "request_id": "Chinese Painting_request-1", "description": "{}"},
                    {"image_path": "Ukiyo-e/example.jpg", "request_id": "Ukiyo-e_request-1", "description": "{}"},
                ]),
                encoding="utf-8",
            )
            (root / "Chinese Painting.tar.gz").write_bytes(b"stub")
            (root / "Ukiyo-e.tar.gz").write_bytes(b"stub")

            inventory = discover_local_emoart_inventory(root)

        self.assertEqual(inventory["annotation_rows"], 2)
        self.assertEqual(inventory["tar_count"], 2)
        self.assertEqual(inventory["styles"], ["Chinese Painting", "Ukiyo-e"])
```

- [ ] **Step 2: Run tests to verify import failure**

Run:

```bash
python3 -m unittest tests/test_track2_v22_official_author_scorer.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `affectiveart.track2_v22_official_author_scorer`.

## Task 2: Implement Evidence Boundary and Local Inventory

**Files:**
- Create: `affectiveart/track2_v22_official_author_scorer.py`
- Modify: `tests/test_track2_v22_official_author_scorer.py`

- [ ] **Step 1: Add module with boundary functions**

Create:

```python
from __future__ import annotations

import csv
import json
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OFFICIAL_HOST_MARKERS = ("codabench.org", "openreview.net", "affectiveart")
AUTHOR_HOST_MARKERS = ("zhiliangzhang.github.io", "github.com/zhiliangzhang", "huggingface.co/datasets/printblue", "hongxiaxie.net", "github.com/aimmemotion")
AUXILIARY_MARKERS = ("artemisdataset.org", "EmoSet", "MMArt", "BAM", "Artpedia", "MART", "2101.07396", "2307.07961")
DEFAULT_EMOART_ROOT = Path(os.environ.get("EMOART_130K_ROOT", "/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k"))
TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
TRACK2_SUBMISSION_KEYS = ("sample_id", "emotion", "emotional_valence", "emotional_arousal_level", *TEXT_FIELDS)
FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip", "submission.json", "submission.zip"}
POS_LOW_EMOTIONS = {"calm", "content", "glad"}


def classify_source_scope(source: str | Path) -> str:
    text = str(source)
    if "EmoArt-130k" in text and Path(text).exists():
        return "local_author_data"
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in OFFICIAL_HOST_MARKERS):
        return "official"
    if any(marker.lower() in lowered for marker in AUTHOR_HOST_MARKERS):
        return "author"
    if any(marker.lower() in lowered for marker in AUXILIARY_MARKERS):
        return "auxiliary"
    return "auxiliary"


def source_can_arbitrate_label(scope: str) -> bool:
    return scope in {"official", "author", "local_author_data"}
```

- [ ] **Step 2: Implement local inventory**

Add:

```python
def discover_local_emoart_inventory(root: str | Path = DEFAULT_EMOART_ROOT) -> dict[str, Any]:
    root = Path(root)
    annotation = root / "Annotation.json"
    if not root.exists():
        raise FileNotFoundError(f"EmoArt root does not exist: {root}")
    if not annotation.exists():
        raise FileNotFoundError(f"Annotation.json does not exist: {annotation}")
    rows = json.loads(annotation.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Annotation.json must contain a list: {annotation}")
    tar_files = sorted(root.glob("*.tar.gz"))
    styles = sorted(path.name[:-7] for path in tar_files)
    first_keys = sorted(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
    return {
        "root": str(root),
        "annotation_path": str(annotation),
        "annotation_rows": len(rows),
        "annotation_size_mb": round(annotation.stat().st_size / 1024 / 1024, 3),
        "tar_count": len(tar_files),
        "tar_total_gb": round(sum(path.stat().st_size for path in tar_files) / 1024 / 1024 / 1024, 3),
        "styles": styles,
        "first_keys": first_keys,
    }
```

- [ ] **Step 3: Run boundary tests**

Run:

```bash
python3 -m unittest tests/test_track2_v22_official_author_scorer.py -v
```

Expected: PASS for the first three tests.

- [ ] **Step 4: Commit Task 2**

Run:

```bash
git add affectiveart/track2_v22_official_author_scorer.py tests/test_track2_v22_official_author_scorer.py
git commit -m "feat: add track2 v22 official author evidence boundary"
```

## Task 3: Style Prior and Calmshift Calibration Tests

**Files:**
- Modify: `tests/test_track2_v22_official_author_scorer.py`
- Modify later: `affectiveart/track2_v22_official_author_scorer.py`

- [ ] **Step 1: Add tests for style prior parsing and official feedback score**

Append:

```python
from affectiveart.track2_v22_official_author_scorer import (
    OfficialScore,
    build_style_emotion_prior,
    score_calmshift_change,
)


class Track2V22ScoringTests(unittest.TestCase):
    def test_build_style_emotion_prior_parses_description_json(self):
        rows = [
            {
                "image_path": "Chinese Painting/a.jpg",
                "description": json.dumps({"dominant_emotion": "Calm", "emotional_valence": "Positive", "emotional_arousal_level": "Low"}),
            },
            {
                "image_path": "Chinese Painting/b.jpg",
                "description": json.dumps({"dominant_emotion": "Content", "emotional_valence": "Positive", "emotional_arousal_level": "Low"}),
            },
            {
                "image_path": "Ukiyo-e/c.jpg",
                "description": json.dumps({"dominant_emotion": "Calm", "emotional_valence": "Positive", "emotional_arousal_level": "Low"}),
            },
        ]

        prior = build_style_emotion_prior(rows)

        self.assertEqual(prior["Chinese Painting"]["total"], 2)
        self.assertAlmostEqual(prior["Chinese Painting"]["emotion_share"]["calm"], 0.5)
        self.assertEqual(prior["Ukiyo-e"]["top_emotion"], "calm")

    def test_score_calmshift_prioritizes_official_confirmed_direction(self):
        score = score_calmshift_change(
            {
                "sample_id": "track2_0001",
                "current_emotion": "content",
                "proposed_emotion": "calm",
                "support_score": 1.2,
                "model_vote_count": 2,
                "max_confidence": 0.75,
                "public_style_support_score": 0.8,
            },
            style_prior={"top_emotion": "calm", "emotion_share": {"calm": 0.72}, "positive_low_share": 0.95},
            anchor=OfficialScore("779605", 0.836408, 0.723150, 0.949667, 0.57, 0.309338),
            confirmed=OfficialScore("785979", 0.842559, 0.740034, 0.945083, 0.66, 0.320642),
        )

        self.assertEqual(score["decision"], "allow")
        self.assertEqual(score["transition"], "content->calm")
        self.assertIn("official_v21_confirmed_content_to_calm", score["reasons"])
        self.assertGreater(score["score"], 2.0)
```

- [ ] **Step 2: Run tests to verify missing functions**

Run:

```bash
python3 -m unittest tests/test_track2_v22_official_author_scorer.py -v
```

Expected: FAIL for missing `OfficialScore`, `build_style_emotion_prior`, and `score_calmshift_change`.

## Task 4: Implement Style Prior and Official Feedback Scoring

**Files:**
- Modify: `affectiveart/track2_v22_official_author_scorer.py`
- Test: `tests/test_track2_v22_official_author_scorer.py`

- [ ] **Step 1: Add score dataclass and parsing helpers**

Add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class OfficialScore:
    submission_id: str
    overall: float
    classification: float
    description: float
    emotion_accuracy: float
    emotion_macro_f1: float


def _canonical_emotion(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "contentment":
        return "content"
    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
```

- [ ] **Step 2: Add style prior builder**

Add:

```python
def build_style_emotion_prior(annotation_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    va_grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in annotation_rows:
        style = _style_from_image_path(row.get("image_path", ""))
        payload = _parse_description_payload(row.get("description"))
        emotion = _canonical_emotion(payload.get("dominant_emotion") or payload.get("emotion"))
        if not style or not emotion:
            continue
        grouped[style][emotion] += 1
        valence = str(payload.get("emotional_valence") or payload.get("valence") or "").strip().lower()
        arousal = str(payload.get("emotional_arousal_level") or payload.get("arousal") or "").strip().lower()
        if valence == "positive" and arousal == "low":
            va_grouped[style]["positive_low"] += 1
    prior: dict[str, dict[str, Any]] = {}
    for style, counts in grouped.items():
        total = sum(counts.values())
        emotion_share = {emotion: count / total for emotion, count in counts.items()}
        top_emotion, top_count = counts.most_common(1)[0]
        prior[style] = {
            "style": style,
            "total": total,
            "top_emotion": top_emotion,
            "top_emotion_share": top_count / total,
            "emotion_share": dict(sorted(emotion_share.items())),
            "positive_low_share": va_grouped[style]["positive_low"] / total if total else 0.0,
        }
    return dict(sorted(prior.items()))


def _style_from_image_path(value: Any) -> str:
    text = str(value or "").replace("\\\\", "/")
    return text.split("/", 1)[0].strip()


def _parse_description_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
```

- [ ] **Step 3: Add calmshift scoring**

Add:

```python
def score_calmshift_change(
    row: dict[str, Any],
    *,
    style_prior: dict[str, Any],
    anchor: OfficialScore,
    confirmed: OfficialScore,
) -> dict[str, Any]:
    current = _canonical_emotion(row.get("current_emotion"))
    proposed = _canonical_emotion(row.get("proposed_emotion"))
    transition = f"{current}->{proposed}"
    reasons: list[str] = []
    score = 0.0
    if transition != "content->calm":
        return {"decision": "hold", "transition": transition, "score": 0.0, "reasons": ["not_official_confirmed_direction"]}
    official_gain = confirmed.emotion_accuracy - anchor.emotion_accuracy
    if official_gain >= 0.085:
        score += 1.4
        reasons.append("official_v21_confirmed_content_to_calm")
    calm_share = _safe_float((style_prior.get("emotion_share") or {}).get("calm"))
    positive_low_share = _safe_float(style_prior.get("positive_low_share"))
    if style_prior.get("top_emotion") == "calm":
        score += 0.55
        reasons.append("author_style_top_emotion_calm")
    if calm_share >= 0.45:
        score += 0.35
        reasons.append("author_style_calm_share_high")
    if positive_low_share >= 0.80:
        score += 0.25
        reasons.append("author_style_positive_low_prior")
    score += min(0.35, _safe_float(row.get("support_score")) * 0.08)
    score += min(0.25, _safe_float(row.get("public_style_support_score")) * 0.10)
    decision = "allow" if score >= 1.8 else "hold"
    return {"decision": decision, "transition": transition, "score": round(score, 6), "reasons": reasons or ["weak_author_official_support"]}
```

- [ ] **Step 4: Run scoring tests**

Run:

```bash
python3 -m unittest tests/test_track2_v22_official_author_scorer.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add affectiveart/track2_v22_official_author_scorer.py tests/test_track2_v22_official_author_scorer.py
git commit -m "feat: add track2 v22 official author scoring"
```

## Task 5: Candidate Ladder Tests

**Files:**
- Modify: `tests/test_track2_v22_official_author_scorer.py`
- Modify later: `affectiveart/track2_v22_official_author_scorer.py`

- [ ] **Step 1: Add tests for ladder selection and output safety**

Append:

```python
from affectiveart.track2_v22_official_author_scorer import (
    build_v22_ladder_changes,
    write_v22_candidate_outputs,
)


def _submission_row(sample_id: str, emotion: str = "content") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": "Positive",
        "emotional_arousal_level": "Low",
        "overall_caption": "A quiet artwork with balanced visual rhythm and restrained emotional atmosphere.",
        "brushstroke": "Soft brushwork supports a restrained mood.",
        "composition": "Balanced composition organizes the scene.",
        "color": "Muted colors support a gentle atmosphere.",
        "line": "Controlled lines keep the visual rhythm calm.",
        "light": "Soft light reinforces the low-arousal feeling.",
    }


class Track2V22CandidateTests(unittest.TestCase):
    def test_build_v22_ladder_changes_selects_top_scored_content_to_calm(self):
        scored = [
            {"sample_id": f"track2_{idx:04d}", "current_emotion": "content", "proposed_emotion": "calm", "score": 3.0 - idx * 0.01, "decision": "allow"}
            for idx in range(10)
        ]

        selected = build_v22_ladder_changes(scored, ladder_size=5)

        self.assertEqual(len(selected), 5)
        self.assertEqual(selected[0]["sample_id"], "track2_0000")
        self.assertEqual(selected[-1]["sample_id"], "track2_0004")

    def test_write_v22_candidate_rejects_formal_submission_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                write_v22_candidate_outputs(
                    base_rows=[_submission_row("track2_0001")],
                    selected_changes=[],
                    out_json=root / "submission.json",
                    out_zip=root / "track2_submission_v22_candidate.zip",
                    report_json=root / "report.json",
                    report_md=root / "report.md",
                    ladder_name="calmshift120",
                )
```

- [ ] **Step 2: Run tests to verify missing candidate functions**

Run:

```bash
python3 -m unittest tests/test_track2_v22_official_author_scorer.py -v
```

Expected: FAIL for missing `build_v22_ladder_changes` and `write_v22_candidate_outputs`.

## Task 6: Implement Candidate Ladder Output

**Files:**
- Modify: `affectiveart/track2_v22_official_author_scorer.py`
- Test: `tests/test_track2_v22_official_author_scorer.py`

- [ ] **Step 1: Add ladder selection**

Add:

```python
def build_v22_ladder_changes(scored_rows: list[dict[str, Any]], *, ladder_size: int | str) -> list[dict[str, Any]]:
    allowed = [
        row for row in scored_rows
        if row.get("decision") == "allow" and row.get("current_emotion") == "content" and row.get("proposed_emotion") == "calm"
    ]
    allowed.sort(key=lambda row: (-_safe_float(row.get("score")), str(row.get("sample_id", ""))))
    if ladder_size == "all":
        return allowed
    return allowed[: int(ladder_size)]
```

- [ ] **Step 2: Add candidate writer**

Add:

```python
def write_v22_candidate_outputs(
    *,
    base_rows: list[dict[str, Any]],
    selected_changes: list[dict[str, Any]],
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
    ladder_name: str,
) -> dict[str, Any]:
    out_json = Path(out_json)
    out_zip = Path(out_zip)
    if out_json.name in FORMAL_SUBMISSION_NAMES or out_zip.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError("v22 candidate writer refuses formal submission filenames")
    change_by_id = {str(row["sample_id"]): row for row in selected_changes}
    output_rows: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    for row in base_rows:
        new_row = {key: row.get(key, "") for key in TRACK2_SUBMISSION_KEYS}
        change = change_by_id.get(str(new_row["sample_id"]))
        if change:
            new_row["emotion"] = "calm"
            new_row["emotional_valence"] = "Positive"
            new_row["emotional_arousal_level"] = "Low"
            applied.append(change)
        output_rows.append(new_row)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(out_json, "submission.json")
    report = {
        "method": "track2_v22_official_author_scorer",
        "ladder_name": ladder_name,
        "base_rows": len(base_rows),
        "selected_changes": len(selected_changes),
        "applied_changes": len(applied),
        "out_json": str(out_json),
        "out_zip": str(out_zip),
        "changes": applied,
    }
    Path(report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(report_md).write_text(_candidate_report_md(report), encoding="utf-8")
    return report


def _candidate_report_md(report: dict[str, Any]) -> str:
    return "\n".join([
        f"# Track2 v22 Candidate {report['ladder_name']}",
        "",
        f"- Base rows: {report['base_rows']}",
        f"- Selected changes: {report['selected_changes']}",
        f"- Applied changes: {report['applied_changes']}",
        f"- JSON: `{report['out_json']}`",
        f"- ZIP: `{report['out_zip']}`",
        "",
    ])
```

- [ ] **Step 3: Run candidate tests**

Run:

```bash
python3 -m unittest tests/test_track2_v22_official_author_scorer.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit Task 6**

Run:

```bash
git add affectiveart/track2_v22_official_author_scorer.py tests/test_track2_v22_official_author_scorer.py
git commit -m "feat: add track2 v22 candidate ladder writer"
```

## Task 7: CLI and Full Artifact Generation

**Files:**
- Create: `scripts/track2_v22_official_author_scorer.py`
- Modify: `affectiveart/track2_v22_official_author_scorer.py`
- Test: `tests/test_track2_v22_official_author_scorer.py`

- [ ] **Step 1: Add CLI script**

Create:

```python
#!/usr/bin/env python3
from __future__ import annotations

from affectiveart.track2_v22_official_author_scorer import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add `main()` with `inventory` and `build-candidates`**

Add:

```python
def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Build Track2 v22 official/author-scored candidates.")
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--emoart-root", default=str(DEFAULT_EMOART_ROOT))
    inv.add_argument("--out-dir", default="experiments/track2_v22_official_author_scorer_20260608")
    build = sub.add_parser("build-candidates")
    build.add_argument("--base-json", default="submissions/track2_submission_v15_desc_expand300_candidate.json")
    build.add_argument("--v17-evidence", default="experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json")
    build.add_argument("--emoart-root", default=str(DEFAULT_EMOART_ROOT))
    build.add_argument("--out-dir", default="experiments/track2_v22_official_author_scorer_20260608")
    build.add_argument("--submission-dir", default="submissions")
    args = parser.parse_args(argv)
    if args.command == "inventory":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inventory = discover_local_emoart_inventory(args.emoart_root)
        (out_dir / "local_emoart130k_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.command == "build-candidates":
        build_v22_candidate_suite(
            base_json=Path(args.base_json),
            v17_evidence=Path(args.v17_evidence),
            emoart_root=Path(args.emoart_root),
            out_dir=Path(args.out_dir),
            submission_dir=Path(args.submission_dir),
        )
        return 0
    return 2
```

- [ ] **Step 3: Add suite builder**

Add:

```python
def build_v22_candidate_suite(
    *,
    base_json: Path,
    v17_evidence: Path,
    emoart_root: Path,
    out_dir: Path,
    submission_dir: Path,
) -> dict[str, Any]:
    base_rows = _load_json_list(base_json)
    evidence_rows = _load_json_list(v17_evidence)
    annotation_rows = _load_json_list(emoart_root / "Annotation.json")
    prior = build_style_emotion_prior(annotation_rows)
    anchor = OfficialScore("779605", 0.836408, 0.723150, 0.949667, 0.570000, 0.309338)
    confirmed = OfficialScore("785979", 0.842559, 0.740034, 0.945083, 0.660000, 0.320642)
    scored = []
    for row in evidence_rows:
        style_key = str(row.get("public_style") or row.get("style") or row.get("style_family") or "").strip()
        style_prior = prior.get(style_key, {})
        scored_row = dict(row)
        scored_row.update(score_calmshift_change(row, style_prior=style_prior, anchor=anchor, confirmed=confirmed))
        scored.append(scored_row)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v22_scored_evidence.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reports = []
    for ladder_name, ladder_size in [("calmshift120", 120), ("calmshift150", 150), ("calmshiftall", "all")]:
        selected = build_v22_ladder_changes(scored, ladder_size=ladder_size)
        report = write_v22_candidate_outputs(
            base_rows=base_rows,
            selected_changes=selected,
            out_json=submission_dir / f"track2_submission_v22_official_author_{ladder_name}_candidate.json",
            out_zip=submission_dir / f"track2_submission_v22_official_author_{ladder_name}_candidate.zip",
            report_json=out_dir / f"{ladder_name}_candidate_report.json",
            report_md=out_dir / f"{ladder_name}_candidate_report.md",
            ladder_name=ladder_name,
        )
        reports.append(report)
    scoreboard = {"method": "track2_v22_official_author_scorer", "candidate_reports": reports}
    (out_dir / "v22_ladder_scoreboard.json").write_text(json.dumps(scoreboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "v22_ladder_scoreboard.md").write_text(_scoreboard_md(scoreboard), encoding="utf-8")
    return scoreboard


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _scoreboard_md(scoreboard: dict[str, Any]) -> str:
    lines = ["# Track2 v22 Official/Author Ladder Scoreboard", ""]
    lines.append("| candidate | selected | zip |")
    lines.append("| --- | ---: | --- |")
    for report in scoreboard["candidate_reports"]:
        lines.append(f"| {report['ladder_name']} | {report['selected_changes']} | `{report['out_zip']}` |")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python3 -m unittest tests/test_track2_v22_official_author_scorer.py -v
```

Expected: PASS.

- [ ] **Step 5: Run inventory**

Run:

```bash
python3 scripts/track2_v22_official_author_scorer.py inventory \
  --emoart-root /Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k \
  --out-dir experiments/track2_v22_official_author_scorer_20260608
```

Expected:

- `local_emoart130k_inventory.json` exists.
- It reports 56 tar files and more than 130000 annotation rows.

- [ ] **Step 6: Run candidate build**

Run:

```bash
python3 scripts/track2_v22_official_author_scorer.py build-candidates \
  --base-json submissions/track2_submission_v15_desc_expand300_candidate.json \
  --v17-evidence experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json \
  --emoart-root /Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k \
  --out-dir experiments/track2_v22_official_author_scorer_20260608 \
  --submission-dir submissions
```

Expected:

- Three v22 candidate JSON files.
- Three v22 candidate ZIP files.
- A ladder scoreboard under `experiments/track2_v22_official_author_scorer_20260608/`.

- [ ] **Step 7: Commit Task 7**

Run:

```bash
git add affectiveart/track2_v22_official_author_scorer.py scripts/track2_v22_official_author_scorer.py tests/test_track2_v22_official_author_scorer.py experiments/track2_v22_official_author_scorer_20260608 submissions/track2_submission_v22_official_author_*_candidate.json submissions/track2_submission_v22_official_author_*_candidate.zip
git commit -m "feat: build track2 v22 official author candidates"
```

## Task 8: Final Validation and Recommendation

**Files:**
- Read/write generated artifacts only.
- Do not overwrite `submissions/track2_submission.json`.
- Do not upload to Codabench.

- [ ] **Step 1: Validate all candidate JSON files**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v22_official_author_calmshift120_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v22_official_author_calmshift150_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v22_official_author_calmshiftall_candidate.json
```

Expected: each validator exits 0.

- [ ] **Step 2: Run Track2-only tests**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_track2_v22_official_author_scorer.py' -v
python3 -m unittest discover -s tests -p 'test_track2_v21_championship_recalibration.py' -v
```

Expected: PASS.

- [ ] **Step 3: Write final recommendation**

Create `experiments/track2_v22_official_author_scorer_20260608/v22_final_recommendation_zh.md` with:

```markdown
# Track2 v22 Final Recommendation

## Recommendation

Submit exactly one ZIP:

`submissions/track2_submission_v22_official_author_<selected>_candidate.zip`

## Why

- The selected candidate preserves the official v21-confirmed `content->calm` direction.
- It uses only official/author evidence for label arbitration.
- It avoids broad cross-quadrant changes.
- It keeps submission format safe and does not overwrite formal package names.

## Risk

- This is still a proxy decision, not hidden-gold reconstruction.
- The final submission should not be repeated or combined with unvalidated manual edits.
```

Replace `<selected>` with the chosen ladder after reviewing the scoreboard. Preferred default is `calmshift150` if `calmshiftall` is much larger and less bounded; otherwise use the highest scored ladder with no format or distribution risk.

- [ ] **Step 4: Run final diff check**

Run:

```bash
git diff --check -- affectiveart/track2_v22_official_author_scorer.py scripts/track2_v22_official_author_scorer.py tests/test_track2_v22_official_author_scorer.py experiments/track2_v22_official_author_scorer_20260608
```

Expected: no output.

- [ ] **Step 5: Commit final recommendation**

Run:

```bash
git add experiments/track2_v22_official_author_scorer_20260608/v22_final_recommendation_zh.md
git commit -m "docs: recommend track2 v22 final candidate"
```
