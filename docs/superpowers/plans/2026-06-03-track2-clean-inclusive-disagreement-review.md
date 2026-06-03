# Track2 Clean/Inclusive Disagreement Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track2-only HTML/CSV review packet for clean-vs-inclusive public-style prediction disagreements, without writing a candidate submission or touching the formal package.

**Architecture:** Add a focused review module that derives disagreement rows from current Track2 submission rows plus clean/inclusive prediction payloads. Add a CLI that copies test images into local HTML assets, writes a review HTML page, writes an editable decision CSV, and emits a JSON/Markdown report. Keep candidate generation out of scope for this plan.

**Tech Stack:** Python stdlib, existing Track2 JSON format, existing `affectiveart.track2_visual_audit.find_track2_image_member`, unittest, local static HTML served by existing browser workflow.

---

## File Structure

- Create `affectiveart/track2_disagreement_review.py`
  - Pure functions for loading/indexing rows.
  - Builds clean/inclusive disagreement review rows.
  - Renders decision CSV, JSON/Markdown report, and review HTML.
  - Extracts image assets from `data/raw/Track2_testset.zip`.
- Create `scripts/build_track2_clean_inclusive_disagreement_review.py`
  - Thin CLI wrapper around the module.
  - Defaults to the existing 20260603 public-style outputs.
- Create `tests/test_track2_disagreement_review.py`
  - Synthetic tests for disagreement selection, CSV columns, and HTML output.
- Output path at runtime:
  - `experiments/track2_final_push_20260603/clean_inclusive_disagreement/html_review/track2_clean_inclusive_disagreement_review.html`
  - `experiments/track2_final_push_20260603/clean_inclusive_disagreement/track2_clean_inclusive_disagreement_decisions.csv`
  - `experiments/track2_final_push_20260603/clean_inclusive_disagreement/track2_clean_inclusive_disagreement_report.json`
  - `experiments/track2_final_push_20260603/clean_inclusive_disagreement/track2_clean_inclusive_disagreement_report.md`

## Task 1: Build Disagreement Row Model

**Files:**
- Create: `affectiveart/track2_disagreement_review.py`
- Test: `tests/test_track2_disagreement_review.py`

- [ ] **Step 1: Write failing tests for row selection**

Add this test file:

```python
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from affectiveart.track2_disagreement_review import (
    DECISION_COLUMNS,
    build_disagreement_review_rows,
    write_disagreement_review_outputs,
)


def current_row(sample_id, emotion="content", valence="Positive", arousal="Low"):
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": f"{sample_id} caption",
        "brushstroke": "brush",
        "composition": "composition",
        "color": "color",
        "line": "line",
        "light": "light",
    }


def pred(sample_id, current, emotion, confidence=0.8, margin=0.2, knn=None):
    return {
        "sample_id": sample_id,
        "current": current,
        "emotion": emotion,
        "confidence": confidence,
        "margin": margin,
        "knn_emotion": knn or emotion,
        "knn_confidence": 0.7,
        "top3": [{"emotion": emotion, "probability": confidence}],
        "knn_top3": [{"emotion": knn or emotion, "votes": 10, "similarity_sum": 9.1}],
    }


class Track2DisagreementReviewTest(unittest.TestCase):
    def test_build_rows_selects_clean_inclusive_emotion_disagreements_only(self):
        current = [
            current_row("track2_keep", "calm"),
            current_row("track2_inclusive_only", "content"),
            current_row("track2_clean_only", "content"),
            current_row("track2_conflict", "content"),
            current_row("track2_agree_change", "content"),
        ]
        clean = {
            "entries": [
                pred("track2_keep", "calm", "calm"),
                pred("track2_inclusive_only", "content", "content"),
                pred("track2_clean_only", "content", "calm"),
                pred("track2_conflict", "content", "calm"),
                pred("track2_agree_change", "content", "calm"),
            ]
        }
        inclusive = {
            "entries": [
                pred("track2_keep", "calm", "calm"),
                pred("track2_inclusive_only", "content", "calm"),
                pred("track2_clean_only", "content", "content"),
                pred("track2_conflict", "content", "tired"),
                pred("track2_agree_change", "content", "calm"),
            ]
        }

        rows = build_disagreement_review_rows(
            current,
            clean,
            inclusive,
            high_similarity_sample_ids={"track2_inclusive_only"},
            image_members={
                "track2_inclusive_only": "images/track2_inclusive_only.jpg",
                "track2_clean_only": "images/track2_clean_only.jpg",
                "track2_conflict": "images/track2_conflict.jpg",
            },
        )

        self.assertEqual([row["sample_id"] for row in rows], [
            "track2_inclusive_only",
            "track2_clean_only",
            "track2_conflict",
        ])
        by_id = {row["sample_id"]: row for row in rows}
        self.assertEqual(by_id["track2_inclusive_only"]["category"], "inclusive_only_change")
        self.assertEqual(by_id["track2_clean_only"]["category"], "clean_only_change")
        self.assertEqual(by_id["track2_conflict"]["category"], "clean_inclusive_conflict")
        self.assertTrue(by_id["track2_inclusive_only"]["high_similarity_public_reference"])
        self.assertEqual(by_id["track2_inclusive_only"]["recommended_decision"], "hold")
        self.assertEqual(by_id["track2_inclusive_only"]["manual_decision"], "")
        self.assertEqual(by_id["track2_inclusive_only"]["reviewer_rationale"], "")

    def test_write_outputs_writes_html_csv_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            current_json = tmp_path / "current.json"
            clean_json = tmp_path / "clean.json"
            inclusive_json = tmp_path / "inclusive.json"
            image_zip = tmp_path / "track2.zip"
            out_dir = tmp_path / "out"
            current_json.write_text(json.dumps([
                current_row("track2_0001", "content"),
                current_row("track2_0002", "content"),
            ]), encoding="utf-8")
            clean_json.write_text(json.dumps({"entries": [
                pred("track2_0001", "content", "content"),
                pred("track2_0002", "content", "calm"),
            ]}), encoding="utf-8")
            inclusive_json.write_text(json.dumps({"entries": [
                pred("track2_0001", "content", "calm"),
                pred("track2_0002", "content", "content"),
            ]}), encoding="utf-8")
            img1 = tmp_path / "track2_0001.jpg"
            img2 = tmp_path / "track2_0002.jpg"
            Image.new("RGB", (64, 48), (200, 100, 20)).save(img1)
            Image.new("RGB", (64, 48), (20, 100, 200)).save(img2)
            with zipfile.ZipFile(image_zip, "w") as zf:
                zf.write(img1, "images/track2_0001.jpg")
                zf.write(img2, "track2_testset/images/track2_0002.jpg")

            report = write_disagreement_review_outputs(
                current_json=current_json,
                clean_predictions_json=clean_json,
                inclusive_predictions_json=inclusive_json,
                image_zip=image_zip,
                out_dir=out_dir,
                high_similarity_sample_ids={"track2_0001"},
            )

            self.assertEqual(report["row_count"], 2)
            self.assertEqual(report["category_counts"]["inclusive_only_change"], 1)
            self.assertEqual(report["category_counts"]["clean_only_change"], 1)
            self.assertTrue(Path(report["outputs"]["html"]).exists())
            self.assertTrue(Path(report["outputs"]["csv"]).exists())
            self.assertTrue(Path(report["outputs"]["json"]).exists())
            self.assertTrue(Path(report["outputs"]["markdown"]).exists())
            html_text = Path(report["outputs"]["html"]).read_text(encoding="utf-8")
            self.assertIn("Track2 Clean/Inclusive Disagreement Review", html_text)
            self.assertIn("track2_0001", html_text)
            csv_text = Path(report["outputs"]["csv"]).read_text(encoding="utf-8")
            self.assertIn(",".join(DECISION_COLUMNS), csv_text.splitlines()[0])
            self.assertTrue((out_dir / "html_review" / "assets" / "track2_0001.jpg").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_track2_disagreement_review -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'affectiveart.track2_disagreement_review'`.

- [ ] **Step 3: Implement row building and output writer**

Create `affectiveart/track2_disagreement_review.py` with these interfaces:

```python
from __future__ import annotations

import csv
import html
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from affectiveart.track2_visual_audit import find_track2_image_member


DECISION_COLUMNS = [
    "sample_id",
    "category",
    "current_emotion",
    "current_valence",
    "current_arousal",
    "clean_emotion",
    "clean_confidence",
    "clean_margin",
    "clean_knn_emotion",
    "clean_knn_confidence",
    "inclusive_emotion",
    "inclusive_confidence",
    "inclusive_margin",
    "inclusive_knn_emotion",
    "inclusive_knn_confidence",
    "high_similarity_public_reference",
    "recommended_decision",
    "manual_decision",
    "reviewer_rationale",
]


def build_disagreement_review_rows(
    current_rows: list[dict[str, Any]],
    clean_payload: dict[str, Any],
    inclusive_payload: dict[str, Any],
    *,
    high_similarity_sample_ids: set[str] | None = None,
    image_members: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    high_similarity_sample_ids = high_similarity_sample_ids or set()
    image_members = image_members or {}
    current_by_id = _index_current_rows(current_rows)
    clean_by_id = _index_prediction_entries(clean_payload)
    inclusive_by_id = _index_prediction_entries(inclusive_payload)
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(current_by_id):
        clean = clean_by_id.get(sample_id)
        inclusive = inclusive_by_id.get(sample_id)
        if not clean or not inclusive:
            continue
        clean_emotion = str(clean.get("emotion", ""))
        inclusive_emotion = str(inclusive.get("emotion", ""))
        if clean_emotion == inclusive_emotion:
            continue
        current = current_by_id[sample_id]
        current_emotion = str(current.get("emotion", ""))
        row = {
            "sample_id": sample_id,
            "category": _category(current_emotion, clean_emotion, inclusive_emotion),
            "current_emotion": current_emotion,
            "current_valence": str(current.get("emotional_valence", "")),
            "current_arousal": str(current.get("emotional_arousal_level", "")),
            "clean": _prediction_summary(clean),
            "inclusive": _prediction_summary(inclusive),
            "high_similarity_public_reference": sample_id in high_similarity_sample_ids,
            "image_member": image_members.get(sample_id, ""),
            "recommended_decision": "hold",
            "manual_decision": "",
            "reviewer_rationale": "",
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (_category_rank(row["category"]), row["sample_id"]))


def write_disagreement_review_outputs(
    *,
    current_json: str | Path,
    clean_predictions_json: str | Path,
    inclusive_predictions_json: str | Path,
    image_zip: str | Path,
    out_dir: str | Path,
    high_similarity_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    current_rows = _read_json_list(current_json)
    clean_payload = _read_json_object(clean_predictions_json)
    inclusive_payload = _read_json_object(inclusive_predictions_json)
    out_dir = Path(out_dir)
    html_dir = out_dir / "html_review"
    assets_dir = html_dir / "assets"
    html_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_members = _image_members_by_sample_id(image_zip, [str(row["sample_id"]) for row in current_rows])
    rows = build_disagreement_review_rows(
        current_rows,
        clean_payload,
        inclusive_payload,
        high_similarity_sample_ids=high_similarity_sample_ids,
        image_members=image_members,
    )
    _extract_assets(image_zip, rows, assets_dir)
    report = _report(rows, current_json, clean_predictions_json, inclusive_predictions_json, image_zip, out_dir)
    _write_decision_csv(Path(report["outputs"]["csv"]), rows)
    Path(report["outputs"]["html"]).write_text(render_html(rows, report), encoding="utf-8")
    Path(report["outputs"]["json"]).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(report["outputs"]["markdown"]).write_text(render_markdown(report), encoding="utf-8")
    return report
```

The implementation must also include private helpers used by these functions:

```python
def _category(current: str, clean: str, inclusive: str) -> str:
    clean_changes = clean != current
    inclusive_changes = inclusive != current
    if inclusive_changes and not clean_changes:
        return "inclusive_only_change"
    if clean_changes and not inclusive_changes:
        return "clean_only_change"
    return "clean_inclusive_conflict"


def _prediction_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "emotion": str(row.get("emotion", "")),
        "confidence": float(row.get("confidence", 0.0)),
        "margin": float(row.get("margin", 0.0)),
        "knn_emotion": str(row.get("knn_emotion", "")),
        "knn_confidence": float(row.get("knn_confidence", 0.0)),
        "top3": list(row.get("top3", [])),
        "knn_top3": list(row.get("knn_top3", [])),
    }


def _category_rank(category: str) -> int:
    return {
        "inclusive_only_change": 0,
        "clean_only_change": 1,
        "clean_inclusive_conflict": 2,
    }.get(category, 99)
```

Render HTML with:

- Sticky header and summary counts.
- One card per disagreement row.
- Image, current label, clean prediction, inclusive prediction.
- SOP block for `calm/content` and `frustrated/aroused`.
- Empty manual decision/rationale controls for human review.
- Filter buttons for all, inclusive-only, clean-only, conflict, high-similarity.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests.test_track2_disagreement_review -v
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 5: Commit task 1**

Run:

```bash
git add affectiveart/track2_disagreement_review.py tests/test_track2_disagreement_review.py
git commit -m "feat: build track2 disagreement review rows"
```

## Task 2: Add CLI Wrapper

**Files:**
- Create: `scripts/build_track2_clean_inclusive_disagreement_review.py`
- Modify: `tests/test_track2_disagreement_review.py`

- [ ] **Step 1: Add CLI smoke test**

Append this test to `Track2DisagreementReviewTest`:

```python
    def test_cli_writes_review_outputs(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            current_json = tmp_path / "current.json"
            clean_json = tmp_path / "clean.json"
            inclusive_json = tmp_path / "inclusive.json"
            image_zip = tmp_path / "track2.zip"
            out_dir = tmp_path / "out"
            current_json.write_text(json.dumps([current_row("track2_0001", "content")]), encoding="utf-8")
            clean_json.write_text(json.dumps({"entries": [pred("track2_0001", "content", "content")]}), encoding="utf-8")
            inclusive_json.write_text(json.dumps({"entries": [pred("track2_0001", "content", "calm")]}), encoding="utf-8")
            img1 = tmp_path / "track2_0001.jpg"
            Image.new("RGB", (64, 48), (200, 100, 20)).save(img1)
            with zipfile.ZipFile(image_zip, "w") as zf:
                zf.write(img1, "images/track2_0001.jpg")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_track2_clean_inclusive_disagreement_review.py",
                    "--current-json",
                    str(current_json),
                    "--clean-predictions-json",
                    str(clean_json),
                    "--inclusive-predictions-json",
                    str(inclusive_json),
                    "--image-zip",
                    str(image_zip),
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
            )

            self.assertIn("track2_clean_inclusive_disagreement_review.html", result.stdout)
            self.assertTrue((out_dir / "html_review" / "track2_clean_inclusive_disagreement_review.html").exists())
            self.assertTrue((out_dir / "track2_clean_inclusive_disagreement_decisions.csv").exists())
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_track2_disagreement_review.Track2DisagreementReviewTest.test_cli_writes_review_outputs -v
```

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Create CLI script**

Create `scripts/build_track2_clean_inclusive_disagreement_review.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_disagreement_review import write_disagreement_review_outputs
from affectiveart.track2_public_style_distillation import load_high_similarity_sample_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track2 clean/inclusive disagreement HTML review packet.")
    parser.add_argument("--current-json", type=Path, default=Path("submissions/track2_submission.json"))
    parser.add_argument(
        "--clean-predictions-json",
        type=Path,
        default=Path("experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/clean_predictions.json"),
    )
    parser.add_argument(
        "--inclusive-predictions-json",
        type=Path,
        default=Path("experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/inclusive_predictions.json"),
    )
    parser.add_argument(
        "--high-similarity-json",
        type=Path,
        default=Path("experiments/track2_public_style_distillation_20260603/public_leak_exclusion_manifest.json"),
    )
    parser.add_argument("--image-zip", type=Path, default=Path("data/raw/Track2_testset.zip"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/track2_final_push_20260603/clean_inclusive_disagreement"),
    )
    args = parser.parse_args()

    high_similarity_ids = load_high_similarity_sample_ids(args.high_similarity_json)
    report = write_disagreement_review_outputs(
        current_json=args.current_json,
        clean_predictions_json=args.clean_predictions_json,
        inclusive_predictions_json=args.inclusive_predictions_json,
        image_zip=args.image_zip,
        out_dir=args.out_dir,
        high_similarity_sample_ids=high_similarity_ids,
    )
    print(json.dumps({
        "row_count": report["row_count"],
        "outputs": report["outputs"],
        "category_counts": report["category_counts"],
        "high_similarity_count": report["high_similarity_count"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI smoke test**

Run:

```bash
python3 -m unittest tests.test_track2_disagreement_review.Track2DisagreementReviewTest.test_cli_writes_review_outputs -v
```

Expected: PASS.

- [ ] **Step 5: Commit task 2**

Run:

```bash
git add scripts/build_track2_clean_inclusive_disagreement_review.py tests/test_track2_disagreement_review.py
git commit -m "feat: add track2 disagreement review cli"
```

## Task 3: Generate Real Review Packet

**Files:**
- Runtime outputs under `experiments/track2_final_push_20260603/clean_inclusive_disagreement/`

- [ ] **Step 1: Run the real CLI**

Run:

```bash
python3 scripts/build_track2_clean_inclusive_disagreement_review.py
```

Expected JSON stdout:

- `row_count` is `56`.
- `category_counts` includes `inclusive_only_change: 24`.
- `outputs.html` points to `experiments/track2_final_push_20260603/clean_inclusive_disagreement/html_review/track2_clean_inclusive_disagreement_review.html`.

- [ ] **Step 2: Check output files**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
base = Path("experiments/track2_final_push_20260603/clean_inclusive_disagreement")
files = [
    base / "html_review" / "track2_clean_inclusive_disagreement_review.html",
    base / "track2_clean_inclusive_disagreement_decisions.csv",
    base / "track2_clean_inclusive_disagreement_report.json",
    base / "track2_clean_inclusive_disagreement_report.md",
]
for path in files:
    print(path, path.exists(), path.stat().st_size if path.exists() else 0)
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(1)
PY
```

Expected: all files exist and have non-zero size.

- [ ] **Step 3: Open in browser or local server**

If a local server is already running for review pages, reuse it. Otherwise run:

```bash
cd experiments/track2_final_push_20260603/clean_inclusive_disagreement/html_review
python3 -m http.server 8772 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8772/track2_clean_inclusive_disagreement_review.html
```

Expected: the page shows 56 review cards, image assets render, and filters work.

- [ ] **Step 4: Do not create a candidate**

Confirm no file matching this pattern exists:

```bash
ls submissions/track2_submission_clean_inclusive_review_candidate.* 2>/dev/null && exit 1 || exit 0
```

Expected: exit code `0`.

- [ ] **Step 5: Commit task 3 outputs if review packet is acceptable**

Run:

```bash
git add experiments/track2_final_push_20260603/clean_inclusive_disagreement
git commit -m "review: add track2 clean inclusive disagreement packet"
```

If the experiment directory is intentionally kept untracked, skip this commit and report the exact output paths.

## Task 4: Final Verification

**Files:**
- All files from Tasks 1-3.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_track2_disagreement_review -v
```

Expected: all tests pass.

- [ ] **Step 2: Run Track2-only regression tests**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
```

Expected: all Track2 tests pass. Existing SWIG deprecation warnings are acceptable.

- [ ] **Step 3: Compile new script and module**

Run:

```bash
python3 -m py_compile affectiveart/track2_disagreement_review.py scripts/build_track2_clean_inclusive_disagreement_review.py
```

Expected: exit code `0`.

- [ ] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code `0`.

- [ ] **Step 5: Final handoff**

Report:

- HTML review URL or file path.
- Decision CSV path.
- JSON/Markdown report paths.
- Exact row count and category counts.
- Confirmation that no formal submission package was overwritten.
- Confirmation that no candidate JSON/ZIP was produced by this plan.
