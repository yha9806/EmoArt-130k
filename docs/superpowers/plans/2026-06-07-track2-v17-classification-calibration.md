# Track2 v17 Classification Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track2-only v17 classification calibration experiment that creates local safe/balanced/aggressive emotion-label candidates on top of `v15_desc_expand300` and gates them before any Codabench submission.

**Architecture:** v17 builds a normalized evidence matrix from existing model predictions, public-reference audits, and official feedback; applies strict candidate selection rules; writes side-path candidate JSON/ZIP files; and runs local/fused shadow scoring reports. It does not rewrite description fields, touch Track1, or overwrite formal submission files.

**Tech Stack:** Python standard library, existing `affectiveart.challenge`, existing Track2 schema helpers, existing local/fused shadow evaluator CLIs, `unittest`.

---

## File Structure

- Create `affectiveart/track2_v17_classification_calibration.py`
  - Owns v17 dataclasses, prediction loading, evidence aggregation, gate logic, candidate writing, and Markdown reports.
- Create `scripts/track2_v17_classification_calibration.py`
  - Thin CLI wrapper around the module, matching existing script style.
- Create `tests/test_track2_v17_classification_calibration.py`
  - Unit tests for evidence aggregation, official-feedback penalties, gates, writer safety, valence/arousal repair, and CLI smoke behavior.
- Output under `experiments/track2_v17_classification_calibration_20260607/`
  - `evidence_matrix.csv/.json`
  - `candidate_reports/*.json/.md`
  - `local_shadow_compare/*`
  - `fused_shadow_compare/*`
  - `final_gate_report.md`
- Candidate files under `submissions/`
  - `track2_submission_v17_safe_candidate.json/.zip`
  - `track2_submission_v17_balanced_candidate.json/.zip`
  - `track2_submission_v17_aggressive_probe_candidate.json/.zip`

## Defaults

Use these default inputs:

```text
base_json=submissions/track2_submission_v15_desc_expand300_candidate.json
official_scores=experiments/track2_official_results_20260606/track2_known_official_exact_scores_from_ledger_20260606.csv
pairwise_diffs=experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv
siglip2=experiments/track2_emoart130k_siglip2/predictions.json
clip=experiments/track2_emoart130k_clip/predictions.json
dinov2=experiments/track2_emoart130k_dinov2/predictions.json
gemini35=experiments/track2_moe_specialist_ensemble_20260603/dry_run_v2/gemini35_vlm_specialist_predictions.json
hard96_selective=experiments/track2_macro_f1_champion_20260511/selective_v2/hard96_selective_predictions.json
public_clean=experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/clean_predictions.json
public_inclusive=experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/inclusive_predictions.json
duplicate_audit=experiments/track2_emoart130k_clip/deep_duplicate_audit_20260511/track2_deep_duplicate_top10_audit.json
neighbor_audit=experiments/track2_emoart130k_clip/overlap_reference/ge095_all/ge095_public_neighbor_audit.json
```

---

## Task 1: Evidence Matrix Core

**Files:**
- Create: `affectiveart/track2_v17_classification_calibration.py`
- Create: `scripts/track2_v17_classification_calibration.py`
- Create: `tests/test_track2_v17_classification_calibration.py`

- [ ] **Step 1: Write failing tests for prediction normalization and aggregation**

Add this test file:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_v17_classification_calibration import (
    build_evidence_rows,
    load_prediction_sources,
    parse_official_failed_transition_counts,
)


class Track2V17ClassificationCalibrationTests(unittest.TestCase):
    def test_load_prediction_sources_accepts_list_and_entries_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            list_path = root / "list.json"
            entries_path = root / "entries.json"
            list_path.write_text(
                json.dumps([
                    {"sample_id": "track2_0001", "emotion": "calm", "confidence": 0.8, "margin": 0.2}
                ]),
                encoding="utf-8",
            )
            entries_path.write_text(
                json.dumps({
                    "entries": [
                        {"sample_id": "track2_0001", "target_emotion": "content", "confidence": 0.7}
                    ]
                }),
                encoding="utf-8",
            )
            rows = load_prediction_sources({"clip": list_path, "siglip2": entries_path})
        self.assertEqual(len(rows["track2_0001"]), 2)
        self.assertEqual({row["source"] for row in rows["track2_0001"]}, {"clip", "siglip2"})

    def test_parse_official_failed_transition_counts_uses_negative_batches_only(self) -> None:
        rows = [
            {
                "candidate_a": "781601_failed",
                "candidate_b": "779605_anchor",
                "emotion_label_changes": "85",
                "top_emotion_transitions": "calm->content:43; content->calm:20",
            }
        ]
        scores = {
            "779605": {"classification": 0.723150},
            "781601": {"classification": 0.719137},
        }
        counts = parse_official_failed_transition_counts(pairwise_rows=rows, score_rows=scores)
        self.assertEqual(counts["calm->content"], 43)
        self.assertEqual(counts["content->calm"], 20)

    def test_build_evidence_rows_groups_model_votes_by_proposed_label(self) -> None:
        base_rows = [
            {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
            }
        ]
        predictions = {
            "track2_0001": [
                {"source": "clip", "emotion": "calm", "confidence": 0.9, "margin": 0.3},
                {"source": "siglip2", "emotion": "calm", "confidence": 0.8, "margin": 0.1},
                {"source": "dinov2", "emotion": "content", "confidence": 0.7, "margin": 0.1},
            ]
        }
        evidence = build_evidence_rows(
            base_rows=base_rows,
            predictions_by_sample=predictions,
            duplicate_rows=[],
            failed_transition_counts={},
        )
        calm = [row for row in evidence if row["proposed_emotion"] == "calm"][0]
        self.assertEqual(calm["transition"], "content->calm")
        self.assertEqual(calm["model_vote_count"], 2)
        self.assertEqual(calm["model_sources"], "clip,siglip2")
        self.assertGreater(calm["support_score"], 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest tests/test_track2_v17_classification_calibration.py -v
```

Expected: import error for missing `affectiveart.track2_v17_classification_calibration`.

- [ ] **Step 3: Implement dataclasses and loaders**

Create `affectiveart/track2_v17_classification_calibration.py` with:

```python
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_EMOTIONS, TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_audit import HIGH_AROUSAL_EMOTIONS, NEGATIVE_EMOTIONS, strict_track2_label_issues


FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}
DEFAULT_EXPERIMENT_DIR = Path("experiments/track2_v17_classification_calibration_20260607")
DEFAULT_BASE_JSON = Path("submissions/track2_submission_v15_desc_expand300_candidate.json")
DEFAULT_OFFICIAL_SCORES = Path(
    "experiments/track2_official_results_20260606/track2_known_official_exact_scores_from_ledger_20260606.csv"
)
DEFAULT_PAIRWISE_DIFFS = Path(
    "experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv"
)
DEFAULT_PREDICTION_SOURCES = {
    "siglip2": Path("experiments/track2_emoart130k_siglip2/predictions.json"),
    "clip": Path("experiments/track2_emoart130k_clip/predictions.json"),
    "dinov2": Path("experiments/track2_emoart130k_dinov2/predictions.json"),
    "gemini35": Path("experiments/track2_moe_specialist_ensemble_20260603/dry_run_v2/gemini35_vlm_specialist_predictions.json"),
    "hard96_selective": Path("experiments/track2_macro_f1_champion_20260511/selective_v2/hard96_selective_predictions.json"),
    "public_clean": Path("experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/clean_predictions.json"),
    "public_inclusive": Path("experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/inclusive_predictions.json"),
}
DEFAULT_DUPLICATE_SOURCES = [
    Path("experiments/track2_emoart130k_clip/deep_duplicate_audit_20260511/track2_deep_duplicate_top10_audit.json"),
    Path("experiments/track2_emoart130k_clip/overlap_reference/ge095_all/ge095_public_neighbor_audit.json"),
]


def load_prediction_sources(paths: dict[str, str | Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("entries", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            sample_id = str(row.get("sample_id") or row.get("request_id") or "").strip()
            emotion = _canonical_emotion(row.get("target_emotion") or row.get("emotion"))
            if not sample_id or not emotion:
                continue
            grouped[sample_id].append(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "emotion": emotion,
                    "confidence": _safe_float(row.get("confidence")),
                    "margin": _safe_float(row.get("margin")),
                    "decision": str(row.get("decision", "")).strip().lower(),
                    "rationale": str(row.get("rationale") or row.get("reason") or "").strip(),
                }
            )
    return dict(grouped)


def parse_official_failed_transition_counts(
    *,
    pairwise_rows: list[dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in pairwise_rows:
        candidate_id = _extract_submission_id(row.get("candidate_a"))
        anchor_id = _extract_submission_id(row.get("candidate_b"))
        if candidate_id not in score_rows or anchor_id not in score_rows:
            continue
        candidate_class = _safe_float(score_rows[candidate_id].get("classification"))
        anchor_class = _safe_float(score_rows[anchor_id].get("classification"))
        if candidate_class >= anchor_class:
            continue
        for transition, count in _parse_transition_counts(row.get("top_emotion_transitions", "")).items():
            counts[transition] += count
    return dict(sorted(counts.items()))
```

- [ ] **Step 4: Implement evidence aggregation**

Append:

```python
def build_evidence_rows(
    *,
    base_rows: list[dict[str, Any]],
    predictions_by_sample: dict[str, list[dict[str, Any]]],
    duplicate_rows: list[dict[str, Any]],
    failed_transition_counts: dict[str, int],
) -> list[dict[str, Any]]:
    duplicates_by_id = _index_rows_by_sample_id(duplicate_rows)
    evidence_rows: list[dict[str, Any]] = []
    for base in base_rows:
        sample_id = str(base.get("sample_id", "")).strip()
        current = _canonical_emotion(base.get("emotion"))
        if not sample_id or not current:
            continue
        by_emotion: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions_by_sample.get(sample_id, []):
            emotion = _canonical_emotion(prediction.get("emotion"))
            if emotion and emotion != current:
                by_emotion[emotion].append(prediction)
        duplicate_payloads = _duplicate_payloads_for_sample(duplicates_by_id.get(sample_id, []), current=current)
        for emotion, payloads in duplicate_payloads.items():
            by_emotion[emotion].extend(payloads)
        for proposed, votes in sorted(by_emotion.items()):
            transition = f"{current}->{proposed}"
            sources = sorted({str(vote.get("source", "")) for vote in votes if str(vote.get("source", "")).strip()})
            model_sources = sorted(source for source in sources if not source.startswith("public_"))
            exact_duplicate = any(vote.get("evidence_type") == "exact_public_duplicate" for vote in votes)
            near_duplicate = any(vote.get("evidence_type") == "near_public_duplicate" for vote in votes)
            support_score = sum(_safe_float(vote.get("confidence")) + 0.25 * max(0.0, _safe_float(vote.get("margin"))) for vote in votes)
            evidence_rows.append(
                {
                    "sample_id": sample_id,
                    "current_emotion": current,
                    "proposed_emotion": proposed,
                    "transition": transition,
                    "same_valence": _valence(current) == _valence(proposed),
                    "same_arousal": _arousal(current) == _arousal(proposed),
                    "model_vote_count": len(model_sources),
                    "model_sources": ",".join(model_sources),
                    "all_sources": ",".join(sources),
                    "support_score": round(float(support_score), 6),
                    "max_confidence": round(max([_safe_float(vote.get("confidence")) for vote in votes] or [0.0]), 6),
                    "exact_duplicate": exact_duplicate,
                    "near_duplicate": near_duplicate,
                    "failed_transition_count": int(failed_transition_counts.get(transition, 0)),
                    "rationale": " | ".join(str(vote.get("rationale", "")).strip() for vote in votes if str(vote.get("rationale", "")).strip())[:800],
                }
            )
    return sorted(
        evidence_rows,
        key=lambda row: (-float(row["support_score"]), int(row["failed_transition_count"]), row["sample_id"], row["proposed_emotion"]),
    )
```

- [ ] **Step 5: Implement helpers used by Task 1**

Append:

```python
def load_track2_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected Track2 JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_official_score_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in load_csv_rows(path):
        submission_id = str(row.get("submission_id", "")).strip()
        if submission_id:
            rows[submission_id] = {
                "overall": _safe_float(row.get("official_overall")),
                "classification": _safe_float(row.get("official_classification")),
                "description": _safe_float(row.get("official_description")),
            }
    return rows


def load_duplicate_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
        if isinstance(entries, list):
            rows.extend(dict(row) for row in entries if isinstance(row, dict))
    return rows


def _duplicate_payloads_for_sample(rows: list[dict[str, Any]], *, current: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        emotion = _canonical_emotion(row.get("public_emotion") or row.get("nearest_emotion"))
        if not emotion or emotion == current:
            continue
        cosine = _safe_float(row.get("clip_cosine") or row.get("top1_clip_cosine"))
        suspect_duplicate = row.get("suspect_duplicate") is True
        if suspect_duplicate or cosine >= 0.985:
            evidence_type = "exact_public_duplicate"
            confidence = max(cosine, 0.985)
        elif cosine >= 0.95:
            evidence_type = "near_public_duplicate"
            confidence = cosine
        else:
            continue
        grouped[emotion].append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "source": f"public_{evidence_type}",
                "emotion": emotion,
                "confidence": confidence,
                "margin": 0.0,
                "evidence_type": evidence_type,
                "rationale": f"{evidence_type} supports {emotion}",
            }
        )
    return grouped


def _index_rows_by_sample_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            grouped[sample_id].append(row)
    return dict(grouped)


def _canonical_emotion(value: Any) -> str:
    emotion = str(value or "").strip().lower()
    if emotion == "contentment":
        emotion = "content"
    return emotion if emotion in TRACK2_JSON_EMOTIONS else ""


def _valence(emotion: str) -> str:
    return "Negative" if emotion in NEGATIVE_EMOTIONS else "Positive"


def _arousal(emotion: str) -> str:
    return "High" if emotion in HIGH_AROUSAL_EMOTIONS else "Low"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_submission_id(value: Any) -> str:
    match = re.match(r"^(\d{6,})", str(value or "").strip())
    return match.group(1) if match else ""


def _parse_transition_counts(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in str(value or "").split(";"):
        item = part.strip()
        if not item or ":" not in item:
            continue
        transition, count_text = item.rsplit(":", 1)
        transition = transition.strip()
        if "->" in transition:
            counts[transition] = int(_safe_float(count_text))
    return counts
```

- [ ] **Step 6: Add CLI skeleton**

Create `scripts/track2_v17_classification_calibration.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_v17_classification_calibration import main


if __name__ == "__main__":
    main()
```

Add this minimal `main()` to the module:

```python
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Track2 v17 classification calibration tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-evidence", help="Build v17 evidence matrix")
    build.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    build.add_argument("--official-scores", type=Path, default=DEFAULT_OFFICIAL_SCORES)
    build.add_argument("--pairwise-diffs", type=Path, default=DEFAULT_PAIRWISE_DIFFS)
    build.add_argument("--out-json", type=Path, default=DEFAULT_EXPERIMENT_DIR / "evidence_matrix.json")
    build.add_argument("--out-csv", type=Path, default=DEFAULT_EXPERIMENT_DIR / "evidence_matrix.csv")
    args = parser.parse_args(argv)
    if args.command == "build-evidence":
        rows = write_evidence_matrix_outputs(
            base_json=args.base_json,
            official_scores=args.official_scores,
            pairwise_diffs=args.pairwise_diffs,
            out_json=args.out_json,
            out_csv=args.out_csv,
        )
        print(json.dumps({"evidence_rows": len(rows), "out_json": str(args.out_json), "out_csv": str(args.out_csv)}, indent=2))
```

Append output helpers:

```python
def write_evidence_matrix_outputs(
    *,
    base_json: str | Path,
    official_scores: str | Path,
    pairwise_diffs: str | Path,
    out_json: str | Path,
    out_csv: str | Path,
    prediction_sources: dict[str, str | Path] | None = None,
    duplicate_sources: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    base_rows = load_track2_rows(base_json)
    predictions = load_prediction_sources(prediction_sources or DEFAULT_PREDICTION_SOURCES)
    duplicates = load_duplicate_rows(duplicate_sources or DEFAULT_DUPLICATE_SOURCES)
    failed_counts = parse_official_failed_transition_counts(
        pairwise_rows=load_csv_rows(pairwise_diffs),
        score_rows=load_official_score_rows(official_scores),
    )
    evidence = build_evidence_rows(
        base_rows=base_rows,
        predictions_by_sample=predictions,
        duplicate_rows=duplicates,
        failed_transition_counts=failed_counts,
    )
    _write_json(Path(out_json), evidence)
    _write_csv(Path(out_csv), evidence)
    return evidence


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
```

- [ ] **Step 7: Verify Task 1**

Run:

```bash
python3 -m unittest tests/test_track2_v17_classification_calibration.py -v
python3 scripts/track2_v17_classification_calibration.py build-evidence \
  --out-json experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json \
  --out-csv experiments/track2_v17_classification_calibration_20260607/evidence_matrix.csv
git diff --check -- affectiveart/track2_v17_classification_calibration.py scripts/track2_v17_classification_calibration.py tests/test_track2_v17_classification_calibration.py
```

Expected:

- Unit tests pass.
- CLI prints a positive `evidence_rows` count.
- Diff check passes.

- [ ] **Step 8: Commit Task 1**

Run:

```bash
git add affectiveart/track2_v17_classification_calibration.py scripts/track2_v17_classification_calibration.py tests/test_track2_v17_classification_calibration.py experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json experiments/track2_v17_classification_calibration_20260607/evidence_matrix.csv
git commit -m "feat: build track2 v17 evidence matrix"
```

---

## Task 2: Acceptance Gates And Candidate Selection

**Files:**
- Modify: `affectiveart/track2_v17_classification_calibration.py`
- Modify: `tests/test_track2_v17_classification_calibration.py`

- [ ] **Step 1: Add failing tests for safe, balanced, and aggressive gates**

Append tests:

```python
from affectiveart.track2_v17_classification_calibration import (
    select_v17_changes,
    v17_gate_for_evidence,
)


class Track2V17GateTests(unittest.TestCase):
    def test_safe_gate_accepts_two_model_families_without_failed_transition(self) -> None:
        row = {
            "sample_id": "track2_0001",
            "current_emotion": "content",
            "proposed_emotion": "glad",
            "transition": "content->glad",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 2,
            "model_sources": "clip,siglip2",
            "support_score": 2.2,
            "max_confidence": 0.82,
            "exact_duplicate": False,
            "near_duplicate": False,
            "failed_transition_count": 0,
        }
        gate = v17_gate_for_evidence(row, profile="safe", distribution={"content": 20, "glad": 2})
        self.assertEqual(gate["decision"], "accept")

    def test_safe_gate_blocks_failed_bulk_transition_without_exact_duplicate(self) -> None:
        row = {
            "sample_id": "track2_0002",
            "current_emotion": "calm",
            "proposed_emotion": "content",
            "transition": "calm->content",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 3,
            "model_sources": "clip,siglip2,dinov2",
            "support_score": 3.4,
            "max_confidence": 0.95,
            "exact_duplicate": False,
            "near_duplicate": False,
            "failed_transition_count": 43,
        }
        gate = v17_gate_for_evidence(row, profile="safe", distribution={"calm": 50, "content": 20})
        self.assertEqual(gate["decision"], "block")
        self.assertIn("failed_transition_family", gate["reasons"])

    def test_exact_duplicate_can_override_failed_transition(self) -> None:
        row = {
            "sample_id": "track2_0003",
            "current_emotion": "content",
            "proposed_emotion": "calm",
            "transition": "content->calm",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 0,
            "model_sources": "",
            "support_score": 0.99,
            "max_confidence": 0.99,
            "exact_duplicate": True,
            "near_duplicate": False,
            "failed_transition_count": 20,
        }
        gate = v17_gate_for_evidence(row, profile="safe", distribution={"content": 20, "calm": 20})
        self.assertEqual(gate["decision"], "accept")

    def test_select_v17_changes_caps_transition_family(self) -> None:
        rows = [
            {
                "sample_id": f"track2_{index:04d}",
                "current_emotion": "content",
                "proposed_emotion": "calm",
                "transition": "content->calm",
                "same_valence": True,
                "same_arousal": True,
                "model_vote_count": 3,
                "model_sources": "clip,siglip2,dinov2",
                "support_score": 3.5,
                "max_confidence": 0.9,
                "exact_duplicate": False,
                "near_duplicate": True,
                "failed_transition_count": 0,
            }
            for index in range(10)
        ]
        selected = select_v17_changes(rows, profile="balanced", current_distribution={"content": 100, "calm": 100})
        self.assertLessEqual(sum(1 for row in selected if row["transition"] == "content->calm"), 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest tests/test_track2_v17_classification_calibration.py -v
```

Expected: import errors for missing `v17_gate_for_evidence` and `select_v17_changes`.

- [ ] **Step 3: Implement gate logic**

Append:

```python
PROFILE_CONFIG = {
    "safe": {
        "min_model_votes": 2,
        "min_support_score": 1.8,
        "min_confidence": 0.72,
        "transition_cap": 2,
        "content_calm_cap": 2,
        "allow_near_duplicate": True,
    },
    "balanced": {
        "min_model_votes": 2,
        "min_support_score": 1.4,
        "min_confidence": 0.65,
        "transition_cap": 4,
        "content_calm_cap": 4,
        "allow_near_duplicate": True,
    },
    "aggressive_probe": {
        "min_model_votes": 1,
        "min_support_score": 0.9,
        "min_confidence": 0.55,
        "transition_cap": 8,
        "content_calm_cap": 8,
        "allow_near_duplicate": True,
    },
}


def v17_gate_for_evidence(
    row: dict[str, Any],
    *,
    profile: str,
    distribution: dict[str, int],
) -> dict[str, Any]:
    config = _profile_config(profile)
    reasons: list[str] = []
    proposed = _canonical_emotion(row.get("proposed_emotion"))
    current = _canonical_emotion(row.get("current_emotion"))
    if not proposed or not current or proposed == current:
        reasons.append("invalid_or_unchanged_emotion")
    if _safe_float(row.get("max_confidence")) < float(config["min_confidence"]):
        reasons.append("low_confidence")
    if _safe_float(row.get("support_score")) < float(config["min_support_score"]):
        reasons.append("low_support_score")
    if int(row.get("model_vote_count") or 0) < int(config["min_model_votes"]):
        reasons.append("insufficient_model_families")
    if int(row.get("failed_transition_count") or 0) >= 10 and not bool(row.get("exact_duplicate")):
        reasons.append("failed_transition_family")
    if bool(row.get("near_duplicate")) and bool(config["allow_near_duplicate"]):
        reasons = [reason for reason in reasons if reason in {"failed_transition_family"}]
    if bool(row.get("exact_duplicate")):
        reasons = []
    if _would_remove_rare_class(current, distribution):
        reasons.append("rare_class_floor")
    decision = "accept" if not reasons else "block"
    return {"decision": decision, "reasons": sorted(set(reasons))}


def select_v17_changes(
    evidence_rows: list[dict[str, Any]],
    *,
    profile: str,
    current_distribution: dict[str, int],
) -> list[dict[str, Any]]:
    config = _profile_config(profile)
    selected: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    seen_samples: set[str] = set()
    for row in sorted(
        evidence_rows,
        key=lambda item: (-float(item.get("support_score", 0.0)), -float(item.get("max_confidence", 0.0)), str(item.get("sample_id", ""))),
    ):
        sample_id = str(row.get("sample_id", ""))
        transition = str(row.get("transition", ""))
        if sample_id in seen_samples:
            continue
        cap = int(config["content_calm_cap"]) if transition in {"content->calm", "calm->content"} else int(config["transition_cap"])
        if transition_counts[transition] >= cap:
            continue
        gate = v17_gate_for_evidence(row, profile=profile, distribution=current_distribution)
        if gate["decision"] != "accept":
            continue
        accepted = dict(row)
        accepted["profile"] = profile
        accepted["gate_reasons"] = ""
        selected.append(accepted)
        seen_samples.add(sample_id)
        transition_counts[transition] += 1
    return selected
```

- [ ] **Step 4: Implement gate helpers**

Append:

```python
def _profile_config(profile: str) -> dict[str, Any]:
    if profile not in PROFILE_CONFIG:
        raise ValueError(f"unknown v17 profile: {profile}")
    return PROFILE_CONFIG[profile]


def _would_remove_rare_class(current: str, distribution: dict[str, int]) -> bool:
    return int(distribution.get(current, 0)) <= 2
```

- [ ] **Step 5: Verify Task 2**

Run:

```bash
python3 -m unittest tests/test_track2_v17_classification_calibration.py -v
git diff --check -- affectiveart/track2_v17_classification_calibration.py tests/test_track2_v17_classification_calibration.py
```

Expected: tests pass and diff check passes.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add affectiveart/track2_v17_classification_calibration.py tests/test_track2_v17_classification_calibration.py
git commit -m "feat: add track2 v17 classification gates"
```

---

## Task 3: Candidate Writer

**Files:**
- Modify: `affectiveart/track2_v17_classification_calibration.py`
- Modify: `scripts/track2_v17_classification_calibration.py`
- Modify: `tests/test_track2_v17_classification_calibration.py`

- [ ] **Step 1: Add failing tests for candidate writing and safety**

Append tests:

```python
import zipfile

from affectiveart.track2_v17_classification_calibration import (
    apply_v17_changes,
    write_v17_candidate_outputs,
)


class Track2V17CandidateWriterTests(unittest.TestCase):
    def test_apply_v17_changes_repairs_valence_and_arousal(self) -> None:
        rows = [
            {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "caption",
                "brushstroke": "brush",
                "composition": "composition",
                "color": "color",
                "line": "line",
                "light": "light",
            }
        ]
        changes = [{"sample_id": "track2_0001", "current_emotion": "content", "proposed_emotion": "excited", "transition": "content->excited"}]
        repaired, report = apply_v17_changes(rows, changes, profile="safe")
        self.assertEqual(repaired[0]["emotion"], "excited")
        self.assertEqual(repaired[0]["emotional_valence"], "Positive")
        self.assertEqual(repaired[0]["emotional_arousal_level"], "High")
        self.assertEqual(report["accepted_label_changes"], 1)
        self.assertEqual(report["label_consistency_issue_count"], 0)

    def test_write_v17_candidate_outputs_writes_zip_with_submission_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "base.json"
            out_json = root / "track2_submission_v17_safe_candidate.json"
            out_zip = root / "track2_submission_v17_safe_candidate.zip"
            rows = [
                {
                    "sample_id": "track2_0001",
                    "emotion": "content",
                    "emotional_valence": "Positive",
                    "emotional_arousal_level": "Low",
                    "overall_caption": "caption",
                    "brushstroke": "brush",
                    "composition": "composition",
                    "color": "color",
                    "line": "line",
                    "light": "light",
                }
            ]
            source.write_text(json.dumps(rows), encoding="utf-8")
            report = write_v17_candidate_outputs(
                base_json=source,
                changes=[{"sample_id": "track2_0001", "current_emotion": "content", "proposed_emotion": "calm", "transition": "content->calm"}],
                profile="safe",
                out_json=out_json,
                out_zip=out_zip,
                report_json=root / "report.json",
                report_md=root / "report.md",
            )
            with zipfile.ZipFile(out_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
        self.assertEqual(report["accepted_label_changes"], 1)

    def test_write_v17_candidate_outputs_rejects_formal_submission_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "base.json"
            source.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_v17_candidate_outputs(
                    base_json=source,
                    changes=[],
                    profile="safe",
                    out_json=root / "track2_submission.json",
                    out_zip=root / "track2_submission_v17_safe_candidate.zip",
                    report_json=root / "report.json",
                    report_md=root / "report.md",
                )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest tests/test_track2_v17_classification_calibration.py -v
```

Expected: import errors for missing writer functions.

- [ ] **Step 3: Implement candidate writer**

Append:

```python
def apply_v17_changes(
    base_rows: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    *,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    changes_by_id = {str(row.get("sample_id", "")): dict(row) for row in changes}
    output: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for row in base_rows:
        repaired = {key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS}
        sample_id = str(repaired.get("sample_id", ""))
        change = changes_by_id.get(sample_id)
        if change is not None:
            proposed = _canonical_emotion(change.get("proposed_emotion"))
            if proposed and proposed != _canonical_emotion(repaired.get("emotion")):
                before = _label_triplet(repaired)
                repaired["emotion"] = proposed
                repaired["emotional_valence"] = _valence(proposed)
                repaired["emotional_arousal_level"] = _arousal(proposed)
                after = _label_triplet(repaired)
                accepted.append(
                    {
                        "sample_id": sample_id,
                        "transition": f"{before[0]}->{after[0]}",
                        "before": {"emotion": before[0], "emotional_valence": before[1], "emotional_arousal_level": before[2]},
                        "after": {"emotion": after[0], "emotional_valence": after[1], "emotional_arousal_level": after[2]},
                        "support_score": _safe_float(change.get("support_score")),
                        "max_confidence": _safe_float(change.get("max_confidence")),
                        "model_sources": str(change.get("model_sources", "")),
                        "rationale": str(change.get("rationale", "")),
                    }
                )
        output.append(repaired)
    distribution = Counter(str(row.get("emotion", "")) for row in output)
    label_issues = [issue for row in output for issue in strict_track2_label_issues(row)]
    report = {
        "method": "track2_v17_classification_calibration_candidate_v1",
        "profile": profile,
        "row_count": len(output),
        "accepted_label_changes": len(accepted),
        "transition_counts": dict(Counter(item["transition"] for item in accepted)),
        "distribution": dict(sorted(distribution.items())),
        "missing_emotions": sorted(TRACK2_JSON_EMOTIONS - set(distribution)),
        "top_emotion": distribution.most_common(1)[0][0] if distribution else "",
        "top_emotion_share": distribution.most_common(1)[0][1] / len(output) if output else 0.0,
        "label_consistency_issue_count": len(label_issues),
        "accepted_changes": accepted,
        "formal_submission_overwritten": False,
    }
    return output, report
```

- [ ] **Step 4: Implement report writing and zip safety**

Append:

```python
def write_v17_candidate_outputs(
    *,
    base_json: str | Path,
    changes: list[dict[str, Any]],
    profile: str,
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
) -> dict[str, Any]:
    out_json_path = Path(out_json)
    out_zip_path = Path(out_zip)
    _assert_safe_candidate_path(out_json_path)
    _assert_safe_candidate_path(out_zip_path)
    rows = load_track2_rows(base_json)
    candidate_rows, report = apply_v17_changes(rows, changes, profile=profile)
    report["base_json"] = str(base_json)
    report["out_json"] = str(out_json)
    report["out_zip"] = str(out_zip)
    _write_json(out_json_path, candidate_rows)
    _write_zip(out_zip_path, out_json_path)
    _write_json(Path(report_json), report)
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text(render_candidate_report_markdown(report), encoding="utf-8")
    return report


def render_candidate_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v17 Candidate Report",
        "",
        f"- Profile: `{report['profile']}`",
        f"- Candidate JSON: `{report.get('out_json', '')}`",
        f"- Candidate ZIP: `{report.get('out_zip', '')}`",
        f"- Accepted label changes: {report['accepted_label_changes']}",
        f"- Label consistency issues: {report['label_consistency_issue_count']}",
        f"- Missing emotions: {', '.join(report['missing_emotions']) or 'none'}",
        f"- Top emotion: {report['top_emotion']} ({float(report['top_emotion_share']):.1%})",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        "",
        "## Accepted Changes",
        "",
    ]
    if not report.get("accepted_changes"):
        lines.append("- none")
    for item in report.get("accepted_changes", [])[:120]:
        lines.append(
            "- "
            f"{item['sample_id']}: {item['transition']}; "
            f"support={float(item.get('support_score', 0.0)):.3f}; "
            f"conf={float(item.get('max_confidence', 0.0)):.3f}; "
            f"sources={item.get('model_sources', '')}"
        )
    return "\n".join(lines) + "\n"


def _assert_safe_candidate_path(path: Path) -> None:
    if path.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError(f"refusing to write formal Track2 submission path: {path}")
    if path.suffix not in {".json", ".zip"}:
        raise ValueError(f"candidate output must be JSON or ZIP: {path}")
    if not path.stem.startswith("track2_submission_v17_") or not path.stem.endswith("_candidate"):
        raise ValueError(f"candidate output must be a v17 side-path candidate: {path}")


def _write_zip(zip_path: Path, json_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, "submission.json")


def _label_triplet(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("emotion", "")),
        str(row.get("emotional_valence", "")),
        str(row.get("emotional_arousal_level", "")),
    )
```

- [ ] **Step 5: Add CLI command `build-candidates`**

Modify `main()` to add:

```python
    candidates = subparsers.add_parser("build-candidates", help="Build v17 safe/balanced/aggressive candidates")
    candidates.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    candidates.add_argument("--evidence-json", type=Path, default=DEFAULT_EXPERIMENT_DIR / "evidence_matrix.json")
    candidates.add_argument("--out-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR / "candidate_reports")
```

Add command branch:

```python
    elif args.command == "build-candidates":
        summary = write_v17_profile_candidates(
            base_json=args.base_json,
            evidence_json=args.evidence_json,
            out_dir=args.out_dir,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
```

Append:

```python
def write_v17_profile_candidates(
    *,
    base_json: str | Path,
    evidence_json: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    base_rows = load_track2_rows(base_json)
    evidence_rows = json.loads(Path(evidence_json).read_text(encoding="utf-8"))
    current_distribution = dict(Counter(str(row.get("emotion", "")) for row in base_rows))
    out_dir = Path(out_dir)
    reports: dict[str, Any] = {}
    for profile in ("safe", "balanced", "aggressive_probe"):
        changes = select_v17_changes(evidence_rows, profile=profile, current_distribution=current_distribution)
        stem = f"track2_submission_v17_{profile}_candidate"
        report = write_v17_candidate_outputs(
            base_json=base_json,
            changes=changes,
            profile=profile,
            out_json=Path("submissions") / f"{stem}.json",
            out_zip=Path("submissions") / f"{stem}.zip",
            report_json=out_dir / f"{profile}_candidate_report.json",
            report_md=out_dir / f"{profile}_candidate_report.md",
        )
        reports[profile] = {
            "candidate_json": report["out_json"],
            "candidate_zip": report["out_zip"],
            "accepted_label_changes": report["accepted_label_changes"],
            "top_emotion": report["top_emotion"],
            "top_emotion_share": report["top_emotion_share"],
            "label_consistency_issue_count": report["label_consistency_issue_count"],
        }
    _write_json(out_dir / "candidate_summary.json", reports)
    return reports
```

- [ ] **Step 6: Verify Task 3**

Run:

```bash
python3 -m unittest tests/test_track2_v17_classification_calibration.py -v
python3 scripts/track2_v17_classification_calibration.py build-candidates \
  --evidence-json experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json \
  --out-dir experiments/track2_v17_classification_calibration_20260607/candidate_reports
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v17_safe_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v17_balanced_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v17_aggressive_probe_candidate.json
git diff --check -- affectiveart/track2_v17_classification_calibration.py scripts/track2_v17_classification_calibration.py tests/test_track2_v17_classification_calibration.py
```

Expected:

- Unit tests pass.
- All three v17 candidate JSON files validate.
- Diff check passes.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add affectiveart/track2_v17_classification_calibration.py scripts/track2_v17_classification_calibration.py tests/test_track2_v17_classification_calibration.py experiments/track2_v17_classification_calibration_20260607/candidate_reports
git commit -m "feat: build track2 v17 classification candidates"
```

Do not commit the generated `submissions/track2_submission_v17_*_candidate.json/.zip` unless the user explicitly chooses to version candidate artifacts. Reports are enough for this task.

---

## Task 4: Scoring And Final Gate

**Files:**
- Modify: `affectiveart/track2_v17_classification_calibration.py`
- Modify: `scripts/track2_v17_classification_calibration.py`
- Modify: `tests/test_track2_v17_classification_calibration.py`

- [ ] **Step 1: Add failing tests for final gate selection**

Append tests:

```python
from affectiveart.track2_v17_classification_calibration import choose_final_gate


class Track2V17FinalGateTests(unittest.TestCase):
    def test_choose_final_gate_prefers_candidate_with_better_lower_bound(self) -> None:
        rows = [
            {"candidate_name": "v15_desc_expand300", "overall_lower": "0.835854", "overall_expected": "0.839902"},
            {"candidate_name": "v17_safe", "overall_lower": "0.836100", "overall_expected": "0.840000"},
        ]
        decision = choose_final_gate(rows, baseline_name="v15_desc_expand300")
        self.assertEqual(decision["decision"], "recommend_submit_v17")
        self.assertEqual(decision["candidate_name"], "v17_safe")

    def test_choose_final_gate_holds_when_lower_bound_is_below_v15(self) -> None:
        rows = [
            {"candidate_name": "v15_desc_expand300", "overall_lower": "0.835854", "overall_expected": "0.839902"},
            {"candidate_name": "v17_balanced", "overall_lower": "0.834000", "overall_expected": "0.841000"},
        ]
        decision = choose_final_gate(rows, baseline_name="v15_desc_expand300")
        self.assertEqual(decision["decision"], "hold_v17_keep_v15")
        self.assertEqual(decision["candidate_name"], "v15_desc_expand300")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest tests/test_track2_v17_classification_calibration.py -v
```

Expected: import error for missing `choose_final_gate`.

- [ ] **Step 3: Implement final gate helper**

Append:

```python
def choose_final_gate(rows: list[dict[str, Any]], *, baseline_name: str = "v15_desc_expand300") -> dict[str, Any]:
    by_name = {str(row.get("candidate_name", "")): row for row in rows}
    if baseline_name not in by_name:
        raise ValueError(f"baseline candidate missing from scorer rows: {baseline_name}")
    baseline = by_name[baseline_name]
    baseline_lower = _safe_float(baseline.get("overall_lower"))
    challengers = [row for row in rows if str(row.get("candidate_name", "")).startswith("v17_")]
    if not challengers:
        return {
            "decision": "hold_v17_keep_v15",
            "candidate_name": baseline_name,
            "reason": "no_v17_candidates_in_score_report",
        }
    best = max(challengers, key=lambda row: (_safe_float(row.get("overall_lower")), _safe_float(row.get("overall_expected"))))
    if _safe_float(best.get("overall_lower")) > baseline_lower:
        return {
            "decision": "recommend_submit_v17",
            "candidate_name": str(best.get("candidate_name", "")),
            "reason": "v17_lower_bound_exceeds_v15",
            "baseline_lower": baseline_lower,
            "candidate_lower": _safe_float(best.get("overall_lower")),
        }
    return {
        "decision": "hold_v17_keep_v15",
        "candidate_name": baseline_name,
        "reason": "no_v17_lower_bound_improvement",
        "baseline_lower": baseline_lower,
        "best_v17_name": str(best.get("candidate_name", "")),
        "best_v17_lower": _safe_float(best.get("overall_lower")),
    }
```

- [ ] **Step 4: Add CLI command `score-gate`**

Modify `main()` to add:

```python
    score_gate = subparsers.add_parser("score-gate", help="Run local/fused scoring command hints and write final v17 gate")
    score_gate.add_argument("--fused-ranking-csv", type=Path, required=True)
    score_gate.add_argument("--out-md", type=Path, default=DEFAULT_EXPERIMENT_DIR / "final_gate_report.md")
    score_gate.add_argument("--out-json", type=Path, default=DEFAULT_EXPERIMENT_DIR / "final_gate_report.json")
```

Add command branch:

```python
    elif args.command == "score-gate":
        report = write_final_gate_report(
            fused_ranking_csv=args.fused_ranking_csv,
            out_md=args.out_md,
            out_json=args.out_json,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
```

Append:

```python
def write_final_gate_report(
    *,
    fused_ranking_csv: str | Path,
    out_md: str | Path,
    out_json: str | Path,
) -> dict[str, Any]:
    rows = load_csv_rows(fused_ranking_csv)
    decision = choose_final_gate(rows, baseline_name="v15_desc_expand300")
    report = {
        "method": "track2_v17_final_gate_v1",
        "fused_ranking_csv": str(fused_ranking_csv),
        **decision,
        "submission_policy": "Do not submit automatically. Submit only after user confirmation.",
    }
    _write_json(Path(out_json), report)
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(render_final_gate_markdown(report, rows), encoding="utf-8")
    return report


def render_final_gate_markdown(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Track2 v17 Final Gate",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Candidate: `{report['candidate_name']}`",
        f"- Reason: `{report['reason']}`",
        f"- Submission policy: {report['submission_policy']}",
        "",
        "## Fused Ranking",
        "",
        "| candidate | overall lower | overall expected | class expected | desc expected | changes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('candidate_name', '')} | {row.get('overall_lower', '')} | {row.get('overall_expected', '')} | "
            f"{row.get('classification_expected', '')} | {row.get('description_expected', '')} | {row.get('changed_rows', '')} |"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 5: Run v17 scoring commands**

Run:

```bash
python3 scripts/track2_fused_shadow_evaluator.py score \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate official_779605_moe_v2_anchor=submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate official_782683_v12_stable_probe=submissions/track2_submission_v12_stable_probe_candidate.json \
  --candidate v15_desc_expand300=submissions/track2_submission_v15_desc_expand300_candidate.json \
  --candidate v17_safe=submissions/track2_submission_v17_safe_candidate.json \
  --candidate v17_balanced=submissions/track2_submission_v17_balanced_candidate.json \
  --candidate v17_aggressive_probe=submissions/track2_submission_v17_aggressive_probe_candidate.json \
  --out-dir experiments/track2_v17_classification_calibration_20260607/fused_shadow_compare
python3 scripts/track2_local_shadow_evaluator.py score \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate official_779605_moe_v2_anchor=submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate v15_desc_expand300=submissions/track2_submission_v15_desc_expand300_candidate.json \
  --candidate v17_safe=submissions/track2_submission_v17_safe_candidate.json \
  --candidate v17_balanced=submissions/track2_submission_v17_balanced_candidate.json \
  --candidate v17_aggressive_probe=submissions/track2_submission_v17_aggressive_probe_candidate.json \
  --out-dir experiments/track2_v17_classification_calibration_20260607/local_shadow_compare
python3 scripts/track2_v17_classification_calibration.py score-gate \
  --fused-ranking-csv experiments/track2_v17_classification_calibration_20260607/fused_shadow_compare/candidate_ranking.csv \
  --out-md experiments/track2_v17_classification_calibration_20260607/final_gate_report.md \
  --out-json experiments/track2_v17_classification_calibration_20260607/final_gate_report.json
```

Expected:

- Fused and local scorers write reports.
- Final gate report chooses `recommend_submit_v17` only if lower bound beats `v15_desc_expand300`; otherwise it chooses `hold_v17_keep_v15`.

- [ ] **Step 6: Verify Task 4**

Run:

```bash
python3 -m unittest tests/test_track2_v17_classification_calibration.py -v
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
git diff --check
```

Expected: tests pass and diff check passes.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add affectiveart/track2_v17_classification_calibration.py scripts/track2_v17_classification_calibration.py tests/test_track2_v17_classification_calibration.py experiments/track2_v17_classification_calibration_20260607/fused_shadow_compare experiments/track2_v17_classification_calibration_20260607/local_shadow_compare experiments/track2_v17_classification_calibration_20260607/final_gate_report.md experiments/track2_v17_classification_calibration_20260607/final_gate_report.json
git commit -m "docs: record track2 v17 scoring gate"
```

---

## Task 5: Final Verification And Handoff

**Files:**
- Read: `experiments/track2_v17_classification_calibration_20260607/final_gate_report.md`
- Read: `experiments/track2_v17_classification_calibration_20260607/candidate_reports/candidate_summary.json`

- [ ] **Step 1: Validate candidate JSON files**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v17_safe_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v17_balanced_candidate.json
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v17_aggressive_probe_candidate.json
```

Expected: `OK` for all three.

- [ ] **Step 2: Check git state for intentional untracked candidate ZIPs**

Run:

```bash
git status --short -- submissions/track2_submission_v17_safe_candidate.json submissions/track2_submission_v17_safe_candidate.zip submissions/track2_submission_v17_balanced_candidate.json submissions/track2_submission_v17_balanced_candidate.zip submissions/track2_submission_v17_aggressive_probe_candidate.json submissions/track2_submission_v17_aggressive_probe_candidate.zip
```

Expected: the candidate files may be untracked. Do not commit them unless the user explicitly asks to version generated submission artifacts.

- [ ] **Step 3: Summarize final gate**

Report in Chinese:

- best local candidate ZIP path
- whether v17 beats `v15_desc_expand300`
- whether final gate says `recommend_submit_v17`, `hold_v17_keep_v15`, or official probe only
- validation status
- exact report paths
- clear statement that no Codabench upload was performed

---

## Self-Review

- Spec coverage: Tasks implement evidence matrix, calibration gate, side-path candidates, scoring, final gate, validation, no auto-submit, and Track2-only boundary.
- Completeness scan: The plan avoids incomplete-plan wording and every code-changing step includes concrete code or exact commands.
- Type consistency: Function names introduced in tests match implementation steps: `load_prediction_sources`, `parse_official_failed_transition_counts`, `build_evidence_rows`, `v17_gate_for_evidence`, `select_v17_changes`, `apply_v17_changes`, `write_v17_candidate_outputs`, `choose_final_gate`.
- Safety: Formal submission overwrite protection is tested and implemented before candidate writing is used.
- Submission budget: The plan ends with local scoring and a user-facing gate; it does not upload to Codabench.
