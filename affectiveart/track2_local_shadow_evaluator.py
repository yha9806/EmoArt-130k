from __future__ import annotations

import csv
import fcntl
import html
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_EMOTIONS, TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_audit import compute_track2_distribution, strict_track2_label_issues
from affectiveart.track2_description_score import audit_description_row


FORMAL_SUBMISSION_PATHS = {
    Path("submissions/track2_submission.json"),
    Path("submissions/track2_submission.zip"),
}


@dataclass(frozen=True)
class OfficialAnchor:
    overall: float = 0.836408
    classification: float = 0.723150
    description: float = 0.949667
    emotion_accuracy: float = 0.570000
    emotion_macro_f1: float = 0.309338
    valence_accuracy: float = 0.883000
    valence_macro_f1: float = 0.823380
    arousal_accuracy: float = 0.913000
    arousal_macro_f1: float = 0.840184


OFFICIAL_ANCHOR = OfficialAnchor()


@dataclass(frozen=True)
class ScoreBand:
    expected: float
    lower: float
    upper: float

    def clamped(self) -> "ScoreBand":
        return ScoreBand(
            expected=_clamp01(self.expected),
            lower=_clamp01(self.lower),
            upper=_clamp01(self.upper),
        )


@dataclass(frozen=True)
class SafetyGateResult:
    candidate_name: str
    passed: bool
    issue_codes: list[str]
    issues: list[dict[str, Any]]
    distribution: dict[str, Any]
    description_issue_count: int


def compute_task_score(*, macro_f1: float, accuracy: float) -> float:
    return 0.5 * float(macro_f1) + 0.5 * float(accuracy)


def compute_classification_score(
    *,
    emotion_macro_f1: float,
    emotion_accuracy: float,
    valence_macro_f1: float,
    valence_accuracy: float,
    arousal_macro_f1: float,
    arousal_accuracy: float,
) -> float:
    emotion_task = compute_task_score(macro_f1=emotion_macro_f1, accuracy=emotion_accuracy)
    valence_task = compute_task_score(macro_f1=valence_macro_f1, accuracy=valence_accuracy)
    arousal_task = compute_task_score(macro_f1=arousal_macro_f1, accuracy=arousal_accuracy)
    return (emotion_task + valence_task + arousal_task) / 3.0


def run_candidate_safety_gate(
    *,
    candidate_name: str,
    candidate_json: Path,
    rows: list[dict[str, Any]],
    expected_row_count: int = 1000,
    formal_submission_paths: set[Path] | None = None,
) -> SafetyGateResult:
    formal_paths = formal_submission_paths or FORMAL_SUBMISSION_PATHS
    issues: list[dict[str, Any]] = []

    if _is_formal_path(candidate_json, formal_paths):
        issues.append({"code": "formal_submission_path", "path": str(candidate_json)})

    if len(rows) != expected_row_count:
        issues.append({"code": "row_count", "expected": expected_row_count, "actual": len(rows)})

    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        issues.append({"code": "missing_sample_id"})
    duplicates = sorted(sample_id for sample_id, count in Counter(sample_ids).items() if sample_id and count > 1)
    if duplicates:
        issues.append({"code": "duplicate_sample_id", "sample_ids": duplicates[:20]})

    required_key_issues = []
    for index, row in enumerate(rows):
        missing = [key for key in TRACK2_JSON_SUBMISSION_KEYS if key not in row or str(row.get(key, "")) == ""]
        if missing:
            required_key_issues.append({"index": index, "sample_id": row.get("sample_id", ""), "missing": missing})
    if required_key_issues:
        issues.append({"code": "missing_required_fields", "rows": required_key_issues[:20]})

    invalid_emotions = sorted({str(row.get("emotion", "")) for row in rows} - TRACK2_JSON_EMOTIONS)
    if invalid_emotions:
        issues.append({"code": "invalid_emotion", "emotions": invalid_emotions})

    label_issues = [
        {
            "sample_id": str(row.get("sample_id", "")),
            "issues": strict_track2_label_issues(row),
        }
        for row in rows
        if strict_track2_label_issues(row)
    ]
    if label_issues:
        issues.append({"code": "label_consistency", "rows": label_issues[:40]})

    distribution = compute_track2_distribution(rows)
    if distribution["missing_emotions"]:
        issues.append({"code": "missing_emotions", "emotions": distribution["missing_emotions"]})

    description_audits = [audit_description_row(row) for row in rows]
    manipulation_rows = [
        item["sample_id"]
        for item in description_audits
        if any(issue.get("code") == "evaluator_manipulation_risk" for issue in item.get("issues", []))
    ]
    if manipulation_rows:
        issues.append({"code": "evaluator_manipulation", "sample_ids": manipulation_rows[:40]})

    return SafetyGateResult(
        candidate_name=candidate_name,
        passed=not issues,
        issue_codes=[str(issue["code"]) for issue in issues],
        issues=issues,
        distribution=distribution,
        description_issue_count=sum(int(item.get("issue_count", 0)) for item in description_audits),
    )


def load_json_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Track2 JSON must contain a list: {path}")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _is_formal_path(path: Path, formal_paths: set[Path]) -> bool:
    normalized = Path(path)
    return any(normalized == item or normalized.as_posix().endswith(item.as_posix()) for item in formal_paths)


def _clamp01(value: float) -> float:
    if math.isnan(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
