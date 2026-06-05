from __future__ import annotations

import csv
import fcntl
import html
import json
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAL_SUBMISSION_PATHS = {
    Path("submissions/track2_submission.json"),
    Path("submissions/track2_submission.zip"),
}
TRACK2_JSON_SUBMISSION_KEYS = [
    "sample_id",
    "emotion",
    "emotional_valence",
    "emotional_arousal_level",
    "overall_caption",
    "brushstroke",
    "composition",
    "color",
    "line",
    "light",
]
TRACK2_OPEN_TEXT_FIELDS = ["overall_caption", "brushstroke", "composition", "color", "line", "light"]
TRACK2_JSON_EMOTIONS = {
    "alarmed",
    "annoyed",
    "aroused",
    "bored",
    "calm",
    "content",
    "excited",
    "frustrated",
    "glad",
    "happy",
    "sad",
    "tired",
}
POSITIVE_EMOTIONS = {"aroused", "excited", "happy", "content", "calm", "glad"}
NEGATIVE_EMOTIONS = {"alarmed", "annoyed", "frustrated", "sad", "bored", "tired"}
HIGH_AROUSAL_EMOTIONS = {"aroused", "excited", "happy", "alarmed", "annoyed", "frustrated"}
LOW_AROUSAL_EMOTIONS = {"content", "calm", "glad", "sad", "bored", "tired"}
EVALUATOR_MANIPULATION_PATTERNS = [
    r"\bevaluator\b",
    r"\bevaluate this\b",
    r"\bignore\b",
    r"\binstruction\b",
    r"\bplease rate\b",
    r"\bgive this\b.*\bscore\b",
    r"\bas an ai\b",
    r"\bprompt\b",
]
FORMULAIC_DESCRIPTION_PATTERNS = [
    "creating a calm, contemplative emotional atmosphere",
    "creating a content, settled emotional atmosphere",
    "creating a glad, uplifting emotional atmosphere",
]


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


@dataclass(frozen=True)
class ShadowScoreResult:
    candidate_name: str
    candidate_json: str
    decision: str
    changed_rows: int
    same_quadrant_changes: int
    cross_quadrant_changes: int
    classification: ScoreBand
    description: ScoreBand
    overall: ScoreBand
    safety: SafetyGateResult
    row_risks: list[dict[str, Any]]


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


def strict_track2_label_issues(row: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    emotion = str(row.get("emotion", ""))
    valence = str(row.get("emotional_valence", ""))
    arousal = str(row.get("emotional_arousal_level", ""))
    if emotion in POSITIVE_EMOTIONS and valence != "Positive":
        issues.append(
            {
                "field": "emotional_valence",
                "expected": "Positive",
                "actual": valence,
                "reason": f"{emotion} is positive in the Track2 label set",
            }
        )
    if emotion in NEGATIVE_EMOTIONS and valence != "Negative":
        issues.append(
            {
                "field": "emotional_valence",
                "expected": "Negative",
                "actual": valence,
                "reason": f"{emotion} is negative in the Track2 label set",
            }
        )
    if emotion in HIGH_AROUSAL_EMOTIONS and arousal != "High":
        issues.append(
            {
                "field": "emotional_arousal_level",
                "expected": "High",
                "actual": arousal,
                "reason": f"{emotion} is high-arousal in the Track2 label set",
            }
        )
    if emotion in LOW_AROUSAL_EMOTIONS and arousal != "Low":
        issues.append(
            {
                "field": "emotional_arousal_level",
                "expected": "Low",
                "actual": arousal,
                "reason": f"{emotion} is low-arousal in the Track2 label set",
            }
        )
    return issues


def compute_track2_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    emotion = Counter(str(row.get("emotion", "")) for row in rows)
    valence = Counter(str(row.get("emotional_valence", "")) for row in rows)
    arousal = Counter(str(row.get("emotional_arousal_level", "")) for row in rows)
    top_emotion, top_count = emotion.most_common(1)[0] if emotion else ("", 0)
    top_two_count = sum(value for _, value in emotion.most_common(2))
    top_share = round(top_count / count, 4) if count else 0.0
    top_two_share = round(top_two_count / count, 4) if count else 0.0
    flags = []
    if top_share >= 0.55:
        flags.append(f"top emotion {top_emotion} covers {top_share:.1%} of rows")
    if top_two_share >= 0.80:
        flags.append(f"top two emotions cover {top_two_share:.1%} of rows")
    return {
        "count": count,
        "emotion": dict(emotion.most_common()),
        "valence": dict(valence.most_common()),
        "arousal": dict(arousal.most_common()),
        "top_emotion": top_emotion,
        "top_emotion_share": top_share,
        "top_two_emotion_share": top_two_share,
        "missing_emotions": sorted(TRACK2_JSON_EMOTIONS - set(emotion)),
        "collapse_flags": flags,
    }


def audit_description_row(row: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    fields = {field: str(row.get(field, "")) for field in TRACK2_OPEN_TEXT_FIELDS}
    combined_lower = "\n".join(fields.values()).lower()

    for pattern in EVALUATOR_MANIPULATION_PATTERNS:
        if re.search(pattern, combined_lower, re.IGNORECASE):
            issues.append({"code": "evaluator_manipulation_risk", "pattern": pattern})

    for phrase in FORMULAIC_DESCRIPTION_PATTERNS:
        if phrase in combined_lower:
            issues.append({"code": "formulaic_emotional_atmosphere_suffix", "phrase": phrase})

    for field, value in fields.items():
        if len(value.split()) < 6:
            issues.append({"code": "underspecified_attribute_text", "field": field})

    return {
        "sample_id": str(row.get("sample_id", "")),
        "emotion": str(row.get("emotion", "")),
        "issue_count": len(issues),
        "issues": issues,
        "field_word_counts": {field: len(value.split()) for field, value in fields.items()},
    }


def run_candidate_safety_gate(
    *,
    candidate_name: str,
    candidate_json: Path,
    rows: list[dict[str, Any]],
    expected_row_count: int = 1000,
    formal_submission_paths: set[Path] | None = None,
    require_all_emotions: bool | None = None,
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
    enforce_all_emotions = expected_row_count >= 1000 if require_all_emotions is None else require_all_emotions
    if enforce_all_emotions and distribution["missing_emotions"]:
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


def score_candidate_rows(
    *,
    candidate_name: str,
    candidate_json: Path,
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    expected_row_count: int = 1000,
    require_all_emotions: bool | None = None,
) -> ShadowScoreResult:
    safety = run_candidate_safety_gate(
        candidate_name=candidate_name,
        candidate_json=candidate_json,
        rows=rows,
        expected_row_count=expected_row_count,
        require_all_emotions=require_all_emotions,
    )
    row_risks = _build_row_risks(baseline_rows, rows)
    changed_rows = len(row_risks)
    same_quadrant = sum(1 for item in row_risks if item["same_quadrant"])
    cross_quadrant = changed_rows - same_quadrant

    classification = _classification_band(
        safety=safety,
        changed_rows=changed_rows,
        same_quadrant=same_quadrant,
        cross_quadrant=cross_quadrant,
    )
    description = _description_band(safety=safety, row_count=max(1, len(rows)))
    overall = ScoreBand(
        expected=0.5 * classification.expected + 0.5 * description.expected,
        lower=0.5 * classification.lower + 0.5 * description.lower,
        upper=0.5 * classification.upper + 0.5 * description.upper,
    ).clamped()
    decision = _decision_for_score(safety=safety, overall=overall, description=description, cross_quadrant=cross_quadrant)

    return ShadowScoreResult(
        candidate_name=candidate_name,
        candidate_json=str(candidate_json),
        decision=decision,
        changed_rows=changed_rows,
        same_quadrant_changes=same_quadrant,
        cross_quadrant_changes=cross_quadrant,
        classification=classification,
        description=description,
        overall=overall,
        safety=safety,
        row_risks=row_risks,
    )


def rank_shadow_candidates(results: list[ShadowScoreResult]) -> list[ShadowScoreResult]:
    return sorted(
        results,
        key=lambda item: (
            item.decision != "recommend_submit",
            -item.overall.lower,
            -item.description.lower,
            item.cross_quadrant_changes,
            item.changed_rows,
            -item.overall.expected,
            item.candidate_name,
        ),
    )


def append_calibration_entry(*, ledger_path: str | Path, entry: dict[str, Any]) -> dict[str, Any]:
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_calibration_entry(entry)
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        existing = ledger_path.read_text(encoding="utf-8") if ledger_path.exists() else ""
        temp_path = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
        temp_path.write_text(existing + json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(ledger_path)
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    return normalized


def load_calibration_summary(ledger_path: str | Path) -> dict[str, Any]:
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return {
            "entry_count": 0,
            "overall_mae": 0.0,
            "classification_mae": 0.0,
            "description_mae": 0.0,
            "entries": [],
        }
    entries = [
        _normalize_calibration_entry(json.loads(line))
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "entry_count": len(entries),
        "overall_mae": _mean_abs_error(entries, "shadow_overall_expected", "official_overall"),
        "classification_mae": _mean_abs_error(entries, "shadow_classification_expected", "official_classification"),
        "description_mae": _mean_abs_error(entries, "shadow_description_expected", "official_description"),
        "entries": entries,
    }


def write_shadow_evaluator_outputs(
    *,
    baseline_json: str | Path,
    candidates: list[dict[str, Any]],
    out_dir: str | Path,
    expected_row_count: int = 1000,
    require_all_emotions: bool | None = None,
) -> dict[str, Any]:
    baseline_json = Path(baseline_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = load_json_rows(baseline_json)
    results: list[ShadowScoreResult] = []
    for candidate in candidates:
        candidate_json = Path(candidate["json"])
        rows = load_json_rows(candidate_json)
        results.append(
            score_candidate_rows(
                candidate_name=str(candidate["name"]),
                candidate_json=candidate_json,
                rows=rows,
                baseline_rows=baseline_rows,
                expected_row_count=expected_row_count,
                require_all_emotions=require_all_emotions,
            )
        )
    ranked = rank_shadow_candidates(results)
    ranking_rows = [_result_to_dict(item) for item in ranked]
    row_risks = [
        {"candidate_name": result.candidate_name, **risk}
        for result in ranked
        for risk in result.row_risks
    ]
    report = {
        "method": "track2_shadow_score_v1_formula_proxy",
        "warning": (
            "This is a local shadow score. It is not the official Codabench score "
            "and must not be treated as hidden-test ground truth."
        ),
        "baseline_json": str(baseline_json),
        "official_anchor": OFFICIAL_ANCHOR.__dict__,
        "ranking": ranking_rows,
        "row_risk_count": len(row_risks),
        "formal_submission_overwritten": False,
    }
    _write_json(out_dir / "shadow_score_report.json", report)
    _write_json(out_dir / "candidate_ranking.json", ranking_rows)
    _write_json(out_dir / "row_risk_matrix.json", row_risks)
    _write_csv(out_dir / "candidate_ranking.csv", ranking_rows)
    _write_csv(out_dir / "row_risk_matrix.csv", row_risks)
    (out_dir / "shadow_score_report.md").write_text(_render_shadow_markdown(report), encoding="utf-8")
    html_dir = out_dir / "html_review"
    html_dir.mkdir(parents=True, exist_ok=True)
    (html_dir / "track2_shadow_evaluator_review.html").write_text(
        _render_shadow_html(report, row_risks),
        encoding="utf-8",
    )
    return report


def _result_to_dict(result: ShadowScoreResult) -> dict[str, Any]:
    return {
        "candidate_name": result.candidate_name,
        "candidate_json": result.candidate_json,
        "decision": result.decision,
        "changed_rows": result.changed_rows,
        "same_quadrant_changes": result.same_quadrant_changes,
        "cross_quadrant_changes": result.cross_quadrant_changes,
        "classification_expected": result.classification.expected,
        "classification_lower": result.classification.lower,
        "classification_upper": result.classification.upper,
        "description_expected": result.description.expected,
        "description_lower": result.description.lower,
        "description_upper": result.description.upper,
        "overall_expected": result.overall.expected,
        "overall_lower": result.overall.lower,
        "overall_upper": result.overall.upper,
        "safety_passed": result.safety.passed,
        "safety_issue_codes": ",".join(result.safety.issue_codes),
        "description_issue_count": result.safety.description_issue_count,
        "top_emotion": result.safety.distribution.get("top_emotion", ""),
        "top_emotion_share": result.safety.distribution.get("top_emotion_share", 0.0),
    }


def _render_shadow_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 Local Shadow Evaluator Report",
        "",
        f"> {report['warning']}",
        "",
        f"- Method: `{report['method']}`",
        f"- Baseline JSON: `{report['baseline_json']}`",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        "",
        "## Candidate Ranking",
        "",
        "| rank | candidate | decision | overall lower | overall expected | class expected | desc expected | changes | cross quadrant |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(report["ranking"], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    str(item["candidate_name"]),
                    str(item["decision"]),
                    f"{float(item['overall_lower']):.6f}",
                    f"{float(item['overall_expected']):.6f}",
                    f"{float(item['classification_expected']):.6f}",
                    f"{float(item['description_expected']):.6f}",
                    str(item["changed_rows"]),
                    str(item["cross_quadrant_changes"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_shadow_html(report: dict[str, Any], row_risks: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item['candidate_name']))}</td>"
        f"<td>{html.escape(str(item['decision']))}</td>"
        f"<td>{float(item['overall_lower']):.6f}</td>"
        f"<td>{float(item['overall_expected']):.6f}</td>"
        f"<td>{int(item['changed_rows'])}</td>"
        f"<td>{int(item['cross_quadrant_changes'])}</td>"
        "</tr>"
        for item in report["ranking"]
    )
    risk_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('candidate_name', '')))}</td>"
        f"<td>{html.escape(str(item.get('sample_id', '')))}</td>"
        f"<td>{html.escape(str(item.get('transition', '')))}</td>"
        f"<td>{html.escape(str(item.get('risk', '')))}</td>"
        "</tr>"
        for item in row_risks[:300]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Track2 Shadow Evaluator</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px; text-align: left; }}
    th {{ background: #eef2f6; }}
    .warning {{ padding: 12px; background: #fff4d6; border: 1px solid #e5c66a; }}
  </style>
</head>
<body>
  <h1>Track2 Local Shadow Evaluator</h1>
  <p class="warning">{html.escape(str(report['warning']))}</p>
  <h2>Candidate Ranking</h2>
  <table>
    <thead><tr><th>candidate</th><th>decision</th><th>overall lower</th><th>overall expected</th><th>changes</th><th>cross quadrant</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Top Row Risks</h2>
  <table>
    <thead><tr><th>candidate</th><th>sample</th><th>transition</th><th>risk</th></tr></thead>
    <tbody>{risk_rows}</tbody>
  </table>
</body>
</html>
"""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_calibration_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "submission_id": str(entry["submission_id"]),
        "file_name": str(entry["file_name"]),
        "shadow_overall_expected": float(entry["shadow_overall_expected"]),
        "shadow_classification_expected": float(entry["shadow_classification_expected"]),
        "shadow_description_expected": float(entry["shadow_description_expected"]),
        "official_overall": float(entry["official_overall"]),
        "official_classification": float(entry["official_classification"]),
        "official_description": float(entry["official_description"]),
        "notes": str(entry.get("notes", "")),
    }
    for key, value in normalized.items():
        if key.endswith("_expected") or key.startswith("official_"):
            if not math.isfinite(float(value)):
                raise ValueError(f"calibration value must be finite: {key}")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"calibration value must be in [0, 1]: {key}")
    if not normalized["submission_id"]:
        raise ValueError("submission_id is required")
    if not normalized["file_name"]:
        raise ValueError("file_name is required")
    return normalized


def _mean_abs_error(entries: list[dict[str, Any]], predicted_key: str, actual_key: str) -> float:
    if not entries:
        return 0.0
    return sum(abs(float(item[predicted_key]) - float(item[actual_key])) for item in entries) / len(entries)


def _classification_band(
    *,
    safety: SafetyGateResult,
    changed_rows: int,
    same_quadrant: int,
    cross_quadrant: int,
) -> ScoreBand:
    if not safety.passed:
        return ScoreBand(expected=0.0, lower=0.0, upper=0.0)
    expected_delta = min(0.055, 0.0012 * same_quadrant - 0.0060 * cross_quadrant)
    uncertainty = 0.0009 * changed_rows + 0.0120 * cross_quadrant
    return ScoreBand(
        expected=OFFICIAL_ANCHOR.classification + expected_delta,
        lower=OFFICIAL_ANCHOR.classification + expected_delta - uncertainty,
        upper=OFFICIAL_ANCHOR.classification + expected_delta + uncertainty + 0.0120,
    ).clamped()


def _description_band(*, safety: SafetyGateResult, row_count: int) -> ScoreBand:
    if not safety.passed:
        return ScoreBand(expected=0.0, lower=0.0, upper=0.0)
    issue_rate = safety.description_issue_count / max(1, row_count)
    expected = OFFICIAL_ANCHOR.description
    lower = expected - min(0.040, issue_rate * 0.030 + 0.006)
    upper = min(1.0, expected + 0.018)
    return ScoreBand(expected=expected, lower=lower, upper=upper).clamped()


def _decision_for_score(
    *,
    safety: SafetyGateResult,
    overall: ScoreBand,
    description: ScoreBand,
    cross_quadrant: int,
) -> str:
    if not safety.passed:
        return "blocked"
    if cross_quadrant and overall.lower < OFFICIAL_ANCHOR.overall:
        return "recommend_hold"
    if description.lower < OFFICIAL_ANCHOR.description - 0.040:
        return "recommend_hold"
    if overall.lower >= OFFICIAL_ANCHOR.overall - 0.040:
        return "recommend_submit"
    return "recommend_hold"


def _build_row_risks(baseline_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_id = {str(row.get("sample_id", "")): row for row in baseline_rows}
    risks: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        before = baseline_by_id.get(sample_id)
        if before is None:
            risks.append(
                {
                    "sample_id": sample_id,
                    "transition": "missing_baseline",
                    "same_quadrant": False,
                    "risk": "missing_baseline",
                }
            )
            continue
        before_label = _label_triplet(before)
        after_label = _label_triplet(row)
        if before_label == after_label:
            continue
        same_quadrant = before_label["valence"] == after_label["valence"] and before_label["arousal"] == after_label["arousal"]
        risks.append(
            {
                "sample_id": sample_id,
                "transition": f"{before_label['emotion']}->{after_label['emotion']}",
                "from": before_label,
                "to": after_label,
                "same_quadrant": same_quadrant,
                "risk": "same_quadrant" if same_quadrant else "cross_quadrant",
            }
        )
    return risks


def _label_triplet(row: dict[str, Any]) -> dict[str, str]:
    return {
        "emotion": str(row.get("emotion", "")),
        "valence": str(row.get("emotional_valence", "")),
        "arousal": str(row.get("emotional_arousal_level", "")),
    }


def _is_formal_path(path: Path, formal_paths: set[Path]) -> bool:
    normalized = Path(path)
    return any(normalized == item or normalized.as_posix().endswith(item.as_posix()) for item in formal_paths)


def _clamp01(value: float) -> float:
    if math.isnan(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
