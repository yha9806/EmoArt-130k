from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_SUBMISSION_KEYS, TRACK2_JSON_TEXT_FIELDS
from affectiveart.track2_description_score import EVALUATOR_MANIPULATION_PATTERNS
from affectiveart.track2_local_shadow_evaluator import write_shadow_evaluator_outputs


FORMAL_SUBMISSION_PATHS = {
    Path("submissions/track2_submission.json"),
    Path("submissions/track2_submission.zip"),
}

_CULTURE_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("chinese", ("chinese", "ink", "scroll", "calligraphy", "bamboo", "gongbi", "literati")),
    ("japanese", ("japanese", "ukiyo-e", "woodblock", "kimono", "nihonga", "rinpa")),
    ("islamic", ("islamic", "persian", "miniature", "arabesque", "calligraphic border")),
    ("indian", ("indian", "mughal", "rajput", "pahari", "miniature")),
    ("korean", ("korean", "joseon", "minhwa")),
    ("mural", ("mural", "fresco", "cave", "dunhuang", "ajanta")),
    ("western", ("renaissance", "christ", "madonna", "oil painting", "portrait", "western")),
]

_OVERDEEP_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("symbolic_overclaim", re.compile(r"\bsymboli[sz](?:es|ed|ing)?\b", re.IGNORECASE)),
    ("embodiment_overclaim", re.compile(r"\bembod(?:y|ies|ied|ying)\b", re.IGNORECASE)),
    ("philosophy_overclaim", re.compile(r"\bphilosoph(?:y|ical|ically)\b", re.IGNORECASE)),
    ("spiritual_overclaim", re.compile(r"\bspiritual(?:\s+\w+){0,3}\b", re.IGNORECASE)),
    ("doctrine_overclaim", re.compile(r"\bdoctrine\b", re.IGNORECASE)),
    ("iconography_overclaim", re.compile(r"\biconograph(?:y|ic)\b", re.IGNORECASE)),
]

_HISTORICAL_CERTAINTY_PATTERN = re.compile(
    r"\b(?:yuan|ming|qing|song|tang|renaissance|baroque|medieval|byzantine)\s+(?:dynasty|period|era|style)?\b",
    re.IGNORECASE,
)

_EMOTION_CONFLICT_TERMS: dict[str, tuple[str, ...]] = {
    "calm": ("fatigue", "exhaustion", "grief", "terror", "rage", "panic", "threatening"),
    "content": ("fatigue", "exhaustion", "grief", "terror", "rage", "panic", "threatening"),
    "glad": ("fatigue", "exhaustion", "grief", "terror", "rage", "panic", "threatening"),
    "sad": ("joyful", "cheerful", "uplifting", "festive", "ecstatic"),
    "bored": ("joyful", "cheerful", "uplifting", "festive", "ecstatic"),
    "tired": ("joyful", "cheerful", "uplifting", "festive", "ecstatic"),
}

_OVERDEEP_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsymboli[sz](?:es|ed|ing)?\b", re.IGNORECASE), "suggests"),
    (re.compile(r"\bembod(?:y|ies|ied|ying)\b", re.IGNORECASE), "conveys"),
    (re.compile(r"\bphilosophical\b", re.IGNORECASE), "reflective"),
    (re.compile(r"\bspiritual doctrine\b", re.IGNORECASE), "quiet mood"),
    (re.compile(r"\bspiritual presence\b", re.IGNORECASE), "quiet focus"),
    (re.compile(r"\biconographic\b", re.IGNORECASE), "decorative"),
]

_POSITIVE_LOW_CONFLICT_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bfatigue\b", re.IGNORECASE), "repose"),
    (re.compile(r"\bexhaustion\b", re.IGNORECASE), "repose"),
    (re.compile(r"\bgrief\b", re.IGNORECASE), "subdued stillness"),
    (re.compile(r"\bterror\b", re.IGNORECASE), "tension"),
    (re.compile(r"\brage\b", re.IGNORECASE), "energy"),
    (re.compile(r"\bpanic\b", re.IGNORECASE), "movement"),
]


def infer_vulca_culture(row: dict[str, Any]) -> str:
    text = _combined_text(row).lower()
    scores: Counter[str] = Counter()
    for culture, cues in _CULTURE_CUES:
        for cue in cues:
            if cue in text:
                scores[culture] += 1
    if not scores:
        return "western"
    return scores.most_common(1)[0][0]


def audit_vulca_entailment_row(row: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    emotion = str(row.get("emotion", "")).lower()
    culture = infer_vulca_culture(row)

    for field in TRACK2_JSON_TEXT_FIELDS:
        value = str(row.get(field, ""))
        lower = value.lower()
        for pattern in EVALUATOR_MANIPULATION_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                issues.append({"code": "evaluator_manipulation_risk", "field": field, "pattern": pattern})

        for family, pattern in _OVERDEEP_PATTERNS:
            if pattern.search(value):
                issues.append({"code": "overdeep_cultural_claim", "field": field, "family": family})

        if _HISTORICAL_CERTAINTY_PATTERN.search(value) and not _has_hedge(value):
            issues.append({"code": "unsupported_historical_certainty", "field": field})

    conflict_terms = _EMOTION_CONFLICT_TERMS.get(emotion, ())
    hits = sorted({term for term in conflict_terms if term in _combined_text(row).lower()})
    if hits:
        issues.append({"code": "emotion_text_conflict", "field": "all_text", "terms": hits})

    counts = Counter(issue["code"] for issue in issues)
    return {
        "sample_id": str(row.get("sample_id", "")),
        "emotion": str(row.get("emotion", "")),
        "culture": culture,
        "issue_count": len(issues),
        "issue_counts": dict(sorted(counts.items())),
        "pseudo_understanding_risk": _pseudo_understanding_risk(issues),
        "issues": issues,
    }


def build_vulca_entailment_candidate_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before_audits = [audit_vulca_entailment_row(row) for row in rows]
    repaired_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []

    for row in rows:
        repaired = dict(row)
        field_changes = []
        for field in TRACK2_JSON_TEXT_FIELDS:
            original = str(repaired.get(field, ""))
            updated = _repair_field(original, emotion=str(repaired.get("emotion", "")))
            if updated != original:
                repaired[field] = updated
                field_changes.append({"field": field, "from": original, "to": updated})
        if field_changes:
            changes.append({"sample_id": str(row.get("sample_id", "")), "field_changes": field_changes})
        repaired_rows.append(_submission_row(repaired))

    after_audits = [audit_vulca_entailment_row(row) for row in repaired_rows]
    report = {
        "method": "track2_vulca_entailment_gate_v1",
        "policy": (
            "Conservative VULCA/Judge++-style text gate. It treats expert-sounding "
            "unsupported cultural, historical, or emotion-conflicting claims as "
            "pseudo-understanding risk and rewrites only open text fields."
        ),
        "row_count": len(rows),
        "changed_rows": len(changes),
        "changed_fields": sum(len(item["field_changes"]) for item in changes),
        "classification_label_changes": _classification_label_changes(rows, repaired_rows),
        "risk_summary_before": _summarize_audits(before_audits),
        "risk_summary_after": _summarize_audits(after_audits),
        "changes": changes,
        "formal_submission_overwritten": False,
    }
    return repaired_rows, report


def write_vulca_entailment_gate_outputs(
    *,
    source_json: str | Path,
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
    baseline_json: str | Path | None = None,
    shadow_out_dir: str | Path | None = None,
    expected_row_count: int = 1000,
) -> dict[str, Any]:
    source_json = Path(source_json)
    out_json = Path(out_json)
    out_zip = Path(out_zip)
    rows = _load_json_rows(source_json)
    repaired_rows, report = build_vulca_entailment_candidate_rows(rows)
    report["source_json"] = str(source_json)
    report["out_json"] = str(out_json)
    report["out_zip"] = str(out_zip)
    report["formal_submission_overwritten"] = _is_formal_path(out_json) or _is_formal_path(out_zip)

    _write_json(out_json, repaired_rows)
    _write_zip(out_zip, out_json)

    if baseline_json is not None and shadow_out_dir is not None:
        shadow_report = write_shadow_evaluator_outputs(
            baseline_json=baseline_json,
            candidates=[{"name": "vulca_entailment_v9", "json": out_json}],
            out_dir=shadow_out_dir,
            expected_row_count=expected_row_count,
        )
        report["shadow_report_json"] = str(Path(shadow_out_dir) / "shadow_score_report.json")
        report["shadow_top"] = shadow_report["ranking"][0]

    _write_json(Path(report_json), report)
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text(render_vulca_entailment_markdown(report), encoding="utf-8")
    return report


def render_vulca_entailment_markdown(report: dict[str, Any]) -> str:
    before = report["risk_summary_before"]
    after = report["risk_summary_after"]
    lines = [
        "# Track2 VULCA Entailment Gate Report",
        "",
        f"- Method: `{report['method']}`",
        f"- Rows: {report['row_count']}",
        f"- Changed rows: {report['changed_rows']}",
        f"- Changed fields: {report['changed_fields']}",
        f"- Classification label changes: {report['classification_label_changes']}",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        "",
        "## Risk Summary",
        "",
        f"- Before issues: {before['total_issues']}",
        f"- After issues: {after['total_issues']}",
        f"- Before average pseudo-understanding risk: {before['mean_pseudo_understanding_risk']:.4f}",
        f"- After average pseudo-understanding risk: {after['mean_pseudo_understanding_risk']:.4f}",
    ]
    if report.get("shadow_top"):
        top = report["shadow_top"]
        lines.extend(
            [
                "",
                "## Shadow Score",
                "",
                f"- Candidate: `{top['candidate_name']}`",
                f"- Decision: `{top['decision']}`",
                f"- Overall expected: {float(top['overall_expected']):.6f}",
                f"- Overall lower: {float(top['overall_lower']):.6f}",
                f"- Description expected: {float(top['description_expected']):.6f}",
            ]
        )
    lines.extend(["", "## Accepted Text Repairs", ""])
    if report.get("changes"):
        for item in report["changes"][:160]:
            fields = ", ".join(change["field"] for change in item["field_changes"])
            lines.append(f"- {item['sample_id']}: {fields}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _repair_field(text: str, *, emotion: str) -> str:
    updated = text
    for pattern, replacement in _OVERDEEP_REPLACEMENTS:
        updated = pattern.sub(replacement, updated)
    if emotion.lower() not in {"calm", "content", "glad"}:
        return _clean_spacing(updated)
    for pattern, replacement in _POSITIVE_LOW_CONFLICT_REPLACEMENTS:
        updated = pattern.sub(replacement, updated)
    return _clean_spacing(updated)


def _summarize_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    total_issues = sum(int(item.get("issue_count", 0)) for item in audits)
    risk_total = sum(float(item.get("pseudo_understanding_risk", 0.0)) for item in audits)
    counts = Counter(code for item in audits for code, count in item.get("issue_counts", {}).items() for _ in range(int(count)))
    return {
        "row_count": len(audits),
        "rows_with_issues": sum(1 for item in audits if int(item.get("issue_count", 0)) > 0),
        "total_issues": total_issues,
        "issue_counts": dict(sorted(counts.items())),
        "mean_pseudo_understanding_risk": risk_total / max(1, len(audits)),
    }


def _pseudo_understanding_risk(issues: list[dict[str, Any]]) -> float:
    weights = {
        "evaluator_manipulation_risk": 1.0,
        "emotion_text_conflict": 0.8,
        "unsupported_historical_certainty": 0.55,
        "overdeep_cultural_claim": 0.35,
    }
    total = sum(weights.get(str(issue.get("code", "")), 0.25) for issue in issues)
    return max(0.0, min(1.0, total))


def _classification_label_changes(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> int:
    labels = ("emotion", "emotional_valence", "emotional_arousal_level")
    before = {str(row.get("sample_id", "")): tuple(str(row.get(label, "")) for label in labels) for row in before_rows}
    after = {str(row.get("sample_id", "")): tuple(str(row.get(label, "")) for label in labels) for row in after_rows}
    return sum(1 for sample_id, labels_before in before.items() if labels_before != after.get(sample_id))


def _submission_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS}


def _combined_text(row: dict[str, Any]) -> str:
    return "\n".join(str(row.get(field, "")) for field in TRACK2_JSON_TEXT_FIELDS)


def _has_hedge(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in ("possibly", "likely", "suggests", "appears", "seems", "style"))


def _clean_spacing(text: str) -> str:
    updated = re.sub(r"\s+", " ", text).strip()
    updated = re.sub(r"\s+\.", ".", updated)
    updated = re.sub(r"\.{2,}", ".", updated)
    return updated


def _is_formal_path(path: Path) -> bool:
    normalized = Path(path)
    return any(normalized == item or normalized.as_posix().endswith(item.as_posix()) for item in FORMAL_SUBMISSION_PATHS)


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected Track2 JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_zip(zip_path: Path, json_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, "submission.json")
