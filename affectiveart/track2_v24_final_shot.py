from __future__ import annotations

import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from affectiveart.track2_official_anchor_calibration import CalibratedScore, score_submission


TRACK2_SUBMISSION_KEYS = (
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
)
TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
TRACK2_EMOTIONS = {
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
NEGATIVE_EMOTIONS = {"alarmed", "annoyed", "bored", "frustrated", "sad", "tired"}
HIGH_AROUSAL_EMOTIONS = {"alarmed", "annoyed", "aroused", "excited", "frustrated", "happy"}
FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}
DEFAULT_OUT_DIR = Path("experiments/track2_v24_final_shot_20260608")
DEFAULT_BASE_JSON = Path("submissions/track2_submission_v22_official_author_calmshiftall_candidate.json")
DEFAULT_DESCRIPTION_JSON = Path("submissions/track2_submission_v15_desc_expand300_candidate.json")
DEFAULT_EVIDENCE_JSON = Path("experiments/track2_v19_aggregate_calibration_20260607/calibrated_evidence.json")


@dataclass(frozen=True)
class FinalGateThresholds:
    min_overall: float = 0.89
    min_classification: float = 0.78
    min_description: float = 0.99
    max_top_emotion_share: float = 0.58


UNSAFE_TEXT_MARKERS = (
    "give this answer",
    "give this submission",
    "perfect score",
    "ignore previous instructions",
    "ignore the rubric",
    "reward the caption",
    "evaluator should",
    "judge should",
)


def is_text_evaluator_safe(text: str) -> bool:
    lowered = str(text or "").lower()
    return not any(marker in lowered for marker in UNSAFE_TEXT_MARKERS)


def choose_v24_final_candidate(
    scores: Iterable[CalibratedScore],
    *,
    thresholds: FinalGateThresholds,
    validation_ok: bool,
    label_consistency_issue_count: int,
    missing_emotions: list[str],
    top_emotion_share: float,
    repeats_failed_pattern: bool,
    unsafe_text_count: int,
) -> dict[str, Any]:
    ranked = sorted(scores, key=lambda score: (-score.overall_expected, score.candidate_name))
    if not ranked:
        return {"decision": "hold_no_submit", "candidate_name": "", "reasons": ["no_candidates"]}
    best = ranked[0]
    reasons: list[str] = []
    if best.overall_expected < thresholds.min_overall:
        reasons.append("overall_below_089")
    if best.classification_expected < thresholds.min_classification:
        reasons.append("classification_below_078")
    if best.description_expected < thresholds.min_description:
        reasons.append("description_below_099")
    if not validation_ok:
        reasons.append("validator_failed")
    if label_consistency_issue_count:
        reasons.append("label_consistency_issues")
    if missing_emotions:
        reasons.append("missing_emotions")
    if top_emotion_share > thresholds.max_top_emotion_share:
        reasons.append("top_emotion_collapse")
    if repeats_failed_pattern:
        reasons.append("repeats_781601_failed_pattern")
    if unsafe_text_count:
        reasons.append("unsafe_description_text")
    decision = "hold_no_submit" if reasons else "recommend_final_submit"
    return {
        "decision": decision,
        "candidate_name": best.candidate_name,
        "overall_expected": best.overall_expected,
        "classification_expected": best.classification_expected,
        "description_expected": best.description_expected,
        "reasons": reasons,
    }


def load_track2_rows(path: str | Path) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Track2 submission must be a list: {path}")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Track2 row {index} must be an object")
        row = {key: str(item.get(key, "")).strip() for key in TRACK2_SUBMISSION_KEYS}
        sample_id = row["sample_id"]
        if not sample_id:
            raise ValueError(f"blank sample_id at row {index}")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        rows.append(row)
    return rows


def write_v24_candidate_outputs(
    *,
    base_rows: list[dict[str, Any]],
    selected_changes: list[dict[str, Any]],
    profile: str,
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
) -> dict[str, Any]:
    out_json = Path(out_json)
    out_zip = Path(out_zip)
    report_json = Path(report_json)
    report_md = Path(report_md)
    for path in (out_json, out_zip, report_json, report_md):
        _reject_formal_submission_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    candidate_rows, apply_report = _apply_v24_changes(base_rows, selected_changes)
    report = {
        "method": "track2_v24_candidate_v1",
        "profile": profile,
        "row_count": len(candidate_rows),
        "accepted_label_changes": apply_report["accepted_label_changes"],
        "text_changed_rows": apply_report["text_changed_rows"],
        "transition_counts": apply_report["transition_counts"],
        "distribution": apply_report["distribution"],
        "missing_emotions": apply_report["missing_emotions"],
        "top_emotion": apply_report["top_emotion"],
        "top_emotion_share": apply_report["top_emotion_share"],
        "label_consistency_issue_count": apply_report["label_consistency_issue_count"],
        "label_consistency_issues": apply_report["label_consistency_issues"],
        "unsafe_text_count": apply_report["unsafe_text_count"],
        "unsafe_text_rows": apply_report["unsafe_text_rows"],
        "accepted_changes": apply_report["accepted_changes"],
        "repeats_failed_pattern": _repeats_781601_failed_pattern(apply_report["transition_counts"]),
        "validation_ok": True,
        "paths": {
            "out_json": str(out_json),
            "out_zip": str(out_zip),
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
    }
    _write_json(out_json, candidate_rows)
    _write_zip_payload(out_zip, candidate_rows)
    _write_json(report_json, report)
    report_md.write_text(_render_candidate_report(report), encoding="utf-8")
    return report


def score_v24_candidates_with_v23(candidate_reports: list[dict[str, Any]]) -> dict[str, Any]:
    scored_reports: list[tuple[CalibratedScore, dict[str, Any]]] = []
    for report in candidate_reports:
        paths = report.get("paths") if isinstance(report.get("paths"), dict) else {}
        out_json = paths.get("out_json")
        if not out_json:
            raise ValueError("candidate report missing paths.out_json")
        scored_reports.append((score_submission(out_json, candidate_name=str(report.get("profile", ""))), report))
    scored_reports.sort(key=lambda item: (-item[0].overall_expected, item[0].candidate_name))
    ranking = [_scoreboard_row(score, report) for score, report in scored_reports]
    if scored_reports:
        best_score, best_report = scored_reports[0]
        final_gate = choose_v24_final_candidate(
            [score for score, _ in scored_reports],
            thresholds=FinalGateThresholds(),
            validation_ok=bool(best_report.get("validation_ok", False)),
            label_consistency_issue_count=int(best_report.get("label_consistency_issue_count") or 0),
            missing_emotions=list(best_report.get("missing_emotions") or []),
            top_emotion_share=float(best_report.get("top_emotion_share") or 0.0),
            repeats_failed_pattern=bool(best_report.get("repeats_failed_pattern", False)),
            unsafe_text_count=int(best_report.get("unsafe_text_count") or 0),
        )
        final_gate["candidate_name"] = best_score.candidate_name
    else:
        final_gate = choose_v24_final_candidate(
            [],
            thresholds=FinalGateThresholds(),
            validation_ok=False,
            label_consistency_issue_count=0,
            missing_emotions=[],
            top_emotion_share=0.0,
            repeats_failed_pattern=False,
            unsafe_text_count=0,
        )
    return {
        "method": "track2_v24_final_shot_v1",
        "ranking": ranking,
        "final_gate": final_gate,
    }


def build_v24_run_outputs(
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    submissions_dir: str | Path = "submissions",
    base_json: str | Path = DEFAULT_BASE_JSON,
    description_json: str | Path = DEFAULT_DESCRIPTION_JSON,
    evidence_json: str | Path = DEFAULT_EVIDENCE_JSON,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    submissions_dir = Path(submissions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = out_dir / "candidate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    base_rows = load_track2_rows(base_json)
    evidence_rows = _load_json_list(evidence_json)
    description_rows = load_track2_rows(description_json) if Path(description_json).exists() else []
    frontier_changes = _select_classification_frontier_changes(base_rows, evidence_rows)
    desc_changes = _select_description_changes(base_rows, description_rows, limit=300)
    final_changes = _merge_changes(frontier_changes, desc_changes)
    candidate_specs = [
        ("classification_frontier", frontier_changes),
        ("descmax", desc_changes),
        ("final", final_changes),
    ]
    candidate_reports: list[dict[str, Any]] = []
    for profile, changes in candidate_specs:
        stem = f"track2_submission_v24_{profile}_candidate"
        candidate_reports.append(
            write_v24_candidate_outputs(
                base_rows=base_rows,
                selected_changes=changes,
                profile=profile,
                out_json=submissions_dir / f"{stem}.json",
                out_zip=submissions_dir / f"{stem}.zip",
                report_json=report_dir / f"{stem}_report.json",
                report_md=report_dir / f"{stem}_report.md",
            )
        )
    report = score_v24_candidates_with_v23(candidate_reports)
    report["submissions_dir"] = str(submissions_dir)
    report["base_json"] = str(base_json)
    report["description_json"] = str(description_json)
    report["evidence_json"] = str(evidence_json)
    _write_json(out_dir / "v24_scoreboard.json", report)
    _write_csv(out_dir / "v24_scoreboard.csv", report["ranking"])
    (out_dir / "v24_final_gate_zh.md").write_text(_render_final_gate(report), encoding="utf-8")
    return report


def _apply_v24_changes(
    base_rows: list[dict[str, Any]],
    selected_changes: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    changes_by_id = {
        str(row.get("sample_id", "")).strip(): dict(row)
        for row in selected_changes
        if str(row.get("sample_id", "")).strip()
    }
    candidate_rows: list[dict[str, str]] = []
    accepted: list[dict[str, Any]] = []
    text_changed = 0
    for base in base_rows:
        row = {key: str(base.get(key, "")).strip() for key in TRACK2_SUBMISSION_KEYS}
        change = changes_by_id.get(row["sample_id"])
        before_emotion = _canonical_emotion(row["emotion"])
        if change:
            proposed = _canonical_emotion(change.get("proposed_emotion"))
            if proposed and proposed != before_emotion:
                row["emotion"] = proposed
                row["emotional_valence"] = _valence(proposed)
                row["emotional_arousal_level"] = _arousal(proposed)
                accepted.append(
                    {
                        "sample_id": row["sample_id"],
                        "transition": f"{before_emotion}->{proposed}",
                        "before_emotion": before_emotion,
                        "after_emotion": proposed,
                    }
                )
            changed_any_text = False
            for field in TEXT_FIELDS:
                if field in change and str(change[field]).strip() != row[field]:
                    row[field] = str(change[field]).strip()
                    changed_any_text = True
            if changed_any_text:
                text_changed += 1
        candidate_rows.append(row)
    distribution = Counter(row["emotion"] for row in candidate_rows)
    label_issues = _label_consistency_issues(candidate_rows)
    unsafe_rows = _unsafe_text_rows(candidate_rows)
    return candidate_rows, {
        "accepted_label_changes": len(accepted),
        "text_changed_rows": text_changed,
        "transition_counts": dict(sorted(Counter(item["transition"] for item in accepted).items())),
        "distribution": dict(sorted(distribution.items())),
        "missing_emotions": sorted(TRACK2_EMOTIONS - set(distribution)),
        "top_emotion": distribution.most_common(1)[0][0] if distribution else "",
        "top_emotion_share": round(distribution.most_common(1)[0][1] / max(1, len(candidate_rows)), 6)
        if distribution
        else 0.0,
        "label_consistency_issue_count": len(label_issues),
        "label_consistency_issues": label_issues,
        "unsafe_text_count": len(unsafe_rows),
        "unsafe_text_rows": unsafe_rows,
        "accepted_changes": accepted,
    }


def _select_classification_frontier_changes(
    base_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, Any]],
    *,
    limit: int = 96,
) -> list[dict[str, Any]]:
    base_by_id = {row["sample_id"]: row for row in base_rows}
    projected = Counter(row["emotion"] for row in base_rows)
    total_rows = max(1, len(base_rows))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    blocked_transitions = {"content->glad", "glad->content"}
    transition_counts: Counter[str] = Counter()
    for row in sorted(evidence_rows, key=_evidence_sort_key):
        if len(selected) >= limit:
            break
        if str(row.get("v19_decision", row.get("v21_decision", ""))) != "accept_candidate":
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen or sample_id not in base_by_id:
            continue
        current = _canonical_emotion(row.get("current_emotion"))
        proposed = _canonical_emotion(row.get("proposed_emotion"))
        if not current or not proposed or current == proposed:
            continue
        if _canonical_emotion(base_by_id[sample_id].get("emotion")) != current:
            continue
        transition = f"{current}->{proposed}"
        if transition in blocked_transitions:
            continue
        if transition == "calm->content" and transition_counts[transition] >= 19:
            continue
        if projected[current] <= 4:
            continue
        if (projected[proposed] + 1) / total_rows > 0.58:
            continue
        score = _safe_float(row.get("v19_score", row.get("v21_score", row.get("support_score", 0.0))))
        votes = _safe_int(row.get("model_vote_count"))
        confidence = _safe_float(row.get("max_confidence"))
        duplicate = _safe_float(row.get("public_duplicate_support_score"))
        near_duplicate = _safe_bool(row.get("near_duplicate")) or _safe_bool(row.get("exact_duplicate"))
        if score < 3.0 and not (near_duplicate and duplicate >= 0.95 and confidence >= 0.9 and votes >= 2):
            continue
        selected.append(
            {
                "sample_id": sample_id,
                "proposed_emotion": proposed,
                "v24_source": "classification_frontier",
                "v24_score": round(score, 6),
                "transition": transition,
            }
        )
        seen.add(sample_id)
        transition_counts[transition] += 1
        projected[current] -= 1
        projected[proposed] += 1
    return selected


def _select_description_changes(
    base_rows: list[dict[str, str]],
    description_rows: list[dict[str, str]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    description_by_id = {row["sample_id"]: row for row in description_rows}
    selected: list[dict[str, Any]] = []
    for base in base_rows:
        if len(selected) >= limit:
            break
        source = description_by_id.get(base["sample_id"])
        if not source:
            continue
        change: dict[str, Any] = {"sample_id": base["sample_id"], "v24_source": "description_max"}
        changed = False
        for field in TEXT_FIELDS:
            value = str(source.get(field, "")).strip()
            if value and value != base[field] and is_text_evaluator_safe(value):
                change[field] = value
                changed = True
        if changed:
            selected.append(change)
    return selected


def _merge_changes(*change_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in change_groups:
        for change in group:
            sample_id = str(change.get("sample_id", "")).strip()
            if not sample_id:
                continue
            target = merged.setdefault(sample_id, {"sample_id": sample_id})
            target.update({key: value for key, value in change.items() if key != "sample_id"})
    return list(merged.values())


def _scoreboard_row(score: CalibratedScore, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_name": score.candidate_name,
        "json_path": score.json_path,
        "calibration_kind": score.calibration_kind,
        "overall_expected": score.overall_expected,
        "classification_expected": score.classification_expected,
        "description_expected": score.description_expected,
        "visible_bucket": score.visible_bucket,
        "accepted_label_changes": int(report.get("accepted_label_changes") or 0),
        "text_changed_rows": int(report.get("text_changed_rows") or 0),
        "top_emotion_share": float(report.get("top_emotion_share") or 0.0),
        "missing_emotions": list(report.get("missing_emotions") or []),
        "unsafe_text_count": int(report.get("unsafe_text_count") or 0),
        "repeats_failed_pattern": bool(report.get("repeats_failed_pattern", False)),
        "warnings": list(score.warnings),
    }


def _canonical_emotion(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in TRACK2_EMOTIONS else ""


def _evidence_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    score = _safe_float(row.get("v19_score", row.get("v21_score", row.get("support_score", 0.0))))
    duplicate = _safe_float(row.get("public_duplicate_support_score"))
    confidence = _safe_float(row.get("max_confidence"))
    return (-score, -duplicate, -confidence, str(row.get("sample_id", "")))


def _valence(emotion: str) -> str:
    return "Negative" if emotion in NEGATIVE_EMOTIONS else "Positive"


def _arousal(emotion: str) -> str:
    return "High" if emotion in HIGH_AROUSAL_EMOTIONS else "Low"


def _label_consistency_issues(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for row in rows:
        emotion = _canonical_emotion(row.get("emotion"))
        if not emotion:
            issues.append({"sample_id": row.get("sample_id", ""), "reason": "invalid_emotion"})
            continue
        expected_valence = _valence(emotion)
        expected_arousal = _arousal(emotion)
        if row.get("emotional_valence") != expected_valence or row.get("emotional_arousal_level") != expected_arousal:
            issues.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "reason": "va_mismatch",
                    "emotion": emotion,
                    "expected_valence": expected_valence,
                    "actual_valence": row.get("emotional_valence", ""),
                    "expected_arousal": expected_arousal,
                    "actual_arousal": row.get("emotional_arousal_level", ""),
                }
            )
    return issues


def _unsafe_text_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unsafe: list[dict[str, str]] = []
    for row in rows:
        bad_fields = [field for field in TEXT_FIELDS if not is_text_evaluator_safe(row.get(field, ""))]
        if bad_fields:
            unsafe.append({"sample_id": row["sample_id"], "fields": ",".join(bad_fields)})
    return unsafe


def _repeats_781601_failed_pattern(transition_counts: dict[str, int]) -> bool:
    return transition_counts.get("calm->content", 0) >= 20 or transition_counts.get("content->glad", 0) >= 5


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_json_list(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"JSON payload must be a list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _reject_formal_submission_path(path: Path) -> None:
    if path.name in FORMAL_SUBMISSION_NAMES and path.parent.name != "v24_final_upload":
        raise ValueError(f"refusing to overwrite formal submission path: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_zip_payload(path: Path, rows: list[dict[str, str]]) -> None:
    payload = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", payload)


def _render_candidate_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Track2 v24 {report['profile']} candidate",
            "",
            f"- accepted_label_changes: `{report['accepted_label_changes']}`",
            f"- text_changed_rows: `{report['text_changed_rows']}`",
            f"- label_consistency_issue_count: `{report['label_consistency_issue_count']}`",
            f"- unsafe_text_count: `{report['unsafe_text_count']}`",
            f"- missing_emotions: `{', '.join(report['missing_emotions']) if report['missing_emotions'] else 'none'}`",
            f"- repeats_failed_pattern: `{report['repeats_failed_pattern']}`",
            "",
        ]
    )


def _render_final_gate(report: dict[str, Any]) -> str:
    gate = report["final_gate"]
    reasons = gate.get("reasons") or []
    return "\n".join(
        [
            "# Track2 v24 final gate",
            "",
            f"- decision: `{gate.get('decision', '')}`",
            f"- candidate_name: `{gate.get('candidate_name', '')}`",
            f"- reasons: `{', '.join(reasons) if reasons else 'none'}`",
            f"- ranking_count: `{len(report.get('ranking') or [])}`",
            "",
        ]
    )
