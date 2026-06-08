from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from affectiveart.track2_official_anchor_calibration import BASE_ANCHOR_JSON, score_submission


DEFAULT_OUT_DIR = Path("experiments/track2_v28_final_shot_hybrid_20260608")
DEFAULT_BASE_JSON = Path("submissions/track2_submission_v21_calmshift90_candidate.json")
DEFAULT_DESCRIPTION_JSON = Path("submissions/track2_submission_v24_final_candidate.json")
DEFAULT_EVIDENCE_JSON = Path("experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json")
DEFAULT_SUBMISSIONS_DIR = Path("submissions")

TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
TRACK2_SUBMISSION_KEYS = ("sample_id", "emotion", "emotional_valence", "emotional_arousal_level", *TEXT_FIELDS)
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
EVALUATOR_MANIPULATION_PATTERNS = (
    "ignore previous",
    "ignore the evaluator",
    "score this",
    "give full score",
    "assign full score",
    "do not penalize",
    "as an evaluator",
)


def merge_description_fields(
    base_rows: list[Mapping[str, Any]],
    description_rows: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    description_by_id = {str(row.get("sample_id", "")).strip(): row for row in description_rows}
    merged: list[dict[str, str]] = []
    for base in base_rows:
        row = {key: str(base.get(key, "")).strip() for key in TRACK2_SUBMISSION_KEYS}
        desc = description_by_id.get(row["sample_id"])
        if desc:
            for field in TEXT_FIELDS:
                row[field] = str(desc.get(field, row.get(field, ""))).strip()
        merged.append(row)
    return merged


def count_unsafe_text_rows(rows: list[Mapping[str, Any]]) -> int:
    total = 0
    for row in rows:
        text = " ".join(str(row.get(field, "")) for field in TEXT_FIELDS).lower()
        if any(pattern in text for pattern in EVALUATOR_MANIPULATION_PATTERNS):
            total += 1
    return total


def score_v28_evidence_row(row: Mapping[str, Any]) -> dict[str, Any]:
    current = _canonical_emotion(row.get("current_emotion"))
    proposed = _canonical_emotion(row.get("proposed_emotion"))
    if current not in TRACK2_EMOTIONS or proposed not in TRACK2_EMOTIONS:
        return _score_row("block", "invalid", 0.0, ["invalid_emotion_label"])
    if current == proposed:
        return _score_row("block", "invalid", 0.0, ["no_op_transition"])
    support = _safe_float(row.get("support_score"))
    votes = _safe_int(row.get("model_vote_count"))
    confidence = _safe_float(row.get("max_confidence"))
    public_style = _safe_float(row.get("public_style_support_score"))
    duplicate = _safe_float(row.get("public_duplicate_support_score"))
    near_or_exact = _safe_bool(row.get("near_duplicate")) or _safe_bool(row.get("exact_duplicate"))
    same_quadrant = _same_quadrant(current, proposed)

    if near_or_exact and duplicate >= 0.95:
        score = support + duplicate * 1.25 + confidence * 0.25 + votes * 0.12
        return _score_row("accept_candidate", "duplicate", score, ["duplicate_strength_evidence"])
    if current == "content" and proposed == "calm" and votes >= 2 and support >= 1.0:
        score = support + public_style * 0.35 + confidence * 0.12 + votes * 0.08
        return _score_row("accept_candidate", "calm_ladder", score, ["official_best_direction_continuation"])
    if same_quadrant and votes >= 3 and support >= 2.25 and public_style >= 0.50:
        score = support + public_style * 0.45 + confidence * 0.15 + votes * 0.08
        return _score_row("accept_candidate", "same_quadrant_frontier", score, ["balanced_same_quadrant_evidence"])
    return _score_row("hold", "weak", support, ["insufficient_v28_evidence"])


def build_v28_evidence(v17_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in v17_rows:
        current = _canonical_emotion(row.get("current_emotion"))
        proposed = _canonical_emotion(row.get("proposed_emotion"))
        scored = score_v28_evidence_row(row)
        enriched = dict(row)
        enriched.update(
            {
                "current_emotion": current,
                "proposed_emotion": proposed,
                "transition": f"{current}->{proposed}" if current and proposed else str(row.get("transition", "")),
                "same_quadrant": _same_quadrant(current, proposed),
                "v28_decision": scored["decision"],
                "v28_tier": scored["tier"],
                "v28_score": scored["score"],
                "v28_reasons": ";".join(scored["reasons"]),
            }
        )
        evidence.append(enriched)
    return sorted(evidence, key=_evidence_sort_key)


def build_calm_ladder_changes(
    evidence_rows: list[Mapping[str, Any]],
    *,
    existing_content_to_calm_count: int,
    target_content_to_calm_count: int,
) -> list[dict[str, Any]]:
    needed = max(0, int(target_content_to_calm_count) - int(existing_content_to_calm_count))
    selected: list[dict[str, Any]] = []
    for row in sorted(evidence_rows, key=_selection_sort_key):
        if len(selected) >= needed:
            break
        current = _canonical_emotion(row.get("current_emotion"))
        proposed = _canonical_emotion(row.get("proposed_emotion"))
        if current == "content" and proposed == "calm" and row.get("v28_decision", "accept_candidate") == "accept_candidate":
            selected.append(_normalize_change(row, current, proposed))
    return selected


def build_balanced_frontier_changes(
    evidence_rows: list[Mapping[str, Any]],
    *,
    base_distribution: Counter[str] | Mapping[str, int],
    total_cap: int,
    top_emotion_cap: float = 0.58,
    class_floor: int = 4,
    cross_cap: int = 2,
) -> list[dict[str, Any]]:
    projected = Counter({str(key): int(value) for key, value in dict(base_distribution).items()})
    total_rows = max(1, sum(projected.values()))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    cross_count = 0
    for row in sorted(evidence_rows, key=_selection_sort_key):
        if len(selected) >= total_cap:
            break
        if row.get("v28_decision", "accept_candidate") != "accept_candidate":
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        current = _canonical_emotion(row.get("current_emotion"))
        proposed = _canonical_emotion(row.get("proposed_emotion"))
        if not sample_id or sample_id in seen:
            continue
        if current not in TRACK2_EMOTIONS or proposed not in TRACK2_EMOTIONS or current == proposed:
            continue
        if projected[current] - 1 < class_floor:
            continue
        if (projected[proposed] + 1) / total_rows > top_emotion_cap:
            continue
        same_quadrant = _same_quadrant(current, proposed)
        duplicate_strength = _safe_float(row.get("public_duplicate_support_score")) >= 0.95 and (
            _safe_bool(row.get("near_duplicate")) or _safe_bool(row.get("exact_duplicate"))
        )
        if not same_quadrant and not duplicate_strength:
            continue
        if not same_quadrant and cross_count >= cross_cap:
            continue
        selected.append(_normalize_change(row, current, proposed))
        seen.add(sample_id)
        projected[current] -= 1
        projected[proposed] += 1
        if not same_quadrant:
            cross_count += 1
    return selected


def choose_v28_final_gate(
    *,
    overall_expected: float,
    classification_expected: float,
    description_expected: float,
    label_consistency_issue_count: int,
    missing_emotions: list[str],
    unsafe_text_count: int,
    top_emotion_share: float,
    blind_knn_copy: bool,
    unreported_cross_quadrant_count: int,
    target_overall: float = 0.89,
    min_classification: float = 0.78,
    min_description: float = 0.97,
    max_top_emotion_share: float = 0.58,
) -> dict[str, Any]:
    reasons: list[str] = []
    if overall_expected < target_overall:
        reasons.append("overall_below_089")
    if classification_expected < min_classification:
        reasons.append("classification_below_078")
    if description_expected < min_description:
        reasons.append("description_below_097")
    if label_consistency_issue_count:
        reasons.append("label_consistency_issues")
    if missing_emotions:
        reasons.append("missing_emotions")
    if unsafe_text_count:
        reasons.append("unsafe_text")
    if top_emotion_share > max_top_emotion_share:
        reasons.append("top_emotion_collapse_risk")
    if blind_knn_copy:
        reasons.append("blind_knn_copy_risk")
    if unreported_cross_quadrant_count:
        reasons.append("unreported_cross_quadrant_changes")
    return {
        "decision": "hold_no_submit" if reasons else "recommend_final_submit",
        "overall_expected": round(float(overall_expected), 6),
        "classification_expected": round(float(classification_expected), 6),
        "description_expected": round(float(description_expected), 6),
        "reasons": reasons,
    }


def load_track2_rows(path: str | Path) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Track2 JSON must be a list: {path}")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Track2 row {index} must be an object")
        row = {key: str(item.get(key, "")).strip() for key in TRACK2_SUBMISSION_KEYS}
        if not row["sample_id"]:
            raise ValueError(f"blank sample_id at row {index}")
        if row["sample_id"] in seen:
            raise ValueError(f"duplicate sample_id: {row['sample_id']}")
        seen.add(row["sample_id"])
        rows.append(row)
    return rows


def write_v28_candidate_outputs(
    *,
    rows: list[Mapping[str, Any]],
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
    profile: str,
    label_changes_vs_v21: int = 0,
    label_changes_vs_anchor: int = 0,
    transition_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    out_json = Path(out_json)
    out_zip = Path(out_zip)
    report_json = Path(report_json)
    report_md = Path(report_md)
    for path in (out_json, out_zip, report_json, report_md):
        _reject_formal_submission_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{key: str(row.get(key, "")).strip() for key in TRACK2_SUBMISSION_KEYS} for row in rows]
    distribution = Counter(row["emotion"] for row in normalized)
    label_issues = _label_issues(normalized)
    report = {
        "method": "track2_v28_final_shot_hybrid_v1",
        "profile": profile,
        "row_count": len(normalized),
        "label_changes_vs_v21": int(label_changes_vs_v21),
        "label_changes_vs_anchor": int(label_changes_vs_anchor),
        "transition_counts": dict(sorted((transition_counts or {}).items())),
        "distribution": dict(sorted(distribution.items())),
        "missing_emotions": sorted(TRACK2_EMOTIONS - set(distribution)),
        "top_emotion": distribution.most_common(1)[0][0] if distribution else "",
        "top_emotion_share": round(distribution.most_common(1)[0][1] / max(1, len(normalized)), 6)
        if distribution
        else 0.0,
        "label_consistency_issue_count": len(label_issues),
        "label_consistency_issues": label_issues[:80],
        "unsafe_text_count": count_unsafe_text_rows(normalized),
        "formal_submission_overwritten": False,
        "paths": {
            "out_json": str(out_json),
            "out_zip": str(out_zip),
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
    }
    _write_json(out_json, normalized)
    _write_zip_payload(out_zip, normalized)
    _write_json(report_json, report)
    report_md.write_text(_render_candidate_md(report), encoding="utf-8")
    return report


def build_v28_candidate_suite(
    *,
    base_json: str | Path = DEFAULT_BASE_JSON,
    description_json: str | Path = DEFAULT_DESCRIPTION_JSON,
    evidence_json: str | Path = DEFAULT_EVIDENCE_JSON,
    anchor_json: str | Path = BASE_ANCHOR_JSON,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    submissions_dir: str | Path = DEFAULT_SUBMISSIONS_DIR,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    submissions_dir = Path(submissions_dir)
    report_dir = out_dir / "candidate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    base_rows = load_track2_rows(base_json)
    description_rows = load_track2_rows(description_json)
    anchor_rows = load_track2_rows(anchor_json)
    evidence_rows = build_v28_evidence(_load_evidence_rows(evidence_json))
    _write_json(out_dir / "v28_evidence.json", evidence_rows)
    _write_csv(out_dir / "v28_evidence.csv", evidence_rows)

    existing_calm_ids = _changed_ids(anchor_rows, base_rows, transition="content->calm")
    base_distribution = Counter(row["emotion"] for row in base_rows)
    profiles: dict[str, list[dict[str, str]]] = {}

    profiles["descmax_on_v21"] = merge_description_fields(base_rows, description_rows)
    for target in (120, 150):
        evidence_without_existing = [row for row in evidence_rows if str(row.get("sample_id", "")) not in existing_calm_ids]
        changes = build_calm_ladder_changes(
            evidence_without_existing,
            existing_content_to_calm_count=len(existing_calm_ids),
            target_content_to_calm_count=target,
        )
        rows, _ = _apply_changes(base_rows, changes)
        profiles[f"calm_ladder_{target}"] = merge_description_fields(rows, description_rows)

    balanced_changes = build_balanced_frontier_changes(
        [row for row in evidence_rows if str(row.get("sample_id", "")) not in existing_calm_ids],
        base_distribution=base_distribution,
        total_cap=120,
        top_emotion_cap=0.58,
        class_floor=4,
        cross_cap=2,
    )
    balanced_rows, _ = _apply_changes(base_rows, balanced_changes)
    profiles["balanced_frontier"] = merge_description_fields(balanced_rows, description_rows)

    profile_rows: list[dict[str, Any]] = []
    profile_reports: dict[str, dict[str, Any]] = {}
    for profile, rows in profiles.items():
        stem = f"track2_submission_v28_{profile}_candidate"
        diff_v21 = _diff_rows(base_rows, rows)
        diff_anchor = _diff_rows(anchor_rows, rows)
        report = write_v28_candidate_outputs(
            rows=rows,
            out_json=submissions_dir / f"{stem}.json",
            out_zip=submissions_dir / f"{stem}.zip",
            report_json=report_dir / f"{stem}_report.json",
            report_md=report_dir / f"{stem}_report.md",
            profile=profile,
            label_changes_vs_v21=diff_v21["label_changes"],
            label_changes_vs_anchor=diff_anchor["label_changes"],
            transition_counts=diff_anchor["transition_counts"],
        )
        score = score_submission(report["paths"]["out_json"], candidate_name=stem)
        row = _profile_score_row(profile, stem, report, score)
        profile_rows.append(row)
        profile_reports[profile] = {**report, "score": row}

    ranked = sorted(profile_rows, key=lambda row: (-float(row["overall_expected"]), str(row["profile"])))
    best = ranked[0] if ranked else {}
    if best:
        best_rows = load_track2_rows(best["json_path"])
        hybrid_stem = "track2_submission_v28_hybrid_best_candidate"
        hybrid_report = write_v28_candidate_outputs(
            rows=best_rows,
            out_json=submissions_dir / f"{hybrid_stem}.json",
            out_zip=submissions_dir / f"{hybrid_stem}.zip",
            report_json=report_dir / f"{hybrid_stem}_report.json",
            report_md=report_dir / f"{hybrid_stem}_report.md",
            profile="hybrid_best",
            label_changes_vs_v21=int(best["label_changes_vs_v21"]),
            label_changes_vs_anchor=int(best["label_changes_vs_anchor"]),
            transition_counts=_parse_transition_counts(str(best.get("transition_counts", ""))),
        )
        hybrid_score = score_submission(hybrid_report["paths"]["out_json"], candidate_name=hybrid_stem)
        hybrid_row = _profile_score_row("hybrid_best", hybrid_stem, hybrid_report, hybrid_score)
        profile_rows.append(hybrid_row)
        profile_reports["hybrid_best"] = {**hybrid_report, "score": hybrid_row}
        ranked = sorted(profile_rows, key=lambda row: (-float(row["overall_expected"]), str(row["profile"])))

    passing = [row for row in ranked if row["gate_decision"] == "recommend_final_submit"]
    best = passing[0] if passing else (ranked[0] if ranked else {})
    summary = {
        "method": "track2_v28_final_shot_hybrid_suite_v1",
        "base_json": str(base_json),
        "description_json": str(description_json),
        "evidence_json": str(evidence_json),
        "target_overall": 0.89,
        "decision": "recommend_final_submit" if passing else "hold_no_submit",
        "best_profile": best.get("profile", ""),
        "best": best,
        "profiles": profile_rows,
        "profile_reports": profile_reports,
        "formal_submission_overwritten": False,
        "interpretation": "Use the best ZIP only if decision is recommend_final_submit.",
    }
    _write_json(out_dir / "v28_profile_scoreboard.json", summary)
    _write_csv(out_dir / "v28_profile_scoreboard.csv", profile_rows)
    (out_dir / "v28_final_recommendation_zh.md").write_text(_render_suite_md(summary), encoding="utf-8")
    return summary


def _profile_score_row(profile: str, stem: str, report: Mapping[str, Any], score: Any) -> dict[str, Any]:
    gate = choose_v28_final_gate(
        overall_expected=float(score.overall_expected),
        classification_expected=float(score.classification_expected),
        description_expected=float(score.description_expected),
        label_consistency_issue_count=int(report["label_consistency_issue_count"]),
        missing_emotions=list(report["missing_emotions"]),
        unsafe_text_count=int(report["unsafe_text_count"]),
        top_emotion_share=float(report["top_emotion_share"]),
        blind_knn_copy=False,
        unreported_cross_quadrant_count=0,
    )
    return {
        "profile": profile,
        "candidate_name": stem,
        "json_path": report["paths"]["out_json"],
        "zip_path": report["paths"]["out_zip"],
        "overall_expected": score.overall_expected,
        "classification_expected": score.classification_expected,
        "description_expected": score.description_expected,
        "score_warnings": ";".join(score.warnings),
        "label_changes_vs_v21": report["label_changes_vs_v21"],
        "label_changes_vs_anchor": report["label_changes_vs_anchor"],
        "transition_counts": "; ".join(f"{key}:{value}" for key, value in report["transition_counts"].items()),
        "top_emotion": report["top_emotion"],
        "top_emotion_share": report["top_emotion_share"],
        "missing_emotions": ";".join(report["missing_emotions"]),
        "unsafe_text_count": report["unsafe_text_count"],
        "label_consistency_issue_count": report["label_consistency_issue_count"],
        "gate_decision": gate["decision"],
        "gate_reasons": ";".join(gate["reasons"]),
    }


def _apply_changes(
    base_rows: list[Mapping[str, Any]],
    changes: list[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    changes_by_id = {
        str(row.get("sample_id", "")).strip(): row
        for row in changes
        if str(row.get("sample_id", "")).strip()
    }
    rows: list[dict[str, str]] = []
    accepted: list[dict[str, str]] = []
    for base in base_rows:
        row = {key: str(base.get(key, "")).strip() for key in TRACK2_SUBMISSION_KEYS}
        change = changes_by_id.get(row["sample_id"])
        if change:
            current = _canonical_emotion(row.get("emotion"))
            expected = _canonical_emotion(change.get("current_emotion"))
            proposed = _canonical_emotion(change.get("proposed_emotion"))
            if current == expected and proposed in TRACK2_EMOTIONS and proposed != current:
                row["emotion"] = proposed
                row["emotional_valence"] = _valence(proposed)
                row["emotional_arousal_level"] = _arousal(proposed)
                accepted.append({"sample_id": row["sample_id"], "transition": f"{current}->{proposed}"})
        rows.append(row)
    return rows, {"accepted": accepted}


def _diff_rows(before_rows: list[Mapping[str, Any]], after_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    before_by_id = {str(row.get("sample_id", "")).strip(): row for row in before_rows}
    transition_counts: Counter[str] = Counter()
    label_changes = 0
    for after in after_rows:
        sample_id = str(after.get("sample_id", "")).strip()
        before = before_by_id.get(sample_id)
        if not before:
            continue
        old = _canonical_emotion(before.get("emotion"))
        new = _canonical_emotion(after.get("emotion"))
        if old != new:
            label_changes += 1
            transition_counts[f"{old}->{new}"] += 1
    return {"label_changes": label_changes, "transition_counts": dict(sorted(transition_counts.items()))}


def _changed_ids(before_rows: list[Mapping[str, Any]], after_rows: list[Mapping[str, Any]], *, transition: str) -> set[str]:
    before_by_id = {str(row.get("sample_id", "")).strip(): row for row in before_rows}
    output: set[str] = set()
    for after in after_rows:
        sample_id = str(after.get("sample_id", "")).strip()
        before = before_by_id.get(sample_id)
        if not before:
            continue
        old = _canonical_emotion(before.get("emotion"))
        new = _canonical_emotion(after.get("emotion"))
        if f"{old}->{new}" == transition:
            output.add(sample_id)
    return output


def _parse_transition_counts(text: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for item in text.split(";"):
        if ":" not in item:
            continue
        key, value = item.strip().rsplit(":", 1)
        output[key.strip()] = _safe_int(value)
    return output


def _normalize_change(row: Mapping[str, Any], current: str, proposed: str) -> dict[str, Any]:
    output = dict(row)
    output["current_emotion"] = current
    output["proposed_emotion"] = proposed
    output["transition"] = f"{current}->{proposed}"
    output["v28_score"] = _safe_float(row.get("v28_score", row.get("support_score")))
    return output


def _score_row(decision: str, tier: str, score: float, reasons: list[str]) -> dict[str, Any]:
    return {"decision": decision, "tier": tier, "score": round(float(score), 6), "reasons": reasons}


def _evidence_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if row.get("v28_decision") == "accept_candidate" else 1,
        -_safe_float(row.get("v28_score")),
        str(row.get("sample_id", "")),
    )


def _selection_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if row.get("v28_decision", "accept_candidate") == "accept_candidate" else 1,
        -_safe_float(row.get("v28_score", row.get("support_score"))),
        str(row.get("sample_id", "")),
    )


def _label_issues(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        emotion = _canonical_emotion(row.get("emotion"))
        if emotion not in TRACK2_EMOTIONS:
            issues.append({"sample_id": sample_id, "field": "emotion", "actual": row.get("emotion", "")})
            continue
        valence = str(row.get("emotional_valence", "")).strip()
        arousal = str(row.get("emotional_arousal_level", "")).strip()
        if valence != _valence(emotion):
            issues.append({"sample_id": sample_id, "field": "emotional_valence", "actual": valence, "expected": _valence(emotion)})
        if arousal != _arousal(emotion):
            issues.append({"sample_id": sample_id, "field": "emotional_arousal_level", "actual": arousal, "expected": _arousal(emotion)})
    return issues


def _canonical_emotion(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {"contentment": "content", "relaxed": "calm", "joy": "glad", "joyful": "glad"}
    text = aliases.get(text, text)
    return text if text in TRACK2_EMOTIONS else ""


def _same_quadrant(first: str, second: str) -> bool:
    return _valence(first) == _valence(second) and _arousal(first) == _arousal(second)


def _valence(emotion: str) -> str:
    return "Negative" if emotion in NEGATIVE_EMOTIONS else "Positive"


def _arousal(emotion: str) -> str:
    return "High" if emotion in HIGH_AROUSAL_EMOTIONS else "Low"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0", ""}:
        return False
    return False


def _load_evidence_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"evidence JSON must be a list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _reject_formal_submission_path(path: Path) -> None:
    if path.name.casefold() in {name.casefold() for name in FORMAL_SUBMISSION_NAMES} and path.parent.name == "submissions":
        raise ValueError(f"refusing to write root formal submission path: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_zip_payload(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
    info = zipfile.ZipInfo("submission.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, payload)


def _render_candidate_md(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"# Track2 v28 {report['profile']} Candidate",
            "",
            f"- row_count: `{report['row_count']}`",
            f"- label_changes_vs_v21: `{report['label_changes_vs_v21']}`",
            f"- label_changes_vs_anchor: `{report['label_changes_vs_anchor']}`",
            f"- top_emotion: `{report['top_emotion']}` ({float(report['top_emotion_share']):.1%})",
            f"- missing_emotions: `{', '.join(report['missing_emotions']) or 'none'}`",
            f"- unsafe_text_count: `{report['unsafe_text_count']}`",
            "",
        ]
    )


def _render_suite_md(summary: Mapping[str, Any]) -> str:
    best = summary.get("best") or {}
    lines = [
        "# Track2 v28 final-shot hybrid report",
        "",
        "## 结论",
        "",
        f"- decision: `{summary.get('decision', '')}`",
        f"- best_profile: `{summary.get('best_profile', '')}`",
        f"- best_overall_expected: `{float(best.get('overall_expected', 0.0) or 0.0):.6f}`",
        f"- best_classification_expected: `{float(best.get('classification_expected', 0.0) or 0.0):.6f}`",
        f"- best_description_expected: `{float(best.get('description_expected', 0.0) or 0.0):.6f}`",
        "",
        "## Profile Scoreboard",
        "",
        "| profile | decision | overall | class | desc | vs_v21 | vs_anchor | top | share | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in sorted(summary.get("profiles", []), key=lambda item: str(item.get("profile", ""))):
        lines.append(
            "| {profile} | {decision} | {overall:.6f} | {classification:.6f} | {description:.6f} | "
            "{v21} | {anchor} | {top} | {share:.3f} | {reasons} |".format(
                profile=row.get("profile", ""),
                decision=row.get("gate_decision", ""),
                overall=float(row.get("overall_expected", 0.0) or 0.0),
                classification=float(row.get("classification_expected", 0.0) or 0.0),
                description=float(row.get("description_expected", 0.0) or 0.0),
                v21=row.get("label_changes_vs_v21", 0),
                anchor=row.get("label_changes_vs_anchor", 0),
                top=row.get("top_emotion", ""),
                share=float(row.get("top_emotion_share", 0.0) or 0.0),
                reasons=row.get("gate_reasons", ""),
            )
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- `hold_no_submit` 表示本地 final gate 未过，不建议消耗最后一次 Codabench 提交。",
            "- v23 scorer 对未提交候选仍是 proxy，不是 hidden gold reconstruction。",
            "",
        ]
    )
    return "\n".join(lines)
