from __future__ import annotations

import csv as _csv
import json as _json
import zipfile as _zipfile
from collections import Counter as _Counter
from dataclasses import dataclass as _dataclass
from math import isfinite as _isfinite
from pathlib import Path as _Path
from typing import Any as _Any


__all__ = [
    "ChampionTargets",
    "DEFAULT_BASE_CANDIDATES",
    "DEFAULT_EXPERIMENT_DIR",
    "DEFAULT_LEADERBOARD",
    "DEFAULT_OFFICIAL_SCORES",
    "DEFAULT_OUT_JSON",
    "DEFAULT_OUT_ZIP",
    "DEFAULT_PAIRWISE_DIFFS",
    "NO_AUTO_SUBMIT_POLICY",
    "OfficialAnchor",
    "TEXT_FIELDS",
    "FORMAL_SUBMISSION_NAMES",
    "TRACK2_JSON_EMOTIONS",
    "TRACK2_JSON_SUBMISSION_KEYS",
    "V18_TOTAL_CHANGE_CAP",
    "V18_TRANSITION_FAMILY_CAPS",
    "choose_v18_final_gate",
    "choose_v18_base",
    "enrich_champion_evidence",
    "load_champion_targets",
    "load_official_anchors",
    "load_track2_rows",
    "merge_description_rows",
    "render_v18_candidate_markdown",
    "select_v18_changes",
    "v18_gate_for_evidence",
    "write_v18_candidate_outputs",
    "write_v18_final_gate_report",
]


DEFAULT_EXPERIMENT_DIR = _Path("experiments/track2_v18_champion_hybrid_20260607")
DEFAULT_LEADERBOARD = _Path(
    "experiments/track2_official_results_20260606/track2_public_leaderboard_20260606.csv"
)
DEFAULT_OFFICIAL_SCORES = _Path(
    "experiments/track2_official_results_20260606/track2_known_official_exact_scores_from_ledger_20260606.csv"
)
DEFAULT_PAIRWISE_DIFFS = _Path(
    "experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv"
)
DEFAULT_BASE_CANDIDATES = {
    "779605": _Path("submissions/track2_submission_moe_v2_accept5_candidate.json"),
    "782683": _Path("submissions/track2_submission_v12_stable_probe_candidate.json"),
    "v15_desc_expand300": _Path("submissions/track2_submission_v15_desc_expand300_candidate.json"),
}
DEFAULT_OUT_JSON = _Path("submissions/track2_submission_v18_champion_hybrid_candidate.json")
DEFAULT_OUT_ZIP = _Path("submissions/track2_submission_v18_champion_hybrid_candidate.zip")
NO_AUTO_SUBMIT_POLICY = True
FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}
TRACK2_JSON_SUBMISSION_KEYS = (
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
TRACK2_JSON_EMOTIONS = {
    "aroused",
    "excited",
    "happy",
    "alarmed",
    "annoyed",
    "frustrated",
    "sad",
    "bored",
    "tired",
    "content",
    "calm",
    "glad",
}
POSITIVE_EMOTIONS = {"aroused", "excited", "happy", "content", "calm", "glad"}
NEGATIVE_EMOTIONS = {"alarmed", "annoyed", "frustrated", "sad", "bored", "tired"}
HIGH_AROUSAL_EMOTIONS = {"aroused", "excited", "happy", "alarmed", "annoyed", "frustrated"}
LOW_AROUSAL_EMOTIONS = {"content", "calm", "glad", "sad", "bored", "tired"}
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
V18_TOTAL_CHANGE_CAP = 36
V18_TRANSITION_FAMILY_CAPS = {
    "calm<->content": 3,
    "content<->glad": 3,
    "calm->glad": 2,
    "happy<->excited": 4,
    "aroused<->excited": 3,
    "sad<->tired": 3,
    "annoyed<->frustrated": 3,
}


@_dataclass(frozen=True)
class OfficialAnchor:
    submission_id: str
    file_name: str
    overall: float
    classification: float
    description: float


@_dataclass(frozen=True)
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


def load_official_anchors(path: str | _Path) -> dict[str, OfficialAnchor]:
    source = _Path(path)
    anchors: dict[str, OfficialAnchor] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in _csv.DictReader(handle):
            submission_id = str(row.get("submission_id", "")).strip()
            if not submission_id:
                continue
            row_context = f"submission_id {submission_id}"
            anchors[submission_id] = OfficialAnchor(
                submission_id=submission_id,
                file_name=str(row.get("file_name", "")).strip(),
                overall=_required_float(
                    row.get("official_overall"),
                    "official_overall",
                    path=source,
                    row_context=row_context,
                ),
                classification=_required_float(
                    row.get("official_classification"),
                    "official_classification",
                    path=source,
                    row_context=row_context,
                ),
                description=_required_float(
                    row.get("official_description"),
                    "official_description",
                    path=source,
                    row_context=row_context,
                ),
            )
    return anchors


def load_champion_targets(path: str | _Path, participant: str = "vulcaart") -> ChampionTargets:
    source = _Path(path)
    rows = _load_csv_rows(source)
    if not rows:
        raise ValueError(f"leaderboard is empty: {source}")
    first = min(rows, key=lambda row: _safe_int(row.get("#"), default=999999))
    current = next((row for row in rows if str(row.get("Participant", "")).strip() == participant), None)
    if current is None:
        raise ValueError(f"participant not found in leaderboard: {participant}")
    return ChampionTargets(
        first_participant=str(first.get("Participant", "")).strip(),
        current_participant=str(current.get("Participant", "")).strip(),
        first_overall=_required_leaderboard_float(first, "Overall Score", source),
        current_overall=_required_leaderboard_float(current, "Overall Score", source),
        first_classification=_required_leaderboard_float(first, "Classification Score", source),
        current_classification=_required_leaderboard_float(current, "Classification Score", source),
        first_description=_required_leaderboard_float(first, "Description Score", source),
        current_description=_required_leaderboard_float(current, "Description Score", source),
        first_emotion_accuracy=_required_leaderboard_float(first, "Emotion Accuracy", source),
        current_emotion_accuracy=_required_leaderboard_float(current, "Emotion Accuracy", source),
        first_emotion_macro_f1=_required_leaderboard_float(first, "Emotion Macro F1", source),
        current_emotion_macro_f1=_required_leaderboard_float(current, "Emotion Macro F1", source),
        first_visual_grounding=_required_leaderboard_float(first, "Visual Grounding", source),
        current_visual_grounding=_required_leaderboard_float(current, "Visual Grounding", source),
        first_attribute_specificity=_required_leaderboard_float(first, "Attribute Specificity", source),
        current_attribute_specificity=_required_leaderboard_float(current, "Attribute Specificity", source),
        first_overall_caption=_required_leaderboard_float(first, "Overall Caption", source),
        current_overall_caption=_required_leaderboard_float(current, "Overall Caption", source),
    )


def choose_v18_base(
    anchors: dict[str, OfficialAnchor],
    available: dict[str, _Path],
) -> dict[str, str]:
    usable: list[tuple[float, str, _Path]] = []
    for submission_id, anchor in anchors.items():
        path = available.get(submission_id)
        if path is not None:
            usable.append((anchor.overall, submission_id, path))

    if not usable:
        fallback = available.get("v15_desc_expand300")
        if fallback is None:
            raise ValueError("no usable v18 base candidates")
        return {
            "submission_id": "v15_desc_expand300",
            "base_json": str(fallback),
            "reason": "fallback_v15_desc_expand300",
        }
    _, submission_id, path = max(usable)
    return {"submission_id": submission_id, "base_json": str(path), "reason": "highest_exact_official_overall"}


def _required_leaderboard_float(row: dict[str, _Any], field: str, path: _Path) -> float:
    participant = str(row.get("Participant", "")).strip() or "<unknown participant>"
    return _required_float(row.get(field), field, path=path, row_context=f"participant {participant}")


def _required_float(value: _Any, field: str, *, path: _Path, row_context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(_invalid_numeric_message(value, field, path, row_context)) from None
    if not _isfinite(parsed):
        raise ValueError(_invalid_numeric_message(value, field, path, row_context))
    return parsed


def _invalid_numeric_message(value: _Any, field: str, path: _Path, row_context: str) -> str:
    return f"invalid numeric field {field!r} for {row_context} in {path}: {value!r}"


def _safe_float(value: _Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if _isfinite(parsed) else default


def _safe_int(value: _Any, default: int = 0) -> int:
    try:
        return int(_safe_float(value, float(default)))
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_bool(value: _Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return True
    if text in {"false", "no", "n"}:
        return False
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return False
    if numeric == 1.0:
        return True
    if numeric == 0.0:
        return False
    return False


def _load_csv_rows(path: str | _Path) -> list[dict[str, _Any]]:
    with _Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in _csv.DictReader(handle)]


def load_track2_rows(path: str | _Path) -> list[dict[str, _Any]]:
    source = _Path(path)
    try:
        payload = _json.loads(source.read_text(encoding="utf-8"))
    except _json.JSONDecodeError as exc:
        raise ValueError(f"invalid Track2 JSON in {source}: {exc}") from None
    if not isinstance(payload, list):
        raise ValueError(f"Track2 submission JSON must be a list: {source}")
    rows: list[dict[str, _Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Track2 row {index} must be an object in {source}")
        row = {key: item.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS}
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"blank sample_id at row {index} in {source}")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id {sample_id!r} in {source}")
        seen_ids.add(sample_id)
        issues = _strict_v18_label_issues(row)
        if issues:
            first = issues[0]
            raise ValueError(
                f"invalid {first['field']} at row {index} in {source}: "
                f"{first.get('actual', '')!r}; expected {first.get('expected', '')!r}"
            )
        rows.append(row)
    return rows


def enrich_champion_evidence(rows: list[dict[str, _Any]]) -> list[dict[str, _Any]]:
    enriched: list[dict[str, _Any]] = []
    for row in rows:
        item = dict(row)
        transition = str(item.get("transition", "")).strip()
        same_valence, same_arousal = _same_quadrant_from_evidence(item)
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
    if transition in {"content->glad", "glad->content"}:
        return "content<->glad"
    if transition == "calm->glad":
        return "calm->glad"
    if transition in {"happy->excited", "excited->happy"}:
        return "happy<->excited"
    if transition in {"aroused->excited", "excited->aroused"}:
        return "aroused<->excited"
    if transition in {"sad->tired", "tired->sad"}:
        return "sad<->tired"
    if transition in {"annoyed->frustrated", "frustrated->annoyed"}:
        return "annoyed<->frustrated"
    return transition


def _champion_sort_key(row: dict[str, _Any]) -> tuple[int, int, float, float, str, str]:
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


def _evidence_transition_labels(row: dict[str, _Any]) -> tuple[str, str, str, str, str, bool, bool]:
    transition = str(row.get("transition", "")).strip()
    parts = [part.strip() for part in transition.split("->")]
    transition_current = ""
    transition_proposed = ""
    valid_transition_shape = False
    if len(parts) == 2 and parts[0] and parts[1]:
        transition_current, transition_proposed = parts
        valid_transition_shape = True
    explicit_current = str(row.get("current_emotion", "")).strip()
    explicit_proposed = str(row.get("proposed_emotion", "")).strip()
    current = explicit_current or transition_current
    proposed = explicit_proposed or transition_proposed
    transition_label_mismatch = valid_transition_shape and (
        (bool(explicit_current) and explicit_current != transition_current)
        or (bool(explicit_proposed) and explicit_proposed != transition_proposed)
    )
    normalized_transition = (
        f"{transition_current}->{transition_proposed}" if valid_transition_shape else transition
    )
    return (
        normalized_transition,
        current,
        proposed,
        transition_current,
        transition_proposed,
        valid_transition_shape,
        transition_label_mismatch,
    )


def v18_gate_for_evidence(
    row: dict[str, _Any],
    *,
    distribution: dict[str, _Any] | _Counter[str],
) -> dict[str, _Any]:
    exact_duplicate = _safe_bool(row.get("exact_duplicate"))
    public_duplicate_support = _safe_float(row.get("public_duplicate_support_score"))
    model_votes = _safe_int(row.get("model_vote_count"))
    support_score = _safe_float(row.get("support_score"))
    (
        transition,
        current,
        proposed,
        transition_current,
        transition_proposed,
        valid_transition_shape,
        transition_label_mismatch,
    ) = _evidence_transition_labels(row)
    same_valence, same_arousal = _same_quadrant_from_labels(current, proposed)
    hard_reasons: list[str] = []
    if (
        not valid_transition_shape
        or transition_current not in TRACK2_JSON_EMOTIONS
        or transition_proposed not in TRACK2_JSON_EMOTIONS
        or current not in TRACK2_JSON_EMOTIONS
        or proposed not in TRACK2_JSON_EMOTIONS
    ):
        hard_reasons.append("invalid_transition")
    if transition_label_mismatch:
        hard_reasons.append("transition_label_mismatch")
    if (current and proposed and current == proposed) or (
        transition_current and transition_proposed and transition_current == transition_proposed
    ):
        hard_reasons.append("no_label_change")
    if current and _would_remove_rare_class(current, distribution):
        hard_reasons.append("rare_current_class_floor")
    if _safe_int(row.get("failed_transition_count")) >= 10 and transition not in MAJORITY_BOUNDARY_TRANSITIONS:
        hard_reasons.append("failed_official_transition_family")
    if hard_reasons:
        return {"decision": "block", "reasons": hard_reasons}

    soft_reasons: list[str] = []
    if not same_valence or not same_arousal:
        soft_reasons.append("unsupported_cross_quadrant")
    if exact_duplicate and public_duplicate_support >= 0.95 and soft_reasons:
        return {"decision": "accept", "reasons": ["exact_duplicate_override"]}
    if model_votes < 2:
        soft_reasons.append("insufficient_model_families")
    if support_score < 1.45:
        soft_reasons.append("low_support_score")
    if exact_duplicate and public_duplicate_support >= 0.95 and soft_reasons:
        return {"decision": "accept", "reasons": ["exact_duplicate_override"]}
    return {"decision": "block" if soft_reasons else "accept", "reasons": soft_reasons or ["meets_v18_thresholds"]}


def select_v18_changes(
    evidence_rows: list[dict[str, _Any]],
    *,
    current_distribution: dict[str, _Any] | _Counter[str],
    total_cap: int = V18_TOTAL_CHANGE_CAP,
) -> list[dict[str, _Any]]:
    selected: list[dict[str, _Any]] = []
    selected_ids: set[str] = set()
    family_counts: _Counter[str] = _Counter()
    projected = _Counter({str(key): _safe_int(value) for key, value in dict(current_distribution).items()})
    for row in sorted(evidence_rows, key=_champion_sort_key):
        if len(selected) >= total_cap:
            break
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in selected_ids:
            continue
        gate = v18_gate_for_evidence(row, distribution=projected)
        if gate["decision"] != "accept":
            continue
        transition, current, proposed, _, _, _, _ = _evidence_transition_labels(row)
        family_transition = f"{current}->{proposed}" if current and proposed else transition
        family = _transition_family(family_transition)
        cap = V18_TRANSITION_FAMILY_CAPS.get(family)
        if cap is not None and family_counts[family] >= cap:
            continue
        item = dict(row)
        item["gate_decision"] = "accept"
        item["gate_reasons"] = ",".join(gate["reasons"])
        item["transition_family"] = family
        selected.append(item)
        selected_ids.add(sample_id)
        family_counts[family] += 1
        if current and proposed:
            projected[current] -= 1
            projected[proposed] += 1
    return selected


def _would_remove_rare_class(
    emotion: str,
    distribution: dict[str, _Any] | _Counter[str],
    floor: int = 3,
) -> bool:
    if emotion not in TRACK2_JSON_EMOTIONS:
        return False
    return _safe_int(dict(distribution).get(emotion)) <= floor


TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
BANNED_DESCRIPTION_TERMS = (
    "evaluator",
    "score",
    "scoring",
    "evaluation",
    "evaluate",
    "award",
    "judge",
    "as an ai",
)


def merge_description_rows(
    base_rows: list[dict[str, _Any]],
    text_rows: list[dict[str, _Any]],
) -> tuple[list[dict[str, _Any]], dict[str, _Any]]:
    text_by_id = {str(row.get("sample_id", "")).strip(): row for row in text_rows}
    merged: list[dict[str, _Any]] = []
    changed_rows = 0
    rejected_rows = 0
    for base in base_rows:
        sample_id = str(base.get("sample_id", "")).strip()
        source = text_by_id.get(sample_id)
        row = dict(base)
        row_changed = False
        row_rejected = False
        saw_text = False
        if source:
            for field in TEXT_FIELDS:
                candidate_text = str(source.get(field, "")).strip()
                if not candidate_text:
                    continue
                saw_text = True
                current_text = str(base.get(field, "")).strip()
                if _is_better_description_text(candidate_text, current_text):
                    row[field] = candidate_text
                    row_changed = True
                elif _has_banned_description_text(candidate_text):
                    row_rejected = True
            if row_rejected or (not row_changed and saw_text):
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
    if _has_banned_description_text(lowered):
        return False
    return True


def _has_banned_description_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in BANNED_DESCRIPTION_TERMS)


def write_v18_candidate_outputs(
    *,
    base_json: str | _Path,
    text_json: str | _Path | None,
    changes: list[dict[str, _Any]],
    out_json: str | _Path,
    out_zip: str | _Path,
    report_json: str | _Path,
    report_md: str | _Path,
) -> dict[str, _Any]:
    _assert_safe_side_path(out_json)
    _assert_safe_side_path(out_zip)
    _assert_safe_side_path(report_json)
    _assert_safe_side_path(report_md)
    base_rows = load_track2_rows(base_json)
    text_rows = load_track2_rows(text_json) if text_json else base_rows
    merged_rows, description_report = merge_description_rows(base_rows, text_rows)
    candidate_rows, classification_report = _apply_v18_changes(merged_rows, changes)
    report = {
        "method": "track2_v18_champion_hybrid_candidate_v1",
        "base_json": str(base_json),
        "text_json": str(text_json or base_json),
        "out_json": str(out_json),
        "out_zip": str(out_zip),
        "row_count": len(candidate_rows),
        "input_change_count": len(changes),
        "accepted_label_changes": classification_report["accepted_label_changes"],
        "description_merge": description_report,
        "transition_counts": classification_report["transition_counts"],
        "distribution": classification_report["distribution"],
        "missing_emotions": classification_report["missing_emotions"],
        "top_emotion": classification_report["top_emotion"],
        "top_emotion_share": classification_report["top_emotion_share"],
        "label_consistency_issue_count": classification_report["label_consistency_issue_count"],
        "label_consistency_issues": classification_report["label_consistency_issues"],
        "formal_submission_overwritten": False,
        "accepted_changes": classification_report["accepted_changes"],
    }
    _write_json(out_json, candidate_rows)
    _write_zip(out_zip, out_json)
    _write_json(report_json, report)
    report_md_path = _Path(report_md)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(render_v18_candidate_markdown(report), encoding="utf-8")
    return report


def _apply_v18_changes(
    base_rows: list[dict[str, _Any]],
    changes: list[dict[str, _Any]],
) -> tuple[list[dict[str, _Any]], dict[str, _Any]]:
    changes_by_id = {
        str(row.get("sample_id", "")).strip(): dict(row)
        for row in changes
        if str(row.get("sample_id", "")).strip()
    }
    candidate_rows: list[dict[str, _Any]] = []
    accepted_changes: list[dict[str, _Any]] = []

    for base in base_rows:
        row = dict(base)
        sample_id = str(row.get("sample_id", "")).strip()
        change = changes_by_id.get(sample_id)
        if change:
            current = _canonical_emotion(row.get("emotion"))
            proposed = _canonical_emotion(_first_present(change, ("proposed_emotion", "target_emotion", "emotion")))
            expected_current = _canonical_emotion(change.get("current_emotion"))
            gate_decision = str(change.get("gate_decision", "")).strip().lower()
            if (
                proposed
                and proposed != current
                and proposed in TRACK2_JSON_EMOTIONS
                and (not expected_current or expected_current == current)
                and gate_decision == "accept"
            ):
                before = _label_triplet(row)
                row["emotion"] = proposed
                row["emotional_valence"] = _valence(proposed)
                row["emotional_arousal_level"] = _arousal(proposed)
                after = _label_triplet(row)
                transition = str(change.get("transition", "")).strip() or f"{before[0]}->{after[0]}"
                accepted_changes.append(
                    {
                        "sample_id": sample_id,
                        "transition": transition,
                        "before": {
                            "emotion": before[0],
                            "emotional_valence": before[1],
                            "emotional_arousal_level": before[2],
                        },
                        "after": {
                            "emotion": after[0],
                            "emotional_valence": after[1],
                            "emotional_arousal_level": after[2],
                        },
                        "support_score": _safe_float(change.get("support_score")),
                        "max_confidence": _safe_float(change.get("max_confidence")),
                        "model_vote_count": _safe_int(change.get("model_vote_count")),
                        "model_sources": str(change.get("model_sources", "")),
                        "all_sources": str(change.get("all_sources", "")),
                        "gate_decision": str(change.get("gate_decision", "")),
                        "gate_reasons": str(change.get("gate_reasons", "")),
                        "rationale": str(change.get("rationale", "")),
                    }
                )
        candidate_rows.append({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})

    distribution = _Counter(str(row.get("emotion", "")) for row in candidate_rows)
    label_issues = [issue for row in candidate_rows for issue in _strict_v18_label_issues(row)]
    return candidate_rows, {
        "accepted_label_changes": len(accepted_changes),
        "transition_counts": dict(_Counter(item["transition"] for item in accepted_changes)),
        "distribution": dict(sorted(distribution.items())),
        "missing_emotions": sorted(TRACK2_JSON_EMOTIONS - set(distribution)),
        "top_emotion": distribution.most_common(1)[0][0] if distribution else "",
        "top_emotion_share": distribution.most_common(1)[0][1] / len(candidate_rows) if candidate_rows else 0.0,
        "label_consistency_issue_count": len(label_issues),
        "label_consistency_issues": label_issues[:80],
        "accepted_changes": accepted_changes,
    }


def render_v18_candidate_markdown(report: dict[str, _Any]) -> str:
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


def choose_v18_final_gate(
    *,
    candidate_report: dict[str, _Any],
    fused_row: dict[str, _Any],
    anchor: dict[str, _Any],
    emotion_accuracy_proxy_delta: float,
) -> dict[str, _Any]:
    reasons: list[str] = []
    description_merge = candidate_report.get("description_merge", {})
    if not isinstance(description_merge, dict):
        description_merge = {}
    if _safe_int(candidate_report.get("label_consistency_issue_count")) != 0:
        reasons.append("label_consistency_issues")
    if _safe_int(description_merge.get("description_changed_rows")) <= 0:
        reasons.append("description_not_maximized")
    if _safe_float(fused_row.get("description_lower")) < _safe_float(anchor.get("description")) - 0.004:
        reasons.append("description_lower_below_anchor")
    if _safe_float(fused_row.get("classification_lower")) < _safe_float(anchor.get("classification")) - 0.006:
        reasons.append("classification_lower_below_tolerance")
    if _safe_float(fused_row.get("overall_lower")) <= _safe_float(anchor.get("overall")):
        reasons.append("overall_lower_not_above_anchor")
    if _safe_float(emotion_accuracy_proxy_delta) <= 0.005:
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
        "candidate_classification_lower": _safe_float(fused_row.get("classification_lower")),
        "candidate_description_lower": _safe_float(fused_row.get("description_lower")),
        "anchor_overall": _safe_float(anchor.get("overall")),
        "anchor_classification": _safe_float(anchor.get("classification")),
        "anchor_description": _safe_float(anchor.get("description")),
        "accepted_label_changes": _safe_int(candidate_report.get("accepted_label_changes")),
        "description_changed_rows": _safe_int(description_merge.get("description_changed_rows")),
        "emotion_accuracy_proxy_delta": _safe_float(emotion_accuracy_proxy_delta),
        "caveat": "Local/fused shadow scorer is a risk control, not the official Codabench scorer.",
        "no_auto_submit": True,
    }


def write_v18_final_gate_report(
    report: dict[str, _Any],
    out_json: str | _Path,
    out_md: str | _Path,
) -> None:
    _assert_safe_side_path(out_json)
    _assert_safe_side_path(out_md)
    _write_json(out_json, report)
    lines = [
        "# Track2 v18 Final Gate",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Candidate: `{report['candidate_name']}`",
        f"- Candidate overall lower: {float(report['candidate_overall_lower']):.6f}",
        f"- Anchor overall: {float(report['anchor_overall']):.6f}",
        f"- Emotion accuracy proxy delta: {float(report['emotion_accuracy_proxy_delta']):.6f}",
        f"- No auto-submit: {report['no_auto_submit']}",
        f"- Caveat: {report['caveat']}",
        "",
        "## Reasons",
        "",
    ]
    for reason in report["reasons"]:
        lines.append(f"- {reason}")
    out_md_path = _Path(out_md)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assert_safe_side_path(path: str | _Path) -> None:
    candidate = _Path(path)
    if candidate.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError(f"refusing to write formal submission path: {candidate}")


def _write_json(path: str | _Path, payload: _Any) -> None:
    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_zip(zip_path: str | _Path, json_path: str | _Path) -> None:
    out = _Path(zip_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with _zipfile.ZipFile(out, "w", compression=_zipfile.ZIP_DEFLATED) as archive:
        info = _zipfile.ZipInfo("submission.json")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.compress_type = _zipfile.ZIP_DEFLATED
        archive.writestr(info, _Path(json_path).read_bytes())


def _strict_v18_label_issues(row: dict[str, _Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    emotion = str(row.get("emotion", ""))
    valence = str(row.get("emotional_valence", ""))
    arousal = str(row.get("emotional_arousal_level", ""))
    if emotion not in TRACK2_JSON_EMOTIONS:
        return [
            {
                "sample_id": str(row.get("sample_id", "")),
                "field": "emotion",
                "expected": "|".join(sorted(TRACK2_JSON_EMOTIONS)),
                "actual": emotion,
                "reason": "emotion is not in the challenge label set",
            }
        ]
    if valence not in {"Positive", "Negative"}:
        issues.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "field": "emotional_valence",
                "expected": "Positive|Negative",
                "actual": valence,
                "reason": "emotional_valence must be one of the challenge values",
            }
        )
    if arousal not in {"High", "Low"}:
        issues.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "field": "emotional_arousal_level",
                "expected": "High|Low",
                "actual": arousal,
                "reason": "emotional_arousal_level must be one of the challenge values",
            }
        )
    if emotion in POSITIVE_EMOTIONS and valence != "Positive":
        issues.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "field": "emotional_valence",
                "expected": "Positive",
                "actual": valence,
                "reason": f"{emotion} is normally positive in the challenge label set",
            }
        )
    if emotion in NEGATIVE_EMOTIONS and valence != "Negative":
        issues.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "field": "emotional_valence",
                "expected": "Negative",
                "actual": valence,
                "reason": f"{emotion} is normally negative in the challenge label set",
            }
        )
    if emotion in HIGH_AROUSAL_EMOTIONS and arousal != "High":
        issues.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "field": "emotional_arousal_level",
                "expected": "High",
                "actual": arousal,
                "reason": f"{emotion} is normally high-arousal in the challenge label set",
            }
        )
    if emotion in LOW_AROUSAL_EMOTIONS and arousal != "Low":
        issues.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "field": "emotional_arousal_level",
                "expected": "Low",
                "actual": arousal,
                "reason": f"{emotion} is normally low-arousal in the challenge label set",
            }
        )
    return issues


def _canonical_emotion(value: _Any) -> str:
    emotion = str(value or "").strip().lower()
    return emotion if emotion in TRACK2_JSON_EMOTIONS else ""


def _same_quadrant_from_evidence(row: dict[str, _Any]) -> tuple[bool, bool]:
    _, current, proposed, _, _, valid_transition_shape, transition_label_mismatch = _evidence_transition_labels(row)
    if (
        valid_transition_shape
        and not transition_label_mismatch
        and current in TRACK2_JSON_EMOTIONS
        and proposed in TRACK2_JSON_EMOTIONS
    ):
        return _same_quadrant_from_labels(current, proposed)
    return _safe_bool(row.get("same_valence")), _safe_bool(row.get("same_arousal"))


def _same_quadrant_from_labels(current: str, proposed: str) -> tuple[bool, bool]:
    if current not in TRACK2_JSON_EMOTIONS or proposed not in TRACK2_JSON_EMOTIONS:
        return False, False
    return _valence(current) == _valence(proposed), _arousal(current) == _arousal(proposed)


def _first_present(row: dict[str, _Any], fields: tuple[str, ...]) -> _Any:
    for field in fields:
        value = row.get(field)
        if str(value or "").strip():
            return value
    return ""


def _label_triplet(row: dict[str, _Any]) -> tuple[str, str, str]:
    return (
        str(row.get("emotion", "")).strip(),
        str(row.get("emotional_valence", "")).strip(),
        str(row.get("emotional_arousal_level", "")).strip(),
    )


def _valence(emotion: str) -> str:
    return "Negative" if emotion in NEGATIVE_EMOTIONS else "Positive"


def _arousal(emotion: str) -> str:
    return "High" if emotion in HIGH_AROUSAL_EMOTIONS else "Low"
