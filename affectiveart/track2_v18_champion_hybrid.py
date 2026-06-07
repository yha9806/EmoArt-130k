from __future__ import annotations

import csv as _csv
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
    "TRACK2_JSON_EMOTIONS",
    "TRACK2_JSON_SUBMISSION_KEYS",
    "V18_TOTAL_CHANGE_CAP",
    "V18_TRANSITION_FAMILY_CAPS",
    "choose_v18_base",
    "enrich_champion_evidence",
    "load_champion_targets",
    "load_official_anchors",
    "merge_description_rows",
    "select_v18_changes",
    "v18_gate_for_evidence",
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
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: _Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
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


def enrich_champion_evidence(rows: list[dict[str, _Any]]) -> list[dict[str, _Any]]:
    enriched: list[dict[str, _Any]] = []
    for row in rows:
        item = dict(row)
        transition = str(item.get("transition", "")).strip()
        same_valence = _safe_bool(item.get("same_valence"))
        same_arousal = _safe_bool(item.get("same_arousal"))
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
    same_valence = _safe_bool(row.get("same_valence"))
    same_arousal = _safe_bool(row.get("same_arousal"))
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
