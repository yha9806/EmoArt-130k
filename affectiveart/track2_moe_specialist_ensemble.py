from __future__ import annotations

from collections import Counter
import math
from dataclasses import dataclass
from typing import Any

from affectiveart.challenge import TRACK2_JSON_EMOTIONS
from affectiveart.track2_audit import (
    HIGH_AROUSAL_EMOTIONS,
    LOW_AROUSAL_EMOTIONS,
    NEGATIVE_EMOTIONS,
    POSITIVE_EMOTIONS,
    compute_track2_distribution,
)


VALID_TRACK2_EMOTIONS = TRACK2_JSON_EMOTIONS
SINGLE_SOURCE_SPECIALIST_ROLES = frozenset(
    {"boundary", "tail", "va", "description", "specialist"}
)


@dataclass(frozen=True)
class GateThresholds:
    min_supporting_sources: int = 2
    min_support_confidence: float = 0.50
    high_confidence: float = 0.86
    min_margin: float = 0.12
    min_macro_f1_gain: float = 0.015
    min_hardcase_macro_f1_gain: float = 0.030
    max_accuracy_drop: float = 0.010
    max_valence_accuracy_drop: float = 0.005
    max_arousal_accuracy_drop: float = 0.005
    max_top_emotion_share_delta: float = 0.020


def expected_label_for_emotion(emotion: str) -> tuple[str, str]:
    normalized = str(emotion).strip().lower()
    if normalized in POSITIVE_EMOTIONS:
        valence = "Positive"
    elif normalized in NEGATIVE_EMOTIONS:
        valence = "Negative"
    else:
        raise ValueError(f"unsupported Track2 emotion: {emotion}")

    if normalized in HIGH_AROUSAL_EMOTIONS:
        arousal = "High"
    elif normalized in LOW_AROUSAL_EMOTIONS:
        arousal = "Low"
    else:
        raise ValueError(f"unsupported Track2 emotion: {emotion}")

    return valence, arousal


def normalize_expert_entries(payload: Any, source: str, role: str) -> list[dict[str, Any]]:
    rows = _extract_payload_rows(payload)
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        emotion = str(row.get("emotion", "")).strip().lower()
        if emotion not in VALID_TRACK2_EMOTIONS:
            continue
        confidence = _safe_float(row.get("confidence"))

        normalized_rows.append(
            {
                "sample_id": sample_id,
                "source": str(source),
                "role": str(role),
                "emotion": emotion,
                "confidence": confidence,
                "margin": _safe_float(row.get("margin")),
                "top3": _normalize_top3(
                    row.get("top3"),
                    fallback=emotion,
                    fallback_probability=confidence,
                ),
            }
        )
    return normalized_rows


def build_gate_decision(
    current_row: dict[str, Any],
    expert_rows: list[dict[str, Any]],
    *,
    thresholds: GateThresholds | None = None,
    description_audit: dict[str, Any] | None = None,
    high_similarity_public_reference: bool = False,
) -> dict[str, Any]:
    thresholds = thresholds or GateThresholds()
    sample_id = str(current_row.get("sample_id", "")).strip()
    current_emotion = str(current_row.get("emotion", "")).strip().lower()
    relevant_rows = _relevant_expert_rows(sample_id, expert_rows)
    evidence = _summarize_evidence(relevant_rows)
    proposed_emotion = _choose_proposal(current_emotion, evidence)

    if not proposed_emotion:
        return _decision_row(
            current_row,
            relevant_rows,
            "keep_current",
            current_emotion,
            ["no_supported_change"],
        )

    reasons: list[str] = []
    support = evidence.get(proposed_emotion, [])
    supporting_source_count = _supporting_source_count(support)
    quality_supporting_source_count = _supporting_source_count(
        _quality_support_rows(support, thresholds)
    )
    strong_opposition = _strong_opposition(
        proposed_emotion,
        evidence,
        thresholds,
    )
    if supporting_source_count >= thresholds.min_supporting_sources:
        if quality_supporting_source_count >= thresholds.min_supporting_sources:
            reasons.append(f"supported_by_{quality_supporting_source_count}_sources")
        else:
            reasons.append("support_below_quality_bar")
    elif _has_single_high_confidence_source(
        proposed_emotion,
        evidence,
        thresholds,
    ):
        if strong_opposition:
            reasons.append("insufficient_independent_support")
        elif _has_single_high_confidence_specialist_source(
            proposed_emotion,
            evidence,
            thresholds,
        ):
            reasons.append("single_high_confidence_source_without_strong_opposition")
        else:
            reasons.append("single_source_not_specialist")
    else:
        reasons.append("insufficient_independent_support")
    if strong_opposition:
        reasons.append("strong_opposition")

    if _description_verdict(description_audit) == "contradiction":
        reasons.append("description_contradiction")
    if high_similarity_public_reference:
        reasons.append("high_similarity_requires_explicit_review")

    blocking_reasons = {
        "insufficient_independent_support",
        "single_source_not_specialist",
        "strong_opposition",
        "support_below_quality_bar",
        "description_contradiction",
        "high_similarity_requires_explicit_review",
    }
    decision = "hold" if blocking_reasons.intersection(reasons) else "accept_change"
    proposed_valence, proposed_arousal = expected_label_for_emotion(proposed_emotion)
    return _decision_row(
        current_row,
        relevant_rows,
        decision,
        proposed_emotion,
        reasons,
        proposed_valence=proposed_valence,
        proposed_arousal=proposed_arousal,
    )


def build_dry_run_report(
    current_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    *,
    queue_sample_ids,
    high_similarity_sample_ids=None,
    description_audit_by_id=None,
) -> dict[str, Any]:
    high_similarity_ids = set(high_similarity_sample_ids or set())
    description_audits = description_audit_by_id or {}
    current_by_id = {
        str(row.get("sample_id", "")).strip(): row
        for row in current_rows
        if isinstance(row, dict) and str(row.get("sample_id", "")).strip()
    }

    decisions: list[dict[str, Any]] = []
    for sample_id in sorted(dict.fromkeys(queue_sample_ids)):
        sample_id = str(sample_id).strip()
        if sample_id not in current_by_id:
            continue
        decisions.append(
            build_gate_decision(
                current_by_id[sample_id],
                expert_rows,
                description_audit=description_audits.get(sample_id),
                high_similarity_public_reference=sample_id in high_similarity_ids,
            )
        )

    decision_counts = Counter(row["decision"] for row in decisions)
    accepted_transition_counts = Counter(
        f"{row['current_emotion']}->{row['proposed_emotion']}"
        for row in decisions
        if row["decision"] == "accept_change"
    )
    return {
        "method": "track2_moe_specialist_dry_run_v1",
        "row_count": len(decisions),
        "decision_counts": dict(decision_counts),
        "accepted_transition_counts": dict(accepted_transition_counts),
        "baseline_distribution": compute_track2_distribution(current_rows),
        "formal_submission_overwritten": False,
        "candidate_json_written": False,
        "candidate_zip_written": False,
        "rows": decisions,
    }


def _relevant_expert_rows(
    sample_id: str,
    expert_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    relevant_rows: list[dict[str, Any]] = []
    for row in expert_rows or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("sample_id", "")).strip() != sample_id:
            continue
        normalized = _normalize_gate_evidence_row(row)
        if normalized:
            relevant_rows.append(normalized)
    return relevant_rows


def _normalize_gate_evidence_row(row: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = str(row.get("sample_id", "")).strip()
    raw_source = row.get("source", "")
    source = "" if raw_source is None else str(raw_source).strip()
    emotion = str(row.get("emotion", "")).strip().lower()
    if not sample_id or not source or emotion not in VALID_TRACK2_EMOTIONS:
        return None
    confidence = _safe_float(row.get("confidence"))
    return {
        "sample_id": sample_id,
        "source": source,
        "role": str(row.get("role", "")).strip(),
        "emotion": emotion,
        "confidence": confidence,
        "margin": _safe_float(row.get("margin")),
        "top3": _normalize_top3(
            row.get("top3"),
            fallback=emotion,
            fallback_probability=confidence,
        ),
    }


def _summarize_evidence(
    expert_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {}
    for row in expert_rows:
        emotion = str(row.get("emotion", "")).strip().lower()
        if emotion in VALID_TRACK2_EMOTIONS:
            evidence.setdefault(emotion, []).append(row)
    return evidence


def _choose_proposal(
    current_emotion: str,
    evidence: dict[str, list[dict[str, Any]]],
) -> str:
    candidates = [emotion for emotion in evidence if emotion != current_emotion]
    if not candidates:
        return ""
    return sorted(
        candidates,
        key=lambda emotion: (
            -_supporting_source_count(evidence[emotion]),
            -sum(_safe_float(row.get("confidence")) for row in evidence[emotion]),
            emotion,
        ),
    )[0]


def _supporting_source_count(rows: list[dict[str, Any]]) -> int:
    return len({_source_key(row) for row in rows})


def _source_key(row: dict[str, Any]) -> str:
    return str(row.get("source", "")).strip()


def _has_single_high_confidence_source(
    proposed_emotion: str,
    evidence: dict[str, list[dict[str, Any]]],
    thresholds: GateThresholds,
) -> bool:
    support = evidence.get(proposed_emotion, [])
    if _supporting_source_count(support) != 1:
        return False
    return any(
        _safe_float(row.get("confidence")) >= thresholds.high_confidence
        and _safe_float(row.get("margin")) >= thresholds.min_margin
        for row in support
    )


def _has_single_high_confidence_specialist_source(
    proposed_emotion: str,
    evidence: dict[str, list[dict[str, Any]]],
    thresholds: GateThresholds,
) -> bool:
    support = evidence.get(proposed_emotion, [])
    if _supporting_source_count(support) != 1:
        return False
    return any(
        _safe_float(row.get("confidence")) >= thresholds.high_confidence
        and _safe_float(row.get("margin")) >= thresholds.min_margin
        and _is_single_source_specialist_role(row)
        for row in support
    )


def _is_single_source_specialist_role(row: dict[str, Any]) -> bool:
    return (
        str(row.get("role", "")).strip().lower()
        in SINGLE_SOURCE_SPECIALIST_ROLES
    )


def _quality_support_rows(
    rows: list[dict[str, Any]],
    thresholds: GateThresholds,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _safe_float(row.get("confidence")) >= thresholds.min_support_confidence
        and _safe_float(row.get("margin")) >= thresholds.min_margin
    ]


def _strong_opposition(
    proposed_emotion: str,
    evidence: dict[str, list[dict[str, Any]]],
    thresholds: GateThresholds,
) -> bool:
    for emotion, rows in evidence.items():
        if emotion == proposed_emotion:
            continue
        for row in rows:
            if (
                _safe_float(row.get("confidence")) >= thresholds.high_confidence
                and _safe_float(row.get("margin")) >= thresholds.min_margin
            ):
                return True
    return False


def _description_verdict(description_audit: dict[str, Any] | None) -> str:
    if not isinstance(description_audit, dict):
        return ""
    return str(description_audit.get("verdict", "")).strip().lower()


def _decision_row(
    current_row: dict[str, Any],
    expert_rows: list[dict[str, Any]],
    decision: str,
    proposed_emotion: str,
    reasons: list[str],
    *,
    proposed_valence: str | None = None,
    proposed_arousal: str | None = None,
) -> dict[str, Any]:
    if proposed_valence is None or proposed_arousal is None:
        proposed_valence, proposed_arousal = expected_label_for_emotion(
            proposed_emotion
        )
    return {
        "sample_id": str(current_row.get("sample_id", "")).strip(),
        "decision": decision,
        "current_emotion": str(current_row.get("emotion", "")).strip().lower(),
        "current_valence": str(current_row.get("emotional_valence", "")).strip(),
        "current_arousal": str(
            current_row.get("emotional_arousal_level", "")
        ).strip(),
        "proposed_emotion": proposed_emotion,
        "proposed_valence": proposed_valence,
        "proposed_arousal": proposed_arousal,
        "reasons": list(reasons),
        "expert_evidence": sorted(
            expert_rows,
            key=lambda row: (
                str(row.get("source", "")),
                str(row.get("emotion", "")),
                str(row.get("role", "")),
            ),
        ),
    }


def _extract_payload_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("entries", "predictions", "rows"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _normalize_top3(
    value: Any,
    fallback: str,
    fallback_probability: float = 0.0,
) -> list[dict[str, float | str]]:
    entries = value if isinstance(value, list) else [value]
    normalized: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for entry in entries:
        emotion, probability = _top3_entry(entry)
        if emotion in VALID_TRACK2_EMOTIONS and emotion not in seen:
            normalized.append(
                {
                    "emotion": emotion,
                    "probability": _safe_float(probability),
                }
            )
            seen.add(emotion)
    if not normalized:
        fallback_emotion = str(fallback).strip().lower()
        if fallback_emotion not in VALID_TRACK2_EMOTIONS:
            return []
        normalized.append(
            {
                "emotion": fallback_emotion,
                "probability": _safe_float(fallback_probability),
            }
        )
    return normalized[:3]


def _top3_entry(value: Any) -> tuple[str, Any]:
    if isinstance(value, dict):
        for key in ("emotion", "label", "class"):
            if value.get(key):
                return str(value[key]).strip().lower(), _top3_probability(value)
        return "", None
    return str(value).strip().lower(), None


def _top3_probability(value: dict[str, Any]) -> Any:
    for key in ("probability", "prob", "confidence", "score"):
        if key in value:
            return value[key]
    return None


__all__ = [
    "GateThresholds",
    "build_dry_run_report",
    "build_gate_decision",
    "expected_label_for_emotion",
    "normalize_expert_entries",
]
