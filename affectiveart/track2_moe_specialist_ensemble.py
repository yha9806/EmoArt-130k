from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from affectiveart.challenge import TRACK2_JSON_EMOTIONS
from affectiveart.track2_audit import (
    HIGH_AROUSAL_EMOTIONS,
    LOW_AROUSAL_EMOTIONS,
    NEGATIVE_EMOTIONS,
    POSITIVE_EMOTIONS,
)


VALID_TRACK2_EMOTIONS = TRACK2_JSON_EMOTIONS


@dataclass(frozen=True)
class GateThresholds:
    min_supporting_sources: int = 2
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
    "expected_label_for_emotion",
    "normalize_expert_entries",
]
