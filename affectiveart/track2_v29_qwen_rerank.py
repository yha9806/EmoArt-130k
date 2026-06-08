from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VALID_EMOTIONS = {
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

EMOTION_TO_VA = {
    "alarmed": ("Negative", "High"),
    "annoyed": ("Negative", "High"),
    "aroused": ("Positive", "High"),
    "bored": ("Negative", "Low"),
    "calm": ("Positive", "Low"),
    "content": ("Positive", "Low"),
    "excited": ("Positive", "High"),
    "frustrated": ("Negative", "High"),
    "glad": ("Positive", "Low"),
    "happy": ("Positive", "High"),
    "sad": ("Negative", "Low"),
    "tired": ("Negative", "Low"),
}


@dataclass(frozen=True)
class RerankEvidence:
    sample_id: str
    current_emotion: str
    proposed_emotion: str
    current_valence: str
    proposed_valence: str
    current_arousal: str
    proposed_arousal: str
    confidence: float
    source: str


def runtime_capability(*, model_path: str | None = None) -> dict[str, Any]:
    reasons: list[str] = []
    qwen_available = False
    if model_path:
        qwen_available = Path(model_path).exists()
        if not qwen_available:
            reasons.append("missing_model_path")
    else:
        reasons.append("model_path_not_configured")
    return {"qwen_available": qwen_available, "reasons": reasons}


def load_evidence(path: str | Path) -> list[RerankEvidence]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return load_evidence_csv(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("evidence", "rows", "changes", "accepted_changes"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported rerank evidence payload: {path}")
    rows: list[RerankEvidence] = []
    for item in payload:
        if isinstance(item, dict):
            try:
                rows.append(normalize_evidence_row(item))
            except ValueError:
                continue
    return rows


def load_evidence_csv(path: str | Path) -> list[RerankEvidence]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows: list[RerankEvidence] = []
        for row in csv.DictReader(handle):
            try:
                rows.append(normalize_evidence_row(row))
            except ValueError:
                continue
        return rows


def normalize_evidence_row(row: dict[str, Any]) -> RerankEvidence:
    current = _emotion(row.get("current_emotion") or row.get("from_emotion") or row.get("old_emotion"))
    proposed = _emotion(row.get("proposed_emotion") or row.get("to_emotion") or row.get("new_emotion") or row.get("emotion"))
    if proposed not in VALID_EMOTIONS:
        raise ValueError(f"Invalid proposed emotion: {proposed}")
    if current not in VALID_EMOTIONS:
        raise ValueError(f"Invalid current emotion: {current}")
    proposed_valence, proposed_arousal = EMOTION_TO_VA[proposed]
    current_valence, current_arousal = EMOTION_TO_VA[current]
    return RerankEvidence(
        sample_id=str(row.get("sample_id", "")).strip(),
        current_emotion=current,
        proposed_emotion=proposed,
        current_valence=str(row.get("current_valence") or current_valence),
        proposed_valence=str(row.get("proposed_valence") or proposed_valence),
        current_arousal=str(row.get("current_arousal") or current_arousal),
        proposed_arousal=str(row.get("proposed_arousal") or proposed_arousal),
        confidence=_safe_float(row.get("confidence") or row.get("score") or row.get("support_score")),
        source=str(row.get("source") or row.get("method") or "unknown"),
    )


def choose_label_changes(
    evidence_rows: Iterable[RerankEvidence],
    *,
    base_distribution: Counter[str],
    total_rows: int,
    top_emotion_cap: float = 0.60,
    class_floor: int = 2,
    min_confidence: float = 0.9,
) -> list[RerankEvidence]:
    selected: list[RerankEvidence] = []
    distribution = Counter(base_distribution)
    seen: set[str] = set()
    for evidence in sorted(evidence_rows, key=lambda item: (-item.confidence, item.sample_id)):
        if evidence.confidence < min_confidence:
            continue
        if evidence.sample_id in seen:
            continue
        if evidence.current_emotion == evidence.proposed_emotion:
            continue
        if distribution[evidence.current_emotion] <= class_floor:
            continue
        if (distribution[evidence.proposed_emotion] + 1) / max(total_rows, 1) > top_emotion_cap:
            continue
        distribution[evidence.current_emotion] -= 1
        distribution[evidence.proposed_emotion] += 1
        selected.append(evidence)
        seen.add(evidence.sample_id)
    return selected


def evidence_to_label_change_map(evidence_rows: Iterable[RerankEvidence]) -> dict[str, dict[str, str]]:
    return {
        row.sample_id: {
            "emotion": row.proposed_emotion,
            "emotional_valence": row.proposed_valence,
            "emotional_arousal_level": row.proposed_arousal,
        }
        for row in evidence_rows
    }


def _emotion(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"contentment", "contented"}:
        return "content"
    return text


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
