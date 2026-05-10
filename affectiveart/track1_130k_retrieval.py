from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RetrievedReference:
    request_id: str
    style: str
    emotion: str
    score: float
    source_text: str
    compiler_target: str


def score_reference_for_packet(packet: dict[str, Any], reference: dict[str, Any]) -> float:
    packet_text = _norm(
        " ".join(
            [
                str(packet.get("caption", "")),
                " ".join(packet.get("hard_requirements", []) or []),
                " ".join(packet.get("style_clues", []) or []),
                " ".join(packet.get("emotion_clues", []) or []),
            ]
        )
    )
    ref_text = _norm(
        " ".join(
            [
                str(reference.get("style", "")),
                str(reference.get("emotion", "")),
                str(reference.get("source_text", "")),
                str(reference.get("compiler_target", "")),
            ]
        )
    )
    score = 0.0
    for style in packet.get("style_clues", []) or []:
        if _norm(style) and _norm(style) in _norm(str(reference.get("style", ""))):
            score += 3.0
    for emotion in packet.get("emotion_clues", []) or []:
        if _norm(emotion) and _norm(emotion) == _norm(str(reference.get("emotion", ""))):
            score += 1.0
    for requirement in packet.get("hard_requirements", []) or []:
        tokens = [token for token in _norm(requirement).split() if len(token) >= 4]
        if tokens and all(token in ref_text for token in tokens[:3]):
            score += 1.5
    overlap = set(packet_text.split()).intersection(ref_text.split())
    score += min(len(overlap) * 0.1, 2.0)
    return round(score, 4)


def select_references(
    packet: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[RetrievedReference]:
    scored = []
    for row in rows:
        score = score_reference_for_packet(packet, row)
        scored.append(
            RetrievedReference(
                request_id=str(row.get("request_id", "")),
                style=str(row.get("style", "")),
                emotion=str(row.get("emotion", "")),
                score=score,
                source_text=str(row.get("source_text", "")),
                compiler_target=str(row.get("compiler_target", "")),
            )
        )
    scored.sort(key=lambda item: (item.score, item.request_id), reverse=True)
    return scored[:top_k]


def _norm(value: str) -> str:
    return " ".join(str(value or "").lower().replace("-", " ").split())
