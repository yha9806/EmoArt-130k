from __future__ import annotations

from collections import Counter
from typing import Any


CALM_WORDS = ("serene", "quiet", "still", "peaceful", "calm", "misty", "tranquil")


def select_vulca_jepa_review_samples(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        priority = _review_priority(row)
        if priority <= 0:
            continue
        enriched = dict(row)
        enriched["review_priority"] = round(priority, 6)
        scored.append(enriched)
    scored.sort(key=lambda row: (row["review_priority"], row.get("sample_id", "")), reverse=True)
    return scored[:limit]


def build_disagreement_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    current_vs_model = 0
    calm_content = 0
    knn_disagreements = 0
    target_counter: Counter[str] = Counter()
    for row in rows:
        current = str(row.get("current", ""))
        emotion = str(row.get("emotion", ""))
        knn_emotion = str(row.get("knn_emotion", emotion))
        if current and emotion and current != emotion:
            current_vs_model += 1
        if current == "content" and emotion == "calm":
            calm_content += 1
        if emotion and knn_emotion and emotion != knn_emotion:
            knn_disagreements += 1
        if emotion:
            target_counter[emotion] += 1
    return {
        "row_count": len(rows),
        "model_current_disagreements": current_vs_model,
        "content_to_calm_candidates": calm_content,
        "model_knn_disagreements": knn_disagreements,
        "model_distribution": dict(sorted(target_counter.items())),
    }


def _review_priority(row: dict[str, Any]) -> float:
    current = str(row.get("current", ""))
    emotion = str(row.get("emotion", ""))
    knn_emotion = str(row.get("knn_emotion", emotion))
    if not current or not emotion or current == emotion:
        return 0.0
    priority = 0.0
    if emotion == knn_emotion:
        priority += float(row.get("knn_confidence", 0.0))
    priority += 0.15 * int(row.get("model_vote_count", 0))
    caption = str(row.get("overall_caption", "")).lower()
    if current == "content" and emotion == "calm" and any(word in caption for word in CALM_WORDS):
        priority += 0.5
    return priority
