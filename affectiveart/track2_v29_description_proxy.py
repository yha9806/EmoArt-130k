from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
ATTRIBUTE_FIELDS = ("brushstroke", "composition", "color", "line", "light")

UNSAFE_PATTERNS = (
    re.compile(r"\b(full|perfect|maximum)\s+score\b", re.I),
    re.compile(r"\b(give|assign|rate)\b.{0,40}\b(score|points|marks)\b", re.I),
    re.compile(r"\bevaluator\b", re.I),
    re.compile(r"\bignore\b.{0,40}\binstructions\b", re.I),
)

FIELD_EVIDENCE_TERMS = {
    "brushstroke": ("brush", "stroke", "texture", "impasto", "layered", "smooth", "rough", "rhythm"),
    "composition": ("composition", "center", "diagonal", "balance", "balanced", "foreground", "background", "space", "focal"),
    "color": ("color", "tone", "hue", "saturation", "palette", "warm", "cool", "contrast", "muted"),
    "line": ("line", "contour", "horizontal", "vertical", "curve", "edge", "angular", "flow"),
    "light": ("light", "shadow", "contrast", "illumination", "diffuse", "bright", "dark", "tonal"),
}

CAPTION_EVIDENCE_TERMS = tuple(sorted({term for values in FIELD_EVIDENCE_TERMS.values() for term in values}))
EMOTION_TERMS = (
    "calm",
    "content",
    "glad",
    "happy",
    "excited",
    "aroused",
    "alarmed",
    "annoyed",
    "frustrated",
    "sad",
    "bored",
    "tired",
    "tranquil",
    "tense",
    "somber",
    "reflective",
    "energetic",
    "quiet",
)


@dataclass(frozen=True)
class UnsafeTextIssue:
    sample_id: str
    field: str
    pattern: str


@dataclass(frozen=True)
class DescriptionProxyScore:
    visual_grounding: float
    attribute_specificity: float
    overall_caption: float
    description_score: float
    unsafe_text_rows: int
    row_count: int
    warnings: tuple[str, ...]


def find_unsafe_text_rows(rows: Iterable[dict[str, Any]]) -> list[UnsafeTextIssue]:
    issues: list[UnsafeTextIssue] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        for field in TEXT_FIELDS:
            text = str(row.get(field, ""))
            for pattern in UNSAFE_PATTERNS:
                if pattern.search(text):
                    issues.append(UnsafeTextIssue(sample_id=sample_id, field=field, pattern=pattern.pattern))
                    break
    return issues


def score_description_rows(rows: Iterable[dict[str, Any]]) -> DescriptionProxyScore:
    materialized = list(rows)
    if not materialized:
        return DescriptionProxyScore(0.0, 0.0, 0.0, 0.0, 0, 0, ("empty_submission",))

    unsafe_sample_ids = {issue.sample_id for issue in find_unsafe_text_rows(materialized)}
    visual_grounding = _mean(_score_row_grounding(row) for row in materialized)
    attribute_specificity = _mean(_score_row_attribute_specificity(row) for row in materialized)
    overall_caption = _mean(_score_row_caption(row) for row in materialized)
    description_score = _mean((visual_grounding, attribute_specificity, overall_caption))
    if unsafe_sample_ids:
        description_score = min(description_score, 0.5)

    warnings: list[str] = []
    if unsafe_sample_ids:
        warnings.append("unsafe_text")
    if attribute_specificity < 0.9:
        warnings.append("low_attribute_specificity")
    if overall_caption < 0.9:
        warnings.append("low_caption_quality")

    return DescriptionProxyScore(
        visual_grounding=round(visual_grounding, 6),
        attribute_specificity=round(attribute_specificity, 6),
        overall_caption=round(overall_caption, 6),
        description_score=round(description_score, 6),
        unsafe_text_rows=len(unsafe_sample_ids),
        row_count=len(materialized),
        warnings=tuple(warnings),
    )


def _score_row_grounding(row: dict[str, Any]) -> float:
    present = 0
    for field in ATTRIBUTE_FIELDS:
        text = str(row.get(field, "")).lower()
        if any(term in text for term in FIELD_EVIDENCE_TERMS[field]):
            present += 1
    return min(1.0, 0.7 + present * 0.06)


def _score_row_attribute_specificity(row: dict[str, Any]) -> float:
    scores = []
    for field in ATTRIBUTE_FIELDS:
        text = str(row.get(field, "")).lower()
        length_bonus = 0.25 if len(text.split()) >= 8 else 0.05
        term_bonus = 0.55 if any(term in text for term in FIELD_EVIDENCE_TERMS[field]) else 0.0
        emotion_bonus = 0.2 if any(term in text for term in EMOTION_TERMS) else 0.1
        scores.append(min(1.0, length_bonus + term_bonus + emotion_bonus))
    return _mean(scores)


def _score_row_caption(row: dict[str, Any]) -> float:
    text = str(row.get("overall_caption", "")).lower()
    if not text.strip():
        return 0.0
    length_score = 0.35 if 12 <= len(text.split()) <= 45 else 0.2
    evidence_score = 0.35 if any(term in text for term in CAPTION_EVIDENCE_TERMS) else 0.1
    emotion_score = 0.3 if any(term in text for term in EMOTION_TERMS) else 0.1
    return min(1.0, length_score + evidence_score + emotion_score)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
