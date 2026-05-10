from __future__ import annotations

import re
from typing import Any


FORBIDDEN_ARTIFACTS = [
    "gallery wall",
    "museum installation",
    "framed photo mockup",
    "catalog page",
    "sample id",
    "filename label",
    "watermark",
    "unrequested readable text",
]


TEXT_PATTERNS = (
    ("calligraphy", r"\bcalligraphy\b"),
    ("inscription", r"\binscriptions?\b|\binscribed\b"),
    ("Cyrillic lettering", r"\bcyrillic\b|\blettering\b|\bheadline\b|\bslogan\b"),
    ("seals", r"\bseals?\b"),
    ("date or numerals", r"\b(year|date|numerals?|numbers?|digits?)\b"),
)


REQUIREMENT_PATTERNS = (
    ("Soviet soldiers", r"\bsoviet soldiers?\b"),
    ("Prussian figures", r"\bprussian figures?\b"),
    ("rifles and bayonets", r"\brifles?\b|\bbayonets?\b"),
    ("aged folded paper", r"\baged\b.*\bfolded paper\b|\bfolded paper\b"),
    ("graph paper", r"\bgraph paper\b"),
    ("rectangular frame", r"\brectangular frame\b"),
    ("branching lines", r"\bbranching lines?\b"),
    ("tree-like network", r"\btree-like network\b|\btree like network\b"),
    ("hearts and geometric marks", r"\bhearts?\b.*\bgeometric\b|\bgeometric\b.*\bhearts?\b"),
    ("lotus blossoms", r"\blotus blossoms?\b|\blotus\b"),
    ("slender stems", r"\bslender stems?\b"),
    ("small leaves", r"\bsmall leaves\b"),
)


def detect_artwork_category(caption: str) -> str:
    text = caption.lower()
    if "poster" in text:
        return "poster"
    if "scroll" in text:
        return "scroll"
    if "graph paper" in text or "pencil" in text or "hand-drawn" in text:
        return "drawing_on_paper"
    if "album leaf" in text:
        return "album_leaf"
    if "engraving" in text or "woodblock" in text or "print" in text:
        return "print"
    return "painting"


def compile_prompt_packet(sample_id: str, caption: str) -> dict[str, Any]:
    allowed_text = _extract_allowed_text(caption)
    hard_requirements = _extract_hard_requirements(caption)
    category = detect_artwork_category(caption)
    risk_score = _risk_score(caption, allowed_text, category)
    return {
        "sample_id": sample_id,
        "caption": " ".join(caption.strip().split()),
        "artwork_category": category,
        "output_is_artwork_itself": True,
        "hard_requirements": hard_requirements,
        "allowed_text": allowed_text,
        "forbidden_artifacts": list(FORBIDDEN_ARTIFACTS),
        "style_clues": _extract_style_clues(caption),
        "emotion_clues": _extract_emotion_clues(caption),
        "risk_score": round(risk_score, 3),
    }


def _extract_allowed_text(caption: str) -> list[str]:
    found = []
    for label, pattern in TEXT_PATTERNS:
        if re.search(pattern, caption, flags=re.IGNORECASE):
            found.append(label)
    return found


def _extract_hard_requirements(caption: str) -> list[str]:
    found = []
    for label, pattern in REQUIREMENT_PATTERNS:
        if re.search(pattern, caption, flags=re.IGNORECASE):
            found.append(label)
    if not found:
        found.append(caption.split(",", 1)[0].strip())
    return found


def _extract_style_clues(caption: str) -> list[str]:
    clues = []
    for clue in (
        "Socialist Realism",
        "Gongbi",
        "Ukiyo-e",
        "ink and wash",
        "Baroque",
        "Renaissance",
        "monochrome pencil",
        "watercolor",
    ):
        if clue.lower() in caption.lower():
            clues.append(clue)
    return clues


def _extract_emotion_clues(caption: str) -> list[str]:
    text = caption.lower()
    return [
        clue
        for clue in ("calm", "dramatic", "somber", "joyful", "tense", "melancholic", "serene")
        if clue in text
    ]


def _risk_score(caption: str, allowed_text: list[str], category: str) -> float:
    score = 0.0
    if allowed_text:
        score += 1.0
    if category in {"poster", "scroll", "drawing_on_paper", "album_leaf"}:
        score += 1.0
    for clue in ("graph paper", "rectangular frame", "calligraphy", "Cyrillic", "propaganda", "border"):
        if clue.lower() in caption.lower():
            score += 0.5
    return score
