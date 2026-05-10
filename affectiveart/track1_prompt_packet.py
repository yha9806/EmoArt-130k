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


SURFACE_FEATURE_PATTERNS = (
    ("open album leaf", r"\bopen album leaf\b|\balbum leaf\b"),
    ("blank ruled page", r"\bblank ruled page\b|\bruled page\b"),
    ("pale patterned border", r"\bpale patterned border\b|\bpatterned border\b"),
    ("pale blue patterned border", r"\bpale blue patterned border\b"),
    ("warm brown panel", r"\bwarm brown panel\b"),
    ("aged folded paper", r"\baged\b.*\bfolded paper\b|\baged folded paper\b"),
    ("folded paper", r"\bfolded paper\b"),
    ("aged paper", r"\baged paper\b"),
    ("graph paper surface", r"\bgraph paper\b"),
    ("rectangular frame", r"\brectangular frame\b"),
    ("red seals", r"\bred seals?\b"),
    ("hanging scroll surface", r"\bhanging scroll\b|\bvertical hanging scroll\b"),
)


UNREQUESTED_PHYSICAL_ARTIFACT_FEATURES = [
    "aged paper",
    "folded paper",
    "unrequested white mat border",
    "thick decorative border",
    "paper curl",
    "drop shadow",
    "framed display",
    "wall-mounted presentation",
    "catalog mockup",
]


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
    required_surface_features = _extract_caption_required_surface_features(caption)
    risk_score = _risk_score(caption, allowed_text, category)
    return {
        "sample_id": sample_id,
        "caption": " ".join(caption.strip().split()),
        "artwork_category": category,
        "output_is_artwork_itself": True,
        "hard_requirements": hard_requirements,
        "allowed_text": allowed_text,
        "forbidden_artifacts": list(FORBIDDEN_ARTIFACTS),
        "caption_required_surface_features": required_surface_features,
        "allowed_surface_features": _allowed_surface_features(category),
        "unrequested_physical_artifact_features": _unrequested_physical_artifact_features(
            required_surface_features
        ),
        "style_clues": _extract_style_clues(caption),
        "emotion_clues": _extract_emotion_clues(caption),
        "risk_score": round(risk_score, 3),
    }


def compile_and_rank_packets(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    packets = [
        compile_prompt_packet(str(row["sample_id"]), str(row["caption"]))
        for row in rows
    ]
    packets.sort(key=lambda item: (item["risk_score"], item["sample_id"]), reverse=True)
    return packets


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


def _extract_caption_required_surface_features(caption: str) -> list[str]:
    found = []
    for label, pattern in SURFACE_FEATURE_PATTERNS:
        if re.search(pattern, caption, flags=re.IGNORECASE) and label not in found:
            found.append(label)
    if "aged folded paper" in found:
        found = [item for item in found if item not in {"aged paper", "folded paper"}]
    return found


def _allowed_surface_features(category: str) -> list[str]:
    return {
        "poster": ["poster layout"],
        "album_leaf": ["album leaf flat page surface"],
        "scroll": ["flat scroll paper or silk surface"],
        "drawing_on_paper": ["flat drawing paper surface"],
        "print": ["flat print surface"],
        "painting": ["painted artwork surface"],
    }.get(category, ["artwork surface"])


def _unrequested_physical_artifact_features(required_surface_features: list[str]) -> list[str]:
    blocked = list(UNREQUESTED_PHYSICAL_ARTIFACT_FEATURES)
    required_text = " ".join(required_surface_features)
    if "aged" in required_text:
        blocked = [item for item in blocked if item != "aged paper"]
    if "folded" in required_text:
        blocked = [item for item in blocked if item != "folded paper"]
    return blocked


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
    if category == "drawing_on_paper":
        score += 0.6
    for clue in ("graph paper", "rectangular frame", "calligraphy", "Cyrillic", "propaganda", "border"):
        if clue.lower() in caption.lower():
            score += 0.5
    return score
