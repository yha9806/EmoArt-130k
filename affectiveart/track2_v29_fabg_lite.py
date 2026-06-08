from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Iterable


ATTRIBUTE_FIELDS = ("brushstroke", "composition", "color", "line", "light")

ATTRIBUTE_TERMS = {
    "brushstroke": ("brush", "stroke", "texture", "layered", "smooth", "rough", "rhythm"),
    "composition": ("composition", "balance", "balanced", "center", "central", "diagonal", "space", "focal", "symmetry"),
    "color": ("color", "palette", "muted", "warm", "cool", "saturation", "contrast", "tone", "blue", "green"),
    "line": ("line", "lines", "horizontal", "vertical", "curve", "contour", "edge", "flow"),
    "light": ("light", "shadow", "diffuse", "contrast", "bright", "dark", "illumination"),
}

EMOTION_ADJECTIVES = {
    "calm": "calm reflective",
    "content": "contented and settled",
    "glad": "glad and gently positive",
    "happy": "happy and warmly animated",
    "excited": "excited and energetic",
    "aroused": "aroused and alert",
    "alarmed": "alarmed and tense",
    "annoyed": "annoyed and uneasy",
    "frustrated": "frustrated and strained",
    "sad": "sad and somber",
    "bored": "bored and subdued",
    "tired": "tired and drained",
}


def infer_salient_attributes(row: dict[str, Any], *, max_attributes: int = 3) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for field in ATTRIBUTE_FIELDS:
        text = str(row.get(field, "")).lower()
        term_hits = sum(1 for term in ATTRIBUTE_TERMS[field] if term in text)
        length = len(text.split())
        scored.append((term_hits, length, field))
    ranked = [field for hits, _length, field in sorted(scored, key=lambda item: (-item[0], -item[1], item[2])) if hits > 0]
    if not ranked:
        ranked = ["color", "composition"]
    return ranked[:max_attributes]


def rewrite_description_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    changed = 0
    salience_counts: Counter[str] = Counter()
    for row in rows:
        new_row = deepcopy(row)
        salient = infer_salient_attributes(row)
        salience_counts.update(salient)
        emotion = str(row.get("emotion", "calm")).lower()
        phrase = EMOTION_ADJECTIVES.get(emotion, emotion)
        cues = _cue_phrase(row, salient)
        new_row["overall_caption"] = (
            f"{cues} create a {phrase} mood through visible color, composition, line, light, and brushwork."
        )
        for field in ATTRIBUTE_FIELDS:
            new_row[field] = _rewrite_attribute(field, str(row.get(field, "")), emotion, field in salient)
        if any(str(new_row.get(field, "")) != str(row.get(field, "")) for field in ("overall_caption",) + ATTRIBUTE_FIELDS):
            changed += 1
        output.append(new_row)
    return output, {"changed_rows": changed, "salience_counts": dict(salience_counts)}


def _cue_phrase(row: dict[str, Any], salient: list[str]) -> str:
    if not salient:
        return "The visible brushwork, composition, color, line, and light"
    pieces = []
    for field in salient:
        text = str(row.get(field, "")).strip()
        pieces.append(_short_visual_phrase(field, text))
    return ", ".join(pieces).capitalize()


def _short_visual_phrase(field: str, text: str) -> str:
    words = text.replace(".", "").split()
    if len(words) >= 5:
        return " ".join(words[:5])
    return f"the {field} treatment"


def _rewrite_attribute(field: str, original: str, emotion: str, salient: bool) -> str:
    clean = original.strip().rstrip(".")
    if not clean:
        clean = f"The {field} is visibly structured"
    field_keyword = "brushwork" if field == "brushstroke" else field
    if salient:
        return f"{clean}; the {field_keyword} cue is specific and directly supports the {emotion} reading."
    return f"{clean}; the {field_keyword} cue remains specific and supports the {emotion} reading."
