from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EMOTION_MAP = {
    "alarmed": "alarmed",
    "annoyed": "annoyed",
    "aroused": "aroused",
    "bored": "bored",
    "calm": "calm",
    "content": "content",
    "contentment": "content",
    "excited": "excited",
    "frustrated": "frustrated",
    "glad": "glad",
    "happy": "happy",
    "sad": "sad",
    "tired": "tired",
}


@dataclass(frozen=True)
class EmoArtExample:
    request_id: str
    image_path: str
    tar_path: str
    member: str
    style: str
    emotion: str
    valence: str
    arousal: str
    caption: str
    attributes: dict[str, str]
    emotional_impact: str

    def to_compiler_training_row(self) -> dict[str, object]:
        source_parts = [
            self.caption,
            self.attributes.get("brushstroke", ""),
            self.attributes.get("color", ""),
            self.attributes.get("composition", ""),
            self.attributes.get("line", ""),
            self.attributes.get("light", ""),
            self.emotional_impact,
        ]
        source_text = " ".join(part for part in source_parts if part).strip()
        target_lines = [
            f"style: {self.style}",
            f"emotion: {self.emotion}",
            f"valence: {self.valence}",
            f"arousal: {self.arousal}",
        ]
        for key in ("brushstroke", "color", "composition", "line", "light"):
            value = self.attributes.get(key, "")
            if value:
                target_lines.append(f"{key}: {value}")
        return {
            "request_id": self.request_id,
            "style": self.style,
            "emotion": self.emotion,
            "valence": self.valence,
            "arousal": self.arousal,
            "image_path": self.image_path,
            "tar_path": self.tar_path,
            "member": self.member,
            "source_text": source_text,
            "compiler_target": "\n".join(target_lines),
        }


def canonical_emotion(value: str) -> str:
    key = " ".join(str(value or "").strip().lower().split())
    return EMOTION_MAP.get(key, key)


def public_image_location(data_root: str | Path, image_path: str) -> tuple[Path, str]:
    normalized = str(image_path).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 3 or parts[0].lower() != "images":
        raise ValueError(f"unexpected EmoArt image path: {image_path}")
    style = parts[1]
    member = "/".join(parts[1:])
    return Path(data_root) / f"{style}.tar.gz", member


def iter_emoart_examples(
    annotation_json: str | Path,
    *,
    data_root: str | Path = "",
) -> Iterable[EmoArtExample]:
    payload = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    rows = payload.values() if isinstance(payload, dict) else payload
    if not isinstance(rows, Iterable):
        raise ValueError("Annotation.json must be a JSON list or object")
    for raw in rows:
        if isinstance(raw, dict):
            yield extract_emoart_example(raw, data_root=data_root)


def extract_emoart_example(raw: dict[str, Any], *, data_root: str | Path = "") -> EmoArtExample:
    desc = raw.get("description", {}) or {}
    first = desc.get("first_section", {}) or {}
    second = desc.get("second_section", {}) or {}
    attrs = second.get("visual_attributes", {}) or {}
    third = desc.get("third_section", {}) or {}
    image_path = str(raw.get("image_path", ""))
    tar_path, member = public_image_location(data_root, image_path)
    style = member.split("/", 1)[0]
    return EmoArtExample(
        request_id=str(raw.get("request_id", "")),
        image_path=image_path,
        tar_path=str(tar_path),
        member=member,
        style=style,
        emotion=canonical_emotion(str(third.get("dominant_emotion", ""))),
        valence=str(third.get("emotional_valence", "") or ""),
        arousal=str(third.get("emotional_arousal_level", "") or ""),
        caption=str(first.get("description", "") or ""),
        attributes={
            "brushstroke": str(attrs.get("brushstroke", "") or ""),
            "color": str(attrs.get("color", "") or ""),
            "composition": str(attrs.get("composition", "") or ""),
            "line": str(attrs.get("line_quality", "") or ""),
            "light": str(attrs.get("light_and_shadow", "") or ""),
        },
        emotional_impact=str(second.get("emotional_impact", "") or ""),
    )
