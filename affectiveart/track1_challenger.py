from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def effective_create_returncode(completed_returncode: int, create_json: Path, image_path: Path) -> int:
    if completed_returncode != 0:
        return completed_returncode
    if not create_json.exists() or not image_path.exists():
        return 1
    try:
        payload = json.loads(create_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 1
    if payload.get("status") != "completed":
        return 1
    if str(payload.get("best_image_url", "")).startswith("mock://"):
        return 1
    if not _looks_like_raster_image(image_path):
        return 1
    return 0


def _looks_like_raster_image(image_path: Path) -> bool:
    header = image_path.read_bytes()[:16]
    if header.startswith(b"\xff\xd8\xff"):
        return True
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return True
    return False


def merge_run_summary_rows(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    replacements = {row.get("sample_id"): row for row in new_rows}
    merged = []
    seen = set()
    for row in existing_rows:
        sample_id = row.get("sample_id")
        if sample_id in replacements:
            merged.append(replacements[sample_id])
            seen.add(sample_id)
        else:
            merged.append(row)
    for row in new_rows:
        sample_id = row.get("sample_id")
        if sample_id not in seen:
            merged.append(row)
    return merged
