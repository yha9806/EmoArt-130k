from __future__ import annotations

import json
from pathlib import Path


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
    return 0
