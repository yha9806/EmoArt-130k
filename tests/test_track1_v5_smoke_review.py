from __future__ import annotations

import json
from pathlib import Path

from affectiveart.track1_v5_smoke_review import build_review_rows, write_review_reports


def _manifest(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": {"total": len(rows)}, "rows": rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _row(sample_id: str, status: str, image_path: str, *, model: str = "gemini", strategy: str = "v5") -> dict:
    row = {
        "sample_id": sample_id,
        "status": status,
        "image_path": image_path,
        "model": model,
        "candidate_strategy": strategy,
        "caption": f"Caption for {sample_id}",
        "provider_prompt": "Create one artwork.",
        "reference_image_path": "refs/board.jpg",
    }
    if status == "error":
        row["error"] = "provider failed"
    return row


def test_build_review_rows_prefers_later_successful_retry(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    (current_dir / "track1_0001.jpg").write_bytes(b"jpg")
    first = _manifest(
        tmp_path / "first.json",
        [_row("track1_0001", "error", "", model="imagen-4-ultra", strategy="anti_template_diversifier")],
    )
    retry = _manifest(
        tmp_path / "retry.json",
        [_row("track1_0001", "generated", "retry/track1_0001.png", model="gemini-3.1-flash-image-preview")],
    )
    rows = build_review_rows([first, retry], current_image_dir=current_dir)
    assert len(rows) == 1
    assert rows[0]["candidate_image_path"] == "retry/track1_0001.png"
    assert rows[0]["candidate_model"] == "gemini-3.1-flash-image-preview"
    assert rows[0]["attempt_statuses"] == "error -> generated"


def test_write_review_reports_outputs_chinese_html_and_json(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "track1_0001",
            "caption": "Official caption",
            "candidate_status": "generated",
            "candidate_image_path": str(tmp_path / "candidate.png"),
            "current_image_path": str(tmp_path / "current.jpg"),
            "reference_image_path": str(tmp_path / "reference.jpg"),
            "candidate_model": "gemini-3-pro-image",
            "candidate_strategy": "caption_faithful_guard",
            "attempt_statuses": "generated",
            "notes": "",
        }
    ]
    outputs = write_review_reports(rows, out_dir=tmp_path / "review")
    html = Path(outputs["html"]).read_text(encoding="utf-8")
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert "当前冠军包" in html
    assert "v5候选" in html
    assert "官方 reference board" in html
    assert payload["summary"]["total"] == 1
