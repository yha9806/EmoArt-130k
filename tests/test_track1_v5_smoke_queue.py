from __future__ import annotations

import json
from pathlib import Path

from affectiveart.track1_moe_generation import load_moe_packets
from affectiveart.track1_v5_smoke_queue import (
    DEFAULT_SMOKE_ANCHORS,
    build_smoke_packets,
    select_smoke_rows,
    write_generated_review,
    write_smoke_queue,
)


def _row(sample_id: str, route: str, style: str, aspect: str, risks: list[str] | None = None) -> dict:
    return {
        "sample_id": sample_id,
        "caption": f"Caption for {sample_id}",
        "v5_route": route,
        "style_family": style,
        "v7_family_id": "generic_artwork",
        "risk_tags": risks or [],
        "aspect_plan": {"label": aspect, "width": 1024, "height": 1024, "prompt_directive": "Use square."},
        "recommended_model": "gemini-3.1-flash-image",
        "reference_assets": [],
        "provider_prompt_path": "",
    }


def test_select_smoke_rows_keeps_anchor_hard_cases_first() -> None:
    rows = [
        _row("track1_0001", "reference_family_primary", "ukiyoe_woodblock", "square_artwork"),
        _row("track1_0077", "caption_faithful_guard", "socialist_realism_poster", "portrait_poster"),
        _row("track1_0803", "caption_faithful_guard", "socialist_realism_poster", "portrait_poster"),
    ]
    selected = select_smoke_rows(rows, target_count=2, anchors=["track1_0803", "track1_0077"])
    assert [row["sample_id"] for row in selected] == ["track1_0803", "track1_0077"]


def test_select_smoke_rows_covers_routes_aspects_and_styles() -> None:
    rows = [
        _row("track1_0001", "reference_family_primary", "ukiyoe_woodblock", "square_artwork"),
        _row("track1_0002", "reference_family_primary", "ink_wash_painting", "square_artwork"),
        _row("track1_0003", "anti_template_diversifier", "generic_painting", "square_artwork"),
        _row("track1_0004", "caption_faithful_guard", "ink_wash_scroll", "vertical_scroll"),
        _row("track1_0005", "caption_faithful_guard", "album_leaf_ink", "album_spread"),
        _row("track1_0006", "caption_faithful_guard", "socialist_realism_poster", "portrait_poster"),
    ]
    selected = select_smoke_rows(rows, target_count=6, anchors=[])
    assert {row["v5_route"] for row in selected} == {
        "anti_template_diversifier",
        "caption_faithful_guard",
        "reference_family_primary",
    }
    assert {"vertical_scroll", "album_spread", "portrait_poster"}.issubset(
        {row["aspect_plan"]["label"] for row in selected}
    )
    assert "socialist_realism_poster" in {row["style_family"] for row in selected}


def test_select_smoke_rows_fills_route_targets_after_anchor_block() -> None:
    rows = [
        _row(f"track1_00{i:02d}", "caption_faithful_guard", "socialist_realism_poster", "portrait_poster")
        for i in range(1, 7)
    ]
    rows += [
        _row("track1_0100", "caption_faithful_guard", "ink_wash_scroll", "vertical_scroll", ["support_surface_guard"]),
        _row("track1_0101", "caption_faithful_guard", "album_leaf_ink", "album_spread", ["support_surface_guard"]),
        _row("track1_0102", "caption_faithful_guard", "gongbi_scroll", "horizontal_scroll", ["text_or_symbol_guard"]),
        _row("track1_0103", "caption_faithful_guard", "generic_artwork", "panel_story", ["relation_logic_guard"]),
        _row("track1_0104", "caption_faithful_guard", "watercolor_painting", "square_artwork", ["real_world_reference_guard"]),
    ]
    rows += [
        _row(f"track1_10{i:02d}", "reference_family_primary", "ukiyoe_woodblock", "square_artwork")
        for i in range(1, 9)
    ]
    rows += [
        _row(f"track1_20{i:02d}", "anti_template_diversifier", "generic_painting", "square_artwork")
        for i in range(1, 9)
    ]
    selected = select_smoke_rows(
        rows,
        target_count=12,
        anchors=[f"track1_00{i:02d}" for i in range(1, 7)],
    )
    counts = {route: sum(1 for row in selected if row["v5_route"] == route) for route in {row["v5_route"] for row in rows}}
    assert counts["caption_faithful_guard"] <= 7
    assert counts["reference_family_primary"] >= 3
    assert counts["anti_template_diversifier"] >= 2


def test_select_smoke_rows_prioritizes_missing_aspects_over_duplicate_aspects() -> None:
    rows = [
        _row("track1_0077", "caption_faithful_guard", "socialist_realism_poster", "portrait_poster"),
        _row("track1_0091", "caption_faithful_guard", "socialist_realism_poster", "portrait_poster"),
        _row("track1_0128", "caption_faithful_guard", "socialist_realism_poster", "portrait_poster"),
        _row("track1_0476", "caption_faithful_guard", "socialist_realism_poster", "portrait_poster"),
        _row("track1_0973", "caption_faithful_guard", "socialist_realism_poster", "portrait_poster", ["text_or_symbol_guard"]),
        _row("track1_0115", "caption_faithful_guard", "album_leaf_ink", "panel_story", ["support_surface_guard"]),
        _row("track1_0063", "caption_faithful_guard", "ink_wash_scroll", "vertical_scroll", ["support_surface_guard"]),
    ]
    rows += [
        _row(f"track1_10{i:02d}", "reference_family_primary", "ukiyoe_woodblock", "square_artwork")
        for i in range(1, 6)
    ]
    rows += [
        _row(f"track1_20{i:02d}", "anti_template_diversifier", "generic_painting", "square_artwork")
        for i in range(1, 4)
    ]
    selected = select_smoke_rows(
        rows,
        target_count=14,
        anchors=["track1_0077", "track1_0091", "track1_0128", "track1_0476"],
    )
    assert "panel_story" in {row["aspect_plan"]["label"] for row in selected}


def test_build_smoke_packets_loads_prompt_file_and_review_metadata(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Create the official artwork.\n", encoding="utf-8")
    row = _row("track1_0077", "caption_faithful_guard", "socialist_realism_poster", "portrait_poster")
    row["provider_prompt_path"] = str(prompt)
    packets = build_smoke_packets([row])
    assert packets[0]["provider_prompt"] == "Create the official artwork.\n"
    assert packets[0]["candidate_strategy"] == "caption_faithful_guard"
    assert packets[0]["review_metadata"]["recommended_model"] == "gemini-3.1-flash-image"
    assert packets[0]["review_metadata"]["candidate_count"] == 1


def test_write_smoke_queue_outputs_generation_compatible_jsonl(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Create the official artwork.\n", encoding="utf-8")
    row = _row(DEFAULT_SMOKE_ANCHORS[0], "caption_faithful_guard", "socialist_realism_poster", "portrait_poster")
    row["provider_prompt_path"] = str(prompt)
    outputs = write_smoke_queue([row], out_dir=tmp_path / "queue")
    packets = load_moe_packets(outputs["jsonl"])
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert len(packets) == 1
    assert packets[0]["sample_id"] == DEFAULT_SMOKE_ANCHORS[0]
    assert payload["summary"]["total"] == 1
    assert Path(outputs["html"]).exists()


def test_write_generated_review_outputs_current_candidate_and_reference(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    generated_dir = tmp_path / "generated"
    review_dir = tmp_path / "review"
    current_dir.mkdir()
    (current_dir / "track1_0077.jpg").write_bytes(b"current")
    candidate = generated_dir / "images" / "track1_0077_caption_faithful_guard_c01.png"
    metadata = candidate.with_suffix(".json")
    reference_board = generated_dir / "reference_boards" / "track1_0077_caption_faithful_guard.jpg"
    candidate.parent.mkdir(parents=True)
    reference_board.parent.mkdir(parents=True)
    candidate.write_bytes(b"candidate")
    reference_board.write_bytes(b"reference")
    metadata.write_text(
        json.dumps(
            {
                "image_model": "gemini-3-pro-image",
                "reference_image_fallback_without_reference": True,
                "reference_image_fallback_reason": "BlockedReason.OTHER",
            }
        ),
        encoding="utf-8",
    )
    queue_json = tmp_path / "queue.json"
    queue_json.write_text(
        json.dumps(
            {
                "packets": [
                    {
                        "sample_id": "track1_0077",
                        "caption": "A Soviet poster.",
                        "candidate_strategy": "caption_faithful_guard",
                        "style_family": "socialist_realism_poster",
                        "risk_tags": ["text_or_symbol_guard"],
                        "aspect_plan": {"label": "portrait_poster"},
                        "reference_assets": [str(reference_board)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "candidate_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "sample_id": "track1_0077",
                        "status": "cached",
                        "image_path": str(candidate),
                        "metadata_path": str(metadata),
                        "reference_image_path": str(reference_board),
                        "model": "gemini-3-pro-image",
                        "candidate_strategy": "caption_faithful_guard",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    outputs = write_generated_review(
        queue_json=queue_json,
        manifest_json=manifest,
        current_images_dir=current_dir,
        out_dir=review_dir,
    )
    html = Path(outputs["html"]).read_text(encoding="utf-8")

    assert "current：当前 champion" in html
    assert "candidate：v5 smoke" in html
    assert "reference board" in html
    assert "reference fallback：是" in html
    assert "BlockedReason.OTHER" in html
