from __future__ import annotations

from pathlib import Path

import pytest

from affectiveart.track1_v5_fid_breakthrough import (
    ANTI_TEMPLATE_ROUTE,
    CAPTION_FAITHFUL_ROUTE,
    REFERENCE_FAMILY_ROUTE,
    build_v5_plan,
    build_v5_provider_prompt,
    classify_v5_route,
    summarize_v5_plan,
    write_v5_reports,
)


def test_text_landmark_relation_routes_to_caption_faithful_guard() -> None:
    contract = {
        "sample_id": "track1_0803",
        "caption": "A Socialist Realism poster with Cyrillic text and Kremlin tower.",
        "text_contract": {"required": True, "modes": ["cyrillic"]},
        "reference_contract": {"categories": ["landmark"]},
        "relation_contract": {"required": True, "checks": ["flag_hierarchy"]},
        "surface_contract": "flat_printed_poster",
        "aspect_plan": {"label": "portrait_poster", "width": 768, "height": 1024, "prompt_directive": "Use portrait."},
        "fid_risk": ["poster_distribution_sensitive"],
    }
    route = classify_v5_route(contract, {}, {"reference_assets": ["refs/a.jpg"]})
    assert route["v5_route"] == CAPTION_FAITHFUL_ROUTE
    assert "text_or_symbol_guard" in route["risk_tags"]
    assert "real_world_reference_guard" in route["risk_tags"]
    assert "relation_logic_guard" in route["risk_tags"]
    assert ANTI_TEMPLATE_ROUTE in route["secondary_routes"]


def test_generic_overused_reference_routes_to_anti_template() -> None:
    contract = {
        "sample_id": "track1_0001",
        "caption": "Ukiyo-e sailboat scene before Mount Fuji.",
        "text_contract": {"required": False, "modes": ["none"]},
        "reference_contract": {"categories": []},
        "relation_contract": {"required": False, "checks": []},
        "surface_contract": "artwork_surface",
        "aspect_plan": {"label": "square_artwork", "width": 1024, "height": 1024, "prompt_directive": "Use square."},
        "fid_risk": [],
    }
    route = classify_v5_route(
        contract,
        {},
        {"reference_assets": ["refs/common.jpg"]},
        reference_reuse_counts={"common.jpg": 25},
    )
    assert route["v5_route"] == ANTI_TEMPLATE_ROUTE
    assert "overused_reference_guard" in route["risk_tags"]


def test_generic_low_risk_routes_to_reference_family() -> None:
    contract = {
        "sample_id": "track1_0460",
        "caption": "Ukiyo-e birds among branches with delicate linework.",
        "text_contract": {"required": False, "modes": ["none"]},
        "reference_contract": {"categories": []},
        "relation_contract": {"required": False, "checks": []},
        "surface_contract": "artwork_surface",
        "aspect_plan": {"label": "square_artwork", "width": 1024, "height": 1024, "prompt_directive": "Use square."},
        "fid_risk": [],
    }
    route = classify_v5_route(contract, {}, {"reference_assets": ["refs/rare.jpg"]}, reference_reuse_counts={"rare.jpg": 3})
    assert route["v5_route"] == REFERENCE_FAMILY_ROUTE


def test_provider_prompt_contains_caption_but_not_sample_id_or_placeholders() -> None:
    row = {
        "sample_id": "track1_0460",
        "caption": "Ukiyo-e birds among branches with delicate linework.",
        "v5_route": REFERENCE_FAMILY_ROUTE,
        "secondary_routes": [ANTI_TEMPLATE_ROUTE],
        "aspect_plan": {"label": "square_artwork", "width": 1024, "height": 1024, "prompt_directive": "Use square."},
        "reference_asset_notes": ["official EmoArt-130k retrieval; style=Ukiyo-e; source=Images\\Ukiyo-e\\bird.jpg"],
        "medium_options": ["woodblock_print"],
        "composition_hints": ["branch crop"],
        "hard_constraints": ["caption_content"],
    }
    prompt = build_v5_provider_prompt(row)
    assert "Ukiyo-e birds among branches" in prompt
    assert "track1_0460" not in prompt
    assert "{" not in prompt
    assert "}" not in prompt
    assert "FID-FIRST STYLE DISTRIBUTION" in prompt


def test_build_plan_and_summary_join_sources() -> None:
    contracts = [
        {
            "sample_id": "track1_0460",
            "caption": "Ukiyo-e birds among branches with delicate linework.",
            "style_family": "ukiyoe",
            "surface_contract": "artwork_surface",
            "text_contract": {"required": False, "modes": ["none"]},
            "reference_contract": {"categories": []},
            "relation_contract": {"required": False, "checks": []},
            "aspect_plan": {"label": "square_artwork", "width": 1024, "height": 1024, "prompt_directive": "Use square."},
            "fid_risk": [],
        }
    ]
    expert_routes = [{"sample_id": "track1_0460", "recommended_model": "imagen-4-ultra"}]
    official_routes = [
        {
            "sample_id": "track1_0460",
            "family_id": "generic_artwork",
            "reference_assets": ["refs/rare.jpg"],
            "caption": "ignored fallback",
        }
    ]
    rows = build_v5_plan(contracts, expert_routes, official_routes)
    assert len(rows) == 1
    assert rows[0]["v5_route"] == REFERENCE_FAMILY_ROUTE
    summary = summarize_v5_plan(rows)
    assert summary["total"] == 1
    assert summary["route_counts"] == {REFERENCE_FAMILY_ROUTE: 1}


def test_non_image_model_recommendation_falls_back_to_track1_image_model() -> None:
    contracts = [
        {
            "sample_id": "track1_0002",
            "caption": "An ink wash landscape.",
            "style_family": "ink_wash_painting",
            "surface_contract": "artwork_surface",
            "text_contract": {"required": False, "modes": ["none"]},
            "reference_contract": {"categories": []},
            "relation_contract": {"required": False, "checks": []},
            "aspect_plan": {"label": "square_artwork", "width": 1024, "height": 1024, "prompt_directive": "Use square."},
            "fid_risk": [],
        }
    ]
    expert_routes = [{"sample_id": "track1_0002", "recommended_model": "gemini-3.5-flash"}]
    official_routes = [{"sample_id": "track1_0002", "family_id": "generic_artwork", "reference_assets": ["refs/rare.jpg"]}]
    rows = build_v5_plan(contracts, expert_routes, official_routes)
    assert rows[0]["recommended_model"] == "gemini-3-pro-image"


def test_report_writer_rejects_protected_track1_champion_paths() -> None:
    repo = Path(__file__).resolve().parents[1]
    protected_paths = [
        repo / "submissions" / "track1_submission.json",
        repo / "submissions" / "track1_submission.zip",
        repo / "submissions" / "track1" / "images",
        repo / "submissions" / "track1" / "images" / "nested",
    ]
    for path in protected_paths:
        with pytest.raises(ValueError, match="protected Track1 output path rejected"):
            write_v5_reports([], out_dir=path)
