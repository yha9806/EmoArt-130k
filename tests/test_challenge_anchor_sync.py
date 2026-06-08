from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from affectiveart.challenge_anchor_sync import (
    build_unified_anchor_registry,
    load_csv_rows,
    write_anchor_registry_outputs,
)


def test_build_unified_registry_merges_track1_precise_scores_with_local_metadata() -> None:
    registry = build_unified_anchor_registry(
        track1_api_rows=[
            {
                "participant": "vulcaart",
                "submission_id": "784403",
                "file_name": "hybrid_probe.zip",
                "official_overall": 0.7396402011,
                "official_fid": 105.6638160234,
                "official_fid_score": 0.4862304023,
                "official_aas": 0.99305,
            }
        ],
        track1_anchor_rows=[
            {
                "submission_id": "784403",
                "local_package": "hybrid_redteam_fid_pass4",
                "local_fid_like": "58.348904",
                "official_overall": "0.74",
                "official_fid": "105.66",
            }
        ],
    )

    anchor = registry["anchors_by_key"]["track1:784403"]

    assert anchor["official_overall"] == 0.7396402011
    assert anchor["official_fid"] == 105.6638160234
    assert anchor["local_package"] == "hybrid_redteam_fid_pass4"
    assert anchor["local_fid_like"] == 58.348904
    assert anchor["calibration_kind"] == "own_official"
    assert not registry["consistency"]["blocking_issue_count"]


def test_build_unified_registry_syncs_track2_exact_and_scoreboard_rows() -> None:
    registry = build_unified_anchor_registry(
        track2_exact_rows=[
            {
                "submission_id": "779605",
                "file_name": "track2_submission_moe_v2_accept5_candidate.zip",
                "official_overall": "0.836408",
                "official_classification": "0.723150",
                "official_description": "0.949667",
            }
        ],
        track2_scoreboard_rows=[
            {
                "candidate_name": "official_779605_moe_v2_anchor",
                "calibration_kind": "exact_official",
                "overall_expected": "0.836408",
                "classification_expected": "0.723150",
                "description_expected": "0.949667",
                "anchor_submission_id": "779605",
                "json_path": "/tmp/track2_submission_moe_v2_accept5_candidate.json",
            },
            {
                "candidate_name": "v22_calmshift120",
                "calibration_kind": "estimated",
                "overall_expected": "0.846335",
                "classification_expected": "0.743002",
                "description_expected": "0.949667",
                "anchor_submission_id": "779605",
            },
        ],
    )

    exact = registry["anchors_by_key"]["track2:779605"]
    estimated = registry["anchors_by_key"]["track2_estimated:v22_calmshift120"]

    assert exact["official_overall"] == 0.836408
    assert exact["official_classification"] == 0.72315
    assert exact["json_path"] == "/tmp/track2_submission_moe_v2_accept5_candidate.json"
    assert estimated["calibration_kind"] == "estimated"
    assert estimated["anchor_submission_id"] == "779605"
    assert registry["summary"]["track2_exact_anchor_count"] == 1
    assert registry["summary"]["track2_estimated_anchor_count"] == 1


def test_author_resource_index_uses_source_scan_and_method_cards() -> None:
    registry = build_unified_anchor_registry(
        source_index_rows=[
            {
                "source_type": "profile",
                "name": "Hongxia Xie / AVC Lab",
                "url": "https://www.hongxiaxie.net/",
                "verified_public_fact": "JLU AVC Lab PI; lists EmoArt and EmoVIT.",
                "track2_use": "Collaborator graph seed",
                "status": "verified",
            }
        ],
        public_resource_scan={
            "openalex_works": {
                '"EmoArt"': [
                    {
                        "title": "EmoArt",
                        "publication_year": 2025,
                        "authorships": [
                            {"author": "Cheng Zhang", "institution": "Jilin University"},
                            {"author": "Hongxia Xie", "institution": "Jilin University"},
                        ],
                    }
                ]
            }
        },
        method_cards={
            "cards": [
                {
                    "id": "fabg",
                    "source_group": "Cheng Zhang / Hongxia Xie / Wen-Huang Cheng",
                    "engineering_module": "salience_aware_prompt_compiler",
                    "public_sources": ["https://github.com/zhiliangzhang/FAB-G"],
                }
            ]
        },
    )

    authors = registry["author_index"]

    assert "Hongxia Xie" in authors
    assert "Cheng Zhang" in authors
    assert "Wen-Huang Cheng" in authors
    assert "Jilin University" in authors["Hongxia Xie"]["institutions"]
    assert "fabg" in authors["Hongxia Xie"]["method_card_ids"]
    assert registry["resource_summary"]["method_card_count"] == 1


def test_write_anchor_registry_outputs(tmp_path: Path) -> None:
    registry = build_unified_anchor_registry(
        track1_anchor_rows=[
            {
                "participant": "vulcaart",
                "submission_id": "782831",
                "local_package": "v3_gate7",
                "official_fid_score": "0.55",
                "official_aas": "0.98",
            }
        ]
    )

    write_anchor_registry_outputs(
        registry,
        json_path=tmp_path / "registry.json",
        csv_path=tmp_path / "registry.csv",
        md_path=tmp_path / "registry.md",
    )

    assert json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))["summary"]["anchor_count"] == 1
    assert load_csv_rows(tmp_path / "registry.csv")[0]["anchor_key"] == "track1:782831"
    assert "Unified Challenge Anchor Registry" in (tmp_path / "registry.md").read_text(encoding="utf-8")


def test_cli_writes_unified_registry_outputs(tmp_path: Path) -> None:
    track1_anchors = tmp_path / "track1.csv"
    track1_anchors.write_text(
        "participant,submission_id,local_package,official_fid_score,official_aas\n"
        "vulcaart,782831,v3_gate7,0.55,0.98\n",
        encoding="utf-8",
    )
    track2_exact = tmp_path / "track2.csv"
    track2_exact.write_text(
        "submission_id,file_name,official_overall,official_classification,official_description\n"
        "779605,track2.zip,0.836408,0.723150,0.949667\n",
        encoding="utf-8",
    )
    source_index = tmp_path / "sources.csv"
    source_index.write_text(
        "source_type,name,url,verified_public_fact,track2_use,status\n"
        "profile,Hongxia Xie / AVC Lab,https://www.hongxiaxie.net/,PI page,method lineage,verified\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "challenge_anchor_sync.py"
    spec = importlib.util.spec_from_file_location("challenge_anchor_sync_cli", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    code = module.main(
        [
            "--track1-anchors-csv",
            str(track1_anchors),
            "--track2-exact-csv",
            str(track2_exact),
            "--source-index-csv",
            str(source_index),
            "--out-json",
            str(tmp_path / "registry.json"),
            "--out-csv",
            str(tmp_path / "registry.csv"),
            "--out-md",
            str(tmp_path / "registry.md"),
        ]
    )

    assert code == 0
    assert json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))["summary"]["anchor_count"] == 2
    assert (tmp_path / "registry.csv").exists()
    assert "Hongxia Xie" in (tmp_path / "registry.md").read_text(encoding="utf-8")
