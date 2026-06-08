from __future__ import annotations

from affectiveart.track1_body_logic_gate import (
    build_body_logic_risk_report,
    classify_body_logic_risk,
)


def test_classify_body_logic_risk_marks_0910_grappling_as_p0() -> None:
    result = classify_body_logic_risk(
        "track1_0910",
        "Ukiyo-e scene of muscular men in patterned garments grappling around a yellow wooden "
        "structure beneath gnarled trees, rendered with fine outlines.",
    )

    assert result["priority"] == "P0"
    assert "人体/肢体" in result["categories"]
    assert "动作/姿态" in result["categories"]
    assert "空间关系" in result["categories"]


def test_classify_body_logic_risk_ignores_static_landscape() -> None:
    result = classify_body_logic_risk(
        "track1_0002",
        "Ink wash landscape with mountains, mist, bamboo, and a calm river.",
    )

    assert result["priority"] == ""
    assert result["categories"] == []


def test_build_body_logic_report_preserves_actual_manifest_gap() -> None:
    p1_report = {
        "packages": [
            {
                "package": "current",
                "rows": [
                    {
                        "sample_id": "track1_0910",
                        "caption": "Ukiyo-e scene of muscular men grappling around a yellow wooden structure.",
                        "actual_hash_changed": False,
                        "manifest_changed": False,
                        "manifest_actual_gap": False,
                        "known_placeholder_type": "",
                    }
                ],
            },
            {
                "package": "b120_fixed",
                "rows": [
                    {
                        "sample_id": "track1_0910",
                        "caption": "Ukiyo-e scene of muscular men grappling around a yellow wooden structure.",
                        "actual_hash_changed": True,
                        "manifest_changed": False,
                        "manifest_actual_gap": True,
                        "known_placeholder_type": "",
                    }
                ],
            },
        ]
    }

    report = build_body_logic_risk_report(p1_report)

    assert report["summary"]["total_risk_recall_samples"] == 1
    assert report["summary"]["p0_samples"] == 1
    assert report["summary"]["manifest_gap_samples"] == 1
    row = report["rows"][0]
    assert row["sample_id"] == "track1_0910"
    assert row["priority"] == "P0"
    assert row["packages"]["b120_fixed"]["actual_hash_changed"] is True
    assert row["packages"]["b120_fixed"]["manifest_actual_gap"] is True
