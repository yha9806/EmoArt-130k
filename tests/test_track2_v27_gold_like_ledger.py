from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from affectiveart.track2_v27_gold_like_ledger import (
    choose_v27_final_gate,
    classify_v27_evidence_row,
    select_v27_changes,
    write_v27_candidate_outputs,
)


def _evidence_row(
    current: str = "content",
    proposed: str = "calm",
    *,
    sample_id: str = "track2_0001",
    support_score: float = 2.4,
    public_style_support_score: float = 0.6,
    public_duplicate_support_score: float = 0.0,
    max_confidence: float = 0.92,
    model_vote_count: int = 3,
    near_duplicate: bool = False,
    exact_duplicate: bool = False,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "current_emotion": current,
        "proposed_emotion": proposed,
        "transition": f"{current}->{proposed}",
        "support_score": support_score,
        "public_style_support_score": public_style_support_score,
        "public_duplicate_support_score": public_duplicate_support_score,
        "max_confidence": max_confidence,
        "model_vote_count": model_vote_count,
        "near_duplicate": near_duplicate,
        "exact_duplicate": exact_duplicate,
        "all_sources": "clip,dinov2,public_clean,public_inclusive,siglip2",
    }


def _accepted_row(
    sample_id: str,
    current: str,
    proposed: str,
    score: float,
    *,
    tier: str = "tier1_visual_duplicate",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "current_emotion": current,
        "proposed_emotion": proposed,
        "transition": f"{current}->{proposed}",
        "v27_decision": "accept_candidate",
        "v27_tier": tier,
        "v27_score": score,
    }


def _base_row(sample_id: str, emotion: str = "content") -> dict[str, str]:
    valence = "Negative" if emotion in {"alarmed", "annoyed", "bored", "frustrated", "sad", "tired"} else "Positive"
    arousal = "High" if emotion in {"alarmed", "annoyed", "aroused", "excited", "frustrated", "happy"} else "Low"
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": "A concrete artwork caption with no evaluator instructions.",
        "brushstroke": "Brushwork is described specifically.",
        "composition": "The composition is described specifically.",
        "color": "The color palette is described specifically.",
        "line": "Line quality is described specifically.",
        "light": "Lighting is described specifically.",
    }


class Track2V27GoldLikeLedgerTests(unittest.TestCase):
    def test_classifies_visual_duplicate_as_tier1_accept(self) -> None:
        scored = classify_v27_evidence_row(
            _evidence_row(near_duplicate=True, public_duplicate_support_score=0.97, model_vote_count=2)
        )

        self.assertEqual(scored["decision"], "accept_candidate")
        self.assertEqual(scored["tier"], "tier1_visual_duplicate")

    def test_rejects_weak_knn_copy_without_duplicate_evidence(self) -> None:
        scored = classify_v27_evidence_row(
            _evidence_row(
                near_duplicate=False,
                public_duplicate_support_score=0.0,
                model_vote_count=1,
                support_score=0.8,
                public_style_support_score=0.0,
            )
        )

        self.assertEqual(scored["decision"], "hold")
        self.assertIn("weak_model_or_style_only", scored["reasons"])

    def test_blocks_invalid_emotion_labels_before_va_repair(self) -> None:
        scored = classify_v27_evidence_row(_evidence_row(proposed="mystery"))

        self.assertEqual(scored["decision"], "block")
        self.assertIn("invalid_emotion_label", scored["reasons"])

    def test_select_ignores_hold_rows_even_with_higher_score(self) -> None:
        hold_row = {
            **_accepted_row("track2_0001", "content", "calm", 99.0),
            "v27_decision": "hold",
            "v27_tier": "tier3_weak_model_or_style",
        }
        selected = select_v27_changes(
            [
                hold_row,
                _accepted_row("track2_0002", "content", "calm", 4.0, tier="tier1_visual_duplicate"),
            ],
            base_distribution=Counter({"content": 100, "calm": 100}),
        )

        self.assertEqual([row["sample_id"] for row in selected], ["track2_0002"])

    def test_select_caps_same_series_and_avoids_duplicate_samples(self) -> None:
        rows = [
            _accepted_row("track2_0001", "content", "calm", 4.0, tier="tier2_same_series"),
            _accepted_row("track2_0001", "content", "excited", 3.9, tier="tier1_visual_duplicate"),
            _accepted_row("track2_0002", "calm", "content", 3.8, tier="tier2_same_series"),
        ]

        selected = select_v27_changes(
            rows,
            base_distribution=Counter({"content": 100, "calm": 100}),
            tier2_cap=1,
        )

        self.assertEqual([row["sample_id"] for row in selected], ["track2_0001"])

    def test_write_v27_candidate_rejects_formal_submission_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "submissions"
            out_dir.mkdir()
            with self.assertRaises(ValueError):
                write_v27_candidate_outputs(
                    base_rows=[_base_row("track2_0001")],
                    selected_changes=[],
                    out_json=out_dir / "track2_submission.json",
                    out_zip=out_dir / "track2_submission_v27_gold_like_candidate.zip",
                    report_json=root / "report.json",
                    report_md=root / "report.md",
                    profile="strict",
                )

    def test_write_v27_candidate_writes_zip_and_repairs_va(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_json = root / "track2_submission_v27_gold_like_candidate.json"
            out_zip = root / "track2_submission_v27_gold_like_candidate.zip"
            report = write_v27_candidate_outputs(
                base_rows=[_base_row("track2_0001", "content")],
                selected_changes=[
                    _accepted_row("track2_0001", "content", "alarmed", 4.0, tier="tier1_visual_duplicate")
                ],
                out_json=out_json,
                out_zip=out_zip,
                report_json=root / "report.json",
                report_md=root / "report.md",
                profile="strict",
            )

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["emotion"], "alarmed")
            self.assertEqual(payload[0]["emotional_valence"], "Negative")
            self.assertEqual(payload[0]["emotional_arousal_level"], "High")
            self.assertEqual(report["accepted_label_changes"], 1)
            self.assertFalse(report["formal_submission_overwritten"])
            self.assertEqual(report["label_consistency_issue_count"], 0)
            with zipfile.ZipFile(out_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])

    def test_choose_v27_final_gate_holds_below_target(self) -> None:
        gate = choose_v27_final_gate(
            projected_overall=0.889999,
            projected_classification=0.83,
            projected_description=0.95,
            label_consistency_issue_count=0,
            missing_emotions=[],
            unsafe_text_count=0,
            top_emotion_share=0.57,
            cross_quadrant_change_count=0,
            cross_quadrant_risk_report_count=0,
        )

        self.assertEqual(gate["decision"], "hold_no_submit")
        self.assertIn("below_089_target", gate["reasons"])

    def test_choose_v27_final_gate_recommends_when_all_hard_gates_pass(self) -> None:
        gate = choose_v27_final_gate(
            projected_overall=0.891,
            projected_classification=0.832,
            projected_description=0.95,
            label_consistency_issue_count=0,
            missing_emotions=[],
            unsafe_text_count=0,
            top_emotion_share=0.57,
            cross_quadrant_change_count=2,
            cross_quadrant_risk_report_count=2,
        )

        self.assertEqual(gate["decision"], "recommend_final_submit")
        self.assertEqual(gate["reasons"], [])

    def test_choose_v27_final_gate_holds_unreported_cross_quadrant_risk(self) -> None:
        gate = choose_v27_final_gate(
            projected_overall=0.891,
            projected_classification=0.832,
            projected_description=0.95,
            label_consistency_issue_count=0,
            missing_emotions=[],
            unsafe_text_count=0,
            top_emotion_share=0.57,
            cross_quadrant_change_count=2,
            cross_quadrant_risk_report_count=1,
        )

        self.assertEqual(gate["decision"], "hold_no_submit")
        self.assertIn("cross_quadrant_risk_not_fully_reported", gate["reasons"])

    def test_choose_v27_final_gate_holds_top_emotion_collapse(self) -> None:
        gate = choose_v27_final_gate(
            projected_overall=0.891,
            projected_classification=0.832,
            projected_description=0.95,
            label_consistency_issue_count=0,
            missing_emotions=[],
            unsafe_text_count=0,
            top_emotion_share=0.581,
            cross_quadrant_change_count=0,
            cross_quadrant_risk_report_count=0,
        )

        self.assertEqual(gate["decision"], "hold_no_submit")
        self.assertIn("top_emotion_collapse_risk", gate["reasons"])


if __name__ == "__main__":
    unittest.main()
