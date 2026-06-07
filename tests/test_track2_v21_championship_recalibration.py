from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from affectiveart.track2_v21_championship_recalibration import (
    build_v21_championship_evidence,
    choose_v21_final_gate,
    score_v21_evidence_row,
    select_v21_changes,
    write_v21_candidate_outputs,
)


def _row(
    current: str = "content",
    proposed: str = "calm",
    *,
    sample_id: str = "track2_0001",
    support: float = 2.4,
    votes: int = 3,
    confidence: float = 0.82,
    near: bool = False,
    exact: bool = False,
    dup: float = 0.0,
    public_style: float = 1.1,
    sources: str = "clip,dinov2,public_clean,public_inclusive,siglip2",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "current_emotion": current,
        "proposed_emotion": proposed,
        "transition": f"{current}->{proposed}",
        "support_score": support,
        "model_vote_count": votes,
        "max_confidence": confidence,
        "near_duplicate": near,
        "exact_duplicate": exact,
        "public_duplicate_support_score": dup,
        "public_style_support_score": public_style,
        "all_sources": sources,
    }


def _base_row(sample_id: str, emotion: str = "content") -> dict[str, str]:
    valence = "Negative" if emotion in {"alarmed", "annoyed", "bored", "frustrated", "sad", "tired"} else "Positive"
    arousal = "High" if emotion in {"alarmed", "annoyed", "aroused", "excited", "frustrated", "happy"} else "Low"
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": "A stable artwork description with specific emotional atmosphere.",
        "brushstroke": "Layered brushwork supports the mood.",
        "composition": "The composition is clear and structured.",
        "color": "The color palette is specific and balanced.",
        "line": "Line work reinforces the subject.",
        "light": "Lighting supports the emotional tone.",
    }


class Track2V21ChampionshipRecalibrationTests(unittest.TestCase):
    def test_score_assigns_reference_tier_for_near_duplicate(self) -> None:
        scored = score_v21_evidence_row(
            _row(near=True, dup=0.96, confidence=0.96, support=2.0, votes=2)
        )

        self.assertEqual(scored["decision"], "accept_candidate")
        self.assertEqual(scored["tier"], "reference")

    def test_score_assigns_consensus_tier_for_three_models_and_public_style(self) -> None:
        scored = score_v21_evidence_row(_row(support=2.2, votes=3, public_style=1.2))

        self.assertEqual(scored["decision"], "accept_candidate")
        self.assertEqual(scored["tier"], "consensus")

    def test_score_assigns_expansion_tier_for_two_models_and_confidence(self) -> None:
        scored = score_v21_evidence_row(
            _row(support=1.7, votes=2, confidence=0.84, public_style=0.0, sources="clip,siglip2")
        )

        self.assertEqual(scored["decision"], "accept_candidate")
        self.assertEqual(scored["tier"], "expansion")

    def test_score_holds_weak_one_source_row(self) -> None:
        scored = score_v21_evidence_row(
            _row(support=0.9, votes=1, confidence=0.61, public_style=0.0, sources="clip")
        )

        self.assertEqual(scored["decision"], "hold")
        self.assertIn("insufficient_v21_evidence", scored["reasons"])

    def test_build_v21_championship_evidence_sorts_reference_before_expansion(self) -> None:
        rows = [
            _row(sample_id="track2_0002", support=1.7, votes=2, confidence=0.84, public_style=0.0),
            _row(sample_id="track2_0001", near=True, dup=0.96, confidence=0.96, support=2.0, votes=2),
        ]

        evidence = build_v21_championship_evidence(rows)

        self.assertEqual(evidence[0]["sample_id"], "track2_0001")
        self.assertEqual(evidence[0]["v21_tier"], "reference")

    def test_select_v21_changes_caps_precision80(self) -> None:
        rows = []
        for index in range(120):
            current = "content" if index < 60 else "tired"
            proposed = "calm" if index < 60 else "sad"
            rows.append(
                {
                    **_row(
                        current=current,
                        proposed=proposed,
                        sample_id=f"track2_{index:04d}",
                        support=3.0 - index * 0.001,
                    ),
                    "v21_decision": "accept_candidate",
                    "v21_score": 3.0 - index * 0.001,
                    "v21_tier": "consensus",
                }
            )

        selected = select_v21_changes(
            rows,
            profile="precision80",
            base_distribution=Counter({"content": 200, "tired": 200}),
        )

        self.assertEqual(len(selected), 80)

    def test_select_v21_changes_blocks_top_emotion_collapse(self) -> None:
        rows = [
            {
                **_row(sample_id=f"track2_{index:04d}", support=3.0 - index * 0.001),
                "v21_decision": "accept_candidate",
                "v21_score": 3.0 - index * 0.001,
                "v21_tier": "consensus",
            }
            for index in range(120)
        ]

        selected = select_v21_changes(
            rows,
            profile="lastshot160",
            base_distribution=Counter({"calm": 575, "content": 200, "sad": 225}),
        )

        self.assertLessEqual(len(selected), 5)

    def test_select_v21_changes_enforces_transition_cap(self) -> None:
        rows = [
            {
                **_row(sample_id=f"track2_{index:04d}", support=3.0 - index * 0.001),
                "v21_decision": "accept_candidate",
                "v21_score": 3.0 - index * 0.001,
                "v21_tier": "consensus",
            }
            for index in range(90)
        ]

        selected = select_v21_changes(rows, profile="precision80", base_distribution=Counter({"content": 200}))

        self.assertEqual(len(selected), 40)

    def test_select_v21_changes_calmshift90_only_uses_content_to_calm(self) -> None:
        rows = []
        for index in range(100):
            current = "content" if index < 90 else "tired"
            proposed = "calm" if index < 90 else "sad"
            rows.append(
                {
                    **_row(current=current, proposed=proposed, sample_id=f"track2_{index:04d}"),
                    "v21_decision": "accept_candidate",
                    "v21_score": 3.0 - index * 0.001,
                    "v21_tier": "consensus",
                }
            )

        selected = select_v21_changes(
            rows,
            profile="calmshift90",
            base_distribution=Counter({"content": 200, "tired": 200}),
        )

        self.assertEqual(len(selected), 90)
        self.assertEqual({row["transition"] for row in selected}, {"content->calm"})

    def test_select_v21_changes_calmshift90_ignores_inconsistent_transition_text(self) -> None:
        row = {
            **_row(current="tired", proposed="sad", sample_id="track2_0001"),
            "transition": "content->calm",
            "v21_decision": "accept_candidate",
            "v21_score": 3.0,
            "v21_tier": "consensus",
        }

        selected = select_v21_changes(
            [row],
            profile="calmshift90",
            base_distribution=Counter({"tired": 20, "sad": 20, "content": 200}),
        )

        self.assertEqual(selected, [])

    def test_select_v21_changes_keeps_current_class_floor(self) -> None:
        rows = [
            {
                **_row(sample_id=f"track2_{index:04d}", current="glad", proposed="calm"),
                "v21_decision": "accept_candidate",
                "v21_score": 3.0 - index * 0.001,
                "v21_tier": "consensus",
            }
            for index in range(10)
        ]

        selected = select_v21_changes(
            rows,
            profile="lastshot160",
            base_distribution=Counter({"glad": 5, "calm": 40, "content": 60}),
        )

        self.assertEqual(len(selected), 1)

    def test_write_v21_candidate_rejects_formal_submission_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                write_v21_candidate_outputs(
                    base_rows=[_base_row("track2_0001")],
                    selected_changes=[],
                    out_json=root / "track2_submission.json",
                    out_zip=root / "track2_submission_v21_precision80_candidate.zip",
                    report_json=root / "report.json",
                    report_md=root / "report.md",
                    profile="precision80",
                )

    def test_write_v21_candidate_writes_zip_and_repairs_va(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_json = root / "track2_submission_v21_precision80_candidate.json"
            out_zip = root / "track2_submission_v21_precision80_candidate.zip"
            report = write_v21_candidate_outputs(
                base_rows=[_base_row("track2_0001", "content")],
                selected_changes=[
                    {
                        "sample_id": "track2_0001",
                        "current_emotion": "content",
                        "proposed_emotion": "alarmed",
                        "transition": "content->alarmed",
                        "v21_score": 4.0,
                        "v21_tier": "reference",
                    }
                ],
                out_json=out_json,
                out_zip=out_zip,
                report_json=root / "report.json",
                report_md=root / "report.md",
                profile="precision80",
            )

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["emotion"], "alarmed")
            self.assertEqual(payload[0]["emotional_valence"], "Negative")
            self.assertEqual(payload[0]["emotional_arousal_level"], "High")
            self.assertEqual(report["accepted_label_changes"], 1)
            with zipfile.ZipFile(out_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])

    def test_choose_v21_final_gate_recommends_champion120_when_scale_is_met(self) -> None:
        report = choose_v21_final_gate(
            candidate_name="v21_champion120",
            profile="champion120",
            accepted_label_changes=110,
            label_consistency_issue_count=0,
            missing_emotions=[],
            top_emotion_share=0.55,
            description_anchor_preserved=True,
        )

        self.assertEqual(report["decision"], "recommend_high_variance_second_last_submission")

    def test_choose_v21_final_gate_recommends_calmshift90_directional_probe(self) -> None:
        report = choose_v21_final_gate(
            candidate_name="v21_calmshift90",
            profile="calmshift90",
            accepted_label_changes=90,
            label_consistency_issue_count=0,
            missing_emotions=[],
            top_emotion_share=0.58,
            description_anchor_preserved=True,
        )

        self.assertEqual(report["decision"], "recommend_directional_second_last_submission")

    def test_choose_v21_final_gate_holds_tiny_candidate(self) -> None:
        report = choose_v21_final_gate(
            candidate_name="v21_precision80",
            profile="precision80",
            accepted_label_changes=40,
            label_consistency_issue_count=0,
            missing_emotions=[],
            top_emotion_share=0.55,
            description_anchor_preserved=True,
        )

        self.assertEqual(report["decision"], "hold")
        self.assertIn("accepted_changes_below_80", report["reasons"])


if __name__ == "__main__":
    unittest.main()
