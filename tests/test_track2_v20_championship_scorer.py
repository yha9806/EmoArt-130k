from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from affectiveart.track2_v20_championship_scorer import (
    build_directional_risk_map,
    build_frontier_requirement_report,
    build_v20_championship_evidence,
    choose_v20_final_gate,
    load_csv_rows,
    required_classification_for_overall,
    score_v20_evidence_row,
    select_v20_changes,
    write_v20_candidate_outputs,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _leaderboard_rows() -> list[dict[str, object]]:
    return [
        {
            "Participant": "N&T",
            "Overall Score": "0.89",
            "Classification Score": "0.78",
            "Description Score": "1.0",
            "Emotion Accuracy": "0.80",
            "Emotion Macro F1": "0.31",
        },
        {
            "Participant": "vulcaart",
            "Overall Score": "0.84",
            "Classification Score": "0.72",
            "Description Score": "0.95",
            "Emotion Accuracy": "0.57",
            "Emotion Macro F1": "0.31",
        },
    ]


def _evidence_row(
    current: str = "content",
    proposed: str = "calm",
    *,
    sample_id: str = "track2_0001",
    support: float = 4.0,
    votes: int = 3,
    near: bool = True,
    exact: bool = False,
    duplicate: float = 0.96,
    confidence: float = 0.96,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "current_emotion": current,
        "proposed_emotion": proposed,
        "transition": f"{current}->{proposed}",
        "same_valence": current in {"calm", "content", "glad"} and proposed in {"calm", "content", "glad"},
        "same_arousal": current in {"calm", "content", "glad"} and proposed in {"calm", "content", "glad"},
        "support_score": support,
        "model_vote_count": votes,
        "public_duplicate_support_score": duplicate,
        "max_confidence": confidence,
        "near_duplicate": near,
        "exact_duplicate": exact,
    }


def _base_row(sample_id: str = "track2_0001", emotion: str = "content") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": "Positive",
        "emotional_arousal_level": "Low",
        "overall_caption": "A quiet scene with a restrained emotional atmosphere.",
        "brushstroke": "Soft brushwork supports the calm surface.",
        "composition": "The composition is stable and centered.",
        "color": "Muted colors create a gentle mood.",
        "line": "The line work is controlled and delicate.",
        "light": "Even light keeps the scene subdued.",
    }


class Track2V20ChampionshipScorerTests(unittest.TestCase):
    def test_required_classification_for_target_overall_uses_official_weighting(self) -> None:
        self.assertAlmostEqual(required_classification_for_overall(0.89, 1.0), 0.78)
        self.assertAlmostEqual(required_classification_for_overall(0.89, 0.98), 0.80)

    def test_frontier_report_identifies_classification_and_description_gap(self) -> None:
        report = build_frontier_requirement_report(
            _leaderboard_rows(),
            participant="vulcaart",
            target_overall=0.89,
        )

        self.assertGreaterEqual(report["required_classification_if_description_1.00"], 0.78)
        self.assertGreater(report["emotion_accuracy_gap"], 0.20)
        self.assertEqual(report["macro_f1_gap"], 0.0)
        self.assertEqual(report["frontier_participant"], "N&T")

    def test_load_csv_rows_keeps_utf8_and_lf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leaderboard.csv"
            _write_csv(path, _leaderboard_rows())

            rows = load_csv_rows(path)

        self.assertEqual(rows[0]["Participant"], "N&T")
        self.assertEqual(rows[1]["Participant"], "vulcaart")

    def test_directional_risk_marks_dominant_failed_direction_dangerous(self) -> None:
        risk = build_directional_risk_map(
            [{"top_emotion_transitions": "calm->content:43; content->calm:20"}]
        )

        self.assertEqual(risk["calm->content"]["risk"], "danger")
        self.assertEqual(risk["content->calm"]["risk"], "caution")
        self.assertEqual(risk["calm->content"]["reverse_transition"], "content->calm")

    def test_directional_risk_ignores_reverse_repair_comparison(self) -> None:
        risk = build_directional_risk_map(
            [
                {
                    "candidate_a": "781601_v3_mid_gemini35_desc_192",
                    "candidate_b": "779605_moe_v2_accept5_anchor",
                    "direction": "candidate_b -> candidate_a",
                    "top_emotion_transitions": "calm->content:43; content->calm:20",
                },
                {
                    "candidate_a": "782683_v12_stable_probe",
                    "candidate_b": "781601_v3_mid_gemini35_desc_192",
                    "direction": "candidate_b -> candidate_a",
                    "top_emotion_transitions": "content->calm:43; calm->content:20",
                },
            ]
        )

        self.assertEqual(risk["calm->content"]["risk"], "danger")
        self.assertEqual(risk["content->calm"]["count"], 20)

    def test_directional_risk_blocks_weak_calm_to_content_but_allows_strong_content_to_calm(self) -> None:
        risk = build_directional_risk_map(
            [{"top_emotion_transitions": "calm->content:43; content->calm:20"}]
        )

        weak = score_v20_evidence_row(
            _evidence_row("calm", "content", support=3.0, votes=3, near=False, duplicate=0.0),
            directional_risk=risk,
        )
        strong = score_v20_evidence_row(
            _evidence_row("content", "calm", support=4.0, votes=3, near=True, duplicate=0.96, confidence=0.96),
            directional_risk=risk,
        )

        self.assertEqual(weak["decision"], "block")
        self.assertIn("danger_direction_requires_exact_duplicate", weak["reasons"])
        self.assertEqual(strong["decision"], "accept_candidate")
        self.assertEqual(strong["role"], "championship_caution_accept")

    def test_caution_direction_allows_strong_model_and_public_style_consensus_without_duplicate(self) -> None:
        risk = build_directional_risk_map(
            [{"top_emotion_transitions": "calm->content:43; content->calm:20"}]
        )

        scored = score_v20_evidence_row(
            {
                **_evidence_row("content", "calm", support=3.1, votes=3, near=False, duplicate=0.0, confidence=0.78),
                "all_sources": "clip,dinov2,public_clean,public_inclusive,siglip2",
                "public_style_support_score": 1.4,
            },
            directional_risk=risk,
        )

        self.assertEqual(scored["decision"], "accept_candidate")
        self.assertEqual(scored["role"], "championship_consensus_accept")

    def test_cross_quadrant_requires_exact_duplicate_override(self) -> None:
        risk = build_directional_risk_map([])

        scored = score_v20_evidence_row(
            _evidence_row("content", "alarmed", support=5.0, votes=4, near=True, duplicate=0.97, confidence=0.99),
            directional_risk=risk,
        )

        self.assertEqual(scored["decision"], "block")
        self.assertIn("cross_quadrant_requires_exact_duplicate", scored["reasons"])

    def test_build_championship_evidence_sorts_accepts_first(self) -> None:
        risk = build_directional_risk_map(
            [{"top_emotion_transitions": "calm->content:43; content->calm:20"}]
        )
        rows = [
            _evidence_row("calm", "content", sample_id="track2_0002", support=3.0, near=False, duplicate=0.0),
            _evidence_row("content", "calm", sample_id="track2_0001", support=4.0, near=True, duplicate=0.96),
        ]

        evidence = build_v20_championship_evidence(rows, risk)

        self.assertEqual(evidence[0]["sample_id"], "track2_0001")
        self.assertEqual(evidence[0]["v20_decision"], "accept_candidate")
        self.assertIn("v20_score", evidence[0])

    def test_select_v20_changes_caps_profiles_and_keeps_unique_samples(self) -> None:
        rows = [
            {
                **_evidence_row("content", "calm", sample_id=f"track2_{index:04d}", support=4.0 - index * 0.01),
                "v20_decision": "accept_candidate",
                "v20_score": 4.0 - index * 0.01,
                "v20_role": "championship_caution_accept",
            }
            for index in range(30)
        ]

        selected = select_v20_changes(rows, profile="precision", base_distribution=Counter({"content": 40}))

        self.assertEqual(len(selected), 18)
        self.assertEqual(selected[0]["sample_id"], "track2_0000")

    def test_write_v20_candidate_rejects_formal_submission_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                write_v20_candidate_outputs(
                    base_rows=[_base_row()],
                    selected_changes=[],
                    out_json=root / "submissions" / "track2_submission.json",
                    out_zip=root / "submissions" / "track2_submission_v20_precision_candidate.zip",
                    report_json=root / "experiments" / "candidate.json",
                    report_md=root / "experiments" / "candidate.md",
                    profile="precision",
                )

    def test_write_v20_candidate_writes_deterministic_zip_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_json = root / "submissions" / "track2_submission_v20_precision_candidate.json"
            out_zip = root / "submissions" / "track2_submission_v20_precision_candidate.zip"
            report = write_v20_candidate_outputs(
                base_rows=[_base_row()],
                selected_changes=[
                    {
                        "sample_id": "track2_0001",
                        "current_emotion": "content",
                        "proposed_emotion": "calm",
                        "transition": "content->calm",
                        "v20_score": 4.0,
                        "v20_role": "championship_caution_accept",
                    }
                ],
                out_json=out_json,
                out_zip=out_zip,
                report_json=root / "experiments" / "candidate.json",
                report_md=root / "experiments" / "candidate.md",
                profile="precision",
            )

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["emotion"], "calm")
            self.assertEqual(report["accepted_label_changes"], 1)
            with zipfile.ZipFile(out_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])

    def test_choose_v20_final_gate_requires_championship_path(self) -> None:
        report = choose_v20_final_gate(
            candidate_name="v20_precision",
            accepted_label_changes=2,
            description_anchor_preserved=True,
            label_consistency_issue_count=0,
            missing_emotions=[],
            frontier_classification_gap=0.06,
            profile="precision",
        )

        self.assertEqual(report["decision"], "hold")
        self.assertIn("insufficient_championship_classification_lift", report["reasons"])

    def test_choose_v20_final_gate_marks_probe_high_variance_not_safe_submit(self) -> None:
        report = choose_v20_final_gate(
            candidate_name="v20_champion_probe",
            accepted_label_changes=40,
            description_anchor_preserved=True,
            label_consistency_issue_count=0,
            missing_emotions=[],
            frontier_classification_gap=0.06,
            profile="champion_probe",
        )

        self.assertEqual(report["decision"], "recommend_high_variance_probe")
        self.assertTrue(report["no_auto_submit"])


if __name__ == "__main__":
    unittest.main()
