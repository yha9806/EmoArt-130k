import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_v9_transition_ablation import (
    build_v9_transition_ablation_outputs,
    extract_v9_changes,
    select_ablation_changes,
)


def _row(sample_id: str, emotion: str, valence: str, arousal: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": "A specific artwork caption.",
        "brushstroke": "Visible brushwork supports the image structure.",
        "composition": "Balanced composition with clear subject placement.",
        "color": "Specific colors shape the atmosphere.",
        "line": "Line quality clarifies the forms.",
        "light": "Light and shadow support the focal point.",
    }


class Track2V9TransitionAblationTest(unittest.TestCase):
    def test_extracts_changes_with_evidence_and_human_gate(self):
        baseline = [
            _row("track2_0001", "content", "Positive", "Low"),
            _row("track2_0002", "calm", "Positive", "Low"),
        ]
        v9 = [
            _row("track2_0001", "calm", "Positive", "Low"),
            _row("track2_0002", "content", "Positive", "Low"),
        ]
        evidence = [
            {
                "sample_id": "track2_0001",
                "proposed_emotion": "calm",
                "transition": "content->calm",
                "supporting_source_count": 4,
                "supporting_family_count": 3,
                "public_style_agreement": True,
                "evidence_score": 9,
            }
        ]
        human = [
            {
                "sample_id": "track2_0002",
                "proposed_emotion": "content",
                "human_decision": "accept_proposed",
                "reviewer_confidence_1_5": "5",
                "gate_recommendation": "allow_high_conf",
            }
        ]

        changes = extract_v9_changes(baseline, v9, evidence_rows=evidence, human_gate_rows=human)

        self.assertEqual([item["transition"] for item in changes], ["content->calm", "calm->content"])
        self.assertTrue(changes[0]["public_style_agreement"])
        self.assertTrue(changes[1]["human_high_confidence_accept"])

    def test_selectors_keep_transition_families_separate(self):
        changes = [
            {"sample_id": "a", "transition": "content->calm", "public_style_agreement": False, "human_high_confidence_accept": False, "supporting_source_count": 2, "supporting_family_count": 2},
            {"sample_id": "b", "transition": "calm->content", "public_style_agreement": True, "human_high_confidence_accept": False, "supporting_source_count": 10, "supporting_family_count": 5},
            {"sample_id": "c", "transition": "happy->excited", "public_style_agreement": False, "human_high_confidence_accept": True, "supporting_source_count": 8, "supporting_family_count": 5},
        ]

        self.assertEqual([row["sample_id"] for row in select_ablation_changes(changes, "content_to_calm_only")], ["a"])
        self.assertEqual([row["sample_id"] for row in select_ablation_changes(changes, "no_calm_to_content")], ["a", "c"])
        self.assertEqual([row["sample_id"] for row in select_ablation_changes(changes, "public_duplicate_only")], ["b"])
        self.assertEqual([row["sample_id"] for row in select_ablation_changes(changes, "human_high_confidence_only")], ["c"])
        self.assertEqual([row["sample_id"] for row in select_ablation_changes(changes, "multisource_consensus_only")], ["b", "c"])

    def test_writes_side_path_candidates_and_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_json = root / "baseline.json"
            v9_json = root / "v9.json"
            evidence_json = root / "evidence.json"
            human_csv = root / "human.csv"
            out_dir = root / "out"
            submissions = root / "submissions"
            baseline = [
                _row("track2_0001", "content", "Positive", "Low"),
                _row("track2_0002", "calm", "Positive", "Low"),
            ]
            v9 = [
                _row("track2_0001", "calm", "Positive", "Low"),
                _row("track2_0002", "content", "Positive", "Low"),
            ]
            baseline_json.write_text(json.dumps(baseline), encoding="utf-8")
            v9_json.write_text(json.dumps(v9), encoding="utf-8")
            evidence_json.write_text(json.dumps([]), encoding="utf-8")
            human_csv.write_text("sample_id,proposed_emotion,human_decision,reviewer_confidence_1_5,gate_recommendation\n", encoding="utf-8")

            report = build_v9_transition_ablation_outputs(
                baseline_json=baseline_json,
                v9_json=v9_json,
                evidence_json=evidence_json,
                human_gate_csv=human_csv,
                out_dir=out_dir,
                submission_dir=submissions,
            )

            content_to_calm = report["candidates"]["content_to_calm_only"]
            self.assertEqual(content_to_calm["changed_rows"], 1)
            self.assertTrue(Path(content_to_calm["json"]).name.startswith("track2_submission_v12_"))
            with zipfile.ZipFile(content_to_calm["zip"]) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
