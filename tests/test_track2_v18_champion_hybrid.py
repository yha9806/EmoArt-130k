from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_v18_champion_hybrid import (
    ChampionTargets,
    OfficialAnchor,
    choose_v18_base,
    load_champion_targets,
    load_official_anchors,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class Track2V18ChampionHybridTests(unittest.TestCase):
    def test_module_has_no_codabench_upload_surface(self) -> None:
        import affectiveart.track2_v18_champion_hybrid as module

        public_names = [name for name in dir(module) if not name.startswith("_")]

        self.assertTrue(module.NO_AUTO_SUBMIT_POLICY)
        self.assertFalse(any("codabench" in name.lower() and "upload" in name.lower() for name in public_names))
        self.assertFalse(any("submit_to" in name.lower() for name in public_names))

    def test_public_api_hides_future_v17_helpers(self) -> None:
        import affectiveart.track2_v18_champion_hybrid as module

        public_names = {name for name in dir(module) if not name.startswith("_")}

        self.assertFalse(
            public_names
            & {
                "apply_v17_changes",
                "build_evidence_rows",
                "load_prediction_sources",
                "load_duplicate_rows",
                "parse_official_failed_transition_counts",
            }
        )

    def test_load_official_anchors_parses_exact_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "official.csv"
            _write_csv(
                path,
                [
                    {
                        "submission_id": "779605",
                        "file_name": "track2_submission_moe_v2_accept5_candidate.zip",
                        "official_overall": "0.836408",
                        "official_classification": "0.72315",
                        "official_description": "0.949667",
                    }
                ],
            )

            anchors = load_official_anchors(path)

            self.assertEqual(list(anchors), ["779605"])
            self.assertEqual(anchors["779605"].submission_id, "779605")
            self.assertAlmostEqual(anchors["779605"].overall, 0.836408)
            self.assertAlmostEqual(anchors["779605"].classification, 0.72315)
            self.assertAlmostEqual(anchors["779605"].description, 0.949667)

    def test_load_official_anchors_rejects_malformed_required_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "official.csv"
            _write_csv(
                path,
                [
                    {
                        "submission_id": "779605",
                        "file_name": "track2_submission_moe_v2_accept5_candidate.zip",
                        "official_overall": "not-a-score",
                        "official_classification": "0.72315",
                        "official_description": "0.949667",
                    }
                ],
            )

            with self.assertRaises(ValueError) as context:
                load_official_anchors(path)

            message = str(context.exception)
            self.assertIn("official_overall", message)
            self.assertIn("779605", message)
            self.assertIn(str(path), message)

    def test_load_champion_targets_uses_first_place_and_current_team(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leaderboard.csv"
            _write_csv(
                path,
                [
                    {
                        "#": "1",
                        "Participant": "N&T小分队",
                        "Overall Score": "0.89",
                        "Classification Score": "0.78",
                        "Description Score": "1.0",
                        "Emotion Accuracy": "0.8",
                        "Emotion Macro F1": "0.31",
                        "Visual Grounding": "0.99",
                        "Attribute Specificity": "1.0",
                        "Overall Caption": "0.99",
                    },
                    {
                        "#": "4",
                        "Participant": "vulcaart",
                        "Overall Score": "0.84",
                        "Classification Score": "0.72",
                        "Description Score": "0.95",
                        "Emotion Accuracy": "0.57",
                        "Emotion Macro F1": "0.31",
                        "Visual Grounding": "0.96",
                        "Attribute Specificity": "0.95",
                        "Overall Caption": "0.94",
                    },
                ],
            )

            targets = load_champion_targets(path, participant="vulcaart")

            self.assertIsInstance(targets, ChampionTargets)
            self.assertAlmostEqual(targets.first_overall, 0.89)
            self.assertAlmostEqual(targets.current_overall, 0.84)
            self.assertAlmostEqual(targets.first_emotion_accuracy, 0.80)
            self.assertAlmostEqual(targets.current_emotion_accuracy, 0.57)

    def test_load_champion_targets_rejects_malformed_required_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leaderboard.csv"
            _write_csv(
                path,
                [
                    {
                        "#": "1",
                        "Participant": "N&T小分队",
                        "Overall Score": "0.89",
                        "Classification Score": "0.78",
                        "Description Score": "1.0",
                        "Emotion Accuracy": "0.8",
                        "Emotion Macro F1": "0.31",
                        "Visual Grounding": "0.99",
                        "Attribute Specificity": "1.0",
                        "Overall Caption": "0.99",
                    },
                    {
                        "#": "4",
                        "Participant": "vulcaart",
                        "Overall Score": "not-a-score",
                        "Classification Score": "0.72",
                        "Description Score": "0.95",
                        "Emotion Accuracy": "0.57",
                        "Emotion Macro F1": "0.31",
                        "Visual Grounding": "0.96",
                        "Attribute Specificity": "0.95",
                        "Overall Caption": "0.94",
                    },
                ],
            )

            with self.assertRaises(ValueError) as context:
                load_champion_targets(path, participant="vulcaart")

            message = str(context.exception)
            self.assertIn("Overall Score", message)
            self.assertIn("vulcaart", message)
            self.assertIn(str(path), message)

    def test_choose_v18_base_prefers_highest_exact_official_anchor(self) -> None:
        anchors = {
            "779605": OfficialAnchor("779605", "a.zip", 0.836408, 0.72315, 0.949667),
            "781601": OfficialAnchor("781601", "b.zip", 0.834027, 0.719137, 0.948917),
        }
        available = {
            "779605": Path("submissions/track2_submission_moe_v2_accept5_candidate.json"),
            "781601": Path("submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json"),
        }

        choice = choose_v18_base(anchors, available)

        self.assertEqual(choice["submission_id"], "779605")
        self.assertEqual(choice["base_json"], str(available["779605"]))
        self.assertEqual(choice["reason"], "highest_exact_official_overall")

    def test_choose_v18_base_ties_by_submission_id_not_component_scores(self) -> None:
        anchors = {
            "100": OfficialAnchor("100", "a.zip", 0.9, 0.8, 0.7),
            "200": OfficialAnchor("200", "b.zip", 0.9, 0.7, 0.6),
        }
        available = {
            "100": Path("submissions/a.json"),
            "200": Path("submissions/b.json"),
        }

        choice = choose_v18_base(anchors, available)

        self.assertEqual(choice["submission_id"], "200")
        self.assertEqual(choice["base_json"], str(available["200"]))
        self.assertEqual(choice["reason"], "highest_exact_official_overall")
