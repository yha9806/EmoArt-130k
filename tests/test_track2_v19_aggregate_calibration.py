from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_v19_aggregate_calibration import (
    derive_aggregate_targets,
    load_leaderboard_aggregates,
)


class Track2V19AggregateCalibrationTests(unittest.TestCase):
    def test_load_leaderboard_aggregates_extracts_our_gap_and_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leaderboard.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Participant",
                        "Overall Score",
                        "Classification Score",
                        "Description Score",
                        "Emotion Accuracy",
                        "Emotion Macro F1",
                        "Valence Accuracy",
                        "Valence Macro F1",
                        "Arousal Accuracy",
                        "Arousal Macro F1",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Participant": "N&T",
                        "Overall Score": "0.89",
                        "Classification Score": "0.78",
                        "Description Score": "1.0",
                        "Emotion Accuracy": "0.80",
                        "Emotion Macro F1": "0.31",
                        "Valence Accuracy": "0.90",
                        "Valence Macro F1": "0.86",
                        "Arousal Accuracy": "0.94",
                        "Arousal Macro F1": "0.87",
                    }
                )
                writer.writerow(
                    {
                        "Participant": "vulcaart",
                        "Overall Score": "0.84",
                        "Classification Score": "0.72",
                        "Description Score": "0.95",
                        "Emotion Accuracy": "0.57",
                        "Emotion Macro F1": "0.31",
                        "Valence Accuracy": "0.88",
                        "Valence Macro F1": "0.82",
                        "Arousal Accuracy": "0.91",
                        "Arousal Macro F1": "0.84",
                    }
                )

            rows = load_leaderboard_aggregates(path)
            target = derive_aggregate_targets(rows, participant="vulcaart")

        self.assertEqual(target["our"]["emotion_accuracy"], 0.57)
        self.assertEqual(target["frontier"]["emotion_accuracy"], 0.80)
        self.assertGreater(target["gaps"]["emotion_accuracy"], 0.20)
        self.assertLess(abs(target["gaps"]["emotion_macro_f1"]), 0.02)

    def test_derive_aggregate_targets_rejects_missing_participant(self) -> None:
        rows = [
            {
                "participant": "other",
                "overall": 0.1,
                "classification": 0.1,
                "description": 0.1,
                "emotion_accuracy": 0.1,
                "emotion_macro_f1": 0.1,
                "valence_accuracy": 0.1,
                "valence_macro_f1": 0.1,
                "arousal_accuracy": 0.1,
                "arousal_macro_f1": 0.1,
            }
        ]
        with self.assertRaises(ValueError):
            derive_aggregate_targets(rows, participant="vulcaart")


if __name__ == "__main__":
    unittest.main()
