from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_v19_aggregate_calibration import (
    build_v19_calibrated_evidence,
    derive_aggregate_targets,
    load_leaderboard_aggregates,
    score_v19_evidence_row,
    summarize_781601_regression_guard,
    write_calibrated_evidence_outputs,
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

    def test_score_evidence_penalizes_known_failed_transition(self) -> None:
        row = {
            "sample_id": "track2_0001",
            "current_emotion": "calm",
            "proposed_emotion": "content",
            "transition": "calm->content",
            "support_score": 3.0,
            "model_vote_count": 3,
            "failed_transition_count": 43,
        }
        scored = score_v19_evidence_row(row, failed_transition_penalty_weight=0.04)
        self.assertEqual(scored["decision"], "hold")
        self.assertIn("official_failed_transition_penalty", scored["reasons"])

    def test_score_evidence_requires_row_level_support_even_when_aggregate_pressure_exists(self) -> None:
        row = {
            "sample_id": "track2_0002",
            "current_emotion": "calm",
            "proposed_emotion": "happy",
            "transition": "calm->happy",
            "support_score": 0.0,
            "model_vote_count": 0,
            "failed_transition_count": 0,
        }
        scored = score_v19_evidence_row(row, aggregate_pressure=1.0)
        self.assertEqual(scored["decision"], "hold")
        self.assertIn("insufficient_row_support", scored["reasons"])

    def test_781601_failed_same_quadrant_batch_is_negative_regression_case(self) -> None:
        rows = [
            {
                "transition": "calm->content",
                "support_score": 3.0,
                "model_vote_count": 3,
                "failed_transition_count": 43,
            },
            {
                "transition": "content->calm",
                "support_score": 3.0,
                "model_vote_count": 3,
                "failed_transition_count": 20,
            },
        ]
        report = summarize_781601_regression_guard(rows)
        self.assertEqual(report["decision"], "block_bulk_same_quadrant_repeat")
        self.assertGreaterEqual(report["failed_same_quadrant_count"], 63)

    def test_781601_guard_deduplicates_failed_transition_counts(self) -> None:
        rows = [
            {"transition": "calm->content", "failed_transition_count": 43},
            {"transition": "calm->content", "failed_transition_count": 43},
            {"transition": "content->calm", "failed_transition_count": 20},
        ]
        report = summarize_781601_regression_guard(rows)
        self.assertEqual(report["failed_same_quadrant_count"], 63)

    def test_build_v19_calibrated_evidence_preserves_fields_and_sorts_accepts_first(self) -> None:
        rows = [
            {
                "sample_id": "track2_0002",
                "current_emotion": "calm",
                "proposed_emotion": "happy",
                "transition": "calm->happy",
                "support_score": 0.0,
                "model_vote_count": 0,
                "failed_transition_count": 0,
            },
            {
                "sample_id": "track2_0001",
                "current_emotion": "tired",
                "proposed_emotion": "sad",
                "transition": "tired->sad",
                "support_score": 2.4,
                "model_vote_count": 2,
                "failed_transition_count": 0,
            },
        ]
        calibrated = build_v19_calibrated_evidence(
            rows,
            {"gaps": {"emotion_accuracy": 0.23}, "target_bands": {"emotion_accuracy_min_delta": 0.05}},
        )
        self.assertEqual(calibrated[0]["sample_id"], "track2_0001")
        self.assertEqual(calibrated[0]["v19_decision"], "accept_candidate")
        self.assertIn("v19_score", calibrated[0])
        self.assertEqual(calibrated[1]["v19_decision"], "hold")

    def test_write_calibrated_evidence_outputs_writes_json_csv_and_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v17_path = root / "v17_evidence.json"
            targets_path = root / "aggregate_targets.json"
            out_dir = root / "out"
            v17_path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "track2_0001",
                            "current_emotion": "calm",
                            "proposed_emotion": "content",
                            "transition": "calm->content",
                            "support_score": 3.0,
                            "model_vote_count": 3,
                            "failed_transition_count": 43,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            targets_path.write_text(
                json.dumps({"gaps": {"emotion_accuracy": 0.23}, "target_bands": {}}),
                encoding="utf-8",
            )

            report = write_calibrated_evidence_outputs(
                v17_evidence=v17_path,
                aggregate_targets_json=targets_path,
                out_dir=out_dir,
            )
            self.assertEqual(report["rows"], 1)
            self.assertTrue((out_dir / "calibrated_evidence.json").exists())
            self.assertTrue((out_dir / "calibrated_evidence.csv").exists())
            self.assertTrue((out_dir / "781601_regression_guard.json").exists())
            self.assertNotIn(b"\r", (out_dir / "calibrated_evidence.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
