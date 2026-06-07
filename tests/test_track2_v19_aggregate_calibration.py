from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_v19_aggregate_calibration import (
    apply_v19_changes,
    build_v19_calibrated_evidence,
    derive_aggregate_targets,
    load_leaderboard_aggregates,
    score_v19_evidence_row,
    select_v19_changes,
    summarize_781601_regression_guard,
    summarize_v19_candidate,
    write_calibrated_evidence_outputs,
    write_v19_candidate_ladder_outputs,
    write_v19_candidate_outputs,
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

    def test_select_v19_changes_keeps_highest_scored_unique_sample(self) -> None:
        rows = [
            {
                "sample_id": "track2_0001",
                "current_emotion": "tired",
                "proposed_emotion": "sad",
                "transition": "tired->sad",
                "v19_score": 2.5,
                "v19_decision": "accept_candidate",
            },
            {
                "sample_id": "track2_0001",
                "current_emotion": "tired",
                "proposed_emotion": "calm",
                "transition": "tired->calm",
                "v19_score": 2.4,
                "v19_decision": "accept_candidate",
            },
        ]
        selected = select_v19_changes(rows, profile="precision", base_distribution={"tired": 5})
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["transition"], "tired->sad")

    def test_apply_v19_changes_repairs_valence_and_arousal_from_emotion(self) -> None:
        base_rows = [
            {
                "sample_id": "track2_0001",
                "emotion": "tired",
                "emotional_valence": "Negative",
                "emotional_arousal_level": "Low",
                "caption": "A subdued figure in a quiet interior.",
                "brushstroke": "Soft brushwork.",
                "composition": "Centered composition.",
                "color": "Muted colors.",
                "line": "Restrained lines.",
                "light": "Dim light.",
            }
        ]
        rows = apply_v19_changes(
            base_rows,
            [
                {
                    "sample_id": "track2_0001",
                    "current_emotion": "tired",
                    "proposed_emotion": "excited",
                    "transition": "tired->excited",
                    "v19_score": 3.0,
                    "v19_decision": "accept_candidate",
                }
            ],
        )
        self.assertEqual(rows[0]["emotion"], "excited")
        self.assertEqual(rows[0]["emotional_valence"], "Positive")
        self.assertEqual(rows[0]["emotional_arousal_level"], "High")

    def test_write_v19_candidate_outputs_rejects_formal_submission_names(self) -> None:
        base_rows = [
            {
                "sample_id": "track2_0001",
                "emotion": "tired",
                "emotional_valence": "Negative",
                "emotional_arousal_level": "Low",
                "caption": "A subdued figure in a quiet interior.",
                "brushstroke": "Soft brushwork.",
                "composition": "Centered composition.",
                "color": "Muted colors.",
                "line": "Restrained lines.",
                "light": "Dim light.",
            }
        ]
        with self.assertRaises(ValueError):
            write_v19_candidate_outputs(
                base_rows=base_rows,
                selected_changes=[],
                out_json=Path("submissions/track2_submission.json"),
                out_zip=Path("submissions/track2_submission_v19_precision_candidate.zip"),
                report_json=Path("experiments/v19_report.json"),
                report_md=Path("experiments/v19_report.md"),
                profile="precision",
            )

    def test_write_v19_candidate_outputs_writes_side_path_zip_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_rows = [
                {
                    "sample_id": "track2_0001",
                    "emotion": "tired",
                    "emotional_valence": "Negative",
                    "emotional_arousal_level": "Low",
                    "caption": "A subdued figure in a quiet interior.",
                    "brushstroke": "Soft brushwork.",
                    "composition": "Centered composition.",
                    "color": "Muted colors.",
                    "line": "Restrained lines.",
                    "light": "Dim light.",
                }
            ]
            report = write_v19_candidate_outputs(
                base_rows=base_rows,
                selected_changes=[
                    {
                        "sample_id": "track2_0001",
                        "current_emotion": "tired",
                        "proposed_emotion": "sad",
                        "transition": "tired->sad",
                        "v19_score": 2.5,
                        "v19_decision": "accept_candidate",
                    }
                ],
                out_json=root / "track2_submission_v19_precision_candidate.json",
                out_zip=root / "track2_submission_v19_precision_candidate.zip",
                report_json=root / "candidate_report.json",
                report_md=root / "candidate_report.md",
                profile="precision",
            )
            self.assertEqual(report["accepted_label_changes"], 1)
            self.assertTrue((root / "track2_submission_v19_precision_candidate.json").exists())
            import zipfile

            with zipfile.ZipFile(root / "track2_submission_v19_precision_candidate.zip") as zf:
                self.assertEqual(zf.namelist(), ["submission.json"])

    def test_summarize_v19_candidate_reports_transition_distribution(self) -> None:
        report = summarize_v19_candidate(
            base_rows=[{"sample_id": "track2_0001", "emotion": "tired"}],
            selected_changes=[
                {
                    "sample_id": "track2_0001",
                    "current_emotion": "tired",
                    "proposed_emotion": "sad",
                    "transition": "tired->sad",
                    "v19_score": 2.5,
                    "v19_decision": "accept_candidate",
                }
            ],
            profile="precision",
        )
        self.assertEqual(report["candidates"]["precision"]["accepted_label_changes"], 1)
        self.assertEqual(report["candidates"]["precision"]["transition_counts"], {"tired->sad": 1})

    def test_write_v19_candidate_ladder_outputs_writes_three_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            evidence_json = root / "calibrated_evidence.json"
            out_dir = root / "experiments"
            submissions_dir = root / "submissions"
            base_json.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "track2_0001",
                            "emotion": "tired",
                            "emotional_valence": "Negative",
                            "emotional_arousal_level": "Low",
                            "caption": "A subdued figure in a quiet interior.",
                            "brushstroke": "Soft brushwork.",
                            "composition": "Centered composition.",
                            "color": "Muted colors.",
                            "line": "Restrained lines.",
                            "light": "Dim light.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            evidence_json.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "track2_0001",
                            "current_emotion": "tired",
                            "proposed_emotion": "sad",
                            "transition": "tired->sad",
                            "v19_score": 2.5,
                            "v19_decision": "accept_candidate",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            report = write_v19_candidate_ladder_outputs(
                base_json=base_json,
                calibrated_evidence_json=evidence_json,
                out_dir=out_dir,
                submissions_dir=submissions_dir,
            )

            self.assertEqual(set(report["candidates"]), {"precision", "balanced", "probe"})
            self.assertTrue((submissions_dir / "track2_submission_v19_precision_candidate.zip").exists())
            self.assertTrue((out_dir / "candidate_reports" / "track2_submission_v19_precision_candidate_report.json").exists())


if __name__ == "__main__":
    unittest.main()
