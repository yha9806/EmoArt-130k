from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from affectiveart.track2_v26_public_knn_probe import (
    build_public_knn_candidate_rows,
    build_public_knn_probe_outputs,
    summarize_public_knn_threshold,
)


class Track2V26PublicKnnProbeTests(unittest.TestCase):
    def test_build_public_knn_candidate_rows_copies_thresholded_public_labels(self) -> None:
        base_rows = [
            {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
            },
            {
                "sample_id": "track2_0002",
                "emotion": "tired",
                "emotional_valence": "Negative",
                "emotional_arousal_level": "Low",
            },
        ]
        nearest_rows = [
            {"sample_id": "track2_0001", "nearest_emotion": "calm", "clip_cosine": 0.96},
            {"sample_id": "track2_0002", "nearest_emotion": "sad", "clip_cosine": 0.94},
        ]

        candidate, changes = build_public_knn_candidate_rows(base_rows, nearest_rows, threshold=0.95)

        self.assertEqual(candidate[0]["emotion"], "calm")
        self.assertEqual(candidate[0]["emotional_valence"], "Positive")
        self.assertEqual(candidate[0]["emotional_arousal_level"], "Low")
        self.assertEqual(candidate[1]["emotion"], "tired")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["transition"], "content->calm")

    def test_summarize_public_knn_threshold_reports_distribution_and_transitions(self) -> None:
        base_rows = [
            {"sample_id": "track2_0001", "emotion": "content"},
            {"sample_id": "track2_0002", "emotion": "calm"},
            {"sample_id": "track2_0003", "emotion": "tired"},
        ]
        nearest_rows = [
            {"sample_id": "track2_0001", "nearest_emotion": "calm", "clip_cosine": 0.96},
            {"sample_id": "track2_0002", "nearest_emotion": "content", "clip_cosine": 0.96},
            {"sample_id": "track2_0003", "nearest_emotion": "sad", "clip_cosine": 0.94},
        ]

        summary = summarize_public_knn_threshold(base_rows, nearest_rows, threshold=0.95)

        self.assertEqual(summary["threshold"], 0.95)
        self.assertEqual(summary["neighbor_rows"], 2)
        self.assertEqual(summary["accepted_label_changes"], 2)
        self.assertEqual(summary["transition_counts"], {"calm->content": 1, "content->calm": 1})
        self.assertAlmostEqual(summary["top_emotion_share"], 1 / 3, places=6)

    def test_probe_outputs_reject_submission_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_json = root / "base.json"
            nearest_json = root / "nearest.json"
            base_json.write_text(
                '[{"sample_id":"track2_0001","emotion":"content","emotional_valence":"Positive","emotional_arousal_level":"Low"}]',
                encoding="utf-8",
            )
            nearest_json.write_text(
                '{"entries":[{"sample_id":"track2_0001","nearest_emotion":"calm","clip_cosine":0.96}]}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "diagnostic output"):
                build_public_knn_probe_outputs(
                    out_dir=root / "submissions",
                    base_json=base_json,
                    nearest_json=nearest_json,
                    thresholds=(0.95,),
                )


if __name__ == "__main__":
    unittest.main()
