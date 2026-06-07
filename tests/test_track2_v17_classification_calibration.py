from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_v17_classification_calibration import (
    build_evidence_rows,
    DEFAULT_PREDICTION_SOURCES,
    load_prediction_sources,
    parse_official_failed_transition_counts,
)


class Track2V17ClassificationCalibrationTests(unittest.TestCase):
    def test_load_prediction_sources_accepts_list_and_entries_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            list_path = root / "list.json"
            entries_path = root / "entries.json"
            list_path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "track2_0001",
                            "emotion": "calm",
                            "confidence": 0.8,
                            "margin": 0.2,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            entries_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "sample_id": "track2_0001",
                                "target_emotion": "content",
                                "confidence": 0.7,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = load_prediction_sources({"clip": list_path, "siglip2": entries_path})
        self.assertEqual(len(rows["track2_0001"]), 2)
        self.assertEqual({row["source"] for row in rows["track2_0001"]}, {"clip", "siglip2"})

    def test_default_prediction_sources_exclude_zero_overlap_hard96(self) -> None:
        self.assertNotIn("hard96_selective", DEFAULT_PREDICTION_SOURCES)

    def test_load_prediction_sources_keeps_zero_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "track2_0001",
                            "emotion": "calm",
                            "confidence": 0,
                            "probability": 0.9,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            rows = load_prediction_sources({"clip": path})
        self.assertEqual(rows["track2_0001"][0]["confidence"], 0.0)

    def test_parse_official_failed_transition_counts_uses_negative_batches_only(self) -> None:
        rows = [
            {
                "candidate_a": "781601_failed",
                "candidate_b": "779605_anchor",
                "emotion_label_changes": "85",
                "top_emotion_transitions": "calm->content:43; content->calm:20",
            }
        ]
        scores = {
            "779605": {"classification": 0.723150},
            "781601": {"classification": 0.719137},
        }
        counts = parse_official_failed_transition_counts(pairwise_rows=rows, score_rows=scores)
        self.assertEqual(counts["calm->content"], 43)
        self.assertEqual(counts["content->calm"], 20)

    def test_parse_official_failed_transition_counts_handles_candidate_b_lower(self) -> None:
        rows = [
            {
                "candidate_a": "779605_anchor",
                "candidate_b": "781601_failed",
                "direction": "candidate_b -> candidate_a",
                "top_emotion_transitions": "content->calm:43; calm->content:20",
            }
        ]
        scores = {
            "779605": {"classification": 0.723150},
            "781601": {"classification": 0.719137},
        }
        counts = parse_official_failed_transition_counts(pairwise_rows=rows, score_rows=scores)
        self.assertEqual(counts["calm->content"], 43)
        self.assertEqual(counts["content->calm"], 20)

    def test_parse_official_failed_transition_counts_ignores_non_declines(self) -> None:
        rows = [
            {
                "candidate_a": "782683_probe",
                "candidate_b": "779605_anchor",
                "direction": "candidate_b -> candidate_a",
                "top_emotion_transitions": "calm->content:2",
            }
        ]
        scores = {
            "779605": {"classification": 0.723150},
            "782683": {"classification": 0.723150},
        }
        counts = parse_official_failed_transition_counts(pairwise_rows=rows, score_rows=scores)
        self.assertEqual(counts, {})

    def test_build_evidence_rows_groups_model_votes_by_proposed_label(self) -> None:
        base_rows = [
            {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
            }
        ]
        predictions = {
            "track2_0001": [
                {"source": "clip", "emotion": "calm", "confidence": 0.9, "margin": 0.3},
                {"source": "siglip2", "emotion": "calm", "confidence": 0.8, "margin": 0.1},
                {"source": "dinov2", "emotion": "content", "confidence": 0.7, "margin": 0.1},
            ]
        }
        evidence = build_evidence_rows(
            base_rows=base_rows,
            predictions_by_sample=predictions,
            duplicate_rows=[],
            failed_transition_counts={},
        )
        calm = [row for row in evidence if row["proposed_emotion"] == "calm"][0]
        self.assertEqual(calm["transition"], "content->calm")
        self.assertEqual(calm["model_vote_count"], 2)
        self.assertEqual(calm["model_sources"], "clip,siglip2")
        self.assertGreater(calm["support_score"], 1.0)

    def test_build_evidence_rows_caps_repeated_duplicate_neighbors(self) -> None:
        base_rows = [
            {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
            }
        ]
        duplicate_rows = [
            {"sample_id": "track2_0001", "public_emotion": "calm", "clip_cosine": 0.99},
            {"sample_id": "track2_0001", "public_emotion": "calm", "clip_cosine": 0.98},
            {"sample_id": "track2_0001", "public_emotion": "calm", "clip_cosine": 0.97},
        ]
        evidence = build_evidence_rows(
            base_rows=base_rows,
            predictions_by_sample={},
            duplicate_rows=duplicate_rows,
            failed_transition_counts={},
        )
        calm = [row for row in evidence if row["proposed_emotion"] == "calm"][0]
        self.assertEqual(calm["model_vote_count"], 0)
        self.assertAlmostEqual(calm["support_score"], 0.99)
        self.assertAlmostEqual(calm["public_duplicate_support_score"], 0.99)


if __name__ == "__main__":
    unittest.main()
