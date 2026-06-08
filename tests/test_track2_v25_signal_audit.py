from __future__ import annotations

import unittest

from affectiveart.track2_v25_signal_audit import (
    choose_v25_direction,
    rank_candidate_scores,
    summarize_model_predictions,
)


class Track2V25SignalAuditTests(unittest.TestCase):
    def test_summarize_model_predictions_quantifies_collapse_and_agreement(self) -> None:
        base_rows = [
            {"sample_id": "track2_0001", "emotion": "content"},
            {"sample_id": "track2_0002", "emotion": "content"},
            {"sample_id": "track2_0003", "emotion": "tired"},
            {"sample_id": "track2_0004", "emotion": "calm"},
        ]
        model_entries = {
            "clip": [
                {"sample_id": "track2_0001", "emotion": "calm", "confidence": 0.90, "margin": 0.60},
                {"sample_id": "track2_0002", "emotion": "calm", "confidence": 0.70, "margin": 0.40},
                {"sample_id": "track2_0003", "emotion": "sad", "confidence": 0.85, "margin": 0.55},
                {"sample_id": "track2_0004", "emotion": "calm", "confidence": 0.95, "margin": 0.70},
            ],
            "siglip2": [
                {"sample_id": "track2_0001", "emotion": "calm", "confidence": 0.88, "margin": 0.58},
                {"sample_id": "track2_0002", "emotion": "content", "confidence": 0.52, "margin": 0.12},
                {"sample_id": "track2_0003", "emotion": "sad", "confidence": 0.62, "margin": 0.35},
                {"sample_id": "track2_0004", "emotion": "calm", "confidence": 0.91, "margin": 0.50},
            ],
            "dinov2": [
                {"sample_id": "track2_0001", "emotion": "calm", "confidence": 0.86, "margin": 0.56},
                {"sample_id": "track2_0002", "emotion": "calm", "confidence": 0.65, "margin": 0.28},
                {"sample_id": "track2_0003", "emotion": "sad", "confidence": 0.59, "margin": 0.20},
                {"sample_id": "track2_0004", "emotion": "calm", "confidence": 0.93, "margin": 0.62},
            ],
        }

        summary = summarize_model_predictions(base_rows, model_entries)

        self.assertEqual(summary["models"]["clip"]["changes_vs_base"], 3)
        self.assertEqual(summary["models"]["clip"]["high_confidence_changes"], 2)
        self.assertEqual(summary["models"]["clip"]["top_emotion"], "calm")
        self.assertEqual(summary["models"]["clip"]["top_emotion_share"], 0.75)
        self.assertEqual(summary["three_model_exact_agreement"]["changed_count"], 2)
        self.assertEqual(
            summary["three_model_exact_agreement"]["transition_counts"],
            {"content->calm": 1, "tired->sad": 1},
        )

    def test_rank_candidate_scores_counts_thresholds_and_best(self) -> None:
        ranked = rank_candidate_scores(
            [
                {"candidate_name": "low", "overall_expected": 0.84},
                {"candidate_name": "mid", "overall_expected": 0.861},
                {"candidate_name": "high", "overall_expected": 0.891},
            ],
            target_overall=0.89,
        )

        self.assertEqual(ranked["best"]["candidate_name"], "high")
        self.assertEqual(ranked["above_086"], 2)
        self.assertEqual(ranked["above_target"], 1)

    def test_choose_v25_direction_requires_new_signal_below_target(self) -> None:
        decision = choose_v25_direction(
            best_overall=0.853471,
            target_overall=0.89,
            best_top_emotion_share=0.666,
            three_model_agreement_changed_count=140,
        )

        self.assertEqual(decision["decision"], "needs_new_classification_signal")
        self.assertIn("best_candidate_below_target", decision["reasons"])
        self.assertIn("top_emotion_collapse", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
