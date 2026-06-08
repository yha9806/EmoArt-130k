from __future__ import annotations

import unittest
from collections import Counter

from affectiveart.track2_v29_qwen_rerank import (
    RerankEvidence,
    choose_label_changes,
    normalize_evidence_row,
    runtime_capability,
)


class Track2V29QwenRerankTests(unittest.TestCase):
    def test_normalize_evidence_row_accepts_high_confidence_row(self) -> None:
        row = {
            "sample_id": "track2_0001",
            "current_emotion": "content",
            "proposed_emotion": "calm",
            "current_valence": "Positive",
            "proposed_valence": "Positive",
            "current_arousal": "Low",
            "proposed_arousal": "Low",
            "confidence": "0.93",
            "source": "qwen_rerank",
        }

        evidence = normalize_evidence_row(row)

        self.assertEqual(evidence.sample_id, "track2_0001")
        self.assertEqual(evidence.proposed_emotion, "calm")
        self.assertGreaterEqual(evidence.confidence, 0.9)

    def test_choose_label_changes_respects_top_emotion_cap(self) -> None:
        rows = [
            RerankEvidence(f"track2_{idx:04d}", "content", "calm", "Positive", "Positive", "Low", "Low", 0.99, "qwen")
            for idx in range(10)
        ]

        selected = choose_label_changes(
            rows,
            base_distribution=Counter({"calm": 58, "content": 42}),
            total_rows=100,
            top_emotion_cap=0.60,
            class_floor=2,
        )

        self.assertEqual(len(selected), 2)

    def test_runtime_capability_reports_missing_model_without_failure(self) -> None:
        capability = runtime_capability(model_path="/definitely/missing/model")

        self.assertFalse(capability["qwen_available"])
        self.assertIn("missing_model_path", capability["reasons"])


if __name__ == "__main__":
    unittest.main()
