import unittest

from affectiveart.vulca_jepa_audit import (
    build_disagreement_report,
    select_vulca_jepa_review_samples,
)


class VulcaJepaAuditTest(unittest.TestCase):
    def test_selects_high_confidence_structure_disagreements(self) -> None:
        rows = [
            {
                "sample_id": "track2_0001",
                "current": "content",
                "emotion": "calm",
                "confidence": 0.40,
                "knn_emotion": "calm",
                "knn_confidence": 1.00,
                "model_vote_count": 3,
                "overall_caption": "A serene mountain landscape with a small boat on still water.",
            },
            {
                "sample_id": "track2_0002",
                "current": "sad",
                "emotion": "sad",
                "confidence": 0.85,
                "knn_emotion": "sad",
                "knn_confidence": 1.00,
                "model_vote_count": 3,
                "overall_caption": "A grieving figure in a dark interior.",
            },
        ]

        selected = select_vulca_jepa_review_samples(rows, limit=1)

        self.assertEqual([row["sample_id"] for row in selected], ["track2_0001"])
        self.assertGreater(selected[0]["review_priority"], 0.0)

    def test_report_counts_emotion_disagreements_and_calm_content_cases(self) -> None:
        rows = [
            {
                "sample_id": "a",
                "current": "content",
                "emotion": "calm",
                "knn_emotion": "calm",
                "knn_confidence": 1.0,
            },
            {
                "sample_id": "b",
                "current": "sad",
                "emotion": "sad",
                "knn_emotion": "sad",
                "knn_confidence": 1.0,
            },
        ]

        report = build_disagreement_report(rows)

        self.assertEqual(report["row_count"], 2)
        self.assertEqual(report["model_current_disagreements"], 1)
        self.assertEqual(report["content_to_calm_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
