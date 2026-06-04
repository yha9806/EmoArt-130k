import unittest

from affectiveart.track2_moe_specialist_ensemble import (
    GateThresholds,
    expected_label_for_emotion,
    normalize_expert_entries,
)


class Track2MoeSpecialistEnsembleTest(unittest.TestCase):
    def test_expected_label_for_emotion_maps_track2_quadrants(self):
        self.assertEqual(expected_label_for_emotion("calm"), ("Positive", "Low"))
        self.assertEqual(expected_label_for_emotion("aroused"), ("Positive", "High"))
        self.assertEqual(expected_label_for_emotion("frustrated"), ("Negative", "High"))
        self.assertEqual(expected_label_for_emotion("tired"), ("Negative", "Low"))

    def test_gate_thresholds_default_values(self):
        thresholds = GateThresholds()

        self.assertEqual(thresholds.min_supporting_sources, 2)
        self.assertEqual(thresholds.high_confidence, 0.86)
        self.assertEqual(thresholds.min_macro_f1_gain, 0.015)
        self.assertEqual(thresholds.max_accuracy_drop, 0.010)
        self.assertEqual(thresholds.max_top_emotion_share_delta, 0.020)

    def test_normalize_expert_entries_accepts_dict_payload_with_entries(self):
        payload = {
            "entries": [
                {
                    "sample_id": "track2_0001",
                    "emotion": "Calm",
                    "confidence": "0.91",
                    "margin": "0.31",
                    "top3": [{"emotion": "calm", "probability": 0.91}],
                }
            ]
        }

        rows = normalize_expert_entries(payload, source="siglip2_clean", role="global")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "track2_0001")
        self.assertEqual(rows[0]["source"], "siglip2_clean")
        self.assertEqual(rows[0]["role"], "global")
        self.assertEqual(rows[0]["emotion"], "calm")
        self.assertEqual(rows[0]["confidence"], 0.91)
        self.assertEqual(rows[0]["margin"], 0.31)
        self.assertEqual(rows[0]["top3"][0]["emotion"], "calm")
        self.assertEqual(rows[0]["top3"][0]["probability"], 0.91)


if __name__ == "__main__":
    unittest.main()
