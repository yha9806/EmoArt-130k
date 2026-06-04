import unittest

from affectiveart.challenge import TRACK2_JSON_EMOTIONS
from affectiveart.track2_moe_specialist_ensemble import (
    GateThresholds,
    VALID_TRACK2_EMOTIONS,
    expected_label_for_emotion,
    normalize_expert_entries,
)


def _expert_row(sample_id="track2_0001", emotion="calm", **overrides):
    row = {
        "sample_id": sample_id,
        "emotion": emotion,
        "confidence": 0.91,
        "margin": 0.31,
        "top3": [{"emotion": emotion, "probability": 0.91}],
    }
    row.update(overrides)
    return row


class Track2MoeSpecialistEnsembleTest(unittest.TestCase):
    def test_valid_emotions_are_anchored_to_official_track2_json_enum(self):
        self.assertIs(VALID_TRACK2_EMOTIONS, TRACK2_JSON_EMOTIONS)

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
                _expert_row(emotion="Calm", confidence="0.91", margin="0.31")
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

    def test_normalize_expert_entries_accepts_list_payloads(self):
        rows = normalize_expert_entries(
            [_expert_row("track2_0002", "aroused")],
            source="clip_clean",
            role="specialist",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "track2_0002")
        self.assertEqual(rows[0]["source"], "clip_clean")
        self.assertEqual(rows[0]["role"], "specialist")
        self.assertEqual(rows[0]["emotion"], "aroused")

    def test_normalize_expert_entries_accepts_predictions_and_rows_keys(self):
        prediction_rows = normalize_expert_entries(
            {"predictions": [_expert_row("track2_0003", "frustrated")]},
            source="dinov2_clean",
            role="hardcase",
        )
        row_rows = normalize_expert_entries(
            {"rows": [_expert_row("track2_0004", "tired")]},
            source="siglip2_clean",
            role="global",
        )

        self.assertEqual(prediction_rows[0]["sample_id"], "track2_0003")
        self.assertEqual(prediction_rows[0]["emotion"], "frustrated")
        self.assertEqual(row_rows[0]["sample_id"], "track2_0004")
        self.assertEqual(row_rows[0]["emotion"], "tired")

    def test_normalize_expert_entries_skips_invalid_rows(self):
        payload = [
            "not-a-row",
            _expert_row("", "calm"),
            _expert_row("track2_0005", "mystery"),
            _expert_row("track2_0006", "happy"),
        ]

        rows = normalize_expert_entries(payload, source="siglip2_clean", role="global")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "track2_0006")
        self.assertEqual(rows[0]["emotion"], "happy")

    def test_normalize_expert_entries_bad_numbers_fall_back_to_zero(self):
        payload = [
            _expert_row(
                "track2_0007",
                "calm",
                confidence="not-a-number",
                margin=float("inf"),
            )
        ]

        rows = normalize_expert_entries(payload, source="siglip2_clean", role="global")

        self.assertEqual(rows[0]["confidence"], 0.0)
        self.assertEqual(rows[0]["margin"], 0.0)

    def test_normalize_expert_entries_preserves_top3_dicts(self):
        payload = [
            _expert_row(
                "track2_0008",
                "calm",
                top3=[
                    {"emotion": "Aroused", "probability": "0.52"},
                    {"label": "Calm", "prob": 0.31},
                    {"class": "Tired", "score": "0.17"},
                ],
            )
        ]

        rows = normalize_expert_entries(payload, source="siglip2_clean", role="global")

        self.assertEqual(
            rows[0]["top3"],
            [
                {"emotion": "aroused", "probability": 0.52},
                {"emotion": "calm", "probability": 0.31},
                {"emotion": "tired", "probability": 0.17},
            ],
        )

    def test_normalize_expert_entries_invalid_top3_falls_back_to_selected_emotion(self):
        payload = [
            _expert_row(
                "track2_0009",
                "glad",
                confidence=0.73,
                top3=[
                    {"emotion": "unknown", "probability": 0.8},
                    {"label": "", "probability": 0.2},
                    None,
                ],
            )
        ]

        rows = normalize_expert_entries(payload, source="siglip2_clean", role="global")

        self.assertEqual(rows[0]["top3"], [{"emotion": "glad", "probability": 0.73}])


if __name__ == "__main__":
    unittest.main()
