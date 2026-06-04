import unittest

from affectiveart.challenge import TRACK2_JSON_EMOTIONS
from affectiveart.track2_moe_specialist_ensemble import (
    GateThresholds,
    VALID_TRACK2_EMOTIONS,
    build_gate_decision,
    expected_label_for_emotion,
    normalize_expert_entries,
)


def current_row(
    sample_id="track2_0001",
    emotion="content",
    valence="Positive",
    arousal="Low",
):
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": "A quiet portrait with balanced color and soft light.",
        "brushstroke": "smooth",
        "composition": "centered",
        "color": "warm muted tones",
        "line": "soft",
        "light": "diffuse",
    }


def expert(
    sample_id,
    source,
    emotion,
    confidence=0.9,
    margin=0.2,
    role="global",
):
    return {
        "sample_id": sample_id,
        "source": source,
        "role": role,
        "emotion": emotion,
        "confidence": confidence,
        "margin": margin,
        "top3": [{"emotion": emotion, "probability": confidence}],
    }


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
        self.assertEqual(thresholds.min_support_confidence, 0.50)
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

    def test_gate_accepts_two_source_supported_va_consistent_change(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert("track2_0001", "clip_clean", "calm"),
                expert("track2_0001", "siglip2_clean", "calm", confidence=0.87),
            ],
        )

        self.assertEqual(decision["decision"], "accept_change")
        self.assertEqual(decision["proposed_emotion"], "calm")
        self.assertEqual(decision["proposed_valence"], "Positive")
        self.assertEqual(decision["proposed_arousal"], "Low")
        self.assertIn("supported_by_2_sources", decision["reasons"])

    def test_gate_holds_single_source_change(self):
        decision = build_gate_decision(
            current_row(),
            [expert("track2_0001", "clip_clean", "calm", confidence=0.82, margin=0.11)],
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertEqual(decision["proposed_emotion"], "calm")
        self.assertIn("insufficient_independent_support", decision["reasons"])

    def test_gate_accepts_single_high_confidence_specialist_change(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert(
                    "track2_0001",
                    "boundary_head",
                    "calm",
                    confidence=0.91,
                    margin=0.21,
                    role="specialist",
                )
            ],
        )

        self.assertEqual(decision["decision"], "accept_change")
        self.assertEqual(decision["proposed_emotion"], "calm")
        self.assertIn(
            "single_high_confidence_source_without_strong_opposition",
            decision["reasons"],
        )

    def test_gate_holds_single_high_confidence_global_change(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert(
                    "track2_0001",
                    "clip_clean",
                    "calm",
                    confidence=0.91,
                    margin=0.21,
                    role="global",
                )
            ],
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertEqual(decision["proposed_emotion"], "calm")
        self.assertIn("single_source_not_specialist", decision["reasons"])

    def test_gate_rejects_text_contradiction(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert("track2_0001", "clip_clean", "sad"),
                expert("track2_0001", "siglip2_clean", "sad", confidence=0.88),
            ],
            description_audit={"verdict": "contradiction"},
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertIn("description_contradiction", decision["reasons"])

    def test_gate_holds_high_similarity_change_without_human_approval(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert("track2_0001", "clip_clean", "calm"),
                expert("track2_0001", "siglip2_clean", "calm", confidence=0.88),
            ],
            high_similarity_public_reference=True,
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertIn(
            "high_similarity_requires_explicit_review",
            decision["reasons"],
        )

    def test_gate_holds_two_source_change_below_quality_bar(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert(
                    "track2_0001",
                    "clip_clean",
                    "calm",
                    confidence=0.49,
                    margin=0.0,
                ),
                expert(
                    "track2_0001",
                    "siglip2_clean",
                    "calm",
                    confidence=0.47,
                    margin=0.0,
                ),
            ],
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertEqual(decision["proposed_emotion"], "calm")
        self.assertIn("support_below_quality_bar", decision["reasons"])

    def test_gate_holds_supported_change_with_strong_opposition(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert("track2_0001", "clip_clean", "calm", confidence=0.7),
                expert("track2_0001", "siglip2_clean", "calm", confidence=0.72),
                expert("track2_0001", "dinov2_clean", "sad", confidence=0.91),
            ],
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertEqual(decision["proposed_emotion"], "calm")
        self.assertIn("strong_opposition", decision["reasons"])

    def test_gate_holds_supported_change_with_current_label_opposition(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert("track2_0001", "clip_clean", "calm", confidence=0.7),
                expert("track2_0001", "siglip2_clean", "calm", confidence=0.72),
                expert("track2_0001", "dinov2_clean", "content", confidence=0.91),
            ],
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertEqual(decision["proposed_emotion"], "calm")
        self.assertIn("strong_opposition", decision["reasons"])

    def test_gate_duplicate_source_rows_do_not_count_as_independent_support(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert(
                    "track2_0001",
                    "clip_clean",
                    "calm",
                    confidence=0.82,
                    margin=0.11,
                ),
                expert(
                    "track2_0001",
                    "clip_clean",
                    "calm",
                    confidence=0.81,
                    margin=0.10,
                ),
            ],
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertEqual(decision["proposed_emotion"], "calm")
        self.assertIn("insufficient_independent_support", decision["reasons"])

    def test_gate_ignores_irrelevant_expert_rows_for_other_sample_id(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert("track2_9999", "clip_clean", "calm"),
                expert("track2_9999", "siglip2_clean", "calm"),
            ],
        )

        self.assertEqual(decision["decision"], "keep_current")
        self.assertEqual(decision["proposed_emotion"], "content")
        self.assertEqual(decision["expert_evidence"], [])
        self.assertIn("no_supported_change", decision["reasons"])

    def test_gate_ignores_blank_source_identities(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert("track2_0001", None, "calm"),
                expert("track2_0001", "   ", "calm"),
            ],
        )

        self.assertEqual(decision["decision"], "keep_current")
        self.assertEqual(decision["expert_evidence"], [])
        self.assertIn("no_supported_change", decision["reasons"])

    def test_gate_tie_breaks_same_support_and_confidence_lexically(self):
        decision = build_gate_decision(
            current_row(),
            [
                expert("track2_0001", "clip_clean", "glad", confidence=0.6),
                expert("track2_0001", "siglip2_clean", "glad", confidence=0.6),
                expert("track2_0001", "dinov2_clean", "calm", confidence=0.6),
                expert("track2_0001", "boundary_head", "calm", confidence=0.6),
            ],
        )

        self.assertEqual(decision["decision"], "accept_change")
        self.assertEqual(decision["proposed_emotion"], "calm")

    def test_gate_optional_controls_are_keyword_only(self):
        with self.assertRaises(TypeError):
            build_gate_decision(current_row(), [], GateThresholds())


if __name__ == "__main__":
    unittest.main()
