import csv
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_v6_specialist_selector import (
    ACCEPT_CROSS_MICRO,
    ACCEPT_SAFE,
    HOLD_REVIEW,
    SelectorThresholds,
    build_candidate_rows,
    load_evidence_matrix,
    select_v6_deltas,
    write_v6_outputs,
)


def row(sample_id, emotion, valence=None, arousal=None):
    labels = {
        "alarmed": ("Negative", "High"),
        "annoyed": ("Negative", "High"),
        "aroused": ("Positive", "High"),
        "bored": ("Negative", "Low"),
        "calm": ("Positive", "Low"),
        "content": ("Positive", "Low"),
        "excited": ("Positive", "High"),
        "frustrated": ("Negative", "High"),
        "glad": ("Positive", "Low"),
        "happy": ("Positive", "High"),
        "sad": ("Negative", "Low"),
        "tired": ("Negative", "Low"),
    }
    if valence is None or arousal is None:
        valence, arousal = labels[emotion]
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": f"A {emotion} artwork with visible subject matter and emotional atmosphere.",
        "brushstroke": "Layered brushwork describes the visible forms.",
        "composition": "Balanced composition organizes the subject clearly.",
        "color": "Specific colors shape the emotional atmosphere.",
        "line": "Line quality defines the figure and spatial rhythm.",
        "light": "Light and shadow clarify depth and focus.",
    }


def write_matrix(path, rows):
    fieldnames = sorted({key for item in rows for key in item})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class Track2V6SelectorCoreTest(unittest.TestCase):
    def test_missing_required_delta_fields_default_to_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0001",
                        "current_emotion": "content",
                        "proposed_emotion": "",
                        "supporting_source_count": "4",
                        "supporting_family_count": "3",
                        "evidence_score": "8",
                    }
                ],
            )

            deltas = load_evidence_matrix(matrix_path)
            decisions = select_v6_deltas(deltas)

        self.assertEqual(len(deltas), 1)
        self.assertTrue(deltas[0].malformed)
        self.assertEqual(decisions[0].decision, HOLD_REVIEW)
        self.assertIn("missing_required_delta_field", decisions[0].reason_codes)

    def test_same_quadrant_multi_source_delta_accepts_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0002",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "supporting_sources": "public_style;teacher;siglip2",
                        "supporting_families": "public_style;teacher",
                        "same_quadrant": "true",
                        "public_reference_support": "true",
                        "public_reference_contradiction": "false",
                        "gemini35_objection": "false",
                        "vulca_objection": "false",
                        "evidence_score": "8",
                        "hard96_net_gain": "2",
                        "hard96_net_loss": "0",
                    }
                ],
            )

            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        self.assertEqual(decisions[0].decision, ACCEPT_SAFE)
        self.assertEqual(decisions[0].ladder, "v6_safe_sameq")

    def test_public_reference_contradiction_overrides_model_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0003",
                        "current_emotion": "calm",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "content",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "5",
                        "supporting_family_count": "4",
                        "same_quadrant": "true",
                        "public_reference_support": "false",
                        "public_reference_contradiction": "true",
                        "gemini35_objection": "false",
                        "vulca_objection": "false",
                        "evidence_score": "10",
                    }
                ],
            )

            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        self.assertEqual(decisions[0].decision, HOLD_REVIEW)
        self.assertIn("public_reference_contradiction", decisions[0].reason_codes)

    def test_cross_quadrant_delta_requires_arbitration_and_hard96_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0004",
                        "current_emotion": "annoyed",
                        "current_valence": "Negative",
                        "current_arousal": "High",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "4",
                        "supporting_family_count": "3",
                        "supporting_sources": "teacher;siglip2;dinov2;public_style",
                        "supporting_families": "teacher;embedding;public_style",
                        "same_quadrant": "false",
                        "gemini35_prefers_proposed": "true",
                        "gemini35_fit_margin": "0.41",
                        "public_reference_contradiction": "false",
                        "vulca_objection": "false",
                        "evidence_score": "9",
                        "hard96_net_gain": "2",
                        "hard96_net_loss": "0",
                    }
                ],
            )

            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        self.assertEqual(decisions[0].decision, ACCEPT_CROSS_MICRO)
        self.assertEqual(decisions[0].ladder, "v6_cross_micro")

    def test_cross_quadrant_without_arbitration_stays_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0005",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "frustrated",
                        "proposed_valence": "Negative",
                        "proposed_arousal": "High",
                        "supporting_source_count": "5",
                        "supporting_family_count": "4",
                        "supporting_sources": "teacher;siglip2;dinov2;clip;public_style",
                        "supporting_families": "teacher;embedding;vision;public_style",
                        "same_quadrant": "false",
                        "gemini35_prefers_proposed": "false",
                        "gemini35_fit_margin": "0.10",
                        "evidence_score": "10",
                        "hard96_net_gain": "2",
                        "hard96_net_loss": "0",
                    }
                ],
            )

            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        self.assertEqual(decisions[0].decision, HOLD_REVIEW)
        self.assertIn("missing_cross_arbitration", decisions[0].reason_codes)

    def test_multiple_accepted_labels_for_one_sample_become_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0622",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "supporting_sources": "teacher;siglip2;public_style",
                        "supporting_families": "teacher;public_style",
                        "same_quadrant": "true",
                        "evidence_score": "8",
                    },
                    {
                        "sample_id": "track2_0622",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "glad",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "supporting_sources": "teacher;dinov2;public_style",
                        "supporting_families": "teacher;public_style",
                        "same_quadrant": "true",
                        "evidence_score": "8",
                    },
                ],
            )

            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        self.assertEqual([item.decision for item in decisions], [HOLD_REVIEW, HOLD_REVIEW])
        self.assertTrue(all("proposal_conflict" in item.reason_codes for item in decisions))

    def test_candidate_rows_apply_only_selected_ladder_changes(self):
        baseline = [row("track2_0006", "content"), row("track2_0007", "annoyed")]
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0006",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "supporting_sources": "teacher;siglip2;public_style",
                        "supporting_families": "teacher;public_style",
                        "same_quadrant": "true",
                        "evidence_score": "8",
                    },
                    {
                        "sample_id": "track2_0007",
                        "current_emotion": "annoyed",
                        "current_valence": "Negative",
                        "current_arousal": "High",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "4",
                        "supporting_family_count": "3",
                        "supporting_sources": "teacher;siglip2;dinov2;public_style",
                        "supporting_families": "teacher;embedding;public_style",
                        "same_quadrant": "false",
                        "gemini35_prefers_proposed": "true",
                        "gemini35_fit_margin": "0.41",
                        "evidence_score": "9",
                        "hard96_net_gain": "2",
                        "hard96_net_loss": "0",
                    },
                ],
            )
            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        safe_rows = build_candidate_rows(baseline, decisions, ladder="v6_safe_sameq")
        cross_rows = build_candidate_rows(baseline, decisions, ladder="v6_cross_micro")
        safe_by_id = {item["sample_id"]: item for item in safe_rows}
        cross_by_id = {item["sample_id"]: item for item in cross_rows}

        self.assertEqual(safe_by_id["track2_0006"]["emotion"], "calm")
        self.assertEqual(safe_by_id["track2_0007"]["emotion"], "annoyed")
        self.assertEqual(cross_by_id["track2_0006"]["emotion"], "calm")
        self.assertEqual(cross_by_id["track2_0007"]["emotion"], "calm")
        self.assertEqual(cross_by_id["track2_0007"]["emotional_valence"], "Positive")
        self.assertEqual(cross_by_id["track2_0007"]["emotional_arousal_level"], "Low")

    def test_unknown_candidate_ladder_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_candidate_rows([row("track2_0008", "content")], [], ladder="v6_unknown")

    def test_invalid_proposed_emotion_defaults_to_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0009",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "clam",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "supporting_sources": "teacher;siglip2;public_style",
                        "supporting_families": "teacher;public_style",
                        "same_quadrant": "true",
                        "evidence_score": "9",
                    }
                ],
            )

            deltas = load_evidence_matrix(matrix_path)
            decisions = select_v6_deltas(deltas)

        self.assertTrue(deltas[0].malformed)
        self.assertEqual(decisions[0].decision, HOLD_REVIEW)
        self.assertIn("invalid_track2_emotion", decisions[0].reason_codes)

    def test_numeric_support_counts_without_tokens_default_to_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0010",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "same_quadrant": "true",
                        "evidence_score": "9",
                    }
                ],
            )

            deltas = load_evidence_matrix(matrix_path)
            decisions = select_v6_deltas(deltas)

        self.assertTrue(deltas[0].malformed)
        self.assertEqual(decisions[0].decision, HOLD_REVIEW)
        self.assertIn("missing_support_traceability", decisions[0].reason_codes)

    def test_inflated_support_counts_default_to_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0011",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "4",
                        "supporting_family_count": "3",
                        "supporting_sources": "teacher",
                        "supporting_families": "teacher",
                        "same_quadrant": "true",
                        "evidence_score": "9",
                    }
                ],
            )

            deltas = load_evidence_matrix(matrix_path)
            decisions = select_v6_deltas(deltas)

        self.assertTrue(deltas[0].malformed)
        self.assertEqual(decisions[0].decision, HOLD_REVIEW)
        self.assertIn("support_count_traceability_mismatch", decisions[0].reason_codes)

    def test_cross_micro_caps_cross_quadrant_changes_at_three(self):
        baseline = [row(f"track2_010{index}", "annoyed") for index in range(5)]
        matrix_rows = []
        for index in range(5):
            matrix_rows.append(
                {
                    "sample_id": f"track2_010{index}",
                    "current_emotion": "annoyed",
                    "current_valence": "Negative",
                    "current_arousal": "High",
                    "proposed_emotion": "calm",
                    "proposed_valence": "Positive",
                    "proposed_arousal": "Low",
                    "supporting_source_count": "4",
                    "supporting_family_count": "3",
                    "supporting_sources": "teacher;siglip2;dinov2;public_style",
                    "supporting_families": "teacher;embedding;public_style",
                    "same_quadrant": "false",
                    "gemini35_prefers_proposed": "true",
                    "gemini35_fit_margin": "0.41",
                    "evidence_score": str(14 - index),
                    "hard96_net_gain": "2",
                    "hard96_net_loss": "0",
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(matrix_path, matrix_rows)
            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        cross_rows = build_candidate_rows(baseline, decisions, ladder="v6_cross_micro")
        changed_ids = [
            item["sample_id"]
            for item in cross_rows
            if item["emotion"] == "calm"
        ]
        capped_rows = build_candidate_rows(
            baseline,
            decisions,
            ladder="v6_cross_micro",
            thresholds=SelectorThresholds(max_cross_micro=2),
        )
        capped_changed_ids = [
            item["sample_id"]
            for item in capped_rows
            if item["emotion"] == "calm"
        ]

        self.assertEqual(changed_ids, ["track2_0100", "track2_0101", "track2_0102"])
        self.assertEqual(capped_changed_ids, ["track2_0100", "track2_0101"])

    def test_desc_plus_uses_same_classification_selection_as_cross_micro(self):
        baseline = [row("track2_0110", "content"), row("track2_0111", "annoyed")]
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "evidence.csv"
            write_matrix(
                matrix_path,
                [
                    {
                        "sample_id": "track2_0110",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "3",
                        "supporting_family_count": "2",
                        "supporting_sources": "teacher;siglip2;public_style",
                        "supporting_families": "teacher;public_style",
                        "same_quadrant": "true",
                        "evidence_score": "8",
                    },
                    {
                        "sample_id": "track2_0111",
                        "current_emotion": "annoyed",
                        "current_valence": "Negative",
                        "current_arousal": "High",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "supporting_source_count": "4",
                        "supporting_family_count": "3",
                        "supporting_sources": "teacher;siglip2;dinov2;public_style",
                        "supporting_families": "teacher;embedding;public_style",
                        "same_quadrant": "false",
                        "gemini35_prefers_proposed": "true",
                        "gemini35_fit_margin": "0.41",
                        "evidence_score": "9",
                        "hard96_net_gain": "2",
                        "hard96_net_loss": "0",
                    },
                ],
            )
            decisions = select_v6_deltas(load_evidence_matrix(matrix_path))

        cross_rows = build_candidate_rows(baseline, decisions, ladder="v6_cross_micro")
        desc_rows = build_candidate_rows(baseline, decisions, ladder="v6_desc_plus")

        self.assertEqual(cross_rows, desc_rows)
