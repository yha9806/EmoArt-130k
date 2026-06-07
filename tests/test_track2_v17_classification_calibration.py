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
    select_v17_changes,
    v17_gate_for_evidence,
)


def _evidence_row(**overrides):
    row = {
        "sample_id": "track2_0001",
        "current_emotion": "content",
        "proposed_emotion": "calm",
        "transition": "content->calm",
        "model_vote_count": 2,
        "model_sources": "clip,siglip2",
        "all_sources": "clip,siglip2",
        "support_score": 1.65,
        "model_support_score": 1.65,
        "public_style_support_score": 0.0,
        "public_duplicate_support_score": 0.0,
        "max_confidence": 0.82,
        "exact_duplicate": False,
        "near_duplicate": False,
        "failed_transition_count": 0,
        "current_label_issue_count": 0,
        "rationale": "",
    }
    row.update(overrides)
    return row


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

    def test_safe_gate_accepts_two_model_families_when_not_failed_transition(self) -> None:
        gate = v17_gate_for_evidence(
            _evidence_row(),
            profile="safe",
            distribution={"content": 12, "calm": 12},
        )
        self.assertEqual(gate["decision"], "accept")
        self.assertIn("meets_profile_thresholds", gate["reasons"])

    def test_aggressive_probe_accepts_one_supported_model_family(self) -> None:
        gate = v17_gate_for_evidence(
            _evidence_row(
                model_vote_count=1,
                model_sources="clip",
                all_sources="clip",
                support_score=1.05,
                model_support_score=1.05,
                max_confidence=0.72,
            ),
            profile="aggressive_probe",
            distribution={"content": 12, "calm": 12},
        )
        self.assertEqual(gate["decision"], "accept")
        self.assertIn("meets_profile_thresholds", gate["reasons"])

    def test_aggressive_probe_blocks_failed_bulk_transition(self) -> None:
        gate = v17_gate_for_evidence(
            _evidence_row(
                model_vote_count=1,
                model_sources="clip",
                all_sources="clip",
                support_score=1.2,
                model_support_score=1.2,
                max_confidence=0.78,
                failed_transition_count=10,
            ),
            profile="aggressive_probe",
            distribution={"content": 12, "calm": 12},
        )
        self.assertEqual(gate["decision"], "block")
        self.assertIn("failed_transition_family", gate["reasons"])

    def test_safe_gate_blocks_failed_bulk_transition_without_exact_duplicate(self) -> None:
        gate = v17_gate_for_evidence(
            _evidence_row(failed_transition_count=10),
            profile="safe",
            distribution={"content": 12, "calm": 12},
        )
        self.assertEqual(gate["decision"], "block")
        self.assertIn("failed_transition_family", gate["reasons"])

    def test_exact_duplicate_overrides_failed_transition(self) -> None:
        gate = v17_gate_for_evidence(
            _evidence_row(
                model_vote_count=0,
                model_sources="",
                all_sources="public_exact_public_duplicate",
                support_score=0.99,
                model_support_score=0.0,
                public_duplicate_support_score=0.99,
                max_confidence=0.99,
                exact_duplicate=True,
                failed_transition_count=20,
            ),
            profile="safe",
            distribution={"content": 12, "calm": 12},
        )
        self.assertEqual(gate["decision"], "accept")
        self.assertNotIn("failed_transition_family", gate["reasons"])

    def test_exact_duplicate_override_requires_high_duplicate_confidence(self) -> None:
        gate = v17_gate_for_evidence(
            _evidence_row(
                model_vote_count=0,
                model_sources="",
                all_sources="public_exact_public_duplicate",
                support_score=0.0,
                model_support_score=0.0,
                public_duplicate_support_score=0.0,
                max_confidence=0.99,
                exact_duplicate=True,
                failed_transition_count=20,
            ),
            profile="safe",
            distribution={"content": 12, "calm": 12},
        )
        self.assertEqual(gate["decision"], "block")
        self.assertIn("insufficient_exact_duplicate_support", gate["reasons"])

    def test_near_duplicate_does_not_bypass_failed_transition_family(self) -> None:
        gate = v17_gate_for_evidence(
            _evidence_row(
                support_score=0.97,
                max_confidence=0.97,
                near_duplicate=True,
                failed_transition_count=10,
            ),
            profile="safe",
            distribution={"content": 12, "calm": 12},
        )
        self.assertEqual(gate["decision"], "block")
        self.assertIn("failed_transition_family", gate["reasons"])

    def test_near_duplicate_does_not_bypass_insufficient_model_families(self) -> None:
        gate = v17_gate_for_evidence(
            _evidence_row(
                model_vote_count=0,
                model_sources="",
                all_sources="public_near_public_duplicate",
                support_score=1.1,
                model_support_score=0.0,
                public_duplicate_support_score=1.1,
                max_confidence=0.99,
                near_duplicate=True,
            ),
            profile="safe",
            distribution={"content": 12, "calm": 12},
        )
        self.assertEqual(gate["decision"], "block")
        self.assertIn("insufficient_model_families", gate["reasons"])

    def test_safe_gate_blocks_public_style_only_without_model_votes(self) -> None:
        gate = v17_gate_for_evidence(
            _evidence_row(
                model_vote_count=0,
                model_sources="",
                all_sources="public_clean,public_inclusive",
                support_score=1.8,
                model_support_score=0.0,
                public_style_support_score=1.8,
                max_confidence=0.93,
            ),
            profile="safe",
            distribution={"content": 12, "calm": 12},
        )
        self.assertEqual(gate["decision"], "block")
        self.assertIn("insufficient_model_families", gate["reasons"])

    def test_balanced_gate_blocks_public_style_only_without_model_votes(self) -> None:
        gate = v17_gate_for_evidence(
            _evidence_row(
                model_vote_count=0,
                model_sources="",
                all_sources="public_clean,public_inclusive",
                support_score=1.8,
                model_support_score=0.0,
                public_style_support_score=1.8,
                max_confidence=0.93,
            ),
            profile="balanced",
            distribution={"content": 12, "calm": 12},
        )
        self.assertEqual(gate["decision"], "block")
        self.assertIn("insufficient_model_families", gate["reasons"])

    def test_gate_blocks_removing_rare_current_class(self) -> None:
        gate = v17_gate_for_evidence(
            _evidence_row(),
            profile="safe",
            distribution={"content": 2, "calm": 12},
        )
        self.assertEqual(gate["decision"], "block")
        self.assertIn("rare_current_class_floor", gate["reasons"])

    def test_select_v17_changes_caps_balanced_content_calm_transition_family(self) -> None:
        rows = []
        for index in range(6):
            rows.append(
                _evidence_row(
                    sample_id=f"track2_content_{index:04d}",
                    current_emotion="content",
                    proposed_emotion="calm",
                    transition="content->calm",
                    support_score=3.0 - index * 0.1,
                    max_confidence=0.95 - index * 0.01,
                )
            )
        for index in range(3):
            rows.append(
                _evidence_row(
                    sample_id=f"track2_calm_{index:04d}",
                    current_emotion="calm",
                    proposed_emotion="content",
                    transition="calm->content",
                    support_score=2.2 - index * 0.1,
                    max_confidence=0.89 - index * 0.01,
                )
            )
        selected = select_v17_changes(rows, profile="balanced", current_distribution={"content": 20, "calm": 20})
        family_selected = [
            row
            for row in selected
            if row["transition"] in {"content->calm", "calm->content"}
        ]
        self.assertEqual(len(family_selected), 4)
        self.assertEqual([row["sample_id"] for row in family_selected], [f"track2_content_{index:04d}" for index in range(4)])

    def test_select_v17_changes_keeps_one_highest_supported_change_per_sample(self) -> None:
        rows = [
            _evidence_row(sample_id="track2_same", proposed_emotion="glad", transition="content->glad", support_score=1.7, max_confidence=0.8),
            _evidence_row(sample_id="track2_same", proposed_emotion="calm", transition="content->calm", support_score=2.1, max_confidence=0.9),
        ]
        selected = select_v17_changes(rows, profile="balanced", current_distribution={"content": 10, "calm": 10})
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["proposed_emotion"], "calm")

    def test_select_v17_changes_tracks_projected_distribution_floor(self) -> None:
        rows = [
            _evidence_row(
                sample_id="track2_glad_0001",
                current_emotion="glad",
                proposed_emotion="content",
                transition="glad->content",
                support_score=2.0,
                max_confidence=0.9,
            ),
            _evidence_row(
                sample_id="track2_glad_0002",
                current_emotion="glad",
                proposed_emotion="content",
                transition="glad->content",
                support_score=1.9,
                max_confidence=0.88,
            ),
        ]
        selected = select_v17_changes(rows, profile="balanced", current_distribution={"glad": 3, "content": 10})
        self.assertEqual([row["sample_id"] for row in selected], ["track2_glad_0001"])


if __name__ == "__main__":
    unittest.main()
