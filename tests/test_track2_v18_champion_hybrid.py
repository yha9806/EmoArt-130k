from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_v18_champion_hybrid import (
    ChampionTargets,
    OfficialAnchor,
    choose_v18_base,
    load_champion_targets,
    load_official_anchors,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class Track2V18ChampionHybridTests(unittest.TestCase):
    def test_module_import_does_not_load_affectiveart_challenge(self) -> None:
        code = (
            "import sys\n"
            "import affectiveart.track2_v18_champion_hybrid\n"
            "raise SystemExit(1 if 'affectiveart.challenge' in sys.modules else 0)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )

    def test_module_has_no_codabench_upload_surface(self) -> None:
        import affectiveart.track2_v18_champion_hybrid as module

        public_names = [name for name in dir(module) if not name.startswith("_")]

        self.assertTrue(module.NO_AUTO_SUBMIT_POLICY)
        self.assertFalse(any("codabench" in name.lower() and "upload" in name.lower() for name in public_names))
        self.assertFalse(any("submit_to" in name.lower() for name in public_names))

    def test_public_api_hides_future_v17_helpers(self) -> None:
        import affectiveart.track2_v18_champion_hybrid as module

        public_names = {name for name in dir(module) if not name.startswith("_")}

        self.assertFalse(
            public_names
            & {
                "apply_v17_changes",
                "build_evidence_rows",
                "load_prediction_sources",
                "load_duplicate_rows",
                "parse_official_failed_transition_counts",
            }
        )

    def test_load_official_anchors_parses_exact_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "official.csv"
            _write_csv(
                path,
                [
                    {
                        "submission_id": "779605",
                        "file_name": "track2_submission_moe_v2_accept5_candidate.zip",
                        "official_overall": "0.836408",
                        "official_classification": "0.72315",
                        "official_description": "0.949667",
                    }
                ],
            )

            anchors = load_official_anchors(path)

            self.assertEqual(list(anchors), ["779605"])
            self.assertEqual(anchors["779605"].submission_id, "779605")
            self.assertAlmostEqual(anchors["779605"].overall, 0.836408)
            self.assertAlmostEqual(anchors["779605"].classification, 0.72315)
            self.assertAlmostEqual(anchors["779605"].description, 0.949667)

    def test_load_official_anchors_rejects_malformed_required_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "official.csv"
            _write_csv(
                path,
                [
                    {
                        "submission_id": "779605",
                        "file_name": "track2_submission_moe_v2_accept5_candidate.zip",
                        "official_overall": "not-a-score",
                        "official_classification": "0.72315",
                        "official_description": "0.949667",
                    }
                ],
            )

            with self.assertRaises(ValueError) as context:
                load_official_anchors(path)

            message = str(context.exception)
            self.assertIn("official_overall", message)
            self.assertIn("779605", message)
            self.assertIn(str(path), message)

    def test_load_champion_targets_uses_first_place_and_current_team(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leaderboard.csv"
            _write_csv(
                path,
                [
                    {
                        "#": "1",
                        "Participant": "N&T小分队",
                        "Overall Score": "0.89",
                        "Classification Score": "0.78",
                        "Description Score": "1.0",
                        "Emotion Accuracy": "0.8",
                        "Emotion Macro F1": "0.31",
                        "Visual Grounding": "0.99",
                        "Attribute Specificity": "1.0",
                        "Overall Caption": "0.99",
                    },
                    {
                        "#": "4",
                        "Participant": "vulcaart",
                        "Overall Score": "0.84",
                        "Classification Score": "0.72",
                        "Description Score": "0.95",
                        "Emotion Accuracy": "0.57",
                        "Emotion Macro F1": "0.31",
                        "Visual Grounding": "0.96",
                        "Attribute Specificity": "0.95",
                        "Overall Caption": "0.94",
                    },
                ],
            )

            targets = load_champion_targets(path, participant="vulcaart")

            self.assertIsInstance(targets, ChampionTargets)
            self.assertAlmostEqual(targets.first_overall, 0.89)
            self.assertAlmostEqual(targets.current_overall, 0.84)
            self.assertAlmostEqual(targets.first_emotion_accuracy, 0.80)
            self.assertAlmostEqual(targets.current_emotion_accuracy, 0.57)

    def test_load_champion_targets_rejects_malformed_required_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leaderboard.csv"
            _write_csv(
                path,
                [
                    {
                        "#": "1",
                        "Participant": "N&T小分队",
                        "Overall Score": "0.89",
                        "Classification Score": "0.78",
                        "Description Score": "1.0",
                        "Emotion Accuracy": "0.8",
                        "Emotion Macro F1": "0.31",
                        "Visual Grounding": "0.99",
                        "Attribute Specificity": "1.0",
                        "Overall Caption": "0.99",
                    },
                    {
                        "#": "4",
                        "Participant": "vulcaart",
                        "Overall Score": "not-a-score",
                        "Classification Score": "0.72",
                        "Description Score": "0.95",
                        "Emotion Accuracy": "0.57",
                        "Emotion Macro F1": "0.31",
                        "Visual Grounding": "0.96",
                        "Attribute Specificity": "0.95",
                        "Overall Caption": "0.94",
                    },
                ],
            )

            with self.assertRaises(ValueError) as context:
                load_champion_targets(path, participant="vulcaart")

            message = str(context.exception)
            self.assertIn("Overall Score", message)
            self.assertIn("vulcaart", message)
            self.assertIn(str(path), message)

    def test_choose_v18_base_prefers_highest_exact_official_anchor(self) -> None:
        anchors = {
            "779605": OfficialAnchor("779605", "a.zip", 0.836408, 0.72315, 0.949667),
            "781601": OfficialAnchor("781601", "b.zip", 0.834027, 0.719137, 0.948917),
        }
        available = {
            "779605": Path("submissions/track2_submission_moe_v2_accept5_candidate.json"),
            "781601": Path("submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json"),
        }

        choice = choose_v18_base(anchors, available)

        self.assertEqual(choice["submission_id"], "779605")
        self.assertEqual(choice["base_json"], str(available["779605"]))
        self.assertEqual(choice["reason"], "highest_exact_official_overall")

    def test_choose_v18_base_ties_by_submission_id_not_component_scores(self) -> None:
        anchors = {
            "100": OfficialAnchor("100", "a.zip", 0.9, 0.8, 0.7),
            "200": OfficialAnchor("200", "b.zip", 0.9, 0.7, 0.6),
        }
        available = {
            "100": Path("submissions/a.json"),
            "200": Path("submissions/b.json"),
        }

        choice = choose_v18_base(anchors, available)

        self.assertEqual(choice["submission_id"], "200")
        self.assertEqual(choice["base_json"], str(available["200"]))
        self.assertEqual(choice["reason"], "highest_exact_official_overall")

    def test_merge_description_rows_preserves_labels_and_uses_better_text(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import merge_description_rows

        base = [
            {
                "sample_id": "track2_0001",
                "emotion": "calm",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "A calm view.",
                "brushstroke": "Soft.",
                "composition": "Balanced.",
                "color": "Muted.",
                "line": "Gentle.",
                "light": "Soft.",
            }
        ]
        text = [
            {
                "sample_id": "track2_0001",
                "emotion": "frustrated",
                "emotional_valence": "Negative",
                "emotional_arousal_level": "High",
                "overall_caption": "A quiet landscape uses muted color and open space to create a calm atmosphere.",
                "brushstroke": "Layered, soft brushwork keeps the surface gentle.",
                "composition": "The open balanced arrangement creates visual stability.",
                "color": "Muted greens and pale blues reinforce calmness.",
                "line": "Slow horizontal lines reduce tension.",
                "light": "Diffuse light softens contrast.",
            }
        ]

        merged, report = merge_description_rows(base, text)

        self.assertEqual(merged[0]["emotion"], "calm")
        self.assertEqual(merged[0]["emotional_valence"], "Positive")
        self.assertEqual(merged[0]["emotional_arousal_level"], "Low")
        self.assertIn("quiet landscape", merged[0]["overall_caption"])
        self.assertEqual(report["description_changed_rows"], 1)
        self.assertEqual(report["label_changed_rows"], 0)

    def test_merge_description_rows_keeps_base_when_rewrite_is_template_like(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import merge_description_rows

        row = {
            "sample_id": "track2_0001",
            "emotion": "content",
            "emotional_valence": "Positive",
            "emotional_arousal_level": "Low",
            "overall_caption": "A domestic interior creates a content emotional atmosphere.",
            "brushstroke": "Soft layered paint describes the interior forms.",
            "composition": "The compact arrangement centers attention on the room.",
            "color": "Warm ochre and green tones support comfort.",
            "line": "Curved outlines keep the space relaxed.",
            "light": "Soft light gives the scene warmth.",
        }

        merged, report = merge_description_rows([row], [{**row, "overall_caption": "A content artwork."}])

        self.assertEqual(merged[0]["overall_caption"], row["overall_caption"])
        self.assertEqual(report["description_changed_rows"], 0)
        self.assertEqual(report["rejected_text_rows"], 1)

    def test_merge_description_rows_rejects_judge_oriented_text(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import merge_description_rows

        row = {
            "sample_id": "track2_0001",
            "emotion": "calm",
            "emotional_valence": "Positive",
            "emotional_arousal_level": "Low",
            "overall_caption": "A calm view.",
            "brushstroke": "Soft layered paint describes the quiet forms.",
            "composition": "The balanced arrangement opens the central space.",
            "color": "Muted greens and blues keep the mood calm.",
            "line": "Slow horizontal lines reduce tension.",
            "light": "Diffuse light softens contrast.",
        }

        merged, report = merge_description_rows(
            [row],
            [
                {
                    **row,
                    "overall_caption": (
                        "A judge sees color, composition, light, and atmosphere across the scene."
                    ),
                }
            ],
        )

        self.assertEqual(merged[0]["overall_caption"], row["overall_caption"])
        self.assertEqual(report["description_changed_rows"], 0)
        self.assertEqual(report["rejected_text_rows"], 1)

    def test_merge_description_rows_counts_mixed_accepted_and_rejected_text(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import merge_description_rows

        row = {
            "sample_id": "track2_0001",
            "emotion": "calm",
            "emotional_valence": "Positive",
            "emotional_arousal_level": "Low",
            "overall_caption": "A calm view.",
            "brushstroke": "Soft.",
            "composition": "Balanced.",
            "color": "Muted.",
            "line": "Gentle.",
            "light": "Soft.",
        }
        text = {
            **row,
            "color": "Muted color expands through the open space with a calm atmosphere.",
            "line": "High scoring in evaluation because the line and light cues are clear.",
        }

        merged, report = merge_description_rows([row], [text])

        self.assertEqual(
            merged[0]["color"],
            "Muted color expands through the open space with a calm atmosphere.",
        )
        self.assertEqual(merged[0]["line"], row["line"])
        self.assertEqual(report["description_changed_rows"], 1)
        self.assertEqual(report["rejected_text_rows"], 1)

    def test_merge_description_rows_rejects_high_scoring_in_evaluation_text(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import merge_description_rows

        row = {
            "sample_id": "track2_0001",
            "emotion": "calm",
            "emotional_valence": "Positive",
            "emotional_arousal_level": "Low",
            "overall_caption": "A calm view.",
            "brushstroke": "Soft layered paint describes the quiet forms.",
            "composition": "The balanced arrangement opens the central space.",
            "color": "Muted greens and blues keep the mood calm.",
            "line": "Slow horizontal lines reduce tension.",
            "light": "Diffuse light softens contrast.",
        }
        text = {
            **row,
            "overall_caption": (
                "High scoring in evaluation because color, composition, line, and light all "
                "support the intended atmosphere."
            ),
        }

        merged, report = merge_description_rows([row], [text])

        self.assertEqual(merged[0]["overall_caption"], row["overall_caption"])
        self.assertEqual(report["description_changed_rows"], 0)
        self.assertEqual(report["rejected_text_rows"], 1)

    def test_merge_description_rows_rejects_evaluate_text(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import merge_description_rows

        row = {
            "sample_id": "track2_0001",
            "emotion": "calm",
            "emotional_valence": "Positive",
            "emotional_arousal_level": "Low",
            "overall_caption": "A calm view.",
            "brushstroke": "Soft layered paint describes the quiet forms.",
            "composition": "The balanced arrangement opens the central space.",
            "color": "Muted greens and blues keep the mood calm.",
            "line": "Slow horizontal lines reduce tension.",
            "light": "Diffuse light softens contrast.",
        }
        text = {
            **row,
            "overall_caption": (
                "Please evaluate how color, composition, line, and light would support "
                "the intended atmosphere."
            ),
        }

        merged, report = merge_description_rows([row], [text])

        self.assertEqual(merged[0]["overall_caption"], row["overall_caption"])
        self.assertEqual(report["description_changed_rows"], 0)
        self.assertEqual(report["rejected_text_rows"], 1)

    def test_enrich_champion_evidence_marks_priority_and_risk(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import enrich_champion_evidence

        rows = [
            {
                "sample_id": "track2_0001",
                "current_emotion": "content",
                "proposed_emotion": "calm",
                "transition": "content->calm",
                "same_valence": True,
                "same_arousal": True,
                "model_vote_count": 2,
                "support_score": 1.6,
                "public_duplicate_support_score": 0.0,
                "exact_duplicate": False,
                "near_duplicate": False,
                "failed_transition_count": 43,
            },
            {
                "sample_id": "track2_0002",
                "current_emotion": "annoyed",
                "proposed_emotion": "content",
                "transition": "annoyed->content",
                "same_valence": False,
                "same_arousal": False,
                "model_vote_count": 1,
                "support_score": 1.0,
                "public_duplicate_support_score": 0.0,
                "exact_duplicate": False,
                "near_duplicate": False,
                "failed_transition_count": 0,
            },
        ]

        enriched = enrich_champion_evidence(rows)

        self.assertEqual(enriched[0]["transition_family"], "calm<->content")
        self.assertTrue(enriched[0]["majority_boundary_transition"])
        self.assertEqual(enriched[0]["champion_priority"], "medium")
        self.assertIn("failed_official_transition", enriched[0]["risk_flags"])
        self.assertEqual(enriched[1]["champion_priority"], "low")
        self.assertIn("cross_quadrant", enriched[1]["risk_flags"])

    def test_enrich_champion_evidence_handles_numeric_booleans_without_mutation(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import enrich_champion_evidence

        rows = [
            {
                "sample_id": "track2_0001",
                "current_emotion": "content",
                "proposed_emotion": "calm",
                "transition": "content->calm",
                "same_valence": "1.0",
                "same_arousal": 1.0,
                "model_vote_count": 2,
                "support_score": 1.6,
                "public_duplicate_support_score": 0.0,
                "exact_duplicate": False,
                "near_duplicate": False,
                "failed_transition_count": 0,
            },
            {
                "sample_id": "track2_0002",
                "current_emotion": "content",
                "proposed_emotion": "calm",
                "transition": "content->calm",
                "same_valence": "0.0",
                "same_arousal": 1.0,
                "model_vote_count": 2,
                "support_score": 1.6,
                "public_duplicate_support_score": 0.0,
                "exact_duplicate": False,
                "near_duplicate": False,
                "failed_transition_count": 0,
            },
        ]
        original_rows = [dict(row) for row in rows]

        enriched = enrich_champion_evidence(rows)

        by_id = {str(row["sample_id"]): row for row in enriched}
        self.assertEqual(rows, original_rows)
        self.assertEqual(by_id["track2_0001"]["champion_priority"], "medium")
        self.assertNotIn("cross_quadrant", by_id["track2_0001"]["risk_flags"])
        self.assertEqual(by_id["track2_0002"]["champion_priority"], "low")
        self.assertIn("cross_quadrant", by_id["track2_0002"]["risk_flags"])

    def test_enrich_champion_evidence_prioritizes_numeric_exact_duplicate(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import enrich_champion_evidence

        rows = [
            {
                "sample_id": "track2_0001",
                "current_emotion": "content",
                "proposed_emotion": "calm",
                "transition": "content->calm",
                "same_valence": "1.0",
                "same_arousal": "1.0",
                "model_vote_count": 0,
                "support_score": 0.0,
                "public_duplicate_support_score": 1.0,
                "exact_duplicate": "1.0",
                "near_duplicate": False,
                "failed_transition_count": 0,
            }
        ]

        enriched = enrich_champion_evidence(rows)

        self.assertEqual(enriched[0]["champion_priority"], "high")
        self.assertNotIn("weak_model_family_count", enriched[0]["risk_flags"])
        self.assertNotIn("weak_support_score", enriched[0]["risk_flags"])

    def test_v18_gate_blocks_unsupported_cross_quadrant_change(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content->annoyed",
            "same_valence": False,
            "same_arousal": False,
            "model_vote_count": 3,
            "support_score": 2.0,
            "exact_duplicate": False,
            "near_duplicate": False,
            "failed_transition_count": 0,
            "risk_flags": "cross_quadrant",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "annoyed": 30})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("unsupported_cross_quadrant", gate["reasons"])

    def test_v18_gate_allows_exact_duplicate_even_with_failed_transition(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content->calm",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 0,
            "support_score": 0.2,
            "public_duplicate_support_score": 1.0,
            "exact_duplicate": True,
            "near_duplicate": False,
            "failed_transition_count": 43,
            "risk_flags": "failed_official_transition",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "calm": 490})

        self.assertEqual(gate["decision"], "accept")
        self.assertIn("exact_duplicate_override", gate["reasons"])

    def test_select_v18_changes_caps_calm_content_family(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import select_v18_changes

        rows = []
        for index in range(5):
            rows.append(
                {
                    "sample_id": f"track2_{index:04d}",
                    "current_emotion": "content",
                    "proposed_emotion": "calm",
                    "transition": "content->calm",
                    "same_valence": True,
                    "same_arousal": True,
                    "model_vote_count": 3,
                    "support_score": 2.0,
                    "public_duplicate_support_score": 0.0,
                    "exact_duplicate": False,
                    "near_duplicate": False,
                    "failed_transition_count": 0,
                    "risk_flags": "",
                    "transition_family": "calm<->content",
                }
            )

        selected = select_v18_changes(rows, current_distribution={"content": 238, "calm": 490})

        self.assertEqual(len(selected), 3)
        self.assertTrue(all(row["gate_decision"] == "accept" for row in selected))

    def test_v18_gate_blocks_rare_current_class_floor(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "current_emotion": "bored",
            "proposed_emotion": "calm",
            "transition": "bored->calm",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 3,
            "support_score": 2.0,
            "public_duplicate_support_score": 0.0,
            "exact_duplicate": False,
            "near_duplicate": False,
            "failed_transition_count": 0,
            "risk_flags": "",
        }

        gate = v18_gate_for_evidence(row, distribution={"bored": 3, "calm": 100})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("rare_current_class_floor", gate["reasons"])

    def test_select_v18_changes_skips_duplicates_and_updates_rare_floor(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import select_v18_changes

        rows = [
            {
                "sample_id": "track2_0001",
                "current_emotion": "bored",
                "proposed_emotion": "calm",
                "transition": "bored->calm",
                "same_valence": True,
                "same_arousal": True,
                "model_vote_count": 3,
                "support_score": 2.5,
                "public_duplicate_support_score": 0.0,
                "exact_duplicate": False,
                "near_duplicate": False,
                "failed_transition_count": 0,
                "risk_flags": "",
            },
            {
                "sample_id": "track2_0001",
                "current_emotion": "content",
                "proposed_emotion": "calm",
                "transition": "content->calm",
                "same_valence": True,
                "same_arousal": True,
                "model_vote_count": 3,
                "support_score": 2.4,
                "public_duplicate_support_score": 0.0,
                "exact_duplicate": False,
                "near_duplicate": False,
                "failed_transition_count": 0,
                "risk_flags": "",
            },
            {
                "sample_id": "track2_0002",
                "current_emotion": "bored",
                "proposed_emotion": "calm",
                "transition": "bored->calm",
                "same_valence": True,
                "same_arousal": True,
                "model_vote_count": 3,
                "support_score": 2.3,
                "public_duplicate_support_score": 0.0,
                "exact_duplicate": False,
                "near_duplicate": False,
                "failed_transition_count": 0,
                "risk_flags": "",
            },
            {
                "sample_id": "track2_0003",
                "current_emotion": "content",
                "proposed_emotion": "calm",
                "transition": "content->calm",
                "same_valence": True,
                "same_arousal": True,
                "model_vote_count": 3,
                "support_score": 2.2,
                "public_duplicate_support_score": 0.0,
                "exact_duplicate": False,
                "near_duplicate": False,
                "failed_transition_count": 0,
                "risk_flags": "",
            },
        ]

        selected = select_v18_changes(rows, current_distribution={"bored": 4, "content": 10, "calm": 100})

        self.assertEqual([row["sample_id"] for row in selected], ["track2_0001", "track2_0003"])

    def test_v18_gate_exact_duplicate_requires_high_public_duplicate_support(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "current_emotion": "content",
            "proposed_emotion": "calm",
            "transition": "content->calm",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 0,
            "support_score": 0.2,
            "public_duplicate_support_score": 0.5,
            "exact_duplicate": True,
            "near_duplicate": False,
            "failed_transition_count": 0,
            "risk_flags": "",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "calm": 490})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("insufficient_model_families", gate["reasons"])
        self.assertIn("low_support_score", gate["reasons"])
        self.assertNotIn("exact_duplicate_override", gate["reasons"])

    def test_v18_gate_blocks_failed_transition_only_outside_majority_boundary(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        blocked = {
            "sample_id": "track2_0001",
            "current_emotion": "content",
            "proposed_emotion": "annoyed",
            "transition": "content->annoyed",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 3,
            "support_score": 2.0,
            "public_duplicate_support_score": 0.0,
            "exact_duplicate": False,
            "near_duplicate": False,
            "failed_transition_count": 12,
            "risk_flags": "failed_official_transition",
        }
        allowed = {
            **blocked,
            "sample_id": "track2_0002",
            "proposed_emotion": "calm",
            "transition": "content->calm",
            "failed_transition_count": 43,
        }

        blocked_gate = v18_gate_for_evidence(blocked, distribution={"content": 200, "annoyed": 30, "calm": 490})
        allowed_gate = v18_gate_for_evidence(allowed, distribution={"content": 200, "annoyed": 30, "calm": 490})

        self.assertEqual(blocked_gate["decision"], "block")
        self.assertIn("failed_official_transition_family", blocked_gate["reasons"])
        self.assertEqual(allowed_gate["decision"], "accept")
        self.assertNotIn("failed_official_transition_family", allowed_gate["reasons"])

    def test_v18_gate_blocks_exact_duplicate_with_invalid_label(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content->joyful",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 0,
            "support_score": 0.2,
            "public_duplicate_support_score": 1.0,
            "exact_duplicate": True,
            "near_duplicate": False,
            "failed_transition_count": 0,
            "risk_flags": "",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "joyful": 10})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("invalid_transition", gate["reasons"])
        self.assertNotIn("exact_duplicate_override", gate["reasons"])

    def test_v18_gate_blocks_exact_duplicate_with_malformed_transition(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content-calm",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 0,
            "support_score": 0.2,
            "public_duplicate_support_score": 1.0,
            "exact_duplicate": True,
            "near_duplicate": False,
            "failed_transition_count": 0,
            "risk_flags": "",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "calm": 490})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("invalid_transition", gate["reasons"])
        self.assertNotIn("exact_duplicate_override", gate["reasons"])

    def test_v18_gate_blocks_exact_duplicate_no_op_transition(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content->content",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 0,
            "support_score": 0.2,
            "public_duplicate_support_score": 1.0,
            "exact_duplicate": True,
            "near_duplicate": False,
            "failed_transition_count": 0,
            "risk_flags": "",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 200})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("no_label_change", gate["reasons"])
        self.assertNotIn("exact_duplicate_override", gate["reasons"])

    def test_v18_gate_blocks_exact_duplicate_rare_class_removal(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "bored->calm",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 0,
            "support_score": 0.2,
            "public_duplicate_support_score": 1.0,
            "exact_duplicate": True,
            "near_duplicate": False,
            "failed_transition_count": 0,
            "risk_flags": "",
        }

        gate = v18_gate_for_evidence(row, distribution={"bored": 3, "calm": 100})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("rare_current_class_floor", gate["reasons"])
        self.assertNotIn("exact_duplicate_override", gate["reasons"])

    def test_v18_gate_blocks_exact_duplicate_failed_non_majority_transition(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content->annoyed",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 0,
            "support_score": 0.2,
            "public_duplicate_support_score": 1.0,
            "exact_duplicate": True,
            "near_duplicate": False,
            "failed_transition_count": 12,
            "risk_flags": "failed_official_transition",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "annoyed": 30})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("failed_official_transition_family", gate["reasons"])
        self.assertNotIn("exact_duplicate_override", gate["reasons"])

    def test_v18_gate_infers_missing_current_and_proposed_from_transition(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content->calm",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 3,
            "support_score": 2.0,
            "public_duplicate_support_score": 0.0,
            "exact_duplicate": False,
            "near_duplicate": False,
            "failed_transition_count": 0,
            "risk_flags": "",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 3, "calm": 490})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("rare_current_class_floor", gate["reasons"])
