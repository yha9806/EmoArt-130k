from __future__ import annotations

import copy
import csv
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
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


def _selectable_transition_row(
    sample_id: str,
    transition: str,
    *,
    support_score: float = 2.0,
    transition_family: str | None = None,
) -> dict[str, object]:
    current, proposed = transition.split("->", maxsplit=1)
    row: dict[str, object] = {
        "sample_id": sample_id,
        "current_emotion": current,
        "proposed_emotion": proposed,
        "transition": transition,
        "same_valence": True,
        "same_arousal": True,
        "model_vote_count": 3,
        "support_score": support_score,
        "public_duplicate_support_score": 0.0,
        "exact_duplicate": False,
        "near_duplicate": False,
        "failed_transition_count": 0,
        "risk_flags": "",
    }
    if transition_family is not None:
        row["transition_family"] = transition_family
    return row


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
                "proposed_emotion": "annoyed",
                "transition": "content->annoyed",
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

    def test_v18_gate_derives_quadrant_from_labels_not_stale_metadata(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content->annoyed",
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

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "annoyed": 30})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("unsupported_cross_quadrant", gate["reasons"])

    def test_v18_gate_blocks_nan_support_score(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content->calm",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": 3,
            "support_score": "nan",
            "public_duplicate_support_score": 0.0,
            "exact_duplicate": False,
            "near_duplicate": False,
            "failed_transition_count": 0,
            "risk_flags": "",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "calm": 490})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("low_support_score", gate["reasons"])

    def test_v18_gate_handles_inf_integer_fields_fail_closed(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "transition": "content->calm",
            "same_valence": True,
            "same_arousal": True,
            "model_vote_count": "inf",
            "support_score": 2.0,
            "public_duplicate_support_score": 0.0,
            "exact_duplicate": False,
            "near_duplicate": False,
            "failed_transition_count": "inf",
            "risk_flags": "",
        }

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "calm": 490})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("insufficient_model_families", gate["reasons"])

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

    def test_select_v18_changes_enforces_family_from_transition_not_stale_metadata(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import select_v18_changes

        rows = [
            _selectable_transition_row(
                f"track2_{index:04d}",
                "content->calm",
                support_score=3.0 - index * 0.1,
                transition_family=f"stale-uncapped-{index}",
            )
            for index in range(5)
        ]

        selected = select_v18_changes(rows, current_distribution={"content": 238, "calm": 490})

        self.assertEqual(len(selected), 3)
        self.assertTrue(all(row["transition_family"] == "calm<->content" for row in selected))

    def test_select_v18_changes_caps_content_glad_in_both_directions(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import select_v18_changes

        rows = [
            _selectable_transition_row("track2_0001", "content->glad", support_score=3.0),
            _selectable_transition_row("track2_0002", "glad->content", support_score=2.9),
            _selectable_transition_row("track2_0003", "content->glad", support_score=2.8),
            _selectable_transition_row("track2_0004", "glad->content", support_score=2.7),
            _selectable_transition_row("track2_0005", "glad->content", support_score=2.6),
        ]

        selected = select_v18_changes(rows, current_distribution={"content": 238, "glad": 177})

        self.assertEqual(len(selected), 3)
        self.assertEqual(
            [row["sample_id"] for row in selected],
            ["track2_0001", "track2_0002", "track2_0003"],
        )
        self.assertTrue(all(row["transition_family"] == "content<->glad" for row in selected))

    def test_select_v18_changes_respects_total_cap(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import select_v18_changes

        rows = [
            _selectable_transition_row("track2_0001", "content->glad", support_score=3.0),
            _selectable_transition_row("track2_0002", "calm->content", support_score=2.9),
            _selectable_transition_row("track2_0003", "happy->excited", support_score=2.8),
            _selectable_transition_row("track2_0004", "sad->tired", support_score=2.7),
        ]

        selected = select_v18_changes(
            rows,
            current_distribution={"content": 238, "glad": 177, "calm": 490, "happy": 210, "sad": 220},
            total_cap=2,
        )

        self.assertEqual([row["sample_id"] for row in selected], ["track2_0001", "track2_0002"])

    def test_select_v18_changes_does_not_mutate_input_rows(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import select_v18_changes

        rows = [
            _selectable_transition_row(
                "track2_0001",
                "content->calm",
                support_score=3.0,
                transition_family="stale-uncapped",
            )
        ]
        original_rows = copy.deepcopy(rows)

        selected = select_v18_changes(rows, current_distribution={"content": 238, "calm": 490})

        self.assertEqual(rows, original_rows)
        self.assertEqual(selected[0]["transition_family"], "calm<->content")
        self.assertEqual(rows[0]["transition_family"], "stale-uncapped")

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

    def test_v18_gate_blocks_invalid_transition_label_even_when_fields_are_valid(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "current_emotion": "content",
            "proposed_emotion": "calm",
            "transition": "content->joyful",
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

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "calm": 490})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("invalid_transition", gate["reasons"])

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

    def test_v18_gate_blocks_no_op_transition_even_when_fields_change(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "current_emotion": "content",
            "proposed_emotion": "calm",
            "transition": "content->content",
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

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "calm": 490})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("no_label_change", gate["reasons"])

    def test_v18_gate_blocks_transition_label_mismatch(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import v18_gate_for_evidence

        row = {
            "sample_id": "track2_0001",
            "current_emotion": "content",
            "proposed_emotion": "glad",
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

        gate = v18_gate_for_evidence(row, distribution={"content": 200, "calm": 490, "glad": 100})

        self.assertEqual(gate["decision"], "block")
        self.assertIn("transition_label_mismatch", gate["reasons"])

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

    def test_write_v18_candidate_outputs_writes_side_path_zip_and_report(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import write_v18_candidate_outputs

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            text_json = root / "text.json"
            out_json = root / "track2_submission_v18_champion_hybrid_candidate.json"
            out_zip = root / "track2_submission_v18_champion_hybrid_candidate.zip"
            report_json = root / "candidate_report.json"
            report_md = root / "candidate_report.md"
            row = {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "A calm room.",
                "brushstroke": "Soft strokes.",
                "composition": "Balanced composition.",
                "color": "Muted color.",
                "line": "Gentle line.",
                "light": "Soft light.",
            }
            base_json.write_text(
                json.dumps([row] + [{**row, "sample_id": f"track2_{index:04d}"} for index in range(2, 6)]),
                encoding="utf-8",
            )
            text_json.write_text(
                json.dumps(
                    [
                        {
                            **row,
                            "overall_caption": (
                                "A quiet interior uses muted color and balanced space "
                                "to create contentment."
                            ),
                        }
                    ]
                ),
                encoding="utf-8",
            )
            changes = [
                {
                    "sample_id": "track2_0001",
                    "current_emotion": "content",
                    "proposed_emotion": "calm",
                    "transition": "content->calm",
                    "support_score": 2.0,
                    "max_confidence": 0.9,
                    "model_vote_count": 3,
                    "model_sources": "clip,dinov2,siglip2",
                    "all_sources": "clip,dinov2,siglip2",
                    "gate_decision": "accept",
                    "gate_reasons": "meets_v18_thresholds",
                    "rationale": "strong visual calm cues",
                }
            ]

            report = write_v18_candidate_outputs(
                base_json=base_json,
                text_json=text_json,
                changes=changes,
                out_json=out_json,
                out_zip=out_zip,
                report_json=report_json,
                report_md=report_md,
            )

            self.assertTrue(out_json.exists())
            self.assertTrue(out_zip.exists())
            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())
            rows = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["emotion"], "calm")
            self.assertEqual(rows[0]["emotional_valence"], "Positive")
            self.assertEqual(rows[0]["emotional_arousal_level"], "Low")
            self.assertIn("quiet interior", rows[0]["overall_caption"])
            self.assertEqual(report["accepted_label_changes"], 1)
            self.assertEqual(report["description_merge"]["description_changed_rows"], 1)
            self.assertEqual(report["label_consistency_issue_count"], 0)
            with zipfile.ZipFile(out_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
                self.assertEqual(archive.getinfo("submission.json").date_time, (1980, 1, 1, 0, 0, 0))

    def test_write_v18_candidate_outputs_requires_explicit_accept_gate(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import write_v18_candidate_outputs

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            out_json = root / "candidate.json"
            out_zip = root / "candidate.zip"
            report_json = root / "candidate_report.json"
            report_md = root / "candidate_report.md"
            row = {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "A calm room.",
                "brushstroke": "Soft strokes.",
                "composition": "Balanced composition.",
                "color": "Muted color.",
                "line": "Gentle line.",
                "light": "Soft light.",
            }
            base_json.write_text(
                json.dumps([row] + [{**row, "sample_id": f"track2_{index:04d}"} for index in range(2, 6)]),
                encoding="utf-8",
            )

            report = write_v18_candidate_outputs(
                base_json=base_json,
                text_json=None,
                changes=[
                    {
                        "sample_id": "track2_0001",
                        "current_emotion": "content",
                        "proposed_emotion": "calm",
                        "transition": "content->calm",
                    }
                ],
                out_json=out_json,
                out_zip=out_zip,
                report_json=report_json,
                report_md=report_md,
            )

            rows = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["emotion"], "content")
            self.assertEqual(report["accepted_label_changes"], 0)

    def test_write_v18_candidate_outputs_blocks_formal_submission_name(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import write_v18_candidate_outputs

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            base_json.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "formal submission"):
                write_v18_candidate_outputs(
                    base_json=base_json,
                    text_json=None,
                    changes=[],
                    out_json=root / "track2_submission.json",
                    out_zip=root / "candidate.zip",
                    report_json=root / "candidate_report.json",
                    report_md=root / "candidate_report.md",
                )

    def test_write_v18_candidate_outputs_blocks_formal_report_name(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import write_v18_candidate_outputs

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            base_json.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "formal submission"):
                write_v18_candidate_outputs(
                    base_json=base_json,
                    text_json=None,
                    changes=[],
                    out_json=root / "candidate.json",
                    out_zip=root / "candidate.zip",
                    report_json=root / "track2_submission.json",
                    report_md=root / "candidate_report.md",
                )

    def test_write_v18_candidate_outputs_blocks_case_variant_formal_name(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import write_v18_candidate_outputs

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            base_json.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "formal submission"):
                write_v18_candidate_outputs(
                    base_json=base_json,
                    text_json=None,
                    changes=[],
                    out_json=root / "Track2_Submission.json",
                    out_zip=root / "candidate.zip",
                    report_json=root / "candidate_report.json",
                    report_md=root / "candidate_report.md",
                )

    def test_load_track2_rows_rejects_invalid_labels(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import load_track2_rows

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "track2_0001",
                            "emotion": "joyful",
                            "emotional_valence": "Positive",
                            "emotional_arousal_level": "Low",
                            "overall_caption": "A caption.",
                            "brushstroke": "Brush.",
                            "composition": "Composition.",
                            "color": "Color.",
                            "line": "Line.",
                            "light": "Light.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "invalid emotion"):
                load_track2_rows(path)

    def test_load_track2_rows_rejects_duplicate_sample_id(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import load_track2_rows

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            row = {
                "sample_id": "track2_0001",
                "emotion": "calm",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "A caption.",
                "brushstroke": "Brush.",
                "composition": "Composition.",
                "color": "Color.",
                "line": "Line.",
                "light": "Light.",
            }
            path.write_text(json.dumps([row, row]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                load_track2_rows(path)

    def test_write_v18_candidate_outputs_does_not_import_untracked_pipeline_modules(self) -> None:
        code = (
            "import sys\n"
            "import affectiveart.track2_v18_champion_hybrid\n"
            "blocked = {'affectiveart.challenge', 'affectiveart.track2_audit', "
            "'affectiveart.track2_v17_classification_calibration'}\n"
            "raise SystemExit(1 if blocked & set(sys.modules) else 0)\n"
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

    def test_choose_v18_final_gate_recommends_only_when_proxy_improves(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import choose_v18_final_gate

        gate = choose_v18_final_gate(
            candidate_report={
                "accepted_label_changes": 12,
                "label_consistency_issue_count": 0,
                "description_merge": {"description_changed_rows": 80},
            },
            fused_row={
                "candidate_name": "v18_champion_hybrid",
                "overall_lower": "0.8370",
                "classification_lower": "0.7240",
                "description_lower": "0.9500",
                "decision": "recommend_submit",
            },
            anchor={"overall": 0.836408, "classification": 0.723150, "description": 0.949667},
            emotion_accuracy_proxy_delta=0.025,
        )

        self.assertEqual(gate["decision"], "recommend_submit_v18")
        self.assertEqual(gate["submission_budget_policy"], "first_of_two_remaining")

    def test_choose_v18_final_gate_holds_when_emotion_proxy_does_not_improve(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import choose_v18_final_gate

        gate = choose_v18_final_gate(
            candidate_report={
                "accepted_label_changes": 0,
                "label_consistency_issue_count": 0,
                "description_merge": {"description_changed_rows": 80},
            },
            fused_row={
                "candidate_name": "v18_champion_hybrid",
                "overall_lower": "0.8370",
                "classification_lower": "0.7240",
                "description_lower": "0.9500",
                "decision": "recommend_submit",
            },
            anchor={"overall": 0.836408, "classification": 0.723150, "description": 0.949667},
            emotion_accuracy_proxy_delta=0.0,
        )

        self.assertEqual(gate["decision"], "hold_keep_anchor")
        self.assertIn("emotion_accuracy_proxy_not_improved", gate["reasons"])

    def test_choose_v18_final_gate_holds_on_nan_proxy_metrics(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import choose_v18_final_gate

        gate = choose_v18_final_gate(
            candidate_report={
                "accepted_label_changes": 12,
                "label_consistency_issue_count": 0,
                "description_merge": {"description_changed_rows": 80},
            },
            fused_row={
                "candidate_name": "v18_champion_hybrid",
                "overall_lower": "nan",
                "classification_lower": "nan",
                "description_lower": "nan",
                "decision": "recommend_submit",
            },
            anchor={"overall": 0.836408, "classification": 0.723150, "description": 0.949667},
            emotion_accuracy_proxy_delta=float("nan"),
        )

        self.assertEqual(gate["decision"], "hold_keep_anchor")
        self.assertIn("description_lower_below_anchor", gate["reasons"])
        self.assertIn("classification_lower_below_tolerance", gate["reasons"])
        self.assertIn("emotion_accuracy_proxy_not_improved", gate["reasons"])

    def test_choose_v18_final_gate_holds_when_description_is_not_maximized(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import choose_v18_final_gate

        gate = choose_v18_final_gate(
            candidate_report={
                "accepted_label_changes": 12,
                "label_consistency_issue_count": 0,
                "description_merge": {"description_changed_rows": 0},
            },
            fused_row={
                "candidate_name": "v18_champion_hybrid",
                "overall_lower": "0.8370",
                "classification_lower": "0.7240",
                "description_lower": "0.9500",
                "decision": "recommend_submit",
            },
            anchor={"overall": 0.836408, "classification": 0.723150, "description": 0.949667},
            emotion_accuracy_proxy_delta=0.025,
        )

        self.assertEqual(gate["decision"], "hold_keep_anchor")
        self.assertIn("description_not_maximized", gate["reasons"])

    def test_choose_v18_final_gate_holds_when_description_change_count_is_negative(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import choose_v18_final_gate

        gate = choose_v18_final_gate(
            candidate_report={
                "accepted_label_changes": 12,
                "label_consistency_issue_count": 0,
                "description_merge": {"description_changed_rows": -1},
            },
            fused_row={
                "candidate_name": "v18_champion_hybrid",
                "overall_lower": "0.8370",
                "classification_lower": "0.7240",
                "description_lower": "0.9500",
                "decision": "recommend_submit",
            },
            anchor={"overall": 0.836408, "classification": 0.723150, "description": 0.949667},
            emotion_accuracy_proxy_delta=0.025,
        )

        self.assertEqual(gate["decision"], "hold_keep_anchor")
        self.assertIn("description_not_maximized", gate["reasons"])

    def test_choose_v18_final_gate_holds_when_overall_proxy_is_below_anchor(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import choose_v18_final_gate

        gate = choose_v18_final_gate(
            candidate_report={
                "accepted_label_changes": 12,
                "label_consistency_issue_count": 0,
                "description_merge": {"description_changed_rows": 80},
            },
            fused_row={
                "candidate_name": "v18_champion_hybrid",
                "overall_lower": "0.8360",
                "classification_lower": "0.7240",
                "description_lower": "0.9500",
                "decision": "recommend_submit",
            },
            anchor={"overall": 0.836408, "classification": 0.723150, "description": 0.949667},
            emotion_accuracy_proxy_delta=0.025,
        )

        self.assertEqual(gate["decision"], "hold_keep_anchor")
        self.assertIn("overall_lower_not_above_anchor", gate["reasons"])

    def test_write_v18_final_gate_report_writes_json_and_markdown(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import write_v18_final_gate_report

        report = {
            "method": "track2_v18_final_gate_v1",
            "decision": "recommend_submit_v18",
            "submission_budget_policy": "first_of_two_remaining",
            "reasons": ["passes_v18_two_submission_gate"],
            "candidate_name": "v18_champion_hybrid",
            "candidate_overall_lower": 0.837,
            "anchor_overall": 0.836408,
            "emotion_accuracy_proxy_delta": 0.025,
            "caveat": "Local scorer caveat.",
            "no_auto_submit": True,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_json = root / "final_gate_report.json"
            out_md = root / "final_gate_report.md"

            write_v18_final_gate_report(report, out_json, out_md)

            self.assertEqual(json.loads(out_json.read_text(encoding="utf-8"))["decision"], "recommend_submit_v18")
            self.assertIn("Track2 v18 Final Gate", out_md.read_text(encoding="utf-8"))

    def test_main_build_writes_candidate_and_reports(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import main

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            text_json = root / "text.json"
            out_json = root / "candidate.json"
            out_zip = root / "candidate.zip"
            exp_dir = root / "experiment"
            row = {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "A calm room.",
                "brushstroke": "Soft strokes.",
                "composition": "Balanced composition.",
                "color": "Muted color.",
                "line": "Gentle line.",
                "light": "Soft light.",
            }
            base_rows = [row] + [
                {**row, "sample_id": f"track2_{index:04d}"}
                for index in range(2, 6)
            ]
            base_json.write_text(json.dumps(base_rows), encoding="utf-8")
            text_json.write_text(json.dumps([row]), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "build",
                        "--base-json",
                        str(base_json),
                        "--text-json",
                        str(text_json),
                        "--out-json",
                        str(out_json),
                        "--out-zip",
                        str(out_zip),
                        "--experiment-dir",
                        str(exp_dir),
                        "--skip-evidence",
                    ]
                )

            self.assertTrue(out_json.exists())
            self.assertTrue(out_zip.exists())
            self.assertTrue((exp_dir / "candidate_report.json").exists())

    def test_main_build_loads_v17_accepted_changes_report(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import main

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            changes_json = root / "changes.json"
            out_json = root / "candidate.json"
            out_zip = root / "candidate.zip"
            exp_dir = root / "experiment"
            row = {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "A calm room.",
                "brushstroke": "Soft strokes.",
                "composition": "Balanced composition.",
                "color": "Muted color.",
                "line": "Gentle line.",
                "light": "Soft light.",
            }
            base_rows = [row] + [
                {**row, "sample_id": f"track2_{index:04d}"}
                for index in range(2, 6)
            ]
            base_json.write_text(json.dumps(base_rows), encoding="utf-8")
            changes_json.write_text(
                json.dumps(
                    {
                        "accepted_changes": [
                            {
                                "sample_id": "track2_0001",
                                "transition": "content->calm",
                                "before": {"emotion": "content"},
                                "after": {"emotion": "calm"},
                                "support_score": 2.0,
                                "model_vote_count": 3,
                                "gate_decision": "accept",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "build",
                        "--base-json",
                        str(base_json),
                        "--changes-json",
                        str(changes_json),
                        "--out-json",
                        str(out_json),
                        "--out-zip",
                        str(out_zip),
                        "--experiment-dir",
                        str(exp_dir),
                    ]
                )

            rows = json.loads(out_json.read_text(encoding="utf-8"))
            report = json.loads((exp_dir / "candidate_report.json").read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["emotion"], "calm")
            self.assertEqual(report["accepted_label_changes"], 1)

    def test_main_build_rechecks_external_changes_with_v18_gate(self) -> None:
        from affectiveart.track2_v18_champion_hybrid import main

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            changes_json = root / "changes.json"
            out_json = root / "candidate.json"
            out_zip = root / "candidate.zip"
            exp_dir = root / "experiment"
            base_rows = [
                {
                    "sample_id": "track2_0001",
                    "emotion": "content",
                    "emotional_valence": "Positive",
                    "emotional_arousal_level": "Low",
                    "overall_caption": "A calm room.",
                    "brushstroke": "Soft strokes.",
                    "composition": "Balanced composition.",
                    "color": "Muted color.",
                    "line": "Gentle line.",
                    "light": "Soft light.",
                },
                {
                    "sample_id": "track2_0002",
                    "emotion": "content",
                    "emotional_valence": "Positive",
                    "emotional_arousal_level": "Low",
                    "overall_caption": "A quiet room.",
                    "brushstroke": "Soft strokes.",
                    "composition": "Balanced composition.",
                    "color": "Muted color.",
                    "line": "Gentle line.",
                    "light": "Soft light.",
                },
            ]
            base_rows.extend(
                {
                    "sample_id": f"track2_{index:04d}",
                    "emotion": "content",
                    "emotional_valence": "Positive",
                    "emotional_arousal_level": "Low",
                    "overall_caption": "A quiet room.",
                    "brushstroke": "Soft strokes.",
                    "composition": "Balanced composition.",
                    "color": "Muted color.",
                    "line": "Gentle line.",
                    "light": "Soft light.",
                }
                for index in range(3, 7)
            )
            base_json.write_text(json.dumps(base_rows), encoding="utf-8")
            changes_json.write_text(
                json.dumps(
                    {
                        "accepted_changes": [
                            {
                                "sample_id": "track2_0001",
                                "transition": "content->calm",
                                "before": {"emotion": "content"},
                                "after": {"emotion": "calm"},
                                "support_score": 2.0,
                                "model_vote_count": 3,
                                "gate_decision": "accept",
                            },
                            {
                                "sample_id": "track2_0002",
                                "transition": "content->annoyed",
                                "before": {"emotion": "content"},
                                "after": {"emotion": "annoyed"},
                                "support_score": 2.0,
                                "model_vote_count": 3,
                                "gate_decision": "accept",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "build",
                        "--base-json",
                        str(base_json),
                        "--changes-json",
                        str(changes_json),
                        "--out-json",
                        str(out_json),
                        "--out-zip",
                        str(out_zip),
                        "--experiment-dir",
                        str(exp_dir),
                    ]
                )

            rows = {row["sample_id"]: row for row in json.loads(out_json.read_text(encoding="utf-8"))}
            selected = json.loads((exp_dir / "selected_changes.json").read_text(encoding="utf-8"))
            self.assertEqual(rows["track2_0001"]["emotion"], "calm")
            self.assertEqual(rows["track2_0002"]["emotion"], "content")
            self.assertEqual([row["sample_id"] for row in selected], ["track2_0001"])
