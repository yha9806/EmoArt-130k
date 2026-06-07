from __future__ import annotations

import csv
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.challenge import TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_v17_classification_calibration import (
    apply_v17_changes,
    build_evidence_rows,
    choose_final_gate,
    DEFAULT_PREDICTION_SOURCES,
    load_prediction_sources,
    main,
    parse_official_failed_transition_counts,
    render_final_gate_markdown,
    select_v17_changes,
    v17_gate_for_evidence,
    write_final_gate_report,
    write_v17_candidate_outputs,
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


def _track2_row(**overrides):
    row = {
        "sample_id": "track2_0001",
        "emotion": "content",
        "emotional_valence": "Positive",
        "emotional_arousal_level": "Low",
        "overall_caption": "A quiet figure beside a window.",
        "brushstroke": "Soft layered strokes.",
        "composition": "Centered subject with open space.",
        "color": "Muted green and gold.",
        "line": "Gentle curved lines.",
        "light": "Diffuse afternoon light.",
    }
    row.update(overrides)
    return row


def _ranking_row(candidate_name: str, overall_lower_value: float, **overrides):
    row = {
        "candidate_name": candidate_name,
        "candidate_json": f"submissions/track2_submission_{candidate_name}_candidate.json",
        "overall_expected": overall_lower_value + 0.003,
        "overall_lower": overall_lower_value,
        "overall_upper": overall_lower_value + 0.012,
        "classification_lower": 0.72,
        "description_lower": 0.94,
        "decision": "recommend_submit",
    }
    row.update(overrides)
    return row


def _write_ranking_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class Track2V17ClassificationCalibrationTests(unittest.TestCase):
    def test_choose_final_gate_prefers_v17_candidate_when_lower_bound_improves(self) -> None:
        gate = choose_final_gate(
            [
                _ranking_row("v15_desc_expand300", 0.8334),
                _ranking_row("v17_safe", 0.8335),
                _ranking_row("v17_balanced", 0.8341),
            ]
        )

        self.assertEqual(gate["decision"], "recommend_submit_v17")
        self.assertEqual(gate["baseline_name"], "v15_desc_expand300")
        self.assertEqual(gate["best_v17_name"], "v17_balanced")
        self.assertEqual(gate["candidate_name"], "v17_balanced")
        self.assertEqual(gate["selected_candidate_name"], "v17_balanced")
        self.assertAlmostEqual(gate["baseline_lower"], 0.8334)
        self.assertAlmostEqual(gate["candidate_lower"], 0.8341)
        self.assertTrue(gate["lower_bound_improved"])
        self.assertEqual(gate["submission_policy"], "no_auto_submit")

    def test_choose_final_gate_holds_v15_without_v17_lower_bound_improvement(self) -> None:
        gate = choose_final_gate(
            [
                _ranking_row("v15_desc_expand300", 0.8334),
                _ranking_row("v17_safe", 0.8334),
                _ranking_row("v17_balanced", 0.8331),
            ]
        )

        self.assertEqual(gate["decision"], "hold_v17_keep_v15")
        self.assertEqual(gate["best_v17_name"], "v17_safe")
        self.assertEqual(gate["candidate_name"], "v15_desc_expand300")
        self.assertEqual(gate["selected_candidate_name"], "v15_desc_expand300")
        self.assertAlmostEqual(gate["baseline_lower"], 0.8334)
        self.assertAlmostEqual(gate["candidate_lower"], 0.8334)
        self.assertFalse(gate["lower_bound_improved"])
        self.assertIn("does not strictly exceed", gate["reason"])

    def test_choose_final_gate_raises_when_baseline_missing(self) -> None:
        with self.assertRaises(ValueError):
            choose_final_gate([_ranking_row("v17_safe", 0.8341)])

    def test_choose_final_gate_ignores_non_v17_challengers(self) -> None:
        gate = choose_final_gate(
            [
                _ranking_row("v15_desc_expand300", 0.8334),
                _ranking_row("v16_top48", 0.84),
                _ranking_row("v17_safe", 0.8330),
            ]
        )

        self.assertEqual(gate["decision"], "hold_v17_keep_v15")
        self.assertEqual(gate["best_v17_name"], "v17_safe")
        self.assertAlmostEqual(gate["candidate_lower"], 0.8334)

    def test_choose_final_gate_raises_when_baseline_lower_is_malformed(self) -> None:
        with self.assertRaisesRegex(ValueError, "v15_desc_expand300.*overall_lower"):
            choose_final_gate(
                [
                    _ranking_row("v15_desc_expand300", 0.8334, overall_lower="nan"),
                    _ranking_row("v17_safe", 0.8341),
                ]
            )

    def test_choose_final_gate_skips_malformed_v17_lower_and_records_reason(self) -> None:
        gate = choose_final_gate(
            [
                _ranking_row("v15_desc_expand300", 0.8334),
                _ranking_row("v17_bad", 0.9, overall_lower="not-a-number"),
                _ranking_row("v17_safe", 0.8333),
            ]
        )

        self.assertEqual(gate["decision"], "hold_v17_keep_v15")
        self.assertEqual(gate["best_v17_name"], "v17_safe")
        self.assertEqual(gate["candidate_name"], "v15_desc_expand300")
        self.assertEqual(gate["skipped_v17_rows"][0]["candidate_name"], "v17_bad")
        self.assertIn("overall_lower", gate["skipped_v17_rows"][0]["reason"])

    def test_write_final_gate_report_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ranking_csv = root / "candidate_ranking.csv"
            out_md = root / "final_gate_report.md"
            out_json = root / "final_gate_report.json"
            rows = [
                _ranking_row("v15_desc_expand300", 0.8334),
                _ranking_row("v17_balanced", 0.8341),
            ]
            _write_ranking_csv(ranking_csv, rows)

            report = write_final_gate_report(ranking_csv, out_md, out_json)

            self.assertEqual(report["decision"], "recommend_submit_v17")
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["best_v17_name"], "v17_balanced")
            self.assertEqual(payload["candidate_name"], "v17_balanced")
            self.assertEqual(payload["selected_candidate_name"], "v17_balanced")
            markdown = out_md.read_text(encoding="utf-8")
            self.assertIn("# Track2 v17 Final Gate", markdown)
            self.assertIn("- Candidate: `v17_balanced`", markdown)
            self.assertIn("local/fused shadow gate", markdown)
            self.assertIn("no_auto_submit", markdown)

    def test_render_final_gate_markdown_includes_full_fused_ranking_rows(self) -> None:
        rows = [
            _ranking_row("v15_desc_expand300", 0.8334),
            _ranking_row("official_779605_moe_v2_anchor", 0.8334),
            _ranking_row("v16_top48", 0.84),
            _ranking_row("v17_safe", 0.8332),
            _ranking_row("v17_balanced", 0.8341),
        ]
        report = choose_final_gate(rows)

        markdown = render_final_gate_markdown(report, rows)

        self.assertIn("recommend_submit_v17", markdown)
        self.assertIn("## Fused Ranking", markdown)
        self.assertIn("official_779605_moe_v2_anchor", markdown)
        self.assertIn("v16_top48", markdown)
        self.assertIn("v15_desc_expand300", markdown)
        self.assertLess(markdown.index("v17_balanced"), markdown.index("v17_safe"))

    def test_main_score_gate_writes_explicit_report_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ranking_csv = root / "candidate_ranking.csv"
            out_md = root / "final_gate_report.md"
            out_json = root / "final_gate_report.json"
            _write_ranking_csv(
                ranking_csv,
                [
                    _ranking_row("v15_desc_expand300", 0.8334),
                    _ranking_row("v17_safe", 0.8341),
                ],
            )

            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "score-gate",
                        "--fused-ranking-csv",
                        str(ranking_csv),
                        "--out-md",
                        str(out_md),
                        "--out-json",
                        str(out_json),
                    ]
                )

            self.assertTrue(out_md.exists())
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["best_v17_name"], "v17_safe")
            self.assertEqual(payload["selected_candidate_name"], "v17_safe")

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

    def test_apply_v17_changes_repairs_valence_arousal_for_proposed_emotion(self) -> None:
        candidate_rows, report = apply_v17_changes(
            [_track2_row()],
            [_evidence_row(proposed_emotion="alarmed", transition="content->alarmed")],
            profile="safe",
        )
        self.assertEqual(candidate_rows[0]["emotion"], "alarmed")
        self.assertEqual(candidate_rows[0]["emotional_valence"], "Negative")
        self.assertEqual(candidate_rows[0]["emotional_arousal_level"], "High")
        self.assertEqual(report["accepted_label_changes"], 1)
        self.assertEqual(report["label_consistency_issue_count"], 0)

    def test_apply_v17_changes_ignores_no_op_proposed_emotion(self) -> None:
        candidate_rows, report = apply_v17_changes(
            [_track2_row()],
            [_evidence_row(proposed_emotion="content", transition="content->content")],
            profile="safe",
        )
        self.assertEqual(candidate_rows[0]["emotion"], "content")
        self.assertEqual(report["accepted_label_changes"], 0)
        self.assertEqual(report["accepted_changes"], [])

    def test_apply_v17_changes_preserves_text_fields_and_submission_key_order(self) -> None:
        base = _track2_row(z_extra="ignored", overall_caption="Keep this exact caption.")
        candidate_rows, _report = apply_v17_changes(
            [base],
            [_evidence_row(proposed_emotion="sad", transition="content->sad")],
            profile="balanced",
        )
        self.assertEqual(list(candidate_rows[0]), TRACK2_JSON_SUBMISSION_KEYS)
        self.assertEqual(candidate_rows[0]["overall_caption"], "Keep this exact caption.")
        self.assertEqual(candidate_rows[0]["brushstroke"], base["brushstroke"])
        self.assertNotIn("z_extra", candidate_rows[0])

    def test_write_v17_candidate_outputs_writes_zip_with_submission_json_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            base_json.write_text(json.dumps([_track2_row()]), encoding="utf-8")
            out_json = root / "submissions" / "track2_submission_v17_safe_candidate.json"
            out_zip = root / "submissions" / "track2_submission_v17_safe_candidate.zip"
            report_json = root / "candidate_reports" / "track2_submission_v17_safe_candidate_report.json"
            report_md = root / "candidate_reports" / "track2_submission_v17_safe_candidate_report.md"

            report = write_v17_candidate_outputs(
                base_json=base_json,
                changes=[_evidence_row(proposed_emotion="alarmed", transition="content->alarmed")],
                profile="safe",
                out_json=out_json,
                out_zip=out_zip,
                report_json=report_json,
                report_md=report_md,
            )

            with zipfile.ZipFile(out_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
                self.assertEqual(archive.getinfo("submission.json").date_time, (1980, 1, 1, 0, 0, 0))
                zipped_payload = json.loads(archive.read("submission.json").decode("utf-8"))
            self.assertEqual(zipped_payload, json.loads(out_json.read_text(encoding="utf-8")))
            self.assertEqual(report["accepted_label_changes"], 1)
            self.assertFalse(report["formal_submission_overwritten"])
            self.assertEqual(report["paths"]["out_zip"], str(out_zip))

    def test_write_v17_candidate_outputs_rejects_formal_submission_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            base_json.write_text(json.dumps([_track2_row()]), encoding="utf-8")
            with self.assertRaises(ValueError):
                write_v17_candidate_outputs(
                    base_json=base_json,
                    changes=[],
                    profile="safe",
                    out_json=root / "track2_submission.json",
                    out_zip=root / "submissions" / "track2_submission_v17_safe_candidate.zip",
                    report_json=root / "candidate_reports" / "track2_submission_v17_safe_candidate_report.json",
                    report_md=root / "candidate_reports" / "track2_submission_v17_safe_candidate_report.md",
                )

    def test_write_v17_candidate_outputs_rejects_formal_report_md_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            base_json.write_text(json.dumps([_track2_row()]), encoding="utf-8")
            formal_report_md = root / "track2_submission.json"
            formal_report_md.write_text("do not overwrite\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_v17_candidate_outputs(
                    base_json=base_json,
                    changes=[],
                    profile="safe",
                    out_json=root / "submissions" / "track2_submission_v17_safe_candidate.json",
                    out_zip=root / "submissions" / "track2_submission_v17_safe_candidate.zip",
                    report_json=root / "candidate_reports" / "track2_submission_v17_safe_candidate_report.json",
                    report_md=formal_report_md,
                )
            self.assertEqual(formal_report_md.read_text(encoding="utf-8"), "do not overwrite\n")

    def test_write_v17_candidate_outputs_rejects_non_v17_report_side_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            base_json.write_text(json.dumps([_track2_row()]), encoding="utf-8")
            out_json = root / "submissions" / "track2_submission_v17_safe_candidate.json"
            out_zip = root / "submissions" / "track2_submission_v17_safe_candidate.zip"
            with self.assertRaises(ValueError):
                write_v17_candidate_outputs(
                    base_json=base_json,
                    changes=[],
                    profile="safe",
                    out_json=out_json,
                    out_zip=out_zip,
                    report_json=root / "candidate_reports" / "track2_submission_v16_safe_candidate_report.json",
                    report_md=root / "candidate_reports" / "track2_submission_v17_safe_candidate_report.md",
                )
            self.assertFalse(out_json.exists())
            self.assertFalse(out_zip.exists())

    def test_write_v17_candidate_outputs_rejects_non_v17_side_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_json = root / "base.json"
            base_json.write_text(json.dumps([_track2_row()]), encoding="utf-8")
            with self.assertRaises(ValueError):
                write_v17_candidate_outputs(
                    base_json=base_json,
                    changes=[],
                    profile="safe",
                    out_json=root / "submissions" / "track2_submission_v16_safe_candidate.json",
                    out_zip=root / "submissions" / "track2_submission_v17_safe_candidate.zip",
                    report_json=root / "candidate_reports" / "track2_submission_v17_safe_candidate_report.json",
                    report_md=root / "candidate_reports" / "track2_submission_v17_safe_candidate_report.md",
                )


if __name__ == "__main__":
    unittest.main()
