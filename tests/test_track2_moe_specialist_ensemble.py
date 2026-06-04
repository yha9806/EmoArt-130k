import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from PIL import Image

from affectiveart.challenge import TRACK2_JSON_EMOTIONS
from affectiveart.track2_moe_specialist_ensemble import (
    GateThresholds,
    VALID_TRACK2_EMOTIONS,
    build_dry_run_report,
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


def _write_track2_image_zip(image_zip, sample_ids):
    source_image = image_zip.parent / "source.jpg"
    Image.new("RGB", (64, 48), (190, 170, 140)).save(source_image)
    with zipfile.ZipFile(image_zip, "w") as zf:
        for sample_id in sample_ids:
            zf.write(source_image, f"track2_testset/images/{sample_id}.jpg")


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

    def test_build_dry_run_report_summarizes_gate_decisions(self):
        current_rows = [
            current_row(sample_id="track2_0001", emotion="content"),
            current_row(sample_id="track2_0002", emotion="content"),
        ]
        expert_rows = [
            expert("track2_0001", "clip_clean", "calm", role="global"),
            expert("track2_0001", "boundary_head", "calm", role="boundary"),
            expert("track2_0002", "clip_clean", "sad"),
        ]

        report = build_dry_run_report(
            current_rows,
            expert_rows,
            queue_sample_ids=["track2_0001", "track2_0002"],
            high_similarity_sample_ids=set(),
        )

        self.assertEqual(report["method"], "track2_moe_specialist_dry_run_v1")
        self.assertEqual(report["row_count"], 2)
        self.assertEqual(report["decision_counts"]["accept_change"], 1)
        self.assertEqual(report["decision_counts"]["hold"], 1)
        self.assertEqual(report["decision_counts"]["keep_current"], 0)
        self.assertEqual(report["missing_current_sample_ids"], [])
        self.assertIs(report["formal_submission_overwritten"], False)
        self.assertIs(report["candidate_json_written"], False)
        self.assertIs(report["candidate_zip_written"], False)
        rows_by_id = {row["sample_id"]: row for row in report["rows"]}
        self.assertEqual(rows_by_id["track2_0001"]["proposed_emotion"], "calm")
        self.assertEqual(rows_by_id["track2_0002"]["decision"], "hold")

    def test_build_dry_run_report_deduplicates_normalized_queue_ids(self):
        report = build_dry_run_report(
            [current_row(sample_id="track2_0001", emotion="content")],
            [
                expert("track2_0001", "clip_clean", "calm", role="global"),
                expert("track2_0001", "boundary_head", "calm", role="boundary"),
            ],
            queue_sample_ids=[" track2_0001 ", "track2_0001"],
        )

        self.assertEqual(report["row_count"], 1)
        self.assertEqual([row["sample_id"] for row in report["rows"]], ["track2_0001"])

    def test_build_dry_run_report_lists_missing_current_sample_ids(self):
        report = build_dry_run_report(
            [current_row(sample_id="track2_0001", emotion="content")],
            [],
            queue_sample_ids=[" track2_0001 ", " track2_9999 "],
        )

        self.assertEqual(report["row_count"], 1)
        self.assertEqual(report["missing_current_sample_ids"], ["track2_9999"])

    def test_build_dry_run_report_uses_stable_counts_for_all_hold_queue(self):
        report = build_dry_run_report(
            [current_row(sample_id="track2_0001", emotion="content")],
            [expert("track2_0001", "clip_clean", "sad")],
            queue_sample_ids=["track2_0001"],
        )

        self.assertEqual(
            report["decision_counts"],
            {"accept_change": 0, "hold": 1, "keep_current": 0},
        )
        self.assertEqual(report["accepted_transition_counts"], {})

    def test_build_dry_run_report_passes_safety_gate_controls(self):
        current_rows = [
            current_row(sample_id="track2_0001", emotion="content"),
            current_row(sample_id="track2_0002", emotion="content"),
        ]
        expert_rows = [
            expert("track2_0001", "clip_clean", "calm", role="global"),
            expert("track2_0001", "boundary_head", "calm", role="boundary"),
            expert("track2_0002", "clip_clean", "calm", role="global"),
            expert("track2_0002", "boundary_head", "calm", role="boundary"),
        ]

        report = build_dry_run_report(
            current_rows,
            expert_rows,
            queue_sample_ids=["track2_0001", "track2_0002"],
            high_similarity_sample_ids=[" track2_0001 "],
            description_audit_by_id={
                " track2_0002 ": {"verdict": "contradiction"}
            },
        )

        rows_by_id = {row["sample_id"]: row for row in report["rows"]}
        self.assertEqual(rows_by_id["track2_0001"]["decision"], "hold")
        self.assertIn(
            "high_similarity_requires_explicit_review",
            rows_by_id["track2_0001"]["reasons"],
        )
        self.assertEqual(rows_by_id["track2_0002"]["decision"], "hold")
        self.assertIn(
            "description_contradiction",
            rows_by_id["track2_0002"]["reasons"],
        )

    def test_write_dry_run_outputs_writes_review_artifacts_without_submission_files(self):
        try:
            from affectiveart.track2_moe_specialist_ensemble import write_dry_run_outputs
        except ImportError:
            self.fail("write_dry_run_outputs is missing")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_zip = tmp_path / "track2_images.zip"
            out_dir = tmp_path / "out"
            _write_track2_image_zip(image_zip, ["track2_0001"])

            report = build_dry_run_report(
                [current_row(sample_id="track2_0001", emotion="content")],
                [
                    expert("track2_0001", "clip_clean", "calm", role="global"),
                    expert("track2_0001", "boundary_head", "calm", role="boundary"),
                ],
                queue_sample_ids=["track2_0001"],
            )

            outputs = write_dry_run_outputs(report, image_zip=image_zip, out_dir=out_dir)

            self.assertTrue(Path(outputs["json"]).exists())
            self.assertTrue(Path(outputs["markdown"]).exists())
            self.assertTrue(Path(outputs["html"]).exists())

            html_text = Path(outputs["html"]).read_text(encoding="utf-8")
            self.assertIn("Track2 MoE Specialist Dry-Run", html_text)
            self.assertIn("track2_0001", html_text)
            self.assertIn("accept_change", html_text)
            self.assertTrue(
                (out_dir / "html_review" / "assets" / "track2_0001.jpg").exists()
            )
            self.assertFalse((out_dir / "submission.json").exists())
            self.assertFalse((out_dir / "submission.zip").exists())

    def test_write_dry_run_outputs_rejects_repo_submissions_out_dir_without_creating_it(self):
        from affectiveart.track2_moe_specialist_ensemble import write_dry_run_outputs

        repo_root = Path(__file__).resolve().parents[1]
        blocked_out_dir = repo_root / "submissions" / "track2_moe_policy_guard_test_output"
        self.assertFalse(blocked_out_dir.exists())

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_zip = tmp_path / "track2_images.zip"
            _write_track2_image_zip(image_zip, ["track2_0001"])
            report = build_dry_run_report(
                [current_row(sample_id="track2_0001", emotion="content")],
                [
                    expert("track2_0001", "clip_clean", "calm", role="global"),
                    expert("track2_0001", "boundary_head", "calm", role="boundary"),
                ],
                queue_sample_ids=["track2_0001"],
            )

            with mock.patch(
                "affectiveart.track2_moe_specialist_ensemble.Path.mkdir"
            ) as mkdir_mock, mock.patch(
                "affectiveart.track2_moe_specialist_ensemble._extract_review_assets",
                return_value={},
            ), mock.patch(
                "affectiveart.track2_moe_specialist_ensemble.Path.write_text",
                return_value=None,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "dry-run output must not be under submissions/",
                ):
                    write_dry_run_outputs(
                        report,
                        image_zip=image_zip,
                        out_dir=blocked_out_dir,
                    )
                mkdir_mock.assert_not_called()

        self.assertFalse(blocked_out_dir.exists())

    def test_write_dry_run_outputs_does_not_mutate_input_report(self):
        from affectiveart.track2_moe_specialist_ensemble import write_dry_run_outputs

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_zip = tmp_path / "track2_images.zip"
            out_dir = tmp_path / "out"
            _write_track2_image_zip(image_zip, ["track2_0001"])
            report = build_dry_run_report(
                [current_row(sample_id="track2_0001", emotion="content")],
                [
                    expert("track2_0001", "clip_clean", "calm", role="global"),
                    expert("track2_0001", "boundary_head", "calm", role="boundary"),
                ],
                queue_sample_ids=["track2_0001"],
            )
            original_report = copy.deepcopy(report)

            outputs = write_dry_run_outputs(report, image_zip=image_zip, out_dir=out_dir)

            self.assertEqual(report, original_report)
            output_report = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
            self.assertIn("outputs", output_report)
            self.assertIn("image_asset", output_report["rows"][0])

    def test_render_dry_run_html_escapes_hostile_sample_id_reason_and_evidence(self):
        from affectiveart.track2_moe_specialist_ensemble import render_dry_run_html

        html_text = render_dry_run_html(
            {
                "row_count": 1,
                "decision_counts": {"hold": 1},
                "rows": [
                    {
                        "sample_id": 'track2_<script>alert("x")</script>',
                        "decision": "hold",
                        "current_emotion": "content",
                        "current_valence": "Positive",
                        "current_arousal": "Low",
                        "proposed_emotion": "calm",
                        "proposed_valence": "Positive",
                        "proposed_arousal": "Low",
                        "reasons": ['<img src=x onerror=alert("x")>'],
                        "expert_evidence": [
                            {
                                "source": "<b>clip</b>",
                                "role": "global",
                                "emotion": "calm",
                                "confidence": 0.9,
                                "margin": 0.2,
                                "top3": [
                                    {
                                        "emotion": "<i>calm</i>",
                                        "probability": 0.9,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )

        self.assertNotIn('<script>alert("x")</script>', html_text)
        self.assertNotIn('<img src=x onerror=alert("x")>', html_text)
        self.assertNotIn("<b>clip</b>", html_text)
        self.assertNotIn("<i>calm</i>", html_text)
        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;", html_text)
        self.assertIn(
            "&lt;img src=x onerror=alert(&quot;x&quot;)&gt;",
            html_text,
        )
        self.assertIn("&lt;b&gt;clip&lt;/b&gt;", html_text)
        self.assertIn("&lt;i&gt;calm&lt;/i&gt;", html_text)

    def test_write_dry_run_outputs_uses_distinct_assets_for_colliding_hostile_ids(self):
        from affectiveart.track2_moe_specialist_ensemble import write_dry_run_outputs

        sample_ids = ["bad id", "bad?id"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_zip = tmp_path / "track2_images.zip"
            out_dir = tmp_path / "out"
            _write_track2_image_zip(image_zip, sample_ids)
            report = build_dry_run_report(
                [
                    current_row(sample_id="bad id", emotion="content"),
                    current_row(sample_id="bad?id", emotion="content"),
                ],
                [
                    expert("bad id", "clip_clean", "calm", role="global"),
                    expert("bad id", "boundary_head", "calm", role="boundary"),
                    expert("bad?id", "clip_clean", "calm", role="global"),
                    expert("bad?id", "boundary_head", "calm", role="boundary"),
                ],
                queue_sample_ids=sample_ids,
            )

            outputs = write_dry_run_outputs(report, image_zip=image_zip, out_dir=out_dir)

            output_report = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
            asset_paths = [row["image_asset"] for row in output_report["rows"]]
            self.assertEqual(len(set(asset_paths)), 2)
            for asset_path in asset_paths:
                self.assertRegex(asset_path, r"^assets/bad_id-[0-9a-f]{8}\.jpg$")
                self.assertTrue((out_dir / "html_review" / asset_path).exists())


if __name__ == "__main__":
    unittest.main()
