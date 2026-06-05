import csv
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_v7_hardcase_arbitration import (
    ACCEPT_BORDERLINE,
    ACCEPT_STRICT,
    HOLD,
    V7_LADDERS,
    build_candidate_rows,
    load_arbitration_rows,
    select_v7_arbitrations,
    write_v7_outputs,
)


def row(sample_id, emotion, caption=None):
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
    valence, arousal = labels[emotion]
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": caption or f"A {emotion} artwork with visible emotional atmosphere.",
        "brushstroke": f"Brushwork supports a {emotion} reading without evaluator instructions.",
        "composition": "The composition organizes the subject clearly.",
        "color": "Specific colors shape the emotional atmosphere.",
        "line": "Line quality defines the figure and spatial rhythm.",
        "light": "Light and shadow clarify depth and focus.",
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_evidence(path, rows):
    fieldnames = sorted({key for item in rows for key in item})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class Track2V7ArbitrationCoreTest(unittest.TestCase):
    def test_strict_arbitrated_cross_with_text_override_enters_strict_ladders(self):
        baseline = [row("track2_0547", "annoyed")]
        text_override = [row("track2_0547", "calm", "A quiet abstract drawing creates a restrained calm mood.")]
        evidence = [
            {
                "sample_id": "track2_0547",
                "current_emotion": "annoyed",
                "current_valence": "Negative",
                "current_arousal": "High",
                "proposed_emotion": "calm",
                "proposed_valence": "Positive",
                "proposed_arousal": "Low",
                "transition": "annoyed->calm",
                "supporting_source_count": 10,
                "supporting_family_count": 5,
                "supporting_sources": "gemini35;teacher;siglip2;dinov2;public_style",
                "supporting_families": "gemini35;teacher;embedding;public_style;selective",
                "same_quadrant": False,
                "cross_quadrant_risk": True,
                "gemini35_objection": False,
                "evidence_score": 11,
            }
        ]
        arbitration = [
            {
                "sample_id": "track2_0547",
                "transition": "annoyed->calm",
                "preferred_option": "proposed",
                "confidence": 0.8,
                "current_fit_score": 0.35,
                "proposed_fit_score": 0.68,
                "fit_margin": 0.33,
                "enter_strict_candidate": True,
            }
        ]

        decisions = select_v7_arbitrations(
            baseline_rows=baseline,
            evidence_rows=evidence,
            arbitration_rows=arbitration,
            text_override_rows=text_override,
        )
        strict_rows = build_candidate_rows(baseline, decisions, ladder="v7_cross1")

        self.assertEqual(decisions[0].decision, ACCEPT_STRICT)
        self.assertEqual(decisions[0].ladder, "v7_strict")
        self.assertEqual(strict_rows[0]["emotion"], "calm")
        self.assertEqual(strict_rows[0]["overall_caption"], text_override[0]["overall_caption"])

    def test_strict_cross_without_label_aligned_text_override_is_held(self):
        baseline = [row("track2_0547", "annoyed")]
        evidence = [
            {
                "sample_id": "track2_0547",
                "current_emotion": "annoyed",
                "proposed_emotion": "calm",
                "supporting_source_count": 10,
                "supporting_family_count": 5,
                "supporting_sources": "gemini35;teacher;siglip2;dinov2;public_style",
                "supporting_families": "gemini35;teacher;embedding;public_style;selective",
                "same_quadrant": False,
                "evidence_score": 11,
            }
        ]
        arbitration = [
            {
                "sample_id": "track2_0547",
                "transition": "annoyed->calm",
                "preferred_option": "proposed",
                "confidence": 0.8,
                "fit_margin": 0.33,
                "enter_strict_candidate": True,
            }
        ]

        decisions = select_v7_arbitrations(
            baseline_rows=baseline,
            evidence_rows=evidence,
            arbitration_rows=arbitration,
            text_override_rows=[],
        )

        self.assertEqual(decisions[0].decision, HOLD)
        self.assertIn("missing_label_aligned_text_override", decisions[0].reason_codes)

    def test_proposed_valence_arousal_mismatch_is_held_before_candidate_write(self):
        baseline = [row("track2_0547", "annoyed")]
        text_override = [
            {
                **row("track2_0547", "calm", "A quiet abstract drawing creates a restrained calm mood."),
                "emotional_valence": "Negative",
                "emotional_arousal_level": "Low",
            }
        ]
        evidence = [
            {
                "sample_id": "track2_0547",
                "current_emotion": "annoyed",
                "proposed_emotion": "calm",
                "proposed_valence": "Negative",
                "proposed_arousal": "Low",
                "supporting_source_count": 10,
                "supporting_family_count": 5,
                "supporting_sources": "gemini35;teacher;siglip2;dinov2;public_style",
                "supporting_families": "gemini35;teacher;embedding;public_style;selective",
                "same_quadrant": False,
                "evidence_score": 11,
            }
        ]
        arbitration = [
            {
                "sample_id": "track2_0547",
                "transition": "annoyed->calm",
                "preferred_option": "proposed",
                "confidence": 0.8,
                "fit_margin": 0.33,
                "enter_strict_candidate": True,
            }
        ]

        decisions = select_v7_arbitrations(
            baseline_rows=baseline,
            evidence_rows=evidence,
            arbitration_rows=arbitration,
            text_override_rows=text_override,
        )

        self.assertEqual(decisions[0].decision, HOLD)
        self.assertIn("proposed_va_label_mismatch", decisions[0].reason_codes)

    def test_borderline_proposed_only_enters_diagnostic_ladder(self):
        baseline = [row("track2_0873", "aroused")]
        text_override = [row("track2_0873", "calm", "A restrained abstract image settles into a calm low-arousal mood.")]
        evidence = [
            {
                "sample_id": "track2_0873",
                "current_emotion": "aroused",
                "proposed_emotion": "calm",
                "supporting_source_count": 7,
                "supporting_family_count": 4,
                "supporting_sources": "gemini35;teacher;siglip2;dinov2",
                "supporting_families": "gemini35;teacher;embedding;public_style",
                "same_quadrant": False,
                "evidence_score": 7,
            }
        ]
        arbitration = [
            {
                "sample_id": "track2_0873",
                "transition": "aroused->calm",
                "preferred_option": "proposed",
                "confidence": 0.7,
                "fit_margin": 0.2,
                "enter_strict_candidate": False,
            }
        ]

        decisions = select_v7_arbitrations(
            baseline_rows=baseline,
            evidence_rows=evidence,
            arbitration_rows=arbitration,
            text_override_rows=text_override,
        )
        strict_rows = build_candidate_rows(baseline, decisions, ladder="v7_cross3")
        diagnostic_rows = build_candidate_rows(baseline, decisions, ladder="v7_hardcase_push")

        self.assertEqual(decisions[0].decision, ACCEPT_BORDERLINE)
        self.assertEqual(strict_rows[0]["emotion"], "aroused")
        self.assertEqual(diagnostic_rows[0]["emotion"], "calm")

    def test_dangerous_positive_low_to_negative_is_blocked_without_strict_gate(self):
        baseline = [row("track2_0815", "content")]
        text_override = [row("track2_0815", "annoyed", "A tense image carries an annoyed mood.")]
        evidence = [
            {
                "sample_id": "track2_0815",
                "current_emotion": "content",
                "proposed_emotion": "annoyed",
                "supporting_source_count": 6,
                "supporting_family_count": 3,
                "supporting_sources": "teacher;siglip2;dinov2",
                "supporting_families": "teacher;embedding;public_style",
                "same_quadrant": False,
                "evidence_score": 5,
            }
        ]
        arbitration = [
            {
                "sample_id": "track2_0815",
                "transition": "content->annoyed",
                "preferred_option": "proposed",
                "confidence": 0.9,
                "fit_margin": 0.5,
                "enter_strict_candidate": False,
            }
        ]

        decisions = select_v7_arbitrations(
            baseline_rows=baseline,
            evidence_rows=evidence,
            arbitration_rows=arbitration,
            text_override_rows=text_override,
        )

        self.assertEqual(decisions[0].decision, HOLD)
        self.assertIn("dangerous_positive_low_shift_without_strict_gate", decisions[0].reason_codes)


class Track2V7WriterTest(unittest.TestCase):
    def test_write_outputs_creates_side_path_candidates_and_shadow_report(self):
        baseline = [row("track2_0547", "annoyed"), row("track2_0873", "aroused")]
        text_override = [
            row("track2_0547", "calm", "A quiet abstract drawing creates a restrained calm mood."),
            row("track2_0873", "calm", "A restrained abstract image settles into a calm low-arousal mood."),
        ]
        evidence = [
            {
                "sample_id": "track2_0547",
                "current_emotion": "annoyed",
                "proposed_emotion": "calm",
                "supporting_source_count": "10",
                "supporting_family_count": "5",
                "supporting_sources": "gemini35;teacher;siglip2;dinov2;public_style",
                "supporting_families": "gemini35;teacher;embedding;public_style;selective",
                "same_quadrant": "false",
                "evidence_score": "11",
            },
            {
                "sample_id": "track2_0873",
                "current_emotion": "aroused",
                "proposed_emotion": "calm",
                "supporting_source_count": "7",
                "supporting_family_count": "4",
                "supporting_sources": "gemini35;teacher;siglip2;dinov2",
                "supporting_families": "gemini35;teacher;embedding;public_style",
                "same_quadrant": "false",
                "evidence_score": "7",
            },
        ]
        arbitration = [
            {
                "sample_id": "track2_0547",
                "transition": "annoyed->calm",
                "preferred_option": "proposed",
                "confidence": 0.8,
                "fit_margin": 0.33,
                "enter_strict_candidate": True,
            },
            {
                "sample_id": "track2_0873",
                "transition": "aroused->calm",
                "preferred_option": "proposed",
                "confidence": 0.7,
                "fit_margin": 0.2,
                "enter_strict_candidate": False,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_json = tmp_path / "baseline.json"
            text_json = tmp_path / "text_override.json"
            evidence_csv = tmp_path / "evidence.csv"
            arbitration_json = tmp_path / "arbitration.json"
            out_dir = tmp_path / "out"
            submission_dir = tmp_path / "submissions"
            write_json(baseline_json, baseline)
            write_json(text_json, text_override)
            write_json(arbitration_json, arbitration)
            write_evidence(evidence_csv, evidence)

            report = write_v7_outputs(
                baseline_json=baseline_json,
                evidence_matrix=evidence_csv,
                arbitration_json=arbitration_json,
                text_override_json=text_json,
                out_dir=out_dir,
                submission_dir=submission_dir,
                expected_row_count=2,
                require_all_emotions=False,
            )

            self.assertEqual(report["method"], "track2_v7_hardcase_arbitration")
            self.assertEqual(report["candidates"]["v7_cross1"]["changed_rows"], 1)
            self.assertEqual(report["candidates"]["v7_cross3"]["changed_rows"], 1)
            self.assertEqual(report["candidates"]["v7_hardcase_push"]["changed_rows"], 2)
            self.assertTrue(Path(report["shadow_report_json"]).exists())
            self.assertTrue(Path(report["html_review_path"]).exists())
            for ladder in V7_LADDERS:
                self.assertTrue(Path(report["candidates"][ladder]["json"]).exists())
                self.assertTrue(Path(report["candidates"][ladder]["zip"]).exists())
            with zipfile.ZipFile(report["candidates"]["v7_cross1"]["zip"]) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])

    def test_cli_run_writes_v7_summary(self):
        repo_root = Path(__file__).resolve().parents[1]
        baseline = [row("track2_0547", "annoyed")]
        text_override = [row("track2_0547", "calm", "A quiet abstract drawing creates a restrained calm mood.")]
        evidence = [
            {
                "sample_id": "track2_0547",
                "current_emotion": "annoyed",
                "proposed_emotion": "calm",
                "supporting_source_count": "10",
                "supporting_family_count": "5",
                "supporting_sources": "gemini35;teacher;siglip2;dinov2;public_style",
                "supporting_families": "gemini35;teacher;embedding;public_style;selective",
                "same_quadrant": "false",
                "evidence_score": "11",
            }
        ]
        arbitration = [
            {
                "sample_id": "track2_0547",
                "transition": "annoyed->calm",
                "preferred_option": "proposed",
                "confidence": 0.8,
                "fit_margin": 0.33,
                "enter_strict_candidate": True,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_json = tmp_path / "baseline.json"
            text_json = tmp_path / "text_override.json"
            evidence_csv = tmp_path / "evidence.csv"
            arbitration_json = tmp_path / "arbitration.json"
            out_dir = tmp_path / "out"
            submission_dir = tmp_path / "submissions"
            write_json(baseline_json, baseline)
            write_json(text_json, text_override)
            write_json(arbitration_json, arbitration)
            write_evidence(evidence_csv, evidence)

            result = subprocess.run(
                [
                    "python3",
                    "scripts/track2_v7_hardcase_arbitration.py",
                    "run",
                    "--baseline-json",
                    str(baseline_json),
                    "--evidence-matrix",
                    str(evidence_csv),
                    "--arbitration-json",
                    str(arbitration_json),
                    "--text-override-json",
                    str(text_json),
                    "--out-dir",
                    str(out_dir),
                    "--submission-dir",
                    str(submission_dir),
                    "--expected-row-count",
                    "1",
                    "--allow-missing-emotions-for-smoke",
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("track2 v7 hardcase arbitration", result.stdout)
            self.assertTrue((out_dir / "v7_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
