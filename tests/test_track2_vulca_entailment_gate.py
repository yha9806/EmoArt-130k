import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_vulca_entailment_gate import (
    audit_vulca_entailment_row,
    build_vulca_entailment_candidate_rows,
    infer_vulca_culture,
    write_vulca_entailment_gate_outputs,
)


def row(sample_id, emotion, valence, arousal, caption="specific visual caption"):
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": caption,
        "brushstroke": "layered visible brushwork supports the depicted forms",
        "composition": "balanced composition with clear foreground and background structure",
        "color": "specific color contrasts shape the emotional atmosphere",
        "line": "line quality defines the figures and spatial rhythm",
        "light": "light and shadow clarify depth and focal emphasis",
    }


def full_label_rows():
    labels = [
        ("alarmed", "Negative", "High"),
        ("annoyed", "Negative", "High"),
        ("aroused", "Positive", "High"),
        ("bored", "Negative", "Low"),
        ("calm", "Positive", "Low"),
        ("content", "Positive", "Low"),
        ("excited", "Positive", "High"),
        ("frustrated", "Negative", "High"),
        ("glad", "Positive", "Low"),
        ("happy", "Positive", "High"),
        ("sad", "Negative", "Low"),
        ("tired", "Negative", "Low"),
    ]
    return [
        row(f"track2_{index:04d}", emotion, valence, arousal)
        for index, (emotion, valence, arousal) in enumerate(labels)
    ]


class Track2VulcaEntailmentGateTest(unittest.TestCase):
    def test_infers_vulca_culture_from_track2_text(self):
        chinese = row(
            "track2_0001",
            "calm",
            "Positive",
            "Low",
            caption="A traditional Chinese ink scroll with calligraphy and bamboo.",
        )
        japanese = row(
            "track2_0002",
            "content",
            "Positive",
            "Low",
            caption="A Japanese ukiyo-e woodblock print of a woman in a kimono.",
        )

        self.assertEqual(infer_vulca_culture(chinese), "chinese")
        self.assertEqual(infer_vulca_culture(japanese), "japanese")

    def test_audit_flags_overdeep_and_emotion_conflict_claims(self):
        risky = row(
            "track2_0004",
            "calm",
            "Positive",
            "Low",
            caption=(
                "A quiet ink painting that symbolizes a spiritual doctrine, "
                "while the dark palette suggests fatigue and grief."
            ),
        )

        audit = audit_vulca_entailment_row(risky)
        codes = {issue["code"] for issue in audit["issues"]}

        self.assertIn("overdeep_cultural_claim", codes)
        self.assertIn("emotion_text_conflict", codes)
        self.assertGreater(audit["pseudo_understanding_risk"], 0.0)

    def test_candidate_repair_preserves_labels_and_reduces_risk(self):
        rows = full_label_rows()
        rows[4]["overall_caption"] = (
            "A quiet ink painting that symbolizes a spiritual doctrine, "
            "while the dark palette suggests fatigue and grief."
        )

        repaired, report = build_vulca_entailment_candidate_rows(rows)
        before = report["risk_summary_before"]
        after = report["risk_summary_after"]

        self.assertEqual(report["classification_label_changes"], 0)
        self.assertEqual(repaired[4]["emotion"], "calm")
        self.assertLess(after["total_issues"], before["total_issues"])
        self.assertGreaterEqual(report["changed_rows"], 1)
        self.assertNotIn("spiritual doctrine", repaired[4]["overall_caption"].lower())
        self.assertNotIn("fatigue", repaired[4]["overall_caption"].lower())

    def test_candidate_repair_does_not_soften_negative_emotion_evidence(self):
        rows = full_label_rows()
        rows[0]["overall_caption"] = (
            "A dramatic crowd scene filled with panic and grief around a fallen body."
        )
        rows[10]["overall_caption"] = (
            "A somber mourner bends over the figure in grief and despair."
        )

        repaired, report = build_vulca_entailment_candidate_rows(rows)

        self.assertEqual(report["classification_label_changes"], 0)
        self.assertIn("panic and grief", repaired[0]["overall_caption"].lower())
        self.assertIn("grief and despair", repaired[10]["overall_caption"].lower())

    def test_writes_side_path_candidate_zip_report_and_shadow_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_json = tmp_path / "source.json"
            baseline_json = tmp_path / "baseline.json"
            out_json = tmp_path / "track2_v9_candidate.json"
            out_zip = tmp_path / "track2_v9_candidate.zip"
            report_json = tmp_path / "v9_report.json"
            report_md = tmp_path / "v9_report.md"
            shadow_dir = tmp_path / "shadow"
            rows = full_label_rows()
            rows[4]["overall_caption"] = (
                "A quiet ink painting that symbolizes a spiritual doctrine, "
                "while the dark palette suggests fatigue and grief."
            )
            source_json.write_text(json.dumps(rows), encoding="utf-8")
            baseline_json.write_text(json.dumps(rows), encoding="utf-8")

            report = write_vulca_entailment_gate_outputs(
                source_json=source_json,
                out_json=out_json,
                out_zip=out_zip,
                report_json=report_json,
                report_md=report_md,
                baseline_json=baseline_json,
                shadow_out_dir=shadow_dir,
                expected_row_count=len(rows),
            )

            self.assertEqual(report["method"], "track2_vulca_entailment_gate_v1")
            self.assertTrue(out_json.exists())
            self.assertTrue(out_zip.exists())
            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())
            self.assertTrue((shadow_dir / "shadow_score_report.json").exists())
            with zipfile.ZipFile(out_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
            self.assertFalse(report["formal_submission_overwritten"])

    def test_cli_writes_candidate_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_json = tmp_path / "source.json"
            baseline_json = tmp_path / "baseline.json"
            out_json = tmp_path / "track2_v9_candidate.json"
            out_zip = tmp_path / "track2_v9_candidate.zip"
            report_json = tmp_path / "v9_report.json"
            report_md = tmp_path / "v9_report.md"
            shadow_dir = tmp_path / "shadow"
            rows = full_label_rows()
            rows[4]["overall_caption"] = (
                "A quiet ink painting that symbolizes a spiritual doctrine, "
                "while the dark palette suggests fatigue and grief."
            )
            source_json.write_text(json.dumps(rows), encoding="utf-8")
            baseline_json.write_text(json.dumps(rows), encoding="utf-8")

            result = subprocess.run(
                [
                    "python3",
                    "scripts/track2_vulca_entailment_gate.py",
                    "--source-json",
                    str(source_json),
                    "--out-json",
                    str(out_json),
                    "--out-zip",
                    str(out_zip),
                    "--report-json",
                    str(report_json),
                    "--report-md",
                    str(report_md),
                    "--baseline-json",
                    str(baseline_json),
                    "--shadow-out-dir",
                    str(shadow_dir),
                    "--expected-row-count",
                    str(len(rows)),
                ],
                cwd=Path.cwd(),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("changed_rows=", result.stdout)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_zip.exists())
            self.assertTrue((shadow_dir / "shadow_score_report.json").exists())


if __name__ == "__main__":
    unittest.main()
