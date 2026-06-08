from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_v29_final_shot import (
    apply_label_changes,
    build_candidate_suite,
    choose_v29_gate,
    protect_side_path,
    run_v29_sweep,
    write_candidate_json_and_zip,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(sample_id: str = "track2_0001") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "emotion": "calm",
        "emotional_valence": "Positive",
        "emotional_arousal_level": "Low",
        "overall_caption": "A balanced landscape uses muted color and diffuse light to create a calm mood.",
        "brushstroke": "Soft brushstroke texture supports calm.",
        "composition": "Balanced composition supports calm.",
        "color": "Muted color supports calm.",
        "line": "Horizontal line rhythm supports calm.",
        "light": "Diffuse light supports calm.",
    }


class Track2V29FinalShotTests(unittest.TestCase):
    def test_protect_side_path_rejects_formal_submission_json(self) -> None:
        with self.assertRaises(ValueError):
            protect_side_path(Path("submissions/track2_submission.json"))

    def test_choose_v29_gate_recommends_only_when_thresholds_pass(self) -> None:
        decision = choose_v29_gate(
            overall_expected=0.891,
            classification_expected=0.781,
            description_expected=0.991,
            unsafe_text_rows=0,
            missing_emotions=[],
            label_consistency_issue_count=0,
            top_emotion_share=0.59,
            broad_knn_copy=False,
            unreported_cross_quadrant_count=0,
        )

        self.assertEqual(decision["decision"], "recommend_final_submit")

    def test_write_candidate_json_and_zip_writes_submission_json_at_zip_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "track2_submission_v29_test_candidate.json"
            out_zip = Path(temp_dir) / "track2_submission_v29_test_candidate.zip"

            write_candidate_json_and_zip([_row()], out_json=out_json, out_zip=out_zip)

            self.assertTrue(out_json.exists())
            self.assertTrue(out_zip.exists())
            with zipfile.ZipFile(out_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
                payload = json.loads(archive.read("submission.json").decode("utf-8"))
            self.assertEqual(payload[0]["sample_id"], "track2_0001")

    def test_apply_label_changes_updates_va_without_touching_text(self) -> None:
        row = _row()

        updated = apply_label_changes(
            [row],
            {
                "track2_0001": {
                    "emotion": "content",
                    "emotional_valence": "Positive",
                    "emotional_arousal_level": "Low",
                }
            },
        )

        self.assertEqual(updated[0]["emotion"], "content")
        self.assertEqual(updated[0]["overall_caption"], row["overall_caption"])

    def test_build_candidate_suite_returns_text_and_hybrid_candidates(self) -> None:
        suite = build_candidate_suite(
            [_row()],
            label_change_maps={
                "qwen_hybrid_balanced": {
                    "track2_0001": {
                        "emotion": "content",
                        "emotional_valence": "Positive",
                        "emotional_arousal_level": "Low",
                    }
                }
            },
        )

        self.assertIn("v29_descmax_on_base", suite)
        self.assertIn("v29_qwen_hybrid_balanced", suite)

    def test_run_v29_sweep_writes_ranked_side_path_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profiles = run_v29_sweep(
                [_row("track2_0001")],
                candidate_prefix=root / "submissions" / "track2_submission_v29",
                out_dir=root / "experiments",
                label_change_maps={
                    "qwen_hybrid_balanced": {
                        "track2_0001": {
                            "emotion": "content",
                            "emotional_valence": "Positive",
                            "emotional_arousal_level": "Low",
                        }
                    }
                },
                base_classification_expected=0.78,
                label_change_credit=0.02,
            )

            self.assertGreaterEqual(len(profiles), 2)
            self.assertTrue(Path(profiles[0].json_path).exists())
            self.assertTrue((root / "experiments" / "v29_final_shot_report.json").exists())

    def test_cli_help_runs_from_repo_root(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/track2_v29_final_shot.py", "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Track2 v29", result.stdout)


if __name__ == "__main__":
    unittest.main()
