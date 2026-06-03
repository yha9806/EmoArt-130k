import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from affectiveart.track2_disagreement_review import (
    DECISION_COLUMNS,
    build_disagreement_review_rows,
    write_disagreement_review_outputs,
)


def current_row(sample_id, emotion="content", valence="Positive", arousal="Low"):
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": f"{sample_id} caption",
        "brushstroke": "brush",
        "composition": "composition",
        "color": "color",
        "line": "line",
        "light": "light",
    }


def pred(sample_id, current, emotion, confidence=0.8, margin=0.2, knn=None):
    return {
        "sample_id": sample_id,
        "current": current,
        "emotion": emotion,
        "confidence": confidence,
        "margin": margin,
        "knn_emotion": knn or emotion,
        "knn_confidence": 0.7,
        "top3": [{"emotion": emotion, "probability": confidence}],
        "knn_top3": [{"emotion": knn or emotion, "votes": 10, "similarity_sum": 9.1}],
    }


class Track2DisagreementReviewTest(unittest.TestCase):
    def test_build_rows_selects_clean_inclusive_emotion_disagreements_only(self):
        current = [
            current_row("track2_keep", "calm"),
            current_row("track2_inclusive_only", "content"),
            current_row("track2_clean_only", "content"),
            current_row("track2_conflict", "content"),
            current_row("track2_agree_change", "content"),
        ]
        clean = {
            "entries": [
                pred("track2_keep", "calm", "calm"),
                pred("track2_inclusive_only", "content", "content"),
                pred("track2_clean_only", "content", "calm"),
                pred("track2_conflict", "content", "calm"),
                pred("track2_agree_change", "content", "calm"),
            ]
        }
        inclusive = {
            "entries": [
                pred("track2_keep", "calm", "calm"),
                pred("track2_inclusive_only", "content", "calm"),
                pred("track2_clean_only", "content", "content"),
                pred("track2_conflict", "content", "tired"),
                pred("track2_agree_change", "content", "calm"),
            ]
        }

        rows = build_disagreement_review_rows(
            current,
            clean,
            inclusive,
            high_similarity_sample_ids={"track2_inclusive_only"},
            image_members={
                "track2_inclusive_only": "images/track2_inclusive_only.jpg",
                "track2_clean_only": "images/track2_clean_only.jpg",
                "track2_conflict": "images/track2_conflict.jpg",
            },
        )

        self.assertEqual(
            [row["sample_id"] for row in rows],
            [
                "track2_inclusive_only",
                "track2_clean_only",
                "track2_conflict",
            ],
        )
        by_id = {row["sample_id"]: row for row in rows}
        self.assertEqual(by_id["track2_inclusive_only"]["category"], "inclusive_only_change")
        self.assertEqual(by_id["track2_clean_only"]["category"], "clean_only_change")
        self.assertEqual(by_id["track2_conflict"]["category"], "clean_inclusive_conflict")
        self.assertTrue(by_id["track2_inclusive_only"]["high_similarity_public_reference"])
        self.assertEqual(by_id["track2_inclusive_only"]["recommended_decision"], "hold")
        self.assertEqual(by_id["track2_inclusive_only"]["manual_decision"], "")
        self.assertEqual(by_id["track2_inclusive_only"]["reviewer_rationale"], "")

    def test_build_rows_accepts_plain_prediction_lists(self):
        current = [current_row("track2_0001", "content")]
        clean = [pred("track2_0001", "content", "content")]
        inclusive = [pred("track2_0001", "content", "calm")]

        rows = build_disagreement_review_rows(current, clean, inclusive)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "track2_0001")
        self.assertEqual(rows[0]["inclusive_emotion"], "calm")

    def test_build_rows_tolerates_non_list_optional_prediction_lists(self):
        current = [current_row("track2_0001", "content")]
        clean = pred("track2_0001", "content", "content")
        clean["top3"] = None
        clean["knn_top3"] = {"emotion": "content"}
        inclusive = pred("track2_0001", "content", "calm")
        inclusive.pop("top3")
        inclusive["knn_top3"] = "calm"

        rows = build_disagreement_review_rows(current, [clean], [inclusive])

        self.assertEqual(rows[0]["clean"]["top3"], [])
        self.assertEqual(rows[0]["clean"]["knn_top3"], [])
        self.assertEqual(rows[0]["inclusive"]["top3"], [])
        self.assertEqual(rows[0]["inclusive"]["knn_top3"], [])

    def test_write_outputs_sanitizes_asset_filename_for_malformed_sample_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            current_json = tmp_path / "current.json"
            clean_json = tmp_path / "clean.json"
            inclusive_json = tmp_path / "inclusive.json"
            image_zip = tmp_path / "track2.zip"
            out_dir = tmp_path / "out"
            sample_id = "../../x"
            current_json.write_text(json.dumps([current_row(sample_id, "content")]), encoding="utf-8")
            clean_json.write_text(json.dumps([pred(sample_id, "content", "content")]), encoding="utf-8")
            inclusive_json.write_text(json.dumps([pred(sample_id, "content", "calm")]), encoding="utf-8")
            img = tmp_path / "source.jpg"
            Image.new("RGB", (64, 48), (200, 100, 20)).save(img)
            with zipfile.ZipFile(image_zip, "w") as zf:
                zf.writestr(f"images/{sample_id}.jpg", img.read_bytes())

            report = write_disagreement_review_outputs(
                current_json=current_json,
                clean_predictions_json=clean_json,
                inclusive_predictions_json=inclusive_json,
                image_zip=image_zip,
                out_dir=out_dir,
            )

            self.assertEqual(report["rows"][0]["image_asset"], "assets/.._.._x.jpg")
            self.assertTrue((out_dir / "html_review" / "assets" / ".._.._x.jpg").exists())
            self.assertFalse((out_dir / "x.jpg").exists())

    def test_write_outputs_writes_html_csv_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            current_json = tmp_path / "current.json"
            clean_json = tmp_path / "clean.json"
            inclusive_json = tmp_path / "inclusive.json"
            image_zip = tmp_path / "track2.zip"
            out_dir = tmp_path / "out"
            current_json.write_text(
                json.dumps(
                    [
                        current_row("track2_0001", "content"),
                        current_row("track2_0002", "content"),
                    ]
                ),
                encoding="utf-8",
            )
            clean_json.write_text(
                json.dumps(
                    {
                        "entries": [
                            pred("track2_0001", "content", "content"),
                            pred("track2_0002", "content", "calm"),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            inclusive_json.write_text(
                json.dumps(
                    {
                        "entries": [
                            pred("track2_0001", "content", "calm"),
                            pred("track2_0002", "content", "content"),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            img1 = tmp_path / "track2_0001.jpg"
            img2 = tmp_path / "track2_0002.jpg"
            Image.new("RGB", (64, 48), (200, 100, 20)).save(img1)
            Image.new("RGB", (64, 48), (20, 100, 200)).save(img2)
            with zipfile.ZipFile(image_zip, "w") as zf:
                zf.write(img1, "images/track2_0001.jpg")
                zf.write(img2, "track2_testset/images/track2_0002.jpg")

            report = write_disagreement_review_outputs(
                current_json=current_json,
                clean_predictions_json=clean_json,
                inclusive_predictions_json=inclusive_json,
                image_zip=image_zip,
                out_dir=out_dir,
                high_similarity_sample_ids={"track2_0001"},
            )

            self.assertEqual(report["row_count"], 2)
            self.assertEqual(report["category_counts"]["inclusive_only_change"], 1)
            self.assertEqual(report["category_counts"]["clean_only_change"], 1)
            self.assertEqual(report["high_similarity_count"], 1)
            self.assertEqual(report["inputs"]["current_json"], str(current_json))
            self.assertTrue(Path(report["outputs"]["html"]).exists())
            self.assertTrue(Path(report["outputs"]["csv"]).exists())
            self.assertTrue(Path(report["outputs"]["json"]).exists())
            self.assertTrue(Path(report["outputs"]["markdown"]).exists())

            html_text = Path(report["outputs"]["html"]).read_text(encoding="utf-8")
            self.assertIn("Track2 Clean/Inclusive Disagreement Review", html_text)
            self.assertIn("track2_0001", html_text)
            self.assertIn("Current label", html_text)
            self.assertIn("Clean prediction", html_text)
            self.assertIn("Inclusive prediction", html_text)
            self.assertIn("calm/content", html_text)
            self.assertIn("frustrated/aroused", html_text)
            self.assertIn("data-filter=\"all\"", html_text)
            self.assertIn("data-filter=\"inclusive_only_change\"", html_text)
            self.assertIn("data-filter=\"clean_only_change\"", html_text)
            self.assertIn("data-filter=\"clean_inclusive_conflict\"", html_text)
            self.assertIn("data-filter=\"high_similarity\"", html_text)
            self.assertIn("<select", html_text)
            self.assertIn("<textarea", html_text)

            csv_text = Path(report["outputs"]["csv"]).read_text(encoding="utf-8")
            self.assertEqual(",".join(DECISION_COLUMNS), csv_text.splitlines()[0])
            self.assertTrue((out_dir / "html_review" / "assets" / "track2_0001.jpg").exists())

            report_text = Path(report["outputs"]["json"]).read_text(encoding="utf-8")
            self.assertIn('"rows"', report_text)
            markdown_text = Path(report["outputs"]["markdown"]).read_text(encoding="utf-8")
            self.assertIn("Track2 Clean/Inclusive Disagreement Review", markdown_text)
            self.assertIn(str(report["outputs"]["html"]), markdown_text)

    def test_cli_writes_review_outputs(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(__file__).resolve().parents[1]
            script = repo_root / "scripts" / "build_track2_clean_inclusive_disagreement_review.py"
            tmp_path = Path(tmp)
            current_json = tmp_path / "current.json"
            clean_json = tmp_path / "clean.json"
            inclusive_json = tmp_path / "inclusive.json"
            high_similarity_json = tmp_path / "high_similarity.json"
            image_zip = tmp_path / "track2.zip"
            out_dir = tmp_path / "out"
            current_json.write_text(json.dumps([current_row("track2_0001", "content")]), encoding="utf-8")
            clean_json.write_text(
                json.dumps({"entries": [pred("track2_0001", "content", "content")]}),
                encoding="utf-8",
            )
            inclusive_json.write_text(
                json.dumps({"entries": [pred("track2_0001", "content", "calm")]}),
                encoding="utf-8",
            )
            high_similarity_json.write_text(
                json.dumps({"entries": [{"sample_id": "track2_0001"}]}),
                encoding="utf-8",
            )
            img1 = tmp_path / "track2_0001.jpg"
            Image.new("RGB", (64, 48), (200, 100, 20)).save(img1)
            with zipfile.ZipFile(image_zip, "w") as zf:
                zf.write(img1, "images/track2_0001.jpg")

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--current-json",
                    str(current_json),
                    "--clean-predictions-json",
                    str(clean_json),
                    "--inclusive-predictions-json",
                    str(inclusive_json),
                    "--high-similarity-json",
                    str(high_similarity_json),
                    "--image-zip",
                    str(image_zip),
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            payload = json.loads(result.stdout)
            self.assertEqual(payload["row_count"], 1)
            self.assertEqual(payload["category_counts"]["inclusive_only_change"], 1)
            self.assertEqual(payload["category_counts"]["clean_only_change"], 0)
            self.assertEqual(payload["category_counts"]["clean_inclusive_conflict"], 0)
            self.assertEqual(payload["high_similarity_count"], 1)
            self.assertEqual(set(payload["outputs"]), {"html", "csv", "json", "markdown"})
            self.assertIn("track2_clean_inclusive_disagreement_review.html", payload["outputs"]["html"])
            self.assertTrue((out_dir / "html_review" / "track2_clean_inclusive_disagreement_review.html").exists())
            self.assertTrue((out_dir / "track2_clean_inclusive_disagreement_decisions.csv").exists())

    def test_cli_missing_high_similarity_manifest_fails_cleanly(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(__file__).resolve().parents[1]
            script = repo_root / "scripts" / "build_track2_clean_inclusive_disagreement_review.py"
            tmp_path = Path(tmp)
            current_json = tmp_path / "current.json"
            clean_json = tmp_path / "clean.json"
            inclusive_json = tmp_path / "inclusive.json"
            high_similarity_json = tmp_path / "missing_high_similarity.json"
            image_zip = tmp_path / "track2.zip"
            out_dir = tmp_path / "out"
            current_json.write_text(json.dumps([current_row("track2_0001", "content")]), encoding="utf-8")
            clean_json.write_text(
                json.dumps({"entries": [pred("track2_0001", "content", "content")]}),
                encoding="utf-8",
            )
            inclusive_json.write_text(
                json.dumps({"entries": [pred("track2_0001", "content", "calm")]}),
                encoding="utf-8",
            )
            img1 = tmp_path / "track2_0001.jpg"
            Image.new("RGB", (64, 48), (200, 100, 20)).save(img1)
            with zipfile.ZipFile(image_zip, "w") as zf:
                zf.write(img1, "images/track2_0001.jpg")

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--current-json",
                    str(current_json),
                    "--clean-predictions-json",
                    str(clean_json),
                    "--inclusive-predictions-json",
                    str(inclusive_json),
                    "--high-similarity-json",
                    str(high_similarity_json),
                    "--image-zip",
                    str(image_zip),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing required input", result.stderr)

    def test_cli_null_high_similarity_manifest_fails_cleanly(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(__file__).resolve().parents[1]
            script = repo_root / "scripts" / "build_track2_clean_inclusive_disagreement_review.py"
            tmp_path = Path(tmp)
            current_json = tmp_path / "current.json"
            clean_json = tmp_path / "clean.json"
            inclusive_json = tmp_path / "inclusive.json"
            high_similarity_json = tmp_path / "null_high_similarity.json"
            image_zip = tmp_path / "track2.zip"
            out_dir = tmp_path / "out"
            current_json.write_text(json.dumps([current_row("track2_0001", "content")]), encoding="utf-8")
            clean_json.write_text(
                json.dumps({"entries": [pred("track2_0001", "content", "content")]}),
                encoding="utf-8",
            )
            inclusive_json.write_text(
                json.dumps({"entries": [pred("track2_0001", "content", "calm")]}),
                encoding="utf-8",
            )
            high_similarity_json.write_text("null", encoding="utf-8")
            img1 = tmp_path / "track2_0001.jpg"
            Image.new("RGB", (64, 48), (200, 100, 20)).save(img1)
            with zipfile.ZipFile(image_zip, "w") as zf:
                zf.write(img1, "images/track2_0001.jpg")

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--current-json",
                    str(current_json),
                    "--clean-predictions-json",
                    str(clean_json),
                    "--inclusive-predictions-json",
                    str(inclusive_json),
                    "--high-similarity-json",
                    str(high_similarity_json),
                    "--image-zip",
                    str(image_zip),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid input", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertNotIn("TypeError", result.stderr)


if __name__ == "__main__":
    unittest.main()
