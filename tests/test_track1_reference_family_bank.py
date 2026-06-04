import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from affectiveart.track1_reference_family_bank import (
    build_reference_family_bank,
    classify_aspect,
    infer_reference_family,
    load_reference_rows,
    write_reference_family_reports,
)


class Track1ReferenceFamilyBankTest(unittest.TestCase):
    def test_classify_aspect_labels_landscape_portrait_and_squareish(self):
        self.assertEqual(classify_aspect(1600, 900)["label"], "landscape")
        self.assertEqual(classify_aspect(768, 1024)["label"], "portrait")
        self.assertEqual(classify_aspect(1000, 950)["label"], "square_ish")

        aspect = classify_aspect(1600, 900)
        self.assertEqual(aspect["width"], 1600)
        self.assertEqual(aspect["height"], 900)
        self.assertAlmostEqual(aspect["ratio"], 1600 / 900)

    def test_infer_reference_family_uses_caption_and_path_terms(self):
        caption = "A Socialist Realism poster with Kremlin towers and blue searchlights."
        family = infer_reference_family("track1_0803", caption, "refs/kremlin_searchlight_01.jpg")

        self.assertEqual(family["family_id"], "kremlin_red_square")
        self.assertIn("kremlin", family["matched_terms"])
        self.assertIn("searchlight", family["matched_terms"])
        self.assertIn("searchlight", family["composition_hints"])

    def test_build_reference_family_bank_summarizes_fixture_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (1600, 900), (120, 50, 40)).save(root / "kremlin_landscape.jpg")
            Image.new("RGB", (768, 1024), (40, 80, 120)).save(root / "naval_poster.jpg")
            rows = [
                {
                    "sample_id": "track1_0803",
                    "caption": "A Socialist Realism propaganda poster with Kremlin tower and searchlights.",
                    "reference_path": str(root / "kremlin_landscape.jpg"),
                },
                {
                    "sample_id": "track1_0077",
                    "caption": "A Soviet naval propaganda poster with a sailor hoisting red and white flags.",
                    "path": str(root / "naval_poster.jpg"),
                },
                {
                    "sample_id": "track1_missing",
                    "caption": "A quiet generic artwork study.",
                    "reference_path": str(root / "missing.jpg"),
                },
            ]
            bank = build_reference_family_bank(rows)

        self.assertEqual(bank["summary"]["total"], 3)
        self.assertEqual(bank["summary"]["missing_images"], 1)
        self.assertEqual(bank["summary"]["aspect_labels"]["landscape"], 1)
        self.assertEqual(bank["summary"]["aspect_labels"]["portrait"], 1)
        self.assertIn("kremlin_red_square", bank["families"])
        self.assertIn("naval_aviation_vehicle", bank["families"])
        self.assertIn("generic_artwork", bank["families"])

    def test_write_reports_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (1600, 900), (120, 50, 40)).save(root / "kremlin.jpg")
            input_json = root / "references.json"
            input_json.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "sample_id": "track1_0803",
                                "caption": "A Kremlin tower with searchlights.",
                                "reference_path": str(root / "kremlin.jpg"),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out_json = root / "bank.json"
            out_csv = root / "bank.csv"
            out_md = root / "bank.md"
            rows = load_reference_rows(input_json)
            bank = build_reference_family_bank(rows)
            write_reference_family_reports(bank, json_path=out_json, csv_path=out_csv, md_path=out_md)

            script = Path(__file__).resolve().parents[1] / "scripts" / "track1_reference_family_bank.py"
            spec = importlib.util.spec_from_file_location("track1_reference_family_bank_cli", script)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            code = module.main(
                [
                    "--references-json",
                    str(input_json),
                    "--out-json",
                    str(out_json),
                    "--out-csv",
                    str(out_csv),
                    "--out-md",
                    str(out_md),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_md.exists())
            self.assertIn("kremlin_red_square", out_json.read_text(encoding="utf-8"))
            self.assertIn("kremlin_red_square", out_csv.read_text(encoding="utf-8"))
            self.assertIn("kremlin_red_square", out_md.read_text(encoding="utf-8"))
