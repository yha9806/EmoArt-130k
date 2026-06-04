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

    def test_build_reference_family_bank_includes_deterministic_medium_hints(self):
        rows = [
            {
                "sample_id": "track1_poster",
                "caption": "A Socialist Realism propaganda poster with bold Cyrillic typography.",
                "reference_path": "refs/soviet_poster.jpg",
            },
            {
                "sample_id": "track1_scroll",
                "caption": "Abstract branching lines on graph paper with a folded ruled page edge.",
                "reference_path": "refs/graph_paper.jpg",
            },
            {
                "sample_id": "track1_painting",
                "caption": "A watercolor and ink brushwork study on canvas.",
                "reference_path": "refs/watercolor_canvas.jpg",
            },
            {
                "sample_id": "track1_document",
                "caption": "A surrender treaty document tableau with officers around a table.",
                "reference_path": "refs/surrender_document.jpg",
            },
            {
                "sample_id": "track1_generic",
                "caption": "A quiet generic artwork study.",
                "reference_path": "refs/generic.jpg",
            },
        ]

        bank = build_reference_family_bank(rows)
        repeated_bank = build_reference_family_bank(rows)

        mediums_by_id = {row["sample_id"]: row["medium_hints"] for row in bank["rows"]}
        repeated_mediums_by_id = {row["sample_id"]: row["medium_hints"] for row in repeated_bank["rows"]}
        self.assertEqual(mediums_by_id, repeated_mediums_by_id)
        self.assertEqual(mediums_by_id["track1_poster"][0], "propaganda_poster_print")
        self.assertEqual(mediums_by_id["track1_scroll"][0], "scroll_or_album_paper_support")
        self.assertEqual(mediums_by_id["track1_painting"][0], "painting_or_brushwork_surface")
        self.assertEqual(mediums_by_id["track1_document"][0], "document_or_tableau_surface")
        self.assertEqual(mediums_by_id["track1_generic"][0], "generic_artwork_surface")
        self.assertIn("medium_hints", bank["families"]["scroll_album_paper_support"])
        self.assertIn(
            "scroll_or_album_paper_support",
            bank["families"]["scroll_album_paper_support"]["medium_hints"],
        )

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

    def test_write_reports_include_medium_hints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "sample_id": "track1_0803",
                    "caption": "A Socialist Realism propaganda poster with Kremlin tower and searchlights.",
                    "reference_path": str(root / "missing.jpg"),
                }
            ]
            bank = build_reference_family_bank(rows)
            out_json = root / "bank.json"
            out_csv = root / "bank.csv"
            out_md = root / "bank.md"

            write_reference_family_reports(bank, json_path=out_json, csv_path=out_csv, md_path=out_md)
            json_payload = json.loads(out_json.read_text(encoding="utf-8"))
            csv_text = out_csv.read_text(encoding="utf-8")
            md_text = out_md.read_text(encoding="utf-8")

        self.assertEqual(json_payload["rows"][0]["medium_hints"][0], "propaganda_poster_print")
        self.assertIn("propaganda_poster_print", json_payload["families"]["kremlin_red_square"]["medium_hints"])
        self.assertIn("medium_hints", csv_text)
        self.assertIn("propaganda_poster_print", csv_text)
        self.assertIn("Medium hints", md_text)
        self.assertIn("propaganda_poster_print", md_text)

    def test_rejects_protected_output_paths_before_writing(self):
        bank = build_reference_family_bank(
            [
                {
                    "sample_id": "track1_0803",
                    "caption": "A Kremlin tower with searchlights.",
                    "reference_path": "",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected_paths = [
                root / "submissions" / "track1_submission.json",
                root / "submissions" / "track1_submission.zip",
                root / "submissions" / "track1" / "images" / "track1_0001.png",
            ]
            for protected_path in protected_paths:
                out_json = protected_path if protected_path.suffix == ".json" else root / "bank.json"
                out_csv = protected_path if protected_path.suffix == ".zip" else root / "bank.csv"
                out_md = protected_path if "images" in protected_path.parts else root / "bank.md"
                with self.subTest(protected_path=protected_path):
                    with self.assertRaisesRegex(ValueError, "protected output path"):
                        write_reference_family_reports(
                            bank,
                            json_path=out_json,
                            csv_path=out_csv,
                            md_path=out_md,
                            repo_root=root,
                        )
                    self.assertFalse(out_json.exists())
                    self.assertFalse(out_csv.exists())
                    self.assertFalse(out_md.exists())

    def test_cli_rejects_protected_outputs_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (1600, 900), (120, 50, 40)).save(root / "kremlin.jpg")
            input_json = root / "references.json"
            input_json.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "track1_0803",
                            "caption": "A Kremlin tower with searchlights.",
                            "reference_path": str(root / "kremlin.jpg"),
                        }
                    ]
                ),
                encoding="utf-8",
            )
            out_json = root / "submissions" / "track1_submission.json"
            out_csv = root / "safe.csv"
            out_md = root / "safe.md"
            module = self._load_cli_module()
            module.ROOT = root

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

            self.assertNotEqual(code, 0)
            self.assertFalse(out_json.exists())
            self.assertFalse(out_csv.exists())
            self.assertFalse(out_md.exists())

    def test_load_reference_rows_resolves_relative_paths_from_json_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (1600, 900), (120, 50, 40)).save(root / "rel_image.jpg")
            input_json = root / "references.json"
            input_json.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "track1_0803",
                            "caption": "A Kremlin tower with searchlights.",
                            "reference_path": "rel_image.jpg",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            rows = load_reference_rows(input_json)
            bank = build_reference_family_bank(rows)

        self.assertEqual(Path(rows[0]["reference_path"]), root / "rel_image.jpg")
        self.assertEqual(bank["summary"]["missing_images"], 0)
        self.assertEqual(bank["summary"]["aspect_labels"]["landscape"], 1)

    def test_load_reference_rows_rejects_malformed_rows_with_indexed_errors(self):
        malformed_payloads = [
            (["not a row"], r"row 0.*object"),
            ([{"caption": "Missing sample.", "reference_path": "ref.jpg"}], r"row 0.*sample_id"),
            ([{"sample_id": "track1_0001", "reference_path": "ref.jpg"}], r"row 0.*caption"),
            ([{"sample_id": "track1_0001", "caption": "Missing path."}], r"row 0.*reference_path.*path"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for payload, pattern in malformed_payloads:
                input_json = root / "references.json"
                input_json.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=payload):
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_reference_rows(input_json)

    def _load_cli_module(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "track1_reference_family_bank.py"
        spec = importlib.util.spec_from_file_location("track1_reference_family_bank_cli", script)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
