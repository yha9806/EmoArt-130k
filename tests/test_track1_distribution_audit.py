import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from affectiveart.track1_distribution_audit import (
    audit_candidate_distribution,
    load_manifest_rows,
    write_distribution_audit_reports,
)


class Track1DistributionAuditTest(unittest.TestCase):
    def test_audit_reports_aspects_prompt_phrases_and_image_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            landscape = root / "landscape.png"
            portrait = root / "portrait.png"
            Image.new("RGB", (120, 60), (100, 100, 100)).save(landscape)
            Image.new("RGB", (40, 80), (10, 20, 30)).save(portrait)
            rows = [
                {
                    "sample_id": "track1_0001",
                    "image_path": str(landscape),
                    "candidate_strategy": "fid_diverse",
                    "family_id": "kremlin_red_square",
                    "provider_prompt": "front-facing flat printed poster",
                },
                {
                    "sample_id": "track1_0002",
                    "image_path": str(portrait),
                    "candidate_strategy": "aas_safe",
                    "review_metadata": {"family_id": "naval_aviation_vehicle"},
                    "provider_prompt": "front-facing flat printed poster",
                },
            ]

            report = audit_candidate_distribution(rows)

        self.assertEqual(report["summary"]["total"], 2)
        self.assertEqual(report["summary"]["existing_image_count"], 2)
        self.assertEqual(report["summary"]["missing_image_count"], 0)
        self.assertEqual(report["summary"]["aspect_labels"], {"landscape": 1, "portrait": 1})
        self.assertEqual(report["summary"]["strategy_counts"], {"aas_safe": 1, "fid_diverse": 1})
        self.assertEqual(report["summary"]["family_counts"], {"kremlin_red_square": 1, "naval_aviation_vehicle": 1})
        self.assertEqual(report["prompt_lint"]["phrases"]["front-facing"]["count"], 2)
        self.assertEqual(report["prompt_lint"]["summary"]["status"], "fail")
        self.assertEqual(report["rows"][0]["width"], 120)
        self.assertEqual(report["rows"][0]["height"], 60)
        self.assertEqual(report["rows"][0]["aspect"]["label"], "landscape")
        self.assertEqual(report["rows"][0]["mean_luma"], 100.0)
        self.assertEqual(report["rows"][0]["luma_stddev"], 0.0)
        self.assertEqual(report["rows"][1]["aspect"]["label"], "portrait")

    def test_family_counts_accept_direct_and_nested_metadata(self):
        rows = [
            {
                "sample_id": "track1_0001",
                "provider_prompt": "wide oil painting",
                "family_id": "direct_family",
            },
            {
                "sample_id": "track1_0002",
                "provider_prompt": "vertical scroll",
                "review_metadata": {"family_id": "nested_family"},
            },
            {
                "sample_id": "track1_0003",
                "provider_prompt": "square watercolor",
                "review_metadata": {"family_id": "nested_family"},
            },
        ]

        report = audit_candidate_distribution(rows)

        self.assertEqual(report["summary"]["family_counts"], {"direct_family": 1, "nested_family": 2})

    def test_missing_image_path_or_file_counts_missing_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.png"
            report = audit_candidate_distribution(
                [
                    {"sample_id": "track1_0001", "provider_prompt": "wide landscape"},
                    {"sample_id": "track1_0002", "image_path": str(missing), "provider_prompt": "vertical scroll"},
                ]
            )

        self.assertEqual(report["summary"]["existing_image_count"], 0)
        self.assertEqual(report["summary"]["missing_image_count"], 2)
        self.assertEqual(report["summary"]["aspect_labels"], {"missing": 2})
        self.assertFalse(report["rows"][0]["exists"])
        self.assertFalse(report["rows"][1]["exists"])

    def test_corrupt_image_counts_unreadable_and_keeps_error_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corrupt = root / "corrupt.png"
            out_json = root / "audit.json"
            out_md = root / "audit.md"
            corrupt.write_text("not an image", encoding="utf-8")

            report = audit_candidate_distribution(
                [
                    {
                        "sample_id": "track1_0001",
                        "image_path": str(corrupt),
                        "provider_prompt": "wide landscape",
                    }
                ]
            )
            write_distribution_audit_reports(report, json_path=out_json, md_path=out_md)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            markdown = out_md.read_text(encoding="utf-8")

        self.assertEqual(report["summary"]["existing_image_count"], 0)
        self.assertEqual(report["summary"]["missing_image_count"], 1)
        self.assertEqual(report["summary"]["aspect_labels"], {"unreadable": 1})
        self.assertFalse(report["rows"][0]["exists"])
        self.assertEqual(report["rows"][0]["aspect"]["label"], "unreadable")
        self.assertTrue(report["rows"][0]["image_error"])
        self.assertTrue(payload["rows"][0]["image_error"])
        self.assertIn("unreadable", markdown)

    def test_loader_supports_list_rows_candidates_and_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = {
                "list.json": [{"sample_id": "track1_0001", "provider_prompt": "wide landscape"}],
                "rows.json": {"rows": [{"sample_id": "track1_0002", "prompt": "vertical scroll"}]},
                "candidates.json": {"candidates": [{"sample_id": "track1_0003", "provider_prompt": "oil painting"}]},
                "packets.json": {"packets": [{"sample_id": "track1_0004", "provider_prompt": "document tableau"}]},
            }
            for name, payload in cases.items():
                path = root / name
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(name=name):
                    rows = load_manifest_rows(path)
                    self.assertEqual(len(rows), 1)
                    self.assertTrue(rows[0]["sample_id"].startswith("track1_"))

    def test_loader_rejects_malformed_and_empty_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_shape = root / "bad_shape.json"
            bad_row = root / "bad_row.json"
            missing_id = root / "missing_id.json"
            missing_prompt = root / "missing_prompt.json"
            empty = root / "empty.json"
            bad_shape.write_text(json.dumps({"items": []}), encoding="utf-8")
            bad_row.write_text(json.dumps({"rows": ["not an object"]}), encoding="utf-8")
            missing_id.write_text(json.dumps({"rows": [{"provider_prompt": "wide landscape"}]}), encoding="utf-8")
            missing_prompt.write_text(json.dumps({"rows": [{"sample_id": "track1_0001"}]}), encoding="utf-8")
            empty.write_text(json.dumps({"rows": []}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest JSON"):
                load_manifest_rows(bad_shape)
            with self.assertRaisesRegex(ValueError, "manifest row 0 must be an object"):
                load_manifest_rows(bad_row)
            with self.assertRaisesRegex(ValueError, "manifest row 0 missing required sample_id"):
                load_manifest_rows(missing_id)
            with self.assertRaisesRegex(ValueError, "manifest row 0 missing required provider_prompt or prompt"):
                load_manifest_rows(missing_prompt)
            with self.assertRaisesRegex(ValueError, "no manifest rows"):
                load_manifest_rows(empty)
            with self.assertRaisesRegex(ValueError, "no manifest rows"):
                audit_candidate_distribution([])

    def test_loader_rejects_non_string_prompt_values_with_indexed_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            object_prompt = root / "object_prompt.json"
            list_prompt = root / "list_prompt.json"
            object_prompt.write_text(
                json.dumps({"rows": [{"sample_id": "track1_0001", "provider_prompt": {"text": "wide"}}]}),
                encoding="utf-8",
            )
            list_prompt.write_text(
                json.dumps({"rows": [{"sample_id": "track1_0002", "prompt": ["wide"]}]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "manifest row 0 missing required provider_prompt or prompt"):
                load_manifest_rows(object_prompt)
            with self.assertRaisesRegex(ValueError, "manifest row 0 missing required provider_prompt or prompt"):
                load_manifest_rows(list_prompt)
            with self.assertRaisesRegex(ValueError, "manifest row 0 missing required provider_prompt or prompt"):
                audit_candidate_distribution([{"sample_id": "track1_0003", "provider_prompt": {"text": "wide"}}])

    def test_reports_and_cli_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            manifest = root / "manifest.json"
            out_json = root / "audit.json"
            out_md = root / "audit.md"
            Image.new("RGB", (50, 50), (70, 80, 90)).save(image_path)
            manifest.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "sample_id": "track1_0001",
                                "image_path": str(image_path),
                                "candidate_strategy": "reference_style",
                                "provider_prompt": "square watercolor study",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = audit_candidate_distribution(load_manifest_rows(manifest))
            write_distribution_audit_reports(report, json_path=out_json, md_path=out_md)

            module = self._load_cli_module()
            code = module.main(
                [
                    "--manifest-json",
                    str(manifest),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                ]
            )
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            markdown = out_md.read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertEqual(payload["summary"]["total"], 1)
        self.assertIn("Track1 Distribution Audit", markdown)

    def test_cli_returns_two_for_bad_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            out_json = root / "audit.json"
            out_md = root / "audit.md"
            manifest.write_text(json.dumps({"packets": []}), encoding="utf-8")

            code = self._load_cli_module().main(
                [
                    "--manifest-json",
                    str(manifest),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                ]
            )

        self.assertEqual(code, 2)
        self.assertFalse(out_json.exists())
        self.assertFalse(out_md.exists())

    def test_protected_output_paths_are_rejected_before_writing(self):
        report = audit_candidate_distribution(
            [{"sample_id": "track1_0001", "provider_prompt": "wide landscape"}]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected_paths = [
                root / "submissions" / "track1_submission.json",
                root / "submissions" / "track1_submission.zip",
                root / "submissions" / "track1" / "images" / "track1_0001.png",
            ]
            for protected_path in protected_paths:
                out_json = protected_path if protected_path.suffix in {".json", ".zip"} else root / "audit.json"
                out_md = protected_path if "images" in protected_path.parts else root / "audit.md"
                with self.subTest(protected_path=protected_path):
                    with self.assertRaisesRegex(ValueError, "protected output path"):
                        write_distribution_audit_reports(report, json_path=out_json, md_path=out_md, repo_root=root)
                    self.assertFalse(out_json.exists())
                    self.assertFalse(out_md.exists())

    def _load_cli_module(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "track1_distribution_audit.py"
        spec = importlib.util.spec_from_file_location("track1_distribution_audit_cli", script)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
