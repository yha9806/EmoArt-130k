import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track1_prompt_lint import (
    DEFAULT_PHRASES,
    lint_prompt_batch,
    load_prompt_rows,
    write_prompt_lint_reports,
)


class Track1PromptLintTest(unittest.TestCase):
    def test_detects_front_facing_portrait_and_flat_poster_collapse(self):
        rows = [
            {
                "sample_id": f"track1_{index:04d}",
                "provider_prompt": (
                    "front-facing flat printed poster portrait poster canvas "
                    "medium-distance figures graphic poster composition"
                ),
            }
            for index in range(7)
        ] + [
            {
                "sample_id": "track1_9999",
                "provider_prompt": "loose watercolor landscape with an asymmetrical crop",
            }
        ]

        report = lint_prompt_batch(rows, threshold=0.6)

        self.assertEqual(report["summary"]["status"], "fail")
        self.assertEqual(report["summary"]["total"], 8)
        self.assertIn("front-facing", report["summary"]["failed_phrases"])
        self.assertIn("portrait poster canvas", report["summary"]["failed_phrases"])
        self.assertIn("flat printed poster", report["summary"]["failed_phrases"])
        self.assertEqual(report["phrases"]["front-facing"]["count"], 7)
        self.assertGreaterEqual(report["phrases"]["front-facing"]["ratio"], 0.6)
        self.assertIn("track1_0000", report["phrases"]["front-facing"]["sample_ids"])
        self.assertIn("fewer, larger", DEFAULT_PHRASES)
        self.assertIn("fewer, larger text blocks", DEFAULT_PHRASES)

    def test_clean_diverse_batch_passes(self):
        rows = [
            {"sample_id": "track1_0001", "provider_prompt": "wide oil painting with deep crowd scene"},
            {"sample_id": "track1_0002", "provider_prompt": "vertical scroll with silk support and ink"},
            {"sample_id": "track1_0003", "provider_prompt": "square watercolor composition with soft light"},
            {"sample_id": "track1_0004", "provider_prompt": "document tableau around a formal table"},
        ]

        report = lint_prompt_batch(rows, threshold=0.6)

        self.assertEqual(report["summary"]["status"], "pass")
        self.assertEqual(report["summary"]["failed_phrases"], [])
        self.assertEqual(report["summary"]["warned_phrases"], [])
        self.assertEqual(report["summary"]["justified_phrases"], [])

    def test_exact_threshold_boundary_does_not_fail(self):
        rows = [
            {"sample_id": f"track1_{index:04d}", "provider_prompt": "front-facing figure study"}
            for index in range(3)
        ] + [
            {"sample_id": "track1_0003", "provider_prompt": "wide landscape"},
            {"sample_id": "track1_0004", "provider_prompt": "vertical scroll"},
        ]

        report = lint_prompt_batch(rows, threshold=0.6)

        self.assertEqual(report["phrases"]["front-facing"]["ratio"], 0.6)
        self.assertEqual(report["phrases"]["front-facing"]["status"], "pass")
        self.assertEqual(report["summary"]["status"], "pass")
        self.assertNotIn("front-facing", report["summary"]["failed_phrases"])

    def test_detects_slash_variant_for_fewer_larger_text_blocks(self):
        rows = [
            {
                "sample_id": f"track1_{index:04d}",
                "provider_prompt": "Use fewer/larger text blocks across the poster area.",
            }
            for index in range(4)
        ] + [
            {"sample_id": "track1_0004", "provider_prompt": "loose brushwork without typography"},
        ]

        report = lint_prompt_batch(rows, threshold=0.6)

        self.assertEqual(report["summary"]["status"], "fail")
        self.assertIn("fewer/larger text blocks", report["summary"]["failed_phrases"])
        self.assertEqual(report["phrases"]["fewer/larger text blocks"]["count"], 4)

    def test_caption_required_and_metadata_justification_warns_instead_of_failing(self):
        rows = [
            {
                "sample_id": "track1_0001",
                "caption": "A historical exhibition poster with visible printed poster support.",
                "provider_prompt": "portrait poster canvas with flat printed poster ink",
            },
            {
                "sample_id": "track1_0002",
                "caption": "A campaign poster with bold typography.",
                "provider_prompt": "portrait poster canvas with flat printed poster texture",
                "surface_contract": {"required": True, "surface": "poster"},
            },
            {
                "sample_id": "track1_0003",
                "caption": "A museum poster for a public lecture.",
                "provider_prompt": "portrait poster canvas with flat printed poster layout",
                "justified_template_phrases": ["portrait poster canvas", "flat printed poster"],
            },
        ]

        report = lint_prompt_batch(rows, threshold=0.6)

        self.assertEqual(report["summary"]["status"], "warn")
        self.assertEqual(report["summary"]["failed_phrases"], [])
        self.assertIn("portrait poster canvas", report["summary"]["warned_phrases"])
        self.assertIn("flat printed poster", report["summary"]["justified_phrases"])
        self.assertEqual(report["phrases"]["portrait poster canvas"]["justified_count"], 3)
        self.assertIn("caption_mentions_poster", report["phrases"]["portrait poster canvas"]["justifications"])

    def test_unjustified_matches_keep_high_frequency_phrase_failed(self):
        rows = [
            {
                "sample_id": f"track1_poster_{index:04d}",
                "caption": "A public event poster with visible printed support.",
                "provider_prompt": "portrait poster canvas with thick ink texture",
            }
            for index in range(8)
        ] + [
            {
                "sample_id": "track1_nonposter_0001",
                "caption": "A quiet watercolor landscape.",
                "provider_prompt": "portrait poster canvas with thick ink texture",
            },
            {
                "sample_id": "track1_nonposter_0002",
                "caption": "An oil painting of a crowded factory floor.",
                "provider_prompt": "portrait poster canvas with thick ink texture",
            },
        ]

        report = lint_prompt_batch(rows, threshold=0.6)

        self.assertEqual(report["summary"]["status"], "fail")
        self.assertIn("portrait poster canvas", report["summary"]["failed_phrases"])
        self.assertEqual(report["phrases"]["portrait poster canvas"]["justified_count"], 8)
        self.assertEqual(
            report["phrases"]["portrait poster canvas"]["unjustified_sample_ids"],
            ["track1_nonposter_0001", "track1_nonposter_0002"],
        )

    def test_front_facing_is_not_automatically_justified_for_posters(self):
        rows = [
            {
                "sample_id": f"track1_{index:04d}",
                "caption": "A propaganda poster with bold typography.",
                "provider_prompt": "front-facing propaganda poster",
            }
            for index in range(4)
        ]

        report = lint_prompt_batch(rows, threshold=0.6)

        self.assertEqual(report["summary"]["status"], "fail")
        self.assertIn("front-facing", report["summary"]["failed_phrases"])

    def test_load_prompt_rows_accepts_json_jsonl_and_prompt_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_list = root / "rows.json"
            json_rows = root / "wrapped_rows.json"
            json_packets = root / "packets.json"
            jsonl = root / "rows.jsonl"
            prompt_dir = root / "prompts"
            prompt_dir.mkdir()

            json_list.write_text(
                json.dumps([{"sample_id": "track1_0001", "provider_prompt": "wide landscape"}]),
                encoding="utf-8",
            )
            json_rows.write_text(
                json.dumps({"rows": [{"sample_id": "track1_0002", "provider_prompt": "vertical scroll"}]}),
                encoding="utf-8",
            )
            json_packets.write_text(
                json.dumps({"packets": [{"sample_id": "track1_0003", "provider_prompt": "oil painting"}]}),
                encoding="utf-8",
            )
            jsonl.write_text(
                "\n".join(
                    [
                        json.dumps({"sample_id": "track1_0004", "provider_prompt": "watercolor"}),
                        json.dumps({"sample_id": "track1_0005", "prompt": "document tableau"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (prompt_dir / "track1_0006.txt").write_text("loose ink study", encoding="utf-8")

            self.assertEqual(load_prompt_rows(json_list)[0]["sample_id"], "track1_0001")
            self.assertEqual(load_prompt_rows(json_rows)[0]["sample_id"], "track1_0002")
            self.assertEqual(load_prompt_rows(json_packets)[0]["sample_id"], "track1_0003")
            self.assertEqual([row["sample_id"] for row in load_prompt_rows(jsonl)], ["track1_0004", "track1_0005"])
            self.assertEqual(load_prompt_rows(prompt_dir), [{"sample_id": "track1_0006", "provider_prompt": "loose ink study"}])

    def test_malformed_prompt_rows_reject_with_indexed_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed = root / "rows.json"
            malformed.write_text(json.dumps({"rows": [{"sample_id": "track1_0001"}]}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "prompt row 0"):
                load_prompt_rows(malformed)

            with self.assertRaisesRegex(ValueError, "prompt row 1"):
                lint_prompt_batch(
                    [
                        {"sample_id": "track1_0001", "provider_prompt": "wide landscape"},
                        {"provider_prompt": "missing id"},
                    ]
                )

            with self.assertRaisesRegex(ValueError, "prompt row 0 missing required provider_prompt or prompt"):
                lint_prompt_batch([{"sample_id": "track1_0002", "provider_prompt": {"text": "wide landscape"}}])

            with self.assertRaisesRegex(ValueError, "prompt row 0 missing required provider_prompt or prompt"):
                lint_prompt_batch([{"sample_id": "track1_0003", "prompt": ["wide landscape"]}])

            malformed.write_text(
                json.dumps({"rows": [{"sample_id": "track1_0004", "provider_prompt": {"text": "wide"}}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "prompt row 0 missing required provider_prompt or prompt"):
                load_prompt_rows(malformed)

    def test_empty_inputs_reject_in_api_and_loaders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty_list = root / "empty_list.json"
            empty_rows = root / "empty_rows.json"
            empty_packets = root / "empty_packets.json"
            empty_dir = root / "empty_prompts"
            empty_dir.mkdir()
            empty_list.write_text("[]", encoding="utf-8")
            empty_rows.write_text(json.dumps({"rows": []}), encoding="utf-8")
            empty_packets.write_text(json.dumps({"packets": []}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no prompt rows"):
                lint_prompt_batch([])
            for source in [empty_list, empty_rows, empty_packets, empty_dir]:
                with self.subTest(source=source):
                    with self.assertRaisesRegex(ValueError, "no prompt rows"):
                        load_prompt_rows(source)

    def test_cli_returns_two_for_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets = root / "packets.json"
            packets.write_text(json.dumps({"packets": []}), encoding="utf-8")
            out_json = root / "lint.json"
            out_md = root / "lint.md"
            module = self._load_cli_module()

            code = module.main(
                [
                    "--packets-json",
                    str(packets),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                ]
            )

            self.assertEqual(code, 2)
            self.assertFalse(out_json.exists())
            self.assertFalse(out_md.exists())

    def test_reports_and_cli_return_two_on_fail_without_allow_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets = root / "packets.json"
            out_json = root / "lint.json"
            out_md = root / "lint.md"
            packets.write_text(
                json.dumps(
                    {
                        "packets": [
                            {"sample_id": "track1_0001", "provider_prompt": "front-facing portrait poster canvas"},
                            {"sample_id": "track1_0002", "provider_prompt": "front-facing portrait poster canvas"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = lint_prompt_batch(json.loads(packets.read_text(encoding="utf-8"))["packets"], threshold=0.6)
            write_prompt_lint_reports(report, json_path=out_json, md_path=out_md)

            module = self._load_cli_module()
            code = module.main(
                [
                    "--packets-json",
                    str(packets),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                    "--threshold",
                    "0.6",
                ]
            )
            allow_code = module.main(
                [
                    "--packets-json",
                    str(packets),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                    "--allow-fail",
                ]
            )

            self.assertEqual(code, 2)
            self.assertEqual(allow_code, 0)
            self.assertIn("front-facing", out_json.read_text(encoding="utf-8"))
            self.assertIn("Status: `fail`", out_md.read_text(encoding="utf-8"))

    def test_protected_output_paths_are_rejected_before_writing(self):
        report = lint_prompt_batch(
            [{"sample_id": "track1_0001", "provider_prompt": "front-facing portrait poster canvas"}]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected_paths = [
                root / "submissions" / "track1_submission.json",
                root / "submissions" / "track1_submission.zip",
                root / "submissions" / "track1" / "images" / "track1_0001.png",
            ]
            for protected_path in protected_paths:
                out_json = protected_path if protected_path.suffix in {".json", ".zip"} else root / "lint.json"
                out_md = protected_path if "images" in protected_path.parts else root / "lint.md"
                with self.subTest(protected_path=protected_path):
                    with self.assertRaisesRegex(ValueError, "protected output path"):
                        write_prompt_lint_reports(report, json_path=out_json, md_path=out_md, repo_root=root)
                    self.assertFalse(out_json.exists())
                    self.assertFalse(out_md.exists())

    def _load_cli_module(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "track1_prompt_lint.py"
        spec = importlib.util.spec_from_file_location("track1_prompt_lint_cli", script)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
