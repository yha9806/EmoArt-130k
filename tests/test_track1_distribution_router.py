import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track1_distribution_router import (
    build_distribution_route,
    build_distribution_routes,
    load_contract_rows,
    load_reference_family_bank,
    write_distribution_route_reports,
)


def _contract(sample_id: str, caption: str, **overrides):
    row = {
        "sample_id": sample_id,
        "caption": caption,
        "aspect_plan": {"label": "portrait_poster", "width": 768, "height": 1024},
        "reference_contract": {"required": False, "categories": []},
        "text_contract": {"required": False, "modes": []},
        "relation_contract": {"required": False, "checks": []},
        "fid_risk": [],
    }
    row.update(overrides)
    return row


class Track1DistributionRouterTest(unittest.TestCase):
    def test_kremlin_route_has_hard_landmark_and_style_freedom(self):
        route = build_distribution_route(
            _contract(
                "track1_0803",
                "A Socialist Realism propaganda poster with the Soviet, American, and British flags above a Kremlin tower.",
                reference_contract={"required": True, "categories": ["landmark", "flags"]},
                text_contract={"required": True, "modes": ["cyrillic"]},
            )
        )

        self.assertEqual(route["sample_id"], "track1_0803")
        self.assertEqual(route["family_id"], "kremlin_red_square")
        self.assertIn("landmark_reference", route["hard_constraints"])
        self.assertIn("caption_content", route["hard_constraints"])
        self.assertIn("aspect_variation_allowed", route["style_freedom"])
        self.assertIn("style_variation_allowed", route["style_freedom"])
        self.assertEqual(
            [strategy["strategy"] for strategy in route["candidate_strategies"]],
            ["aas_safe", "reference_style", "fid_diverse"],
        )

    def test_scroll_route_does_not_allow_free_crop_of_support(self):
        route = build_distribution_route(
            _contract(
                "track1_0063",
                "A vertical hanging scroll with silk mounting, roller rods, calligraphy, and an ink landscape.",
            )
        )

        self.assertEqual(route["family_id"], "scroll_album_paper_support")
        self.assertIn("support_surface_required", route["hard_constraints"])
        self.assertIn("preserve_support_aspect", route["style_freedom"])
        self.assertNotIn("free_crop_allowed", route["style_freedom"])

    def test_generic_route_has_safe_fallback_family_and_strategies(self):
        route = build_distribution_route(
            _contract(
                "track1_0999",
                "A calm watercolor painting of a quiet room with soft light and delicate brushwork.",
            )
        )

        self.assertEqual(route["family_id"], "generic_artwork")
        self.assertIn("caption_content", route["hard_constraints"])
        self.assertEqual(
            [strategy["strategy"] for strategy in route["candidate_strategies"]],
            ["aas_safe", "reference_style", "fid_diverse"],
        )

    def test_bank_hints_propagate_without_prompt_metadata_ids(self):
        bank = {
            "rows": [
                {
                    "sample_id": "track1_0999",
                    "family_id": "agriculture_industry_worker",
                    "reference_path": "/tmp/emoart/reference/agriculture.jpg",
                    "aspect": {"label": "landscape", "width": 1600, "height": 900, "ratio": 1.777777},
                    "composition_hints": ["field depth", "working hands"],
                    "medium_hints": ["oil_paint_surface", "aged_canvas"],
                }
            ],
            "families": {
                "agriculture_industry_worker": {
                    "family_id": "agriculture_industry_worker",
                    "composition_hints": ["family field composition"],
                    "medium_hints": ["family medium"],
                    "aspect_labels": {"landscape": 1},
                }
            },
        }

        route = build_distribution_route(
            _contract("track1_0999", "A quiet generic scene."),
            reference_family_bank=bank,
        )

        self.assertEqual(route["family_id"], "agriculture_industry_worker")
        self.assertEqual(route["aspect_hints"][0]["label"], "landscape")
        self.assertIn("field depth", route["composition_hints"])
        self.assertIn("working hands", route["composition_hints"])
        self.assertEqual(route["medium_options"], ["oil_paint_surface", "aged_canvas"])
        self.assertEqual(route["reference_assets"], ["/tmp/emoart/reference/agriculture.jpg"])
        serialized = json.dumps(route)
        self.assertNotIn("reference_path", serialized)
        self.assertNotIn("matched_terms", serialized)

    def test_family_reference_assets_propagate_when_sample_has_no_direct_bank_row(self):
        bank = {
            "rows": [],
            "families": {
                "kremlin_red_square": {
                    "family_id": "kremlin_red_square",
                    "composition_hints": ["red brick tower"],
                    "medium_hints": ["poster paint"],
                    "aspect_labels": {"landscape": 2, "portrait": 1},
                    "reference_assets": [
                        "/tmp/emoart/reference/kremlin_a.jpg",
                        "/tmp/emoart/reference/kremlin_b.jpg",
                    ],
                }
            },
        }

        route = build_distribution_route(
            _contract("track1_0803", "A Kremlin tower with searchlights."),
            reference_family_bank=bank,
        )

        self.assertEqual(
            route["reference_assets"],
            ["/tmp/emoart/reference/kremlin_a.jpg", "/tmp/emoart/reference/kremlin_b.jpg"],
        )

    def test_build_routes_writes_reports_and_cli(self):
        contracts = [
            _contract("track1_0803", "A Kremlin tower with Soviet flags and searchlights."),
            _contract("track1_0077", "A Soviet naval poster with a sailor hoisting red and white flags."),
        ]
        bank = {
            "rows": [
                {
                    "sample_id": "track1_0077",
                    "family_id": "naval_aviation_vehicle",
                    "aspect": {"label": "portrait", "width": 768, "height": 1024},
                    "composition_hints": ["uniformed sailor", "flag diagonal"],
                    "medium_hints": ["propaganda_poster_print"],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts_json = root / "contracts.json"
            bank_json = root / "bank.json"
            out_json = root / "routes.json"
            out_csv = root / "routes.csv"
            out_md = root / "routes.md"
            contracts_json.write_text(json.dumps({"rows": contracts}), encoding="utf-8")
            bank_json.write_text(json.dumps(bank), encoding="utf-8")

            routes = build_distribution_routes(contracts, reference_family_bank=bank)
            write_distribution_route_reports(routes, json_path=out_json, csv_path=out_csv, md_path=out_md)
            self.assertIn("kremlin_red_square", out_md.read_text(encoding="utf-8"))

            module = self._load_cli_module()
            code = module.main(
                [
                    "--contracts-json",
                    str(contracts_json),
                    "--reference-family-bank-json",
                    str(bank_json),
                    "--out-json",
                    str(out_json),
                    "--out-csv",
                    str(out_csv),
                    "--out-md",
                    str(out_md),
                ]
            )

            self.assertEqual(code, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total"], 2)
            self.assertIn("kremlin_red_square", payload["summary"]["families"])
            self.assertIn("naval_aviation_vehicle", payload["summary"]["families"])
            self.assertIn("candidate_strategies", out_csv.read_text(encoding="utf-8"))
            self.assertIn("naval_aviation_vehicle", out_md.read_text(encoding="utf-8"))

    def test_load_contract_rows_accepts_list_rows_or_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = [
                [_contract("track1_0001", "A generic artwork.")],
                {"rows": [_contract("track1_0002", "A generic artwork.")]},
                {"contracts": [_contract("track1_0003", "A generic artwork.")]},
            ]
            for payload in payloads:
                path = root / "contracts.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                rows = load_contract_rows(path)
                self.assertEqual(len(rows), 1)

    def test_report_writing_rejects_protected_output_paths(self):
        routes = [build_distribution_route(_contract("track1_0803", "A Kremlin tower."))]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected_paths = [
                root / "submissions" / "track1_submission.json",
                root / "submissions" / "track1_submission.zip",
                root / "submissions" / "track1" / "images" / "track1_0001.png",
            ]
            for protected_path in protected_paths:
                out_json = protected_path if protected_path.suffix == ".json" else root / "routes.json"
                out_csv = protected_path if protected_path.suffix == ".zip" else root / "routes.csv"
                out_md = protected_path if "images" in protected_path.parts else root / "routes.md"
                with self.subTest(protected_path=protected_path):
                    with self.assertRaisesRegex(ValueError, "protected output path"):
                        write_distribution_route_reports(
                            routes,
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
            contracts_json = root / "contracts.json"
            contracts_json.write_text(
                json.dumps([_contract("track1_0803", "A Kremlin tower.")]),
                encoding="utf-8",
            )
            out_json = root / "submissions" / "track1_submission.json"
            out_csv = root / "safe.csv"
            out_md = root / "safe.md"
            module = self._load_cli_module()
            module.ROOT = root

            code = module.main(
                [
                    "--contracts-json",
                    str(contracts_json),
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

    def test_load_reference_family_bank_accepts_path_or_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank_json = root / "bank.json"
            bank_json.write_text(json.dumps({"rows": []}), encoding="utf-8")

            self.assertIsNone(load_reference_family_bank(None))
            self.assertEqual(load_reference_family_bank(bank_json), {"rows": []})

    def _load_cli_module(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "track1_distribution_router.py"
        spec = importlib.util.spec_from_file_location("track1_distribution_router_cli", script)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
