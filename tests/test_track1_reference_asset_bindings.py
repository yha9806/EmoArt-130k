import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from affectiveart.track1_reference_asset_bindings import (
    attach_reference_assets_to_routes,
    load_reference_asset_index,
    load_route_rows,
    write_reference_asset_route_reports,
)


def _route(sample_id="track1_0803", family_id="kremlin_red_square"):
    return {
        "sample_id": sample_id,
        "caption": "A Kremlin poster.",
        "family_id": family_id,
        "candidate_strategies": [{"strategy": "aas_safe"}],
    }


class Track1ReferenceAssetBindingsTest(unittest.TestCase):
    def test_direct_sample_references_are_bound_as_existing_asset_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            Image.new("RGB", (320, 180), (100, 40, 40)).save(assets / "direct.jpg")
            index = {
                "references": {
                    "track1_0803": [
                        {"file": "direct.jpg", "note": "direct Kremlin reference"},
                    ]
                }
            }

            routes = attach_reference_assets_to_routes([_route()], index, asset_root=assets)

        self.assertEqual(len(routes[0]["reference_assets"]), 1)
        self.assertTrue(routes[0]["reference_assets"][0].endswith("direct.jpg"))
        self.assertEqual(routes[0]["reference_asset_source"], "sample")
        self.assertEqual(routes[0]["reference_asset_notes"], ["direct Kremlin reference"])

    def test_family_fallback_references_are_used_when_sample_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            Image.new("RGB", (320, 180), (100, 40, 40)).save(assets / "family_a.jpg")
            Image.new("RGB", (180, 320), (40, 80, 120)).save(assets / "family_b.jpg")
            index = {
                "references": {},
                "family_references": {
                    "kremlin_red_square": [
                        {"file": "family_a.jpg", "note": "family first"},
                        {"file": "family_b.jpg", "note": "family second"},
                    ]
                },
            }

            routes = attach_reference_assets_to_routes([_route()], index, asset_root=assets, max_assets_per_route=1)

        self.assertEqual(len(routes[0]["reference_assets"]), 1)
        self.assertTrue(routes[0]["reference_assets"][0].endswith("family_a.jpg"))
        self.assertEqual(routes[0]["reference_asset_source"], "family")

    def test_caption_style_references_prevent_generic_family_template_collapse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            Image.new("RGB", (180, 320), (210, 205, 180)).save(assets / "gongbi.jpg")
            index = {
                "references": {},
                "family_references": {},
                "caption_style_references": {
                    "gongbi": [{"file": "gongbi.jpg", "note": "official Gongbi style reference"}],
                },
            }
            route = _route(sample_id="track1_0063", family_id="generic_artwork")
            route["caption"] = "A Gongbi vertical scroll landscape with calligraphy and red seals."

            routes = attach_reference_assets_to_routes([route], index, asset_root=assets)

        self.assertEqual(len(routes[0]["reference_assets"]), 1)
        self.assertTrue(routes[0]["reference_assets"][0].endswith("gongbi.jpg"))
        self.assertEqual(routes[0]["reference_asset_source"], "caption_style")
        self.assertEqual(routes[0]["reference_style_key"], "gongbi")

    def test_writes_reports_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            Image.new("RGB", (320, 180), (100, 40, 40)).save(assets / "direct.jpg")
            routes_json = root / "routes.json"
            index_json = root / "index.json"
            out_json = root / "out.json"
            out_csv = root / "out.csv"
            out_md = root / "out.md"
            routes_json.write_text(json.dumps({"routes": [_route()]}), encoding="utf-8")
            index_json.write_text(
                json.dumps({"references": {"track1_0803": [{"file": "direct.jpg"}]}}),
                encoding="utf-8",
            )

            routes = attach_reference_assets_to_routes(
                load_route_rows(routes_json),
                load_reference_asset_index(index_json),
                asset_root=assets,
            )
            write_reference_asset_route_reports(routes, json_path=out_json, csv_path=out_csv, md_path=out_md)
            module = self._load_cli_module()
            code = module.main(
                [
                    "--routes-json",
                    str(routes_json),
                    "--reference-assets-index-json",
                    str(index_json),
                    "--reference-assets-dir",
                    str(assets),
                    "--out-json",
                    str(out_json),
                    "--out-csv",
                    str(out_csv),
                    "--out-md",
                    str(out_md),
                ]
            )
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            md_text = out_md.read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertEqual(payload["summary"]["routes_with_reference_assets"], 1)
        self.assertIn("direct.jpg", md_text)

    def _load_cli_module(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "track1_attach_reference_assets.py"
        spec = importlib.util.spec_from_file_location("track1_attach_reference_assets_cli", path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
