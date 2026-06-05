import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from affectiveart.track1_reference_role_gate import build_reference_role_gate
from affectiveart.track1_reference_role_repair import (
    repair_reference_routes,
    write_reference_role_repair_artifacts,
)


def _image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 100), color).save(path)
    return str(path)


class Track1ReferenceRoleRepairTest(unittest.TestCase):
    def test_missing_poster_print_role_is_repaired_from_poster_donor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oil_ship = _image(root / "FishingHarbor.jpg", (20, 60, 90))
            poster = _image(root / "Kukryniksy-TASSWindow929.jpg", (170, 30, 30))
            routes = [
                {
                    "sample_id": "track1_0476",
                    "caption": "A Socialist Realism propaganda poster with bold Cyrillic lettering and ships.",
                    "family_id": "naval_aviation_vehicle",
                    "reference_assets": [oil_ship],
                    "reference_asset_notes": ["official retrieval; source=FishingHarbor.jpg"],
                },
                {
                    "sample_id": "track1_0019",
                    "caption": "A Socialist Realism poster of heroic soldiers.",
                    "family_id": "battle_tank_cavalry",
                    "reference_assets": [poster],
                    "reference_asset_notes": ["official retrieval; source=Kukryniksy-TASSWindow929.jpg"],
                },
            ]

            result = repair_reference_routes(routes, repair_roles=["poster_print"], max_assets_per_route=2)

        repaired = {row["sample_id"]: row for row in result["routes"]}
        self.assertTrue(repaired["track1_0476"]["reference_assets"][0].endswith("Kukryniksy-TASSWindow929.jpg"))
        self.assertTrue(repaired["track1_0476"]["reference_role_repaired"])
        self.assertEqual(repaired["track1_0476"]["reference_role_added"], ["poster_print"])
        gate = build_reference_role_gate([repaired["track1_0476"]])
        self.assertNotIn("poster_print", gate["rows"][0]["missing_required_roles"])
        self.assertEqual(result["summary"]["repaired_routes"], 1)

    def test_repair_preserves_existing_poster_reference_without_adding_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poster = _image(root / "Kukryniksy-TASSWindow929.jpg", (170, 30, 30))
            route = {
                "sample_id": "track1_0077",
                "caption": "A Socialist Realism propaganda poster with bold Cyrillic typography.",
                "reference_assets": [poster],
                "reference_asset_notes": ["official retrieval; source=Kukryniksy-TASSWindow929.jpg"],
            }

            result = repair_reference_routes([route], repair_roles=["poster_print"], max_assets_per_route=4)

        self.assertEqual(result["routes"][0]["reference_assets"], [poster])
        self.assertFalse(result["routes"][0].get("reference_role_repaired", False))
        self.assertEqual(result["summary"]["repaired_routes"], 0)

    def test_unresolved_when_no_matching_donor_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oil_ship = _image(root / "FishingHarbor.jpg", (20, 60, 90))
            route = {
                "sample_id": "track1_0476",
                "caption": "A Socialist Realism propaganda poster with bold Cyrillic lettering.",
                "reference_assets": [oil_ship],
            }

            result = repair_reference_routes([route], repair_roles=["poster_print"], max_assets_per_route=4)

        self.assertEqual(result["summary"]["unresolved_routes"], 1)
        self.assertEqual(result["unresolved_rows"][0]["missing_role"], "poster_print")

    def test_repair_does_not_trim_existing_unique_required_role_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oil_a = _image(root / "OilA.jpg", (20, 60, 90))
            oil_b = _image(root / "OilB.jpg", (30, 70, 100))
            oil_c = _image(root / "OilC.jpg", (40, 80, 110))
            train = _image(root / "RailwayTrainWindowPassengers.jpg", (80, 80, 80))
            poster = _image(root / "Kukryniksy-TASSWindow929.jpg", (170, 30, 30))
            routes = [
                {
                    "sample_id": "track1_0091",
                    "caption": (
                        "A Socialist Realism poster of a border guard saluting joyful passengers "
                        "leaning from a train window."
                    ),
                    "reference_assets": [oil_a, oil_b, oil_c, train],
                },
                {
                    "sample_id": "track1_0019",
                    "caption": "A Socialist Realism propaganda poster.",
                    "reference_assets": [poster],
                },
            ]

            result = repair_reference_routes(routes, repair_roles=["poster_print"], max_assets_per_route=4)

        repaired = {row["sample_id"]: row for row in result["routes"]}["track1_0091"]
        names = [Path(asset).name for asset in repaired["reference_assets"]]
        self.assertIn("Kukryniksy-TASSWindow929.jpg", names)
        self.assertIn("RailwayTrainWindowPassengers.jpg", names)
        self.assertNotIn("OilC.jpg", names)
        gate = build_reference_role_gate([repaired])["rows"][0]
        self.assertNotIn("poster_print", gate["missing_required_roles"])
        self.assertNotIn("train_window", gate["missing_required_roles"])

    def test_write_artifacts_outputs_routes_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poster = _image(root / "Kukryniksy-TASSWindow929.jpg", (170, 30, 30))
            result = repair_reference_routes(
                [
                    {
                        "sample_id": "track1_0019",
                        "caption": "A Socialist Realism propaganda poster.",
                        "reference_assets": [poster],
                    }
                ]
            )

            paths = write_reference_role_repair_artifacts(result, out_dir=root / "repair")
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            md = Path(paths["md"]).read_text(encoding="utf-8")

        self.assertEqual(payload["summary"]["total"], 1)
        self.assertIn("Track1 Reference Role Repair", md)

    def test_cli_writes_repaired_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oil_ship = _image(root / "FishingHarbor.jpg", (20, 60, 90))
            poster = _image(root / "Kukryniksy-TASSWindow929.jpg", (170, 30, 30))
            routes_json = root / "routes.json"
            routes_json.write_text(
                json.dumps(
                    {
                        "routes": [
                            {
                                "sample_id": "track1_0476",
                                "caption": "A Socialist Realism propaganda poster.",
                                "reference_assets": [oil_ship],
                            },
                            {
                                "sample_id": "track1_0019",
                                "caption": "A Socialist Realism propaganda poster.",
                                "reference_assets": [poster],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out_dir = root / "out"
            module = _load_cli_script()

            code = module.main(["--routes-json", str(routes_json), "--out-dir", str(out_dir)])

            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "track1_reference_role_repair.json").exists())
            self.assertTrue((out_dir / "track1_distribution_routes_role_repaired.json").exists())


def _load_cli_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "track1_reference_role_repair.py"
    spec = importlib.util.spec_from_file_location("track1_reference_role_repair_cli", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
