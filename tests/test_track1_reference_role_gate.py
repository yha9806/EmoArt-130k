import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from affectiveart.track1_reference_role_gate import (
    build_reference_role_gate,
    write_reference_role_gate_artifacts,
)


def _image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 100), color).save(path)
    return str(path)


class Track1ReferenceRoleGateTest(unittest.TestCase):
    def test_poster_caption_without_poster_reference_fails_medium_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oil_ship = _image(root / "FishingHarbor.jpg", (20, 60, 90))
            route = {
                "sample_id": "track1_0077",
                "caption": "A Socialist Realism Soviet naval propaganda poster with bold Cyrillic typography.",
                "reference_assets": [oil_ship],
                "reference_asset_notes": ["official retrieval; source=Images\\Socialist Realism\\FishingHarbor.jpg"],
            }

            report = build_reference_role_gate([route])

        row = report["rows"][0]
        self.assertEqual(row["status"], "fail")
        self.assertIn("poster_print", row["missing_required_roles"])
        self.assertEqual(report["summary"]["fail"], 1)

    def test_poster_caption_with_tass_reference_passes_medium_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poster = _image(root / "Kukryniksy-TASSWindow929.jpg", (170, 30, 30))
            route = {
                "sample_id": "track1_0019",
                "caption": "A Socialist Realism propaganda poster with bold Cyrillic text.",
                "reference_assets": [poster],
            }

            report = build_reference_role_gate([route])

        row = report["rows"][0]
        self.assertNotIn("poster_print", row["missing_required_roles"])
        self.assertEqual(row["role_checks"][0]["status"], "pass")

    def test_train_window_caption_requires_train_window_reference_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            portrait = _image(root / "PortraitofaMilitaryMan.jpg", (40, 80, 60))
            route = {
                "sample_id": "track1_0091",
                "caption": (
                    "A Socialist Realism poster of a uniformed border guard saluting joyful passengers "
                    "leaning from a train window beside a red and green frontier post."
                ),
                "reference_assets": [portrait],
            }

            report = build_reference_role_gate([route])

        row = report["rows"][0]
        self.assertEqual(row["status"], "fail")
        self.assertIn("train_window", row["missing_required_roles"])

    def test_candidate_reference_fallback_is_hard_failure(self):
        route = {
            "sample_id": "track1_0747",
            "caption": "A Socialist Realism wartime propaganda poster with mounted soldiers and fleeing civilians.",
            "reference_assets": [],
        }
        candidate_manifest = {
            "rows": [
                {
                    "sample_id": "track1_0747",
                    "candidate_strategy": "aas_safe",
                    "reference_image_fallback_without_reference": True,
                    "reference_image_fallback_reason": "Gemini blocked the request",
                }
            ]
        }

        report = build_reference_role_gate([route], candidate_manifest=candidate_manifest)

        row = report["rows"][0]
        self.assertEqual(row["status"], "fail")
        self.assertIn("provider_reference_fallback", row["risk_flags"])
        self.assertIn("Gemini blocked", row["fallback_reason"])

    def test_write_artifacts_creates_chinese_html_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poster = _image(root / "Kukryniksy-TASSWindow929.jpg", (170, 30, 30))
            report = build_reference_role_gate(
                [
                    {
                        "sample_id": 'track1_0001"><script>',
                        "caption": "A propaganda poster <script>bad</script>.",
                        "reference_assets": [poster],
                    }
                ]
            )

            paths = write_reference_role_gate_artifacts(report, out_dir=root / "gate")
            html = Path(paths["html"]).read_text(encoding="utf-8")
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["summary"]["total"], 1)
        self.assertIn("Track1 Reference Role Gate", html)
        self.assertIn("角色质量门", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_cli_writes_role_gate_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            routes_json = root / "routes.json"
            routes_json.write_text(
                json.dumps(
                    {
                        "routes": [
                            {
                                "sample_id": "track1_0091",
                                "caption": "A poster with passengers leaning from a train window.",
                                "reference_assets": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out_dir = root / "out"
            module = _load_cli_script()

            code = module.main(["--routes-json", str(routes_json), "--out-dir", str(out_dir)])

            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "track1_reference_role_gate.json").exists())
            self.assertTrue((out_dir / "track1_reference_role_gate_zh.html").exists())

    def test_cli_returns_error_for_malformed_routes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            routes_json = root / "routes.json"
            routes_json.write_text("{not valid json", encoding="utf-8")
            module = _load_cli_script()

            code = module.main(["--routes-json", str(routes_json), "--out-dir", str(root / "out")])

        self.assertEqual(code, 2)


def _load_cli_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "track1_reference_role_gate.py"
    spec = importlib.util.spec_from_file_location("track1_reference_role_gate_cli", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
