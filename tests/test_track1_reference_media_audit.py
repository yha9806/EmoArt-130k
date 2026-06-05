import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from affectiveart.track1_reference_media_audit import (
    build_reference_media_audit,
    render_reference_media_audit_html,
    write_reference_media_audit_artifacts,
)


def _image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color).save(path)
    return str(path)


class Track1ReferenceMediaAuditTest(unittest.TestCase):
    def test_audit_detects_changed_sample_source_risk_and_tainted_partial_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            old_ship = _image(assets / "oil_ship.jpg", (20, 60, 90))
            old_landscape = _image(assets / "landscape.jpg", (50, 90, 40))
            poster = _image(assets / "Kukryniksy-TASSWindow929.jpg", (160, 30, 20))
            sample_oil = _image(assets / "sample_oil_worker.jpg", (90, 80, 70))
            old_candidate_dir = root / "old_candidates"
            _image(old_candidate_dir / "images" / "track1_0077_aas_safe_c01.png", (1, 2, 3))
            _image(old_candidate_dir / "images" / "track1_0491_aas_safe_c01.png", (4, 5, 6))
            old_routes = [
                {
                    "sample_id": "track1_0077",
                    "caption": "A Socialist Realism naval propaganda poster with bold Cyrillic typography.",
                    "family_id": "naval_aviation_vehicle",
                    "reference_asset_source": "family",
                    "reference_assets": [old_ship, old_landscape],
                },
                {
                    "sample_id": "track1_0491",
                    "caption": "A Socialist Realism agricultural poster with a farm woman.",
                    "family_id": "generic_artwork",
                    "reference_asset_source": "sample",
                    "reference_assets": [sample_oil],
                },
            ]
            new_routes = [
                {
                    "sample_id": "track1_0077",
                    "caption": "A Socialist Realism naval propaganda poster with bold Cyrillic typography.",
                    "family_id": "naval_aviation_vehicle",
                    "reference_asset_source": "caption_media",
                    "reference_style_key": "socialist realism:poster_print",
                    "reference_assets": [poster],
                },
                {
                    "sample_id": "track1_0491",
                    "caption": "A Socialist Realism agricultural poster with a farm woman.",
                    "family_id": "generic_artwork",
                    "reference_asset_source": "sample",
                    "reference_assets": [sample_oil],
                },
            ]

            report = build_reference_media_audit(
                old_routes,
                new_routes,
                partial_candidate_image_dir=old_candidate_dir / "images",
            )

        self.assertEqual(report["summary"]["changed_samples"], 1)
        self.assertEqual(report["changed_rows"][0]["sample_id"], "track1_0077")
        self.assertEqual(report["summary"]["sample_source_manual_review"], 1)
        self.assertEqual(report["sample_source_risks"][0]["sample_id"], "track1_0491")
        self.assertEqual(report["summary"]["tainted_partial_samples"], 2)
        self.assertEqual(report["tainted_partial_samples"], ["track1_0077", "track1_0491"])

    def test_write_artifacts_creates_boards_and_chinese_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            old_asset = _image(assets / "oil_ship.jpg", (20, 60, 90))
            new_asset = _image(assets / "PosterVictory.jpg", (160, 30, 20))
            report = build_reference_media_audit(
                [
                    {
                        "sample_id": "track1_0077",
                        "caption": "A propaganda poster.",
                        "reference_asset_source": "family",
                        "reference_assets": [old_asset],
                    }
                ],
                [
                    {
                        "sample_id": "track1_0077",
                        "caption": "A propaganda poster.",
                        "reference_asset_source": "caption_media",
                        "reference_style_key": "socialist realism:poster_print",
                        "reference_assets": [new_asset],
                    }
                ],
            )
            out_dir = root / "audit"

            paths = write_reference_media_audit_artifacts(report, out_dir=out_dir)
            html = Path(paths["html"]).read_text(encoding="utf-8")
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))

            self.assertEqual(payload["summary"]["changed_samples"], 1)
            self.assertTrue((out_dir / "reference_boards" / "old" / "track1_0077.jpg").exists())
            self.assertTrue((out_dir / "reference_boards" / "new" / "track1_0077.jpg").exists())
            self.assertIn("Track1 Reference Media Audit", html)
            self.assertIn("旧 reference board", html)
            self.assertIn("v2 reference board", html)
            self.assertIn("caption_media", html)

    def test_render_html_escapes_untrusted_caption(self):
        report = build_reference_media_audit(
            [
                {
                    "sample_id": 'track1_0001"><script>alert(1)</script>',
                    "caption": "bad <script>alert(2)</script>",
                    "reference_asset_source": "family",
                    "reference_assets": [],
                }
            ],
            [
                {
                    "sample_id": 'track1_0001"><script>alert(1)</script>',
                    "caption": "bad <script>alert(2)</script>",
                    "reference_asset_source": "caption_media",
                    "reference_assets": [],
                }
            ],
        )

        html = render_reference_media_audit_html(report, out_html=Path("/tmp/audit.html"))

        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_write_artifacts_sanitizes_sample_id_board_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = _image(root / "asset.jpg", (1, 2, 3))
            report = build_reference_media_audit(
                [
                    {
                        "sample_id": "../track1_0001<script>",
                        "caption": "A poster.",
                        "reference_asset_source": "family",
                        "reference_assets": [asset],
                    }
                ],
                [
                    {
                        "sample_id": "../track1_0001<script>",
                        "caption": "A poster.",
                        "reference_asset_source": "caption_media",
                        "reference_assets": [asset],
                    }
                ],
            )
            out_dir = root / "audit"

            write_reference_media_audit_artifacts(report, out_dir=out_dir)

            board_root = (out_dir / "reference_boards").resolve()
            boards = list(board_root.rglob("*.jpg"))
            self.assertEqual(len(boards), 2)
            for board in boards:
                self.assertTrue(board.resolve().is_relative_to(board_root))
                self.assertNotIn("..", board.name)
                self.assertNotIn("<", board.name)

    def test_cli_writes_reference_media_audit_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            old_asset = _image(assets / "oil_ship.jpg", (20, 60, 90))
            new_asset = _image(assets / "PosterVictory.jpg", (160, 30, 20))
            old_routes = root / "old.json"
            new_routes = root / "new.json"
            old_routes.write_text(
                json.dumps(
                    {
                        "routes": [
                            {
                                "sample_id": "track1_0077",
                                "caption": "A propaganda poster.",
                                "reference_asset_source": "family",
                                "reference_assets": [old_asset],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            new_routes.write_text(
                json.dumps(
                    {
                        "routes": [
                            {
                                "sample_id": "track1_0077",
                                "caption": "A propaganda poster.",
                                "reference_asset_source": "caption_media",
                                "reference_assets": [new_asset],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out_dir = root / "audit"
            module = _load_cli_script()

            code = module.main(
                [
                    "--old-routes-json",
                    str(old_routes),
                    "--new-routes-json",
                    str(new_routes),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "track1_reference_media_audit.json").exists())
            self.assertTrue((out_dir / "track1_reference_media_audit_zh.html").exists())


if __name__ == "__main__":
    unittest.main()


def _load_cli_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "track1_reference_media_audit.py"
    spec = importlib.util.spec_from_file_location("track1_reference_media_audit_cli", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module
