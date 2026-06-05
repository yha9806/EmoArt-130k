import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class Track1StrategyReviewHtmlTest(unittest.TestCase):
    def test_cli_writes_chinese_grouped_strategy_review_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "current.jpg"
            old_candidate = root / "old.jpg"
            reference = root / "reference.jpg"
            aas = root / "aas.png"
            reference_style = root / "reference_style.png"
            fid = root / "fid.png"
            Image.new("RGB", (768, 1024), (80, 80, 80)).save(current)
            Image.new("RGB", (768, 1024), (90, 80, 70)).save(old_candidate)
            Image.new("RGB", (1600, 900), (30, 80, 100)).save(reference)
            Image.new("RGB", (768, 1024), (100, 50, 30)).save(aas)
            Image.new("RGB", (1600, 900), (100, 80, 30)).save(reference_style)
            Image.new("RGB", (1200, 900), (30, 100, 50)).save(fid)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "sample_id": "track1_0803",
                                "caption": "Kremlin tower with searchlights.",
                                "candidate_strategy": "aas_safe",
                                "image_path": str(aas),
                                "current_image_path": str(current),
                                "old_candidate_image_path": str(old_candidate),
                                "reference_paths": [str(reference)],
                                "family_id": "kremlin_red_square",
                                "provider_prompt": "strict caption-first prompt",
                            },
                            {
                                "sample_id": "track1_0803",
                                "caption": "Kremlin tower with searchlights.",
                                "candidate_strategy": "reference_style",
                                "image_path": str(reference_style),
                                "current_image_path": str(current),
                                "old_candidate_image_path": str(old_candidate),
                                "reference_paths": [str(reference)],
                                "family_id": "kremlin_red_square",
                                "provider_prompt": "reference family prompt",
                            },
                            {
                                "sample_id": "track1_0803",
                                "caption": "Kremlin tower with searchlights.",
                                "candidate_strategy": "fid_diverse",
                                "image_path": str(fid),
                                "current_image_path": str(current),
                                "old_candidate_image_path": str(old_candidate),
                                "reference_paths": [str(reference)],
                                "family_id": "kremlin_red_square",
                                "provider_prompt": "less templated prompt",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            out_html = root / "review.html"
            module = _load_script()
            code = module.main(["--manifest-json", str(manifest), "--out-html", str(out_html)])
            html = out_html.read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertIn("Track1 三策略候选评审", html)
        self.assertIn("当前提交包", html)
        self.assertIn("旧候选", html)
        self.assertIn("AAS 安全", html)
        self.assertIn("参考风格", html)
        self.assertIn("FID 多样化", html)
        self.assertIn("参考族", html)
        self.assertIn("kremlin_red_square", html)
        self.assertIn("object-fit: contain", html)
        self.assertEqual(html.count("<section class=\"sample-card\""), 1)

    def test_loader_accepts_list_candidates_and_packets_shapes(self):
        module = _load_script()
        self.assertEqual(module.load_review_rows([{"sample_id": "track1_0001"}]), [{"sample_id": "track1_0001"}])
        self.assertEqual(module.load_review_rows({"candidates": [{"sample_id": "track1_0002"}]}), [{"sample_id": "track1_0002"}])
        self.assertEqual(module.load_review_rows({"packets": [{"sample_id": "track1_0003"}]}), [{"sample_id": "track1_0003"}])

    def test_loader_rejects_empty_or_malformed_rows(self):
        module = _load_script()
        with self.assertRaisesRegex(ValueError, "no review rows"):
            module.load_review_rows({"rows": []})
        with self.assertRaisesRegex(ValueError, "review row 0 must be an object"):
            module.load_review_rows({"rows": ["not an object"]})
        with self.assertRaisesRegex(ValueError, "review row 0 missing required sample_id"):
            module.load_review_rows({"rows": [{"caption": "missing id"}]})

    def test_relative_image_paths_are_rebased_from_manifest_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            out_dir = root / "review"
            assets.mkdir()
            out_dir.mkdir()
            Image.new("RGB", (20, 20), (1, 2, 3)).save(assets / "candidate.png")
            module = _load_script()

            html = module.render_html(
                [
                    {
                        "sample_id": "track1_0001",
                        "candidate_strategy": "aas_safe",
                        "image_path": "assets/candidate.png",
                    }
                ],
                out_html=out_dir / "review.html",
                asset_base_dir=root,
            )

        self.assertIn('src="../assets/candidate.png"', html)

    def test_render_html_escapes_untrusted_values(self):
        module = _load_script()
        html = module.render_html(
            [
                {
                    "sample_id": 'track1_0001"><script>alert(1)</script>',
                    "caption": "caption <script>alert(2)</script>",
                    "family_id": "family <b>bad</b>",
                    "candidate_strategy": "aas_safe",
                    "image_path": "assets/<bad>.png",
                    "provider_prompt": "prompt </pre><script>alert(3)</script>",
                }
            ],
            out_html=Path("/tmp/review.html"),
            asset_base_dir=Path("/tmp"),
        )

        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;b&gt;bad&lt;/b&gt;", html)
        self.assertIn("&lt;/pre&gt;", html)

    def test_cli_rejects_protected_submission_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"rows": [{"sample_id": "track1_0001"}]}), encoding="utf-8")
            module = _load_script()
            code = module.main(
                [
                    "--manifest-json",
                    str(manifest),
                    "--out-html",
                    str(Path("submissions/track1_submission.json")),
                ]
            )

        self.assertEqual(code, 2)


def _load_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "track1_render_strategy_review_html.py"
    spec = importlib.util.spec_from_file_location("track1_render_strategy_review_html_cli", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module
