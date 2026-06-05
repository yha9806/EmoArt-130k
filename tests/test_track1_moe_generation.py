import base64
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track1_moe_generation import (
    DEFAULT_FLASH_IMAGE_MODEL,
    build_generation_jobs,
    resolve_image_model,
    run_generation_jobs,
    write_generation_manifest,
)


PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLv"
    "AAAAAElFTkSuQmCC"
)


def _packet(**overrides):
    row = {
        "rank": 1,
        "sample_id": "track1_0803",
        "caption": "A Socialist Realism propaganda poster with Soviet, American, and British flags.",
        "provider_prompt": "Create a single finished artwork image.\nProduce only the image.",
        "review_metadata": {
            "recommended_model": "gemini-3.1-flash-image",
            "candidate_count": 2,
            "primary_expert": "poster_expert",
        },
        "aspect_plan": {"width": 768, "height": 1024, "label": "portrait_poster"},
    }
    row.update(overrides)
    return row


class FakeImageResult:
    image_b64 = PNG_1X1
    metadata = {"fake": True}


class FakeProvider:
    def __init__(self, model):
        self.model = model
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs, "model": self.model})
        return FakeImageResult()


class FakeProviderFactory:
    def __init__(self):
        self.providers = []

    def __call__(self, model):
        provider = FakeProvider(model)
        self.providers.append(provider)
        return provider


class Track1MoeGenerationTest(unittest.TestCase):
    def test_build_jobs_expands_candidate_count_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = build_generation_jobs([_packet()], out_dir=Path(tmp))

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["candidate_index"], 1)
        self.assertEqual(jobs[1]["candidate_index"], 2)
        self.assertTrue(jobs[0]["image_path"].endswith("images/track1_0803_c01.png"))
        self.assertEqual(jobs[0]["width"], 768)
        self.assertEqual(jobs[0]["height"], 1024)
        self.assertEqual(jobs[0]["model"], DEFAULT_FLASH_IMAGE_MODEL)
        self.assertNotIn("track1_0803", jobs[0]["provider_prompt"])

    def test_build_jobs_keeps_strategy_variants_on_distinct_paths(self):
        packets = [
            _packet(candidate_strategy="aas_safe", review_metadata={**_packet()["review_metadata"], "candidate_count": 1}),
            _packet(candidate_strategy="reference_style", review_metadata={**_packet()["review_metadata"], "candidate_count": 1}),
            _packet(candidate_strategy="fid_diverse", review_metadata={**_packet()["review_metadata"], "candidate_count": 1}),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            jobs = build_generation_jobs(packets, out_dir=Path(tmp), max_candidates_per_sample=1)

        image_names = [Path(job["image_path"]).name for job in jobs]
        prompt_names = [Path(job["prompt_path"]).name for job in jobs]
        self.assertEqual(len(set(image_names)), 3)
        self.assertEqual(len(set(prompt_names)), 3)
        self.assertEqual(
            image_names,
            [
                "track1_0803_aas_safe_c01.png",
                "track1_0803_reference_style_c01.png",
                "track1_0803_fid_diverse_c01.png",
            ],
        )

    def test_resolve_image_model_uses_flash_default_unless_pro_override_is_supplied(self):
        self.assertEqual(resolve_image_model("gemini-3.1-flash-image"), DEFAULT_FLASH_IMAGE_MODEL)
        self.assertEqual(resolve_image_model("gemini-3-pro-image"), DEFAULT_FLASH_IMAGE_MODEL)
        self.assertEqual(
            resolve_image_model("gemini-3-pro-image", pro_image_model="gemini-3-pro-image-preview"),
            "gemini-3-pro-image-preview",
        )

    def test_run_generation_jobs_writes_png_metadata_and_manifest_rows(self):
        factory = FakeProviderFactory()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = build_generation_jobs([_packet()], out_dir=root, max_candidates_per_sample=1)
            rows = run_generation_jobs(jobs, provider_factory=factory)
            write_generation_manifest(rows, root / "candidate_manifest.json")
            manifest = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8"))

            image_path = Path(rows[0]["image_path"])
            metadata_path = Path(rows[0]["metadata_path"])

            self.assertEqual(rows[0]["status"], "generated")
            self.assertTrue(image_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(manifest["summary"]["generated"], 1)
            self.assertEqual(len(factory.providers), 1)
            self.assertEqual(factory.providers[0].calls[0]["kwargs"]["raw_prompt"], True)

    def test_cli_dry_run_writes_manifest_without_generating_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets_jsonl = root / "packets.jsonl"
            out_dir = root / "out"
            packets_jsonl.write_text(json.dumps(_packet(), ensure_ascii=False) + "\n", encoding="utf-8")
            path = Path(__file__).resolve().parents[1] / "scripts" / "track1_moe_generate_candidates.py"
            spec = importlib.util.spec_from_file_location("track1_moe_generate_candidates_cli", path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)

            code = module.main(
                [
                    "--packets-jsonl",
                    str(packets_jsonl),
                    "--out-dir",
                    str(out_dir),
                    "--dry-run",
                ]
            )
            manifest = json.loads((out_dir / "candidate_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["summary"]["planned"], 2)
        self.assertFalse(any(Path(row["image_path"]).exists() for row in manifest["rows"]))


if __name__ == "__main__":
    unittest.main()
