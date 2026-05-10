import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track1_challenger import effective_create_returncode, merge_run_summary_rows


class Track1ChallengerTest(unittest.TestCase):
    def test_effective_create_returncode_rejects_failed_json_without_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_json = root / "create.json"
            image_path = root / "image.png"
            create_json.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

            code = effective_create_returncode(0, create_json, image_path)

        self.assertEqual(code, 1)

    def test_effective_create_returncode_accepts_completed_json_with_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_json = root / "create.json"
            image_path = root / "image.png"
            create_json.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            image_path.write_bytes(b"\xff\xd8\xff\xe0jpeg")

            code = effective_create_returncode(0, create_json, image_path)

        self.assertEqual(code, 0)

    def test_effective_create_returncode_rejects_mock_fallback_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_json = root / "create.json"
            image_path = root / "image.png"
            create_json.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "best_image_url": "mock://abc.svg",
                        "evaluation_source": "mock_fallback",
                    }
                ),
                encoding="utf-8",
            )
            image_path.write_bytes(b"\xff\xd8\xff\xe0jpeg")

            code = effective_create_returncode(0, create_json, image_path)

        self.assertEqual(code, 1)

    def test_effective_create_returncode_accepts_real_image_with_mock_evaluation_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_json = root / "create.json"
            image_path = root / "image.png"
            create_json.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "best_image_url": "https://example.test/image.jpg",
                        "evaluation_source": "mock_fallback",
                    }
                ),
                encoding="utf-8",
            )
            image_path.write_bytes(b"\xff\xd8\xff\xe0jpeg")

            code = effective_create_returncode(0, create_json, image_path)

        self.assertEqual(code, 0)

    def test_effective_create_returncode_rejects_svg_written_as_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_json = root / "create.json"
            image_path = root / "image.png"
            create_json.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            image_path.write_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'></svg>")

            code = effective_create_returncode(0, create_json, image_path)

        self.assertEqual(code, 1)

    def test_merge_run_summary_rows_replaces_selected_rows_without_dropping_others(self):
        existing = [
            {"sample_id": "track1_0001", "returncode": 0},
            {"sample_id": "track1_0002", "returncode": 1},
        ]
        new = [{"sample_id": "track1_0002", "returncode": 0}]

        merged = merge_run_summary_rows(existing, new)

        self.assertEqual(
            merged,
            [
                {"sample_id": "track1_0001", "returncode": 0},
                {"sample_id": "track1_0002", "returncode": 0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
