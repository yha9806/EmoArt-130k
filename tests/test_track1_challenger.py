import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track1_challenger import effective_create_returncode


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
            image_path.write_bytes(b"fake")

            code = effective_create_returncode(0, create_json, image_path)

        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
