import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.emoart130k import (
    EmoArtExample,
    canonical_emotion,
    iter_emoart_examples,
    public_image_location,
)


class EmoArt130kTest(unittest.TestCase):
    def test_canonical_emotion_normalizes_public_labels(self):
        self.assertEqual(canonical_emotion("Contentment"), "content")
        self.assertEqual(canonical_emotion(" calm "), "calm")
        self.assertEqual(canonical_emotion("Alarmed"), "alarmed")

    def test_public_image_location_resolves_tar_and_member(self):
        tar_path, member = public_image_location(
            "/data/EmoArt-130k",
            "Images\\Gongbi\\abc.jpg",
        )

        self.assertEqual(tar_path, Path("/data/EmoArt-130k/Gongbi.tar.gz"))
        self.assertEqual(member, "Gongbi/abc.jpg")

    def test_iter_emoart_examples_extracts_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation = root / "Annotation.json"
            annotation.write_text(
                json.dumps(
                    [
                        {
                            "request_id": "Gongbi_request-1",
                            "image_path": "Images\\Gongbi\\001.jpg",
                            "description": {
                                "first_section": {
                                    "description": "A lotus scroll with pale silk ground."
                                },
                                "second_section": {
                                    "visual_attributes": {
                                        "brushstroke": "delicate lines",
                                        "color": "pale beige and pink",
                                        "composition": "vertical scroll composition",
                                        "light_and_shadow": "soft diffuse light",
                                        "line_quality": "fine controlled line",
                                    },
                                    "emotional_impact": "calm and contemplative",
                                },
                                "third_section": {
                                    "dominant_emotion": "Calm",
                                    "emotional_valence": "Positive",
                                    "emotional_arousal_level": "Low",
                                },
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            rows = list(iter_emoart_examples(annotation, data_root=root))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIsInstance(row, EmoArtExample)
        self.assertEqual(row.request_id, "Gongbi_request-1")
        self.assertEqual(row.style, "Gongbi")
        self.assertEqual(row.emotion, "calm")
        self.assertEqual(row.valence, "Positive")
        self.assertEqual(row.arousal, "Low")
        self.assertIn("lotus scroll", row.caption)
        self.assertIn("delicate lines", row.attributes["brushstroke"])

    def test_example_to_compiler_training_row_contains_prompt_signals(self):
        example = EmoArtExample(
            request_id="Gongbi_request-1",
            image_path="Images\\Gongbi\\001.jpg",
            tar_path="/data/Gongbi.tar.gz",
            member="Gongbi/001.jpg",
            style="Gongbi",
            emotion="calm",
            valence="Positive",
            arousal="Low",
            caption="A lotus scroll with pale silk ground.",
            attributes={
                "brushstroke": "delicate lines",
                "color": "pale beige and pink",
                "composition": "vertical scroll composition",
                "line": "fine controlled line",
                "light": "soft diffuse light",
            },
            emotional_impact="calm and contemplative",
        )

        row = example.to_compiler_training_row()

        self.assertEqual(row["request_id"], "Gongbi_request-1")
        self.assertEqual(row["style"], "Gongbi")
        self.assertEqual(row["emotion"], "calm")
        self.assertIn("lotus scroll", row["source_text"])
        self.assertIn("brushstroke: delicate lines", row["compiler_target"])
        self.assertIn("emotion: calm", row["compiler_target"])


if __name__ == "__main__":
    unittest.main()
