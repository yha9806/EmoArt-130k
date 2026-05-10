import unittest

import numpy as np

from scripts import track2_clip_emoart130k as script
from scripts.track2_clip_emoart130k import validate_backbone_for_static_track2


class FakeEncoder:
    def encode_images(self, images):
        return [np.array([float(len(str(image))), 1.0], dtype="float32") for image in images]


class Track2EmbeddingScriptTest(unittest.TestCase):
    def test_flush_batch_uses_encoder_interface_and_places_embeddings_by_index(self):
        embeddings = [None, None, None]

        script.flush_batch(FakeEncoder(), ["aaa", "b"], [2, 0], embeddings)

        np.testing.assert_array_equal(embeddings[0], np.array([1.0, 1.0], dtype="float32"))
        self.assertIsNone(embeddings[1])
        np.testing.assert_array_equal(embeddings[2], np.array([3.0, 1.0], dtype="float32"))

    def test_static_video_batch_repeats_each_image_for_requested_frame_count(self):
        videos = script.static_video_batch(["a", "b"], frame_count=3)

        self.assertEqual(videos, [["a", "a", "a"], ["b", "b", "b"]])


class Track2BackboneGuardTest(unittest.TestCase):
    def test_rejects_video_backbone_without_static_proxy_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "video backbone"):
            validate_backbone_for_static_track2("vjepa2-vitl", allow_static_video_proxy=False)

    def test_allows_video_backbone_when_proxy_flag_is_explicit(self) -> None:
        validate_backbone_for_static_track2("vjepa2-vitl", allow_static_video_proxy=True)

    def test_allows_image_backbone_without_proxy_flag(self) -> None:
        validate_backbone_for_static_track2("ijepa-vith16-1k", allow_static_video_proxy=False)


if __name__ == "__main__":
    unittest.main()
