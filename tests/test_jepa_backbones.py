import unittest

from affectiveart.jepa_backbones import (
    BACKBONES,
    image_backbones,
    lookup_backbone,
    recommended_track2_backbones,
)


class JepaBackboneRegistryTest(unittest.TestCase):
    def test_registry_contains_image_jepa_and_video_jepa_metadata(self) -> None:
        ijepa = lookup_backbone("ijepa-vith16-1k")
        self.assertEqual(ijepa.model_id, "facebook/ijepa_vith16_1k")
        self.assertEqual(ijepa.modality, "image")
        self.assertFalse(ijepa.recommended_for_track2_full)

        vjepa = lookup_backbone("vjepa2-vitl")
        self.assertEqual(vjepa.modality, "video")
        self.assertFalse(vjepa.static_image_safe)

    def test_recommended_track2_backbones_exclude_video_models(self) -> None:
        names = [backbone.name for backbone in recommended_track2_backbones()]
        self.assertIn("siglip2-base-patch16-224", names)
        self.assertIn("dinov2-base", names)
        self.assertNotIn("vjepa2-vitl", names)

    def test_image_backbones_include_ijepa_variants(self) -> None:
        names = [backbone.name for backbone in image_backbones()]
        self.assertIn("ijepa-vith16-1k", names)
        self.assertIn("ijepa-vith14-1k", names)

    def test_lookup_raises_for_unknown_name(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown backbone"):
            lookup_backbone("missing-model")


if __name__ == "__main__":
    unittest.main()
