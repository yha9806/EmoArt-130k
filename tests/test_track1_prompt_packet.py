import unittest

from affectiveart.track1_prompt_packet import (
    compile_prompt_packet,
    detect_artwork_category,
)


class Track1PromptPacketTest(unittest.TestCase):
    def test_detect_artwork_category_prioritizes_surface_types(self):
        self.assertEqual(detect_artwork_category("A Soviet propaganda poster with Cyrillic lettering"), "poster")
        self.assertEqual(detect_artwork_category("A Gongbi vertical hanging scroll with calligraphy"), "scroll")
        self.assertEqual(detect_artwork_category("Branching pencil lines on graph paper"), "drawing_on_paper")
        self.assertEqual(detect_artwork_category("A Baroque oil painting portrait"), "painting")

    def test_compile_prompt_packet_locks_requested_content_and_boundary(self):
        packet = compile_prompt_packet(
            "track1_0151",
            "A Socialist Realism propaganda poster on aged folded paper, with bold Cyrillic lettering and dramatic Soviet soldiers charging with rifles and bayonets against historical Prussian figures, using muted greens, reds, and tan tones.",
        )

        self.assertEqual(packet["sample_id"], "track1_0151")
        self.assertEqual(packet["artwork_category"], "poster")
        self.assertTrue(packet["output_is_artwork_itself"])
        self.assertIn("Soviet soldiers", packet["hard_requirements"])
        self.assertIn("Prussian figures", packet["hard_requirements"])
        self.assertIn("Cyrillic lettering", packet["allowed_text"])
        self.assertIn("gallery wall", packet["forbidden_artifacts"])
        self.assertIn("aged folded paper", packet["caption_required_surface_features"])
        self.assertIn("poster layout", packet["allowed_surface_features"])
        self.assertIn("unrequested white mat border", packet["unrequested_physical_artifact_features"])
        self.assertGreaterEqual(packet["risk_score"], 2.0)

    def test_compile_prompt_packet_does_not_require_surface_artifacts_for_generic_poster(self):
        packet = compile_prompt_packet(
            "track1_0990",
            "A Socialist Realism Soviet propaganda poster with bold Cyrillic text, a hard-hatted industrial worker gesturing toward smoking factories, coal heaps, and a rebuilding mining town in a muted blue and ochre palette.",
        )

        self.assertEqual(packet["caption_required_surface_features"], [])
        self.assertIn("poster layout", packet["allowed_surface_features"])
        self.assertIn("aged paper", packet["unrequested_physical_artifact_features"])
        self.assertIn("thick decorative border", packet["unrequested_physical_artifact_features"])
        self.assertIn("outer gray or black frame", packet["unrequested_physical_artifact_features"])

    def test_compile_prompt_packet_extracts_album_leaf_surface_requirements(self):
        packet = compile_prompt_packet(
            "track1_0429",
            "Ink and wash painting of an open album leaf with a blank ruled page beside a delicate composition of bamboo, bare tree branches, vertical calligraphy, and red seals within a pale patterned border.",
        )

        self.assertIn("open album leaf", packet["caption_required_surface_features"])
        self.assertIn("blank ruled page", packet["caption_required_surface_features"])
        self.assertIn("pale patterned border", packet["caption_required_surface_features"])
        self.assertIn("album leaf flat page surface", packet["allowed_surface_features"])
        self.assertIn("photo-style mat border", packet["unrequested_physical_artifact_features"])

    def test_compile_prompt_packet_forbids_text_when_not_requested(self):
        packet = compile_prompt_packet(
            "track1_0301",
            "Abstract hand-drawn branching lines fill a rectangular frame on graph paper, forming a dense tree-like network with small heart and geometric marks in a simple monochrome pencil style.",
        )

        self.assertEqual(packet["artwork_category"], "drawing_on_paper")
        self.assertEqual(packet["allowed_text"], [])
        self.assertIn("sample id", packet["forbidden_artifacts"])
        self.assertIn("graph paper", packet["hard_requirements"])
        self.assertIn("rectangular frame", packet["hard_requirements"])
        self.assertIn("graph paper surface", packet["caption_required_surface_features"])

    def test_prompt_packet_risk_sort_prefers_text_and_surface_sensitive_samples(self):
        rows = [
            {"sample_id": "track1_0001", "caption": "A calm watercolor lake."},
            {"sample_id": "track1_0002", "caption": "A poster with Cyrillic lettering and soldiers."},
            {"sample_id": "track1_0003", "caption": "A graph paper pencil drawing with a rectangular frame."},
        ]
        from affectiveart.track1_prompt_packet import compile_and_rank_packets

        packets = compile_and_rank_packets(rows)

        self.assertEqual(packets[0]["sample_id"], "track1_0003")
        self.assertEqual(packets[1]["sample_id"], "track1_0002")
        self.assertEqual(packets[-1]["sample_id"], "track1_0001")


if __name__ == "__main__":
    unittest.main()
