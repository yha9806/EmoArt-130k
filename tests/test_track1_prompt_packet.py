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
        self.assertGreaterEqual(packet["risk_score"], 2.0)

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


if __name__ == "__main__":
    unittest.main()
