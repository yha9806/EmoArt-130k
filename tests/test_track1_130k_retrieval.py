import unittest

from affectiveart.track1_130k_retrieval import (
    RetrievedReference,
    score_reference_for_packet,
    select_references,
)


class Track1130kRetrievalTest(unittest.TestCase):
    def test_score_reference_rewards_style_emotion_and_attribute_overlap(self):
        packet = {
            "caption": "A Gongbi vertical hanging scroll with delicate lotus blossoms and calligraphy.",
            "style_clues": ["Gongbi"],
            "emotion_clues": ["calm"],
            "hard_requirements": ["lotus blossoms", "calligraphy"],
        }
        reference = {
            "request_id": "Gongbi_request-1",
            "style": "Gongbi",
            "emotion": "calm",
            "source_text": "A lotus scroll with delicate linework and side calligraphy.",
            "compiler_target": "brushstroke: delicate lines\ncomposition: vertical scroll",
        }

        score = score_reference_for_packet(packet, reference)

        self.assertGreater(score, 5.0)

    def test_select_references_returns_sorted_top_k(self):
        packet = {
            "caption": "A Gongbi lotus scroll with calligraphy.",
            "style_clues": ["Gongbi"],
            "emotion_clues": [],
            "hard_requirements": ["lotus blossoms", "calligraphy"],
        }
        rows = [
            {
                "request_id": "bad",
                "style": "Cubism",
                "emotion": "alarmed",
                "source_text": "A cubist city scene.",
                "compiler_target": "",
            },
            {
                "request_id": "good",
                "style": "Gongbi",
                "emotion": "calm",
                "source_text": "Lotus blossoms with calligraphy in a vertical scroll.",
                "compiler_target": "",
            },
        ]

        refs = select_references(packet, rows, top_k=1)

        self.assertEqual(len(refs), 1)
        self.assertIsInstance(refs[0], RetrievedReference)
        self.assertEqual(refs[0].request_id, "good")


if __name__ == "__main__":
    unittest.main()
