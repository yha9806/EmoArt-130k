from __future__ import annotations

import unittest

from affectiveart.track2_v29_fabg_lite import (
    ATTRIBUTE_FIELDS,
    infer_salient_attributes,
    rewrite_description_rows,
)


def _row() -> dict[str, str]:
    return {
        "sample_id": "track2_0001",
        "emotion": "calm",
        "emotional_valence": "Positive",
        "emotional_arousal_level": "Low",
        "overall_caption": "Old caption.",
        "brushstroke": "Soft brushstrokes create gentle texture.",
        "composition": "Balanced central composition with open space.",
        "color": "Muted blue and green color palette.",
        "line": "Horizontal lines move slowly.",
        "light": "Diffuse light with low contrast.",
    }


class Track2V29FabgLiteTests(unittest.TestCase):
    def test_infer_salient_attributes_prefers_specific_visual_cues(self) -> None:
        row = _row()

        salient = infer_salient_attributes(row, max_attributes=3)

        self.assertIn("color", salient)
        self.assertIn("composition", salient)
        self.assertLessEqual(len(salient), 3)

    def test_rewrite_description_rows_preserves_labels(self) -> None:
        row = _row()

        rewritten, report = rewrite_description_rows([row])

        self.assertEqual(rewritten[0]["emotion"], "calm")
        self.assertEqual(rewritten[0]["emotional_valence"], "Positive")
        self.assertEqual(rewritten[0]["emotional_arousal_level"], "Low")
        self.assertGreater(report["changed_rows"], 0)
        for field in ATTRIBUTE_FIELDS:
            self.assertTrue(rewritten[0][field])

    def test_rewrite_mentions_emotion_and_salient_cues(self) -> None:
        rewritten, _ = rewrite_description_rows([_row()])
        caption = rewritten[0]["overall_caption"].lower()

        self.assertIn("calm", caption)
        self.assertTrue("muted" in caption or "balanced" in caption or "diffuse" in caption)


if __name__ == "__main__":
    unittest.main()
