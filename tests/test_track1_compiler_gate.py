import unittest

from affectiveart.track1_compiler_gate import (
    compile_vulca_prompt,
    decision_from_scores,
)


class Track1CompilerGateTest(unittest.TestCase):
    def test_compile_vulca_prompt_keeps_references_out_of_content_lock_prompt_by_default(self):
        packet = {
            "caption": "A Gongbi vertical hanging scroll with lotus blossoms and side calligraphy.",
            "artwork_category": "scroll",
            "hard_requirements": ["lotus blossoms", "side calligraphy"],
            "allowed_text": ["calligraphy"],
            "forbidden_artifacts": ["gallery wall", "sample id"],
            "retrieved_references": [
                {
                    "style": "Gongbi",
                    "emotion": "calm",
                    "source_text": "Lotus flowers on silk with delicate linework.",
                    "compiler_target": "brushstroke: delicate lines\ncolor: pale beige and pink",
                    "score": 7.2,
                }
            ],
        }

        prompt = compile_vulca_prompt(packet)

        self.assertLess(prompt.index("NON-NEGOTIABLE CONTENT"), prompt.index("GENERATION PRIORITY"))
        self.assertIn("lotus blossoms", prompt)
        self.assertIn("side calligraphy", prompt)
        self.assertIn("output must be the artwork surface itself", prompt.lower())
        self.assertIn("gallery wall", prompt)
        self.assertNotIn("130K STYLE REFERENCES", prompt)
        self.assertNotIn("Reference subjects are not requirements", prompt)
        self.assertNotIn("Lotus flowers on silk with delicate linework", prompt)

    def test_decision_rejects_when_candidate_not_clear_win(self):
        decision = decision_from_scores(
            sample_id="track1_0301",
            baseline={"caption_fidelity": "pass", "visual_quality": 0.9, "style_fidelity": 0.9},
            candidate={"caption_fidelity": "pass", "visual_quality": 0.88, "style_fidelity": 0.9},
        )

        self.assertEqual(decision["decision"], "reject")
        self.assertIn("not a clear win", decision["reason"])

    def test_decision_accepts_clear_candidate_win(self):
        decision = decision_from_scores(
            sample_id="track1_0999",
            baseline={"caption_fidelity": "partial", "visual_quality": 0.55, "style_fidelity": 0.6},
            candidate={"caption_fidelity": "pass", "visual_quality": 0.86, "style_fidelity": 0.82},
        )

        self.assertEqual(decision["decision"], "accept")
        self.assertGreaterEqual(decision["margin"], 0.2)

    def test_summarize_decisions_counts_accepts_and_rejects(self):
        from affectiveart.track1_compiler_gate import summarize_decisions

        summary = summarize_decisions(
            [
                {"sample_id": "track1_0001", "decision": "accept"},
                {"sample_id": "track1_0002", "decision": "reject"},
                {"sample_id": "track1_0003", "decision": "reject"},
            ]
        )

        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["rejected"], 2)
        self.assertEqual(summary["total"], 3)


if __name__ == "__main__":
    unittest.main()
