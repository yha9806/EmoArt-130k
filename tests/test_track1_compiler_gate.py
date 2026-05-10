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
            "caption_required_surface_features": ["hanging scroll surface"],
            "allowed_surface_features": ["flat scroll paper or silk surface"],
            "unrequested_physical_artifact_features": ["gallery wall", "framed display"],
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
        self.assertIn("Required surface features from caption: hanging scroll surface", prompt)
        self.assertIn("Allowed surface treatment: flat scroll paper or silk surface", prompt)
        self.assertIn("Do not add unrequested physical artifact features", prompt)
        self.assertIn("gallery wall", prompt)
        self.assertNotIn("130K STYLE REFERENCES", prompt)
        self.assertNotIn("Reference subjects are not requirements", prompt)
        self.assertNotIn("Lotus flowers on silk with delicate linework", prompt)

    def test_compile_vulca_prompt_blocks_unrequested_surface_artifacts_for_generic_poster(self):
        packet = {
            "caption": "A Socialist Realism Soviet propaganda poster with bold Cyrillic text, a worker, factories, coal heaps, and a rebuilding mining town.",
            "artwork_category": "poster",
            "hard_requirements": ["worker", "factories", "coal heaps", "rebuilding mining town"],
            "allowed_text": ["Cyrillic lettering"],
            "forbidden_artifacts": ["gallery wall"],
            "caption_required_surface_features": [],
            "allowed_surface_features": [
                "poster layout",
                "internal printed margin",
                "graphic border line",
                "poster design border",
            ],
            "unrequested_physical_artifact_features": [
                "aged paper",
                "external decorative frame",
                "unrequested white mat border",
            ],
        }

        prompt = compile_vulca_prompt(packet)

        self.assertIn("Poster may include internal printed margins, typography blocks, or graphic border lines as part of the poster design.", prompt)
        self.assertIn("Do not add an external frame, photo mat, wall display, drop shadow, catalog mockup, or product-photo presentation.", prompt)
        self.assertIn("Do not add unrequested physical artifact features: aged paper, external decorative frame, unrequested white mat border", prompt)

    def test_compile_vulca_prompt_uses_provider_safe_wording_for_caricature_panels(self):
        packet = {
            "caption": "A Socialist Realism Soviet propaganda poster with a bold Cyrillic headline, heroic soldiers, an industrial worker, anti-fascist caricature scenes, and stark wartime panels in muted reds, blues, and beige tones.",
            "artwork_category": "poster",
            "hard_requirements": ["bold Cyrillic headline"],
            "allowed_text": ["Cyrillic lettering"],
            "forbidden_artifacts": ["gallery wall"],
            "caption_required_surface_features": [],
            "allowed_surface_features": ["poster layout"],
            "unrequested_physical_artifact_features": ["aged paper"],
        }

        prompt = compile_vulca_prompt(packet)

        self.assertIn("non-graphic symbolic anti-fascist caricature panels", prompt)
        self.assertNotIn("anti-fascist caricature scenes", prompt)

    def test_compile_vulca_prompt_keeps_album_leaf_border_internal(self):
        packet = {
            "caption": "A Gongbi album leaf with calligraphy, lychee fruits, and a pale patterned border.",
            "artwork_category": "album_leaf",
            "hard_requirements": ["lychee fruits", "pale patterned border"],
            "allowed_text": ["calligraphy"],
            "forbidden_artifacts": ["gallery wall"],
            "caption_required_surface_features": ["open album leaf", "pale patterned border"],
            "allowed_surface_features": ["album leaf flat page surface"],
            "unrequested_physical_artifact_features": [
                "photo-style mat border",
                "outer gray or black frame",
            ],
        }

        prompt = compile_vulca_prompt(packet)

        self.assertIn("Album leaf borders must be printed or painted internal page design only", prompt)
        self.assertIn("Do not surround the album leaf with an outer gray or black frame, photo mat, shadow, or display mount.", prompt)

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
