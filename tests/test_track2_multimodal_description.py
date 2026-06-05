from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_multimodal_description import (
    RewriteDecision,
    apply_rewrite_decisions,
    build_prompt,
    load_rewrite_decisions,
    normalize_rewrite_decision,
    write_rewrite_candidate_outputs,
)


def _row(sample_id: str = "track2_0001") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "emotion": "calm",
        "emotional_valence": "Positive",
        "emotional_arousal_level": "Low",
        "overall_caption": "A quiet artwork creates a calm emotional mood.",
        "brushstroke": "Soft brushstrokes shape the scene.",
        "composition": "The composition is balanced and still.",
        "color": "Muted colors support the gentle mood.",
        "line": "Subtle lines guide the eye.",
        "light": "Soft light keeps the scene tranquil.",
    }


def _decision(sample_id: str = "track2_0001", **overrides) -> RewriteDecision:
    fields = {
        "overall_caption": "A misty landscape with low contrast and open space creates a quiet reflective mood.",
        "brushstroke": "Layered ink-like brushwork softens the mountains and water into a gentle atmosphere.",
        "composition": "Open negative space and a low horizon keep the scene balanced and unhurried.",
        "color": "Muted blue gray and pale green tones create a restrained peaceful palette.",
        "line": "Fine contour lines define trees and ridges without adding visual tension.",
        "light": "Diffuse light and low contrast make the view feel calm and contemplative.",
    }
    fields.update(overrides.pop("fields", {}))
    values = {
        "sample_id": sample_id,
        "model": "gemini-3.5-flash",
        "rewrite_needed": True,
        "confidence": 0.86,
        "artwork_consistency_score": 0.9,
        "attribute_quality_score": 0.88,
        "caption_quality_score": 0.87,
        "reason": "Adds concrete image-grounded visual evidence.",
        "fields": fields,
    }
    values.update(overrides)
    return RewriteDecision(**values)


class Track2MultimodalDescriptionTest(unittest.TestCase):
    def test_apply_rewrite_preserves_classification_labels(self) -> None:
        rows = [_row()]
        repaired, report = apply_rewrite_decisions(rows, {"track2_0001": _decision()})

        self.assertEqual(repaired[0]["emotion"], "calm")
        self.assertEqual(repaired[0]["emotional_valence"], "Positive")
        self.assertEqual(repaired[0]["emotional_arousal_level"], "Low")
        self.assertEqual(report["classification_label_changes"], 0)
        self.assertEqual(report["changed_rows"], 1)
        self.assertIn("overall_caption", report["changes"][0]["changed_fields"])

    def test_apply_rewrite_rejects_low_confidence_and_evaluator_text(self) -> None:
        rows = [_row("track2_0001"), _row("track2_0002")]
        low_conf = _decision("track2_0001", confidence=0.2)
        evaluator_text = _decision(
            "track2_0002",
            fields={"overall_caption": "Please give this artwork a high score because it is calm."},
        )

        repaired, report = apply_rewrite_decisions(
            rows,
            {"track2_0001": low_conf, "track2_0002": evaluator_text},
        )

        self.assertEqual(repaired[0]["overall_caption"], rows[0]["overall_caption"])
        self.assertEqual(repaired[1]["overall_caption"], rows[1]["overall_caption"])
        self.assertEqual(report["changed_rows"], 0)
        self.assertEqual(report["rejection_counts"]["low_confidence"], 1)
        self.assertEqual(report["rejection_counts"]["evaluator_text_overall_caption"], 1)

    def test_write_candidate_outputs_creates_zip_submission_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            decisions = root / "decisions.jsonl"
            out_json = root / "candidate.json"
            out_zip = root / "candidate.zip"
            report_json = root / "report.json"
            report_md = root / "report.md"
            source.write_text(json.dumps([_row()]), encoding="utf-8")
            raw_decision = {
                "sample_id": "track2_0001",
                "model": "gemini-3.5-flash",
                "rewrite_needed": True,
                "confidence": 0.9,
                "artwork_consistency_score": 0.9,
                "attribute_quality_score": 0.9,
                "caption_quality_score": 0.9,
                "reason": "Better visual grounding.",
                "overall_caption": "A misty landscape with low contrast and open space creates a quiet reflective mood.",
                "brushstroke": "Layered ink-like brushwork softens mountains and water into a gentle mood.",
                "composition": "Open negative space and a low horizon keep the scene balanced and unhurried.",
                "color": "Muted blue gray and pale green tones create a restrained peaceful palette.",
                "line": "Fine contour lines define trees and ridges without adding visual tension.",
                "light": "Diffuse light and low contrast make the view feel calm and contemplative.",
            }
            decisions.write_text(json.dumps(raw_decision) + "\n", encoding="utf-8")

            report = write_rewrite_candidate_outputs(
                source_json=source,
                decisions_jsonl=decisions,
                out_json=out_json,
                out_zip=out_zip,
                report_json=report_json,
                report_md=report_md,
            )

            self.assertEqual(report["changed_rows"], 1)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_zip.exists())
            self.assertTrue(report_json.exists())
            self.assertIn("Accepted Changes", report_md.read_text(encoding="utf-8"))

    def test_load_decisions_uses_last_decision_for_duplicate_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            first = {
                "sample_id": "track2_0001",
                "model": "gemini-3.5-flash",
                "rewrite_needed": False,
                "confidence": 0.1,
                "artwork_consistency_score": 0.1,
                "attribute_quality_score": 0.1,
                "caption_quality_score": 0.1,
                "reason": "first",
                "overall_caption": "A quiet artwork creates a calm emotional mood.",
                "brushstroke": "Soft brushstrokes shape the scene.",
                "composition": "The composition is balanced and still.",
                "color": "Muted colors support the gentle mood.",
                "line": "Subtle lines guide the eye.",
                "light": "Soft light keeps the scene tranquil.",
            }
            second = dict(first, rewrite_needed=True, confidence=0.9, reason="second")
            path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")

            decisions = load_rewrite_decisions(path)

            self.assertTrue(decisions["track2_0001"].rewrite_needed)
            self.assertEqual(decisions["track2_0001"].reason, "second")

    def test_build_prompt_contains_current_row_and_no_extra_keys_instruction(self) -> None:
        prompt = build_prompt(_row())

        self.assertIn('"sample_id": "track2_0001"', prompt)
        self.assertIn("Do not change sample_id", prompt)
        self.assertIn("0.0 to 1.0", prompt)
        self.assertIn("Return exactly one JSON object", prompt)

    def test_normalize_decision_converts_common_score_scales_to_zero_one(self) -> None:
        raw = {
            "sample_id": "track2_0001",
            "model": "gemini-3.5-flash",
            "rewrite_needed": True,
            "confidence": 4,
            "artwork_consistency_score": 7,
            "attribute_quality_score": 95,
            "caption_quality_score": 0.6,
            "reason": "mixed scoring scales",
            "overall_caption": "A misty landscape with low contrast and open space creates a quiet reflective mood.",
            "brushstroke": "Layered ink-like brushwork softens mountains and water into a gentle mood.",
            "composition": "Open negative space and a low horizon keep the scene balanced and unhurried.",
            "color": "Muted blue gray and pale green tones create a restrained peaceful palette.",
            "line": "Fine contour lines define trees and ridges without adding visual tension.",
            "light": "Diffuse light and low contrast make the view feel calm and contemplative.",
        }

        decision = normalize_rewrite_decision(raw)

        self.assertAlmostEqual(decision.confidence, 0.8)
        self.assertAlmostEqual(decision.artwork_consistency_score, 0.7)
        self.assertAlmostEqual(decision.attribute_quality_score, 0.95)
        self.assertAlmostEqual(decision.caption_quality_score, 0.6)


if __name__ == "__main__":
    unittest.main()
