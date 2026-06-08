from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_official_anchor_calibration import (
    OFFICIAL_ANCHOR_LEDGER,
    build_official_anchor_scoreboard,
    canonical_submission_fingerprint,
    required_classification_for_target,
    score_submission,
)


ROOT = Path(__file__).resolve().parents[1]


class Track2OfficialAnchorCalibrationTests(unittest.TestCase):
    def test_anchor_ledger_contains_exact_and_visible_official_results(self) -> None:
        self.assertAlmostEqual(OFFICIAL_ANCHOR_LEDGER["779605"].overall, 0.836408, places=6)
        self.assertAlmostEqual(OFFICIAL_ANCHOR_LEDGER["781601"].classification, 0.719137, places=6)
        self.assertEqual(OFFICIAL_ANCHOR_LEDGER["782683"].overall_visible, 0.84)
        self.assertEqual(OFFICIAL_ANCHOR_LEDGER["785979"].calibration_kind, "exact_official")

    def test_fingerprint_is_stable_across_whitespace_and_key_order(self) -> None:
        rows = [
            {
                "sample_id": "track2_0001",
                "emotion": "calm",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "A.",
                "brushstroke": "B.",
                "composition": "C.",
                "color": "D.",
                "line": "E.",
                "light": "F.",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            compact = Path(tmp) / "compact.json"
            pretty = Path(tmp) / "pretty.json"
            compact.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
            pretty.write_text(json.dumps([dict(reversed(list(rows[0].items())))], indent=2), encoding="utf-8")

            self.assertEqual(canonical_submission_fingerprint(compact), canonical_submission_fingerprint(pretty))

    def test_scores_known_official_anchors_in_their_official_bucket(self) -> None:
        anchors = {
            "779605": ROOT / "submissions/track2_submission_moe_v2_accept5_candidate.json",
            "781601": ROOT / "submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json",
            "782683": ROOT / "submissions/track2_submission_v12_stable_probe_candidate.json",
            "785979": ROOT / "submissions/track2_submission_v21_calmshift90_candidate.json",
        }

        scores = {submission_id: score_submission(path) for submission_id, path in anchors.items()}

        self.assertAlmostEqual(scores["779605"].overall_expected, 0.836408, places=6)
        self.assertAlmostEqual(scores["781601"].overall_expected, 0.834027, places=6)
        self.assertEqual(scores["782683"].overall_visible, 0.84)
        self.assertAlmostEqual(scores["785979"].overall_expected, 0.842559, places=6)
        self.assertEqual(scores["781601"].visible_bucket, 0.83)
        self.assertEqual(scores["782683"].calibration_kind, "visible_official")

    def test_v22_candidates_are_estimates_not_exact_anchor_matches(self) -> None:
        score = score_submission(ROOT / "submissions/track2_submission_v22_official_author_calmshiftall_candidate.json")

        self.assertEqual(score.calibration_kind, "estimated")
        self.assertGreater(score.overall_expected, 0.84)
        self.assertLess(score.overall_expected, 0.86)
        self.assertIn("not_hidden_label_reconstruction", score.warnings)

    def test_frontier_requirement_exposes_why_089_needs_more_than_micro_tuning(self) -> None:
        self.assertAlmostEqual(required_classification_for_target(0.89, 1.00), 0.78, places=6)
        self.assertAlmostEqual(required_classification_for_target(0.89, 0.95), 0.83, places=6)

    def test_scoreboard_ranks_candidates_and_reports_anchor_residuals(self) -> None:
        candidates = {
            "official_779605": ROOT / "submissions/track2_submission_moe_v2_accept5_candidate.json",
            "official_781601": ROOT / "submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json",
            "official_782683": ROOT / "submissions/track2_submission_v12_stable_probe_candidate.json",
            "official_785979": ROOT / "submissions/track2_submission_v21_calmshift90_candidate.json",
            "v22_all": ROOT / "submissions/track2_submission_v22_official_author_calmshiftall_candidate.json",
        }

        board = build_official_anchor_scoreboard(candidates)

        self.assertEqual(board["anchor_residual_summary"]["blocking_residual_count"], 0)
        self.assertEqual(board["frontier_requirement"]["required_classification_if_description_095"], 0.83)
        self.assertEqual(board["ranking"][0]["candidate_name"], "v22_all")
        self.assertLess(board["ranking"][0]["overall_expected"], 0.89)

