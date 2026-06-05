import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_fused_shadow_evaluator import (
    rank_fused_shadow_candidates,
    score_fused_candidate_rows,
    write_fused_shadow_evaluator_outputs,
)


def _row(sample_id: str, *, emotion: str = "content", valence: str = "Positive", arousal: str = "Low", caption: str = ""):
    text = caption or "A quiet painting suggests a balanced and peaceful mood."
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "caption": text,
        "object": "The artwork presents clear figurative elements with stable visual focus.",
        "brushstroke": "The brushwork is controlled and supports the calm visual rhythm.",
        "color": "The color palette is restrained and coherent.",
        "composition": "The composition keeps the main forms balanced within the frame.",
        "light": "The lighting is soft and even.",
        "line": "The line work is precise and readable.",
        "overall_caption": text,
    }


class Track2FusedShadowEvaluatorTest(unittest.TestCase):
    def test_vulca_risk_reduction_ranks_higher_when_classification_is_equal(self):
        baseline = [
            _row(
                "track2_0001",
                caption=(
                    "A serene Chinese painting symbolizing spiritual presence and "
                    "philosophical harmony in a historical Song dynasty setting."
                ),
            )
        ]
        v3_like = [
            _row(
                "track2_0001",
                caption="A serene Chinese painting symbolizing spiritual presence and domestic harmony.",
            )
        ]
        v9_like = [
            _row(
                "track2_0001",
                caption="A serene Chinese painting suggests a quiet focus and domestic harmony.",
            )
        ]

        v3_score = score_fused_candidate_rows(
            candidate_name="v3_like",
            candidate_json=Path("submissions/v3_like.json"),
            rows=v3_like,
            baseline_rows=baseline,
            expected_row_count=1,
            require_all_emotions=False,
        )
        v9_score = score_fused_candidate_rows(
            candidate_name="v9_like",
            candidate_json=Path("submissions/v9_like.json"),
            rows=v9_like,
            baseline_rows=baseline,
            expected_row_count=1,
            require_all_emotions=False,
        )

        ranked = rank_fused_shadow_candidates([v3_score, v9_score])

        self.assertEqual([item.candidate_name for item in ranked], ["v9_like", "v3_like"])
        self.assertGreater(v9_score.description.expected, v3_score.description.expected)
        self.assertGreater(v9_score.vulca_description_delta, v3_score.vulca_description_delta)

    def test_vulca_description_gain_does_not_override_classification_hold(self):
        baseline = [
            _row(
                "track2_0001",
                emotion="content",
                valence="Positive",
                arousal="Low",
                caption="A serene Chinese painting symbolizing spiritual presence and harmony.",
            )
        ]
        risky = [
            _row(
                "track2_0001",
                emotion="alarmed",
                valence="Negative",
                arousal="High",
                caption="A dramatic painting suggests a quiet focus and visual harmony.",
            )
        ]

        score = score_fused_candidate_rows(
            candidate_name="risky",
            candidate_json=Path("submissions/risky.json"),
            rows=risky,
            baseline_rows=baseline,
            expected_row_count=1,
            require_all_emotions=False,
        )

        self.assertEqual(score.base_decision, "recommend_hold")
        self.assertEqual(score.decision, "recommend_hold")
        self.assertGreater(score.vulca_description_delta, 0.0)

    def test_cli_writes_fused_report_without_overwriting_formal_submission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            baseline_json = tmp_path / "baseline.json"
            v3_json = tmp_path / "v3.json"
            v9_json = tmp_path / "v9.json"
            baseline_json.write_text(
                json.dumps(
                    [
                        _row(
                            "track2_0001",
                            caption="A Chinese painting symbolizing spiritual presence and harmony.",
                        )
                    ]
                ),
                encoding="utf-8",
            )
            v3_json.write_text(
                json.dumps(
                    [
                        _row(
                            "track2_0001",
                            caption="A Chinese painting symbolizing harmony.",
                        )
                    ]
                ),
                encoding="utf-8",
            )
            v9_json.write_text(
                json.dumps(
                    [
                        _row(
                            "track2_0001",
                            caption="A Chinese painting suggests harmony.",
                        )
                    ]
                ),
                encoding="utf-8",
            )
            out_dir = tmp_path / "fused"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/track2_fused_shadow_evaluator.py",
                    "score",
                    "--baseline-json",
                    str(baseline_json),
                    "--candidate",
                    f"v3={v3_json}",
                    "--candidate",
                    f"v9={v9_json}",
                    "--out-dir",
                    str(out_dir),
                    "--expected-row-count",
                    "1",
                    "--no-require-all-emotions",
                ],
                check=False,
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("top_candidate=v9", result.stdout)
            report = json.loads((out_dir / "fused_shadow_score_report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["formal_submission_overwritten"])
            self.assertEqual(report["ranking"][0]["candidate_name"], "v9")

            direct_report = write_fused_shadow_evaluator_outputs(
                baseline_json=baseline_json,
                candidates=[{"name": "v3", "json": v3_json}, {"name": "v9", "json": v9_json}],
                out_dir=tmp_path / "direct",
                expected_row_count=1,
                require_all_emotions=False,
            )
            self.assertEqual(direct_report["ranking"][0]["candidate_name"], "v9")

