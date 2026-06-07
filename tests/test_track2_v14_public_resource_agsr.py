import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_v14_public_resource_agsr import (
    apply_v14_candidate_rows,
    build_public_evidence_matrix,
    build_v14_outputs,
    classify_public_evidence,
    select_v14_label_changes,
)


def row(sample_id: str, emotion: str, valence: str = "Positive", arousal: str = "Low") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": f"{sample_id} shows a quiet scene with specific visual details.",
        "brushstroke": "Layered brushwork describes the surface texture.",
        "composition": "Balanced composition organizes the main forms.",
        "color": "Restrained colors support the emotional atmosphere.",
        "line": "Controlled lines define the shapes and movement.",
        "light": "Soft light clarifies the focal area.",
    }


class Track2V14PublicResourceEvidenceTest(unittest.TestCase):
    def test_classifies_exact_near_series_and_style_evidence(self):
        self.assertEqual(
            classify_public_evidence(
                {
                    "clip_cosine": 0.992,
                    "dhash_distance": 0,
                    "duplicate_bucket": "visual_duplicate_likely",
                }
            ),
            "exact_same_work",
        )
        self.assertEqual(
            classify_public_evidence(
                {
                    "clip_cosine": 0.965,
                    "duplicate_bucket": "same_work_or_series_review",
                    "topk_majority_count": 8,
                }
            ),
            "near_same_work",
        )
        self.assertEqual(
            classify_public_evidence(
                {
                    "clip_cosine": 0.945,
                    "duplicate_bucket": "same_work_or_series_review",
                    "topk_majority_count": 6,
                }
            ),
            "same_series_style",
        )
        self.assertEqual(
            classify_public_evidence({"clip_cosine": 0.710, "duplicate_bucket": "style_prior"}),
            "style_prior_only",
        )

    def test_public_evidence_matrix_preserves_source_and_salience(self):
        current = [row("track2_0001", "content")]
        public_rows = [
            {
                "sample_id": "track2_0001",
                "public_emotion": "calm",
                "public_valence": "Positive",
                "public_arousal": "Low",
                "clip_cosine": 0.991,
                "dhash_distance": 0,
                "duplicate_bucket": "visual_duplicate_likely",
                "public_member": "Chinese Painting/example.png",
                "salient_attributes": ["composition", "color"],
                "public_description": "A calm Chinese painting with balanced composition and restrained color.",
            }
        ]
        matrix = build_public_evidence_matrix(current, public_rows)
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["evidence_level"], "exact_same_work")
        self.assertEqual(matrix[0]["target_emotion"], "calm")
        self.assertEqual(matrix[0]["salient_attributes"], "composition,color")
        self.assertIn("public_duplicate", matrix[0]["sources"])


class Track2V14SelectionTest(unittest.TestCase):
    def test_selects_exact_and_near_same_work_but_blocks_style_only(self):
        matrix = [
            {
                "sample_id": "track2_exact",
                "current_emotion": "content",
                "target_emotion": "calm",
                "evidence_level": "exact_same_work",
                "same_quadrant": True,
            },
            {
                "sample_id": "track2_near",
                "current_emotion": "content",
                "target_emotion": "calm",
                "evidence_level": "near_same_work",
                "same_quadrant": True,
            },
            {
                "sample_id": "track2_style",
                "current_emotion": "content",
                "target_emotion": "calm",
                "evidence_level": "same_series_style",
                "same_quadrant": True,
            },
            {
                "sample_id": "track2_cross",
                "current_emotion": "calm",
                "target_emotion": "frustrated",
                "evidence_level": "near_same_work",
                "same_quadrant": False,
            },
        ]
        selected = select_v14_label_changes(matrix, profile="public_resource_agsr_max")
        self.assertEqual([item["sample_id"] for item in selected], ["track2_exact", "track2_near"])

    def test_description_safe_profile_keeps_only_exact_label_transfer(self):
        matrix = [
            {
                "sample_id": "track2_exact",
                "current_emotion": "content",
                "target_emotion": "calm",
                "evidence_level": "exact_same_work",
                "same_quadrant": True,
            },
            {
                "sample_id": "track2_near",
                "current_emotion": "content",
                "target_emotion": "calm",
                "evidence_level": "near_same_work",
                "same_quadrant": True,
            },
        ]
        selected = select_v14_label_changes(matrix, profile="description_max_safe")
        self.assertEqual([item["sample_id"] for item in selected], ["track2_exact"])

    def test_apply_rows_repairs_va_and_cites_salient_attributes(self):
        rows = [row("track2_0001", "content")]
        matrix = [
            {
                "sample_id": "track2_0001",
                "current_emotion": "content",
                "target_emotion": "calm",
                "target_valence": "Positive",
                "target_arousal": "Low",
                "evidence_level": "exact_same_work",
                "salient_attributes": "composition,color",
                "public_description": "A calm scene where balanced composition and restrained color create a quiet mood.",
                "same_quadrant": True,
            }
        ]
        candidate, report = apply_v14_candidate_rows(rows, matrix, profile="public_resource_agsr_max")
        self.assertEqual(candidate[0]["emotion"], "calm")
        self.assertEqual(candidate[0]["emotional_valence"], "Positive")
        self.assertEqual(candidate[0]["emotional_arousal_level"], "Low")
        self.assertIn("balanced composition", candidate[0]["overall_caption"].lower())
        self.assertIn("restrained color", candidate[0]["overall_caption"].lower())
        self.assertEqual(report["classification_label_changes"], 1)


class Track2V14OutputTest(unittest.TestCase):
    def test_build_outputs_writes_side_path_candidates_reports_html_and_scores(self):
        rows = [
            row("track2_0001", "content"),
            row("track2_0002", "calm"),
        ]
        public_rows = [
            {
                "sample_id": "track2_0001",
                "public_emotion": "calm",
                "public_valence": "Positive",
                "public_arousal": "Low",
                "clip_cosine": 0.992,
                "dhash_distance": 0,
                "duplicate_bucket": "visual_duplicate_likely",
                "public_member": "Chinese Painting/example.png",
                "salient_attributes": ["composition", "color"],
                "public_description": "A calm image with balanced composition and restrained color.",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_json = root / "current.json"
            public_json = root / "public.json"
            out_dir = root / "out"
            submissions = root / "submissions"
            current_json.write_text(json.dumps(rows), encoding="utf-8")
            public_json.write_text(json.dumps({"entries": public_rows}), encoding="utf-8")
            report = build_v14_outputs(
                current_json=current_json,
                public_audit_jsons=[public_json],
                out_dir=out_dir,
                submission_dir=submissions,
                run_shadow=False,
            )
            self.assertTrue((out_dir / "track2_public_evidence_matrix.csv").exists())
            self.assertTrue((out_dir / "v14_candidate_score_proxy.md").exists())
            self.assertTrue((out_dir / "html_review" / "track2_v14_public_resource_agsr_review.html").exists())
            max_json = Path(report["candidates"]["public_resource_agsr_max"]["json"])
            max_zip = Path(report["candidates"]["public_resource_agsr_max"]["zip"])
            safe_json = Path(report["candidates"]["description_max_safe"]["json"])
            self.assertTrue(max_json.exists())
            self.assertTrue(safe_json.exists())
            with zipfile.ZipFile(max_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])


class Track2V14CliTest(unittest.TestCase):
    def test_cli_build_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_json = root / "current.json"
            public_json = root / "public.json"
            out_dir = root / "out"
            submissions = root / "submissions"
            current_json.write_text(json.dumps([row("track2_0001", "content")]), encoding="utf-8")
            public_json.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "sample_id": "track2_0001",
                                "public_emotion": "calm",
                                "public_valence": "Positive",
                                "public_arousal": "Low",
                                "clip_cosine": 0.991,
                                "dhash_distance": 0,
                                "duplicate_bucket": "visual_duplicate_likely",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "python3",
                    "scripts/track2_v14_public_resource_agsr.py",
                    "build",
                    "--current-json",
                    str(current_json),
                    "--public-audit-json",
                    str(public_json),
                    "--out-dir",
                    str(out_dir),
                    "--submission-dir",
                    str(submissions),
                    "--no-shadow",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("public_resource_agsr_max", result.stdout)


if __name__ == "__main__":
    unittest.main()
