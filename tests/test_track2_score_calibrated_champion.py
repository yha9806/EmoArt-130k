import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_score_calibrated_champion import (
    CandidateSource,
    build_evidence_matrix,
    build_score_calibrated_candidates,
    write_score_calibrated_outputs,
)


def row(sample_id, emotion, valence=None, arousal=None):
    if valence is None or arousal is None:
        valence, arousal = {
            "calm": ("Positive", "Low"),
            "content": ("Positive", "Low"),
            "glad": ("Positive", "Low"),
            "happy": ("Positive", "High"),
            "frustrated": ("Negative", "High"),
            "tired": ("Negative", "Low"),
        }[emotion]
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": valence,
        "emotional_arousal_level": arousal,
        "overall_caption": f"A {emotion} artwork with grounded image details.",
        "brushstroke": "Layered brushwork describes the visible forms.",
        "composition": "Balanced composition organizes the subject clearly.",
        "color": "Specific colors shape the emotional atmosphere.",
        "line": "Line quality defines the figure and spatial rhythm.",
        "light": "Light and shadow clarify depth and focus.",
    }


def candidate_rows(baseline_rows, changes):
    rows = [dict(item) for item in baseline_rows]
    by_id = {item["sample_id"]: item for item in rows}
    for sample_id, emotion in changes.items():
        valence, arousal = {
            "calm": ("Positive", "Low"),
            "content": ("Positive", "Low"),
            "glad": ("Positive", "Low"),
            "happy": ("Positive", "High"),
            "frustrated": ("Negative", "High"),
            "tired": ("Negative", "Low"),
        }[emotion]
        by_id[sample_id]["emotion"] = emotion
        by_id[sample_id]["emotional_valence"] = valence
        by_id[sample_id]["emotional_arousal_level"] = arousal
    return rows


class Track2ScoreCalibratedChampionTest(unittest.TestCase):
    def test_evidence_matrix_aggregates_sources_and_quadrant_risk(self):
        baseline = [
            row("track2_0001", "content"),
            row("track2_0002", "content"),
        ]
        sources = [
            CandidateSource(
                name="public_style_review11",
                family="public_style",
                rows=candidate_rows(baseline, {"track2_0001": "calm", "track2_0002": "frustrated"}),
            ),
            CandidateSource(
                name="teacher_review",
                family="teacher",
                rows=candidate_rows(baseline, {"track2_0001": "calm"}),
            ),
        ]

        matrix = build_evidence_matrix(baseline, sources)
        by_key = {(item["sample_id"], item["proposed_emotion"]): item for item in matrix}

        calm = by_key[("track2_0001", "calm")]
        self.assertEqual(calm["supporting_source_count"], 2)
        self.assertEqual(calm["supporting_family_count"], 2)
        self.assertTrue(calm["same_quadrant"])
        self.assertFalse(calm["cross_quadrant_risk"])
        self.assertEqual(calm["transition"], "content->calm")

        frustrated = by_key[("track2_0002", "frustrated")]
        self.assertFalse(frustrated["same_quadrant"])
        self.assertTrue(frustrated["cross_quadrant_risk"])

    def test_candidate_ladder_keeps_safe_same_quadrant_and_blocks_weak_cross_quadrant(self):
        baseline = [
            row("track2_0001", "content"),
            row("track2_0002", "content"),
            row("track2_0003", "calm"),
        ]
        sources = [
            CandidateSource(
                name="public_style",
                family="public_style",
                rows=candidate_rows(
                    baseline,
                    {
                        "track2_0001": "calm",
                        "track2_0002": "frustrated",
                        "track2_0003": "content",
                    },
                ),
            ),
            CandidateSource(
                name="teacher",
                family="teacher",
                rows=candidate_rows(baseline, {"track2_0001": "calm", "track2_0003": "content"}),
            ),
        ]
        matrix = build_evidence_matrix(baseline, sources)

        candidates = build_score_calibrated_candidates(baseline, matrix)
        safe_by_id = {item["sample_id"]: item for item in candidates["v3_safe_plus"]["rows"]}
        push_by_id = {item["sample_id"]: item for item in candidates["v3_push"]["rows"]}

        self.assertEqual(safe_by_id["track2_0001"]["emotion"], "calm")
        self.assertEqual(safe_by_id["track2_0003"]["emotion"], "content")
        self.assertEqual(safe_by_id["track2_0002"]["emotion"], "content")
        self.assertEqual(candidates["v3_safe_plus"]["changed_rows"], 2)
        self.assertEqual(push_by_id["track2_0002"]["emotion"], "content")

    def test_writer_protects_formal_paths_and_writes_candidate_zips(self):
        baseline = [
            row("track2_0001", "content"),
            row("track2_0002", "calm"),
        ]
        sources = [
            CandidateSource(
                name="public_style",
                family="public_style",
                rows=candidate_rows(baseline, {"track2_0001": "calm"}),
            ),
            CandidateSource(
                name="teacher",
                family="teacher",
                rows=candidate_rows(baseline, {"track2_0001": "calm"}),
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_json = tmp_path / "baseline.json"
            baseline_json.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaises(ValueError):
                write_score_calibrated_outputs(
                    baseline_json=baseline_json,
                    sources=sources,
                    out_dir=tmp_path / "experiments",
                    submission_dir=tmp_path / "submissions",
                    formal_submission_paths={tmp_path / "submissions" / "track2_submission.json"},
                    output_names={"v3_safe_plus": "track2_submission"},
                )

            report = write_score_calibrated_outputs(
                baseline_json=baseline_json,
                sources=sources,
                out_dir=tmp_path / "experiments",
                submission_dir=tmp_path / "submissions",
            )

            safe_json = Path(report["candidates"]["v3_safe_plus"]["json"])
            safe_zip = Path(report["candidates"]["v3_safe_plus"]["zip"])
            self.assertTrue(safe_json.exists())
            self.assertTrue(safe_zip.exists())
            with zipfile.ZipFile(safe_zip) as zf:
                self.assertEqual(zf.namelist(), ["submission.json"])
                payload = json.loads(zf.read("submission.json").decode("utf-8"))
            self.assertEqual(len(payload), 2)
            self.assertEqual(payload[0]["sample_id"], "track2_0001")

    def test_cli_writes_summary_and_side_path_candidates(self):
        baseline = [row("track2_0001", "content"), row("track2_0002", "calm")]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_json = tmp_path / "baseline.json"
            source_json = tmp_path / "source.json"
            out_dir = tmp_path / "out"
            submission_dir = tmp_path / "submissions"
            baseline_json.write_text(json.dumps(baseline), encoding="utf-8")
            source_json.write_text(
                json.dumps(candidate_rows(baseline, {"track2_0001": "calm"})),
                encoding="utf-8",
            )
            repo_root = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [
                    "python3",
                    str(repo_root / "scripts" / "track2_score_calibrated_champion.py"),
                    "--baseline-json",
                    str(baseline_json),
                    "--candidate",
                    f"public_style=public_style={source_json}",
                    "--candidate",
                    f"teacher=teacher={source_json}",
                    "--out-dir",
                    str(out_dir),
                    "--submission-dir",
                    str(submission_dir),
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out_dir / "evidence_matrix.csv").exists())
            self.assertTrue((out_dir / "score_calibrated_summary.json").exists())
            self.assertTrue((submission_dir / "track2_submission_v3_safe_plus_candidate.zip").exists())


if __name__ == "__main__":
    unittest.main()
