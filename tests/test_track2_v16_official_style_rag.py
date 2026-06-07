from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_v16_official_style_rag import (
    OfficialSubmissionScore,
    apply_v16_decisions_to_rows,
    build_rag_queue_rows,
    build_official_counterfactual_report,
    classify_transition_risk,
    load_official_scores,
    official_delta_summary,
    parse_transition_counts,
    should_accept_v16_decision,
    write_v16_candidate_outputs,
    write_official_counterfactual_outputs,
)


class Track2V16OfficialStyleRagTests(unittest.TestCase):
    def test_official_delta_summary_marks_781601_as_negative_batch(self) -> None:
        anchor = OfficialSubmissionScore(
            submission_id="779605",
            overall=0.836408,
            classification=0.723150,
            description=0.949667,
        )
        failed = OfficialSubmissionScore(
            submission_id="781601",
            overall=0.834027,
            classification=0.719137,
            description=0.948917,
        )
        summary = official_delta_summary(anchor=anchor, candidate=failed, changed_rows=85)
        self.assertLess(summary["classification_delta"], 0)
        self.assertLess(summary["classification_delta_per_changed_row"], 0)
        self.assertEqual(summary["batch_verdict"], "negative_official_evidence")

    def test_classify_transition_risk_blocks_failed_bulk_boundaries(self) -> None:
        risk = classify_transition_risk(
            transition="calm->content",
            evidence_sources={"same_quadrant_batch"},
            official_failed_transition_count=43,
        )
        self.assertEqual(risk, "blocked_by_failed_official_batch")

    def test_classify_transition_risk_allows_exact_duplicate(self) -> None:
        risk = classify_transition_risk(
            transition="content->calm",
            evidence_sources={"exact_public_duplicate"},
            official_failed_transition_count=20,
        )
        self.assertEqual(risk, "allow_exact_duplicate")

    def test_build_rag_queue_prioritizes_disagreement_and_blocks_failed_transitions(self) -> None:
        current_rows = [
            {"sample_id": "track2_0001", "emotion": "calm"},
            {"sample_id": "track2_0002", "emotion": "content"},
        ]
        prediction_rows = {
            "track2_0001": {
                "target_emotion": "content",
                "sources": ["same_quadrant_batch"],
                "confidence": 0.9,
            },
            "track2_0002": {
                "target_emotion": "glad",
                "sources": ["rag_teacher", "multibackbone_consensus"],
                "confidence": 0.8,
            },
        }
        queue = build_rag_queue_rows(
            current_rows=current_rows,
            prediction_rows=prediction_rows,
            failed_transition_counts={"calm->content": 43},
        )
        by_id = {row["sample_id"]: row for row in queue}
        self.assertEqual(by_id["track2_0001"]["risk_gate"], "blocked_by_failed_official_batch")
        self.assertEqual(by_id["track2_0002"]["risk_gate"], "allow_strong_consensus")
        self.assertGreater(by_id["track2_0002"]["priority_score"], by_id["track2_0001"]["priority_score"])

    def test_load_official_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scores.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "submission_id",
                        "file_name",
                        "official_overall",
                        "official_classification",
                        "official_description",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "submission_id": "779605",
                        "file_name": "anchor.zip",
                        "official_overall": "0.836408",
                        "official_classification": "0.723150",
                        "official_description": "0.949667",
                    }
                )
            scores = load_official_scores(path)
        self.assertEqual(scores["779605"].submission_id, "779605")
        self.assertAlmostEqual(scores["779605"].classification, 0.723150)

    def test_parse_transition_counts_ignores_malformed_items(self) -> None:
        counts = parse_transition_counts("calm->content:43; malformed; glad->content:4")
        self.assertEqual(counts, {"calm->content": 43, "glad->content": 4})

    def test_empty_official_scores_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_official_counterfactual_report(official_scores={}, pairwise_diff_rows=[])

    def test_write_official_counterfactual_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            official_scores = root / "scores.csv"
            pairwise_diffs = root / "diffs.csv"
            with official_scores.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "submission_id",
                        "file_name",
                        "official_overall",
                        "official_classification",
                        "official_description",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "submission_id": "779605",
                        "file_name": "anchor.zip",
                        "official_overall": "0.836408",
                        "official_classification": "0.723150",
                        "official_description": "0.949667",
                    }
                )
                writer.writerow(
                    {
                        "submission_id": "781601",
                        "file_name": "failed.zip",
                        "official_overall": "0.834027",
                        "official_classification": "0.719137",
                        "official_description": "0.948917",
                    }
                )
            with pairwise_diffs.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "candidate_a",
                        "candidate_b",
                        "emotion_label_changes",
                        "top_emotion_transitions",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "candidate_a": "781601_v3_mid_gemini35_desc_192",
                        "candidate_b": "779605_moe_v2_accept5_anchor",
                        "emotion_label_changes": "85",
                        "top_emotion_transitions": "calm->content:43; content->calm:20",
                    }
                )
            report = write_official_counterfactual_outputs(
                official_scores_csv=official_scores,
                pairwise_diffs_csv=pairwise_diffs,
                out_dir=root / "out",
            )
            payload = json.loads((root / "out" / "official_counterfactual_report.json").read_text())
        self.assertEqual(report["anchor_submission_id"], "779605")
        self.assertEqual(payload["known_failed_batches"][0]["batch_verdict"], "negative_official_evidence")
        self.assertEqual(payload["failed_transition_counts"]["calm->content"], 43)

    def test_should_accept_v16_decision_requires_strong_evidence_for_label_change(self) -> None:
        decision = {
            "sample_id": "track2_0001",
            "current_emotion": "calm",
            "proposed_emotion": "content",
            "confidence": 0.95,
            "evidence_sources": ["same_quadrant_batch"],
            "risk_gate": "blocked_by_failed_official_batch",
        }
        self.assertFalse(should_accept_v16_decision(decision))

    def test_should_accept_v16_decision_allows_rag_and_backbone_consensus(self) -> None:
        decision = {
            "sample_id": "track2_0002",
            "current_emotion": "content",
            "proposed_emotion": "glad",
            "confidence": 0.88,
            "evidence_sources": ["rag_teacher", "multibackbone_consensus"],
            "risk_gate": "allow_strong_consensus",
        }
        self.assertTrue(should_accept_v16_decision(decision))

    def test_apply_v16_decisions_repairs_quadrant_labels_and_caps_changes(self) -> None:
        rows = [
            {
                "sample_id": "track2_0001",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "x",
                "brushstroke": "x",
                "composition": "x",
                "color": "x",
                "line": "x",
                "light": "x",
            },
            {
                "sample_id": "track2_0002",
                "emotion": "content",
                "emotional_valence": "Positive",
                "emotional_arousal_level": "Low",
                "overall_caption": "x",
                "brushstroke": "x",
                "composition": "x",
                "color": "x",
                "line": "x",
                "light": "x",
            },
        ]
        decisions = [
            {
                "sample_id": "track2_0001",
                "current_emotion": "content",
                "proposed_emotion": "calm",
                "confidence": 0.91,
                "evidence_sources": ["rag_teacher", "multibackbone_consensus"],
                "risk_gate": "allow_strong_consensus",
            },
            {
                "sample_id": "track2_0002",
                "current_emotion": "content",
                "proposed_emotion": "glad",
                "confidence": 0.91,
                "evidence_sources": ["rag_teacher", "multibackbone_consensus"],
                "risk_gate": "allow_strong_consensus",
            },
        ]
        repaired, report = apply_v16_decisions_to_rows(rows, decisions, max_changes=1)
        self.assertEqual(report["accepted_label_changes"], 1)
        self.assertEqual(repaired[0]["emotion"], "calm")
        self.assertEqual(repaired[0]["emotional_valence"], "Positive")
        self.assertEqual(repaired[0]["emotional_arousal_level"], "Low")
        self.assertEqual(repaired[1]["emotion"], "content")
        self.assertEqual(report["rejection_counts"]["max_changes_reached"], 1)

    def test_write_v16_candidate_outputs_writes_side_path_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "base.json"
            decisions = root / "decisions.jsonl"
            out_json = root / "track2_submission_v16_test_candidate.json"
            out_zip = root / "track2_submission_v16_test_candidate.zip"
            rows = [
                {
                    "sample_id": "track2_0001",
                    "emotion": "content",
                    "emotional_valence": "Positive",
                    "emotional_arousal_level": "Low",
                    "overall_caption": "caption",
                    "brushstroke": "brush",
                    "composition": "composition",
                    "color": "color",
                    "line": "line",
                    "light": "light",
                }
            ]
            source.write_text(json.dumps(rows), encoding="utf-8")
            decisions.write_text(
                json.dumps(
                    {
                        "sample_id": "track2_0001",
                        "current_emotion": "content",
                        "proposed_emotion": "calm",
                        "confidence": 0.9,
                        "evidence_sources": ["rag_teacher", "multibackbone_consensus"],
                        "risk_gate": "allow_strong_consensus",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = write_v16_candidate_outputs(
                base_json=source,
                decisions_jsonl=decisions,
                out_json=out_json,
                out_zip=out_zip,
                report_json=root / "report.json",
                report_md=root / "report.md",
            )
            with zipfile.ZipFile(out_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
        self.assertEqual(report["accepted_label_changes"], 1)

    def test_write_v16_candidate_outputs_rejects_formal_submission_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "base.json"
            decisions = root / "decisions.jsonl"
            source.write_text("[]", encoding="utf-8")
            decisions.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_v16_candidate_outputs(
                    base_json=source,
                    decisions_jsonl=decisions,
                    out_json=root / "track2_submission.json",
                    out_zip=root / "track2_submission_v16_test_candidate.zip",
                    report_json=root / "report.json",
                    report_md=root / "report.md",
                )


if __name__ == "__main__":
    unittest.main()
