from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_registry_debug import (
    audit_source_index,
    audit_v22_scoreboard,
    score_source_row,
)


class Track2RegistryDebugTests(unittest.TestCase):
    def test_score_source_row_prioritizes_official_and_author_sources(self) -> None:
        official = score_source_row(
            {
                "source_type": "challenge",
                "name": "Codabench Track2",
                "url": "https://www.codabench.org/competitions/16304",
                "verified_public_fact": "Official format and scoring constraints",
                "track2_use": "Official format and scoring constraints",
                "status": "verified",
            }
        )
        author = score_source_row(
            {
                "source_type": "dataset",
                "name": "printblue/EmoArt-130k",
                "url": "https://huggingface.co/datasets/printblue/EmoArt-130k",
                "verified_public_fact": "132664 images; 56 styles; Annotation.json",
                "track2_use": "Training/reference/style prior",
                "status": "verified",
            }
        )
        auxiliary = score_source_row(
            {
                "source_type": "dataset",
                "name": "ArtEmis",
                "url": "https://artemisdataset.org/",
                "verified_public_fact": "Art emotion captions",
                "track2_use": "Description/caption style training; not Track2 gold",
                "status": "verified",
            }
        )

        self.assertEqual(official["source_scope"], "official")
        self.assertEqual(author["source_scope"], "author")
        self.assertEqual(auxiliary["source_scope"], "auxiliary")
        self.assertEqual(official["label_arbitration_allowed"], "yes")
        self.assertEqual(author["label_arbitration_allowed"], "yes")
        self.assertEqual(auxiliary["label_arbitration_allowed"], "no")
        self.assertGreater(float(author["classification_actionability"]), float(auxiliary["classification_actionability"]))
        self.assertGreater(float(official["track2_actionability_score"]), 0.8)

    def test_score_source_row_does_not_treat_auxiliary_official_code_as_official_gold(self) -> None:
        emovit = score_source_row(
            {
                "source_type": "code",
                "name": "aimmemotion/EmoVIT",
                "url": "https://github.com/aimmemotion/EmoVIT",
                "verified_public_fact": "Official CVPR 2024 code and model/data preparation notes",
                "track2_use": "Potential auxiliary emotion teacher",
                "status": "verified",
            }
        )

        self.assertEqual(emovit["source_scope"], "author")
        self.assertEqual(emovit["label_arbitration_allowed"], "no")
        self.assertLess(float(emovit["classification_actionability"]), 0.7)
        self.assertIn("auxiliary_only", emovit["risk_flags"])

    def test_audit_source_index_detects_missing_scoring_columns_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_index = Path(tmp) / "source_index.csv"
            with source_index.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["source_type", "name", "url", "verified_public_fact", "track2_use", "status"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "source_type": "profile",
                        "name": "Duplicated Source",
                        "url": "https://example.com/source",
                        "verified_public_fact": "First role",
                        "track2_use": "Core collaborator graph",
                        "status": "verified",
                    }
                )
                writer.writerow(
                    {
                        "source_type": "lab",
                        "name": "Duplicated Source",
                        "url": "https://example.com/source",
                        "verified_public_fact": "Second role",
                        "track2_use": "Method crawl seed",
                        "status": "verified",
                    }
                )

            report = audit_source_index(source_index)

        self.assertEqual(report["row_count"], 2)
        self.assertIn("missing_actionability_columns", report["issue_codes"])
        self.assertEqual(report["duplicate_url_group_count"], 1)
        self.assertEqual(report["duplicate_name_group_count"], 1)
        self.assertEqual(len(report["scored_rows"]), 2)
        self.assertIn("merge_group", report["scored_rows"][0])

    def test_audit_v22_scoreboard_checks_upload_zip_and_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_json = root / "candidate.json"
            upload_json = root / "track2_submission.json"
            upload_zip = root / "track2_submission.zip"
            rows = [
                {
                    "sample_id": "track2_0001",
                    "emotion": "calm",
                    "emotional_valence": "Positive",
                    "emotional_arousal_level": "Low",
                    "overall_caption": "Caption.",
                    "brushstroke": "Brush.",
                    "composition": "Comp.",
                    "color": "Color.",
                    "line": "Line.",
                    "light": "Light.",
                }
            ]
            payload = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
            candidate_json.write_text(payload.decode("utf-8"), encoding="utf-8")
            upload_json.write_text(payload.decode("utf-8"), encoding="utf-8")
            with zipfile.ZipFile(upload_zip, "w") as archive:
                archive.writestr("submission.json", payload)
            scoreboard = root / "scoreboard.json"
            scoreboard.write_text(
                json.dumps(
                    {
                        "recommended_profile": "calmshiftall",
                        "recommended_candidate": {
                            "paths": {"out_json": str(candidate_json)},
                            "accepted_label_changes": 177,
                            "label_consistency_issue_count": 0,
                            "missing_emotions": [],
                        },
                        "upload_copy": {"json": str(upload_json), "zip": str(upload_zip)},
                        "candidates": {
                            "calmshift120": {"official_projection": {"projected_overall": 0.846}},
                            "calmshiftall": {"official_projection": {"projected_overall": 0.848}},
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = audit_v22_scoreboard(scoreboard)

        self.assertEqual(report["recommended_profile"], "calmshiftall")
        self.assertEqual(report["issue_count"], 0)
        self.assertTrue(report["upload_zip_matches_recommended_json"])
