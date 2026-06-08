from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from affectiveart.track2_v22_official_author_scorer import (
    OfficialScore,
    build_style_emotion_prior,
    build_v22_candidate_suite,
    classify_source_scope,
    discover_local_emoart_inventory,
    score_calmshift_change,
    select_v22_calmshift_changes,
    source_can_arbitrate_label,
)


def _evidence_row(
    *,
    sample_id: str = "track2_0001",
    current: str = "content",
    proposed: str = "calm",
    support: float = 3.2,
    votes: int = 3,
    confidence: float = 0.91,
    duplicate: float = 0.96,
    style: float = 0.7,
    near: bool = True,
    exact: bool = False,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "current_emotion": current,
        "proposed_emotion": proposed,
        "transition": f"{current}->{proposed}",
        "support_score": support,
        "model_vote_count": votes,
        "max_confidence": confidence,
        "public_duplicate_support_score": duplicate,
        "public_style_support_score": style,
        "near_duplicate": near,
        "exact_duplicate": exact,
        "all_sources": "clip,dinov2,public_clean,public_inclusive,public_near_public_duplicate,siglip2",
    }


def _base_row(sample_id: str, emotion: str = "content") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": "Positive",
        "emotional_arousal_level": "Low",
        "overall_caption": "A specific description of the artwork and its atmosphere.",
        "brushstroke": "Controlled brushwork supports the emotional tone.",
        "composition": "The composition is balanced and clearly arranged.",
        "color": "The palette is restrained and coherent.",
        "line": "Line quality is precise and expressive.",
        "light": "Soft lighting reinforces the mood.",
    }


class Track2V22EvidenceBoundaryTests(unittest.TestCase):
    def test_classifies_official_and_author_sources_as_arbitration_sources(self) -> None:
        self.assertEqual(classify_source_scope("https://www.codabench.org/competitions/16304"), "official")
        self.assertEqual(classify_source_scope("https://openreview.net/forum?id=LbbHX8ofXZ"), "official")
        self.assertEqual(classify_source_scope("https://huggingface.co/datasets/printblue/EmoArt-130k"), "author")
        self.assertEqual(classify_source_scope("https://github.com/zhiliangzhang/FAB-G"), "author")
        self.assertEqual(classify_source_scope("/tmp/EmoArt-130k"), "local_author_data")

    def test_blocks_auxiliary_sources_from_label_arbitration(self) -> None:
        self.assertEqual(classify_source_scope("https://arxiv.org/abs/2101.07396"), "auxiliary")
        self.assertFalse(source_can_arbitrate_label("auxiliary"))
        self.assertTrue(source_can_arbitrate_label("official"))
        self.assertTrue(source_can_arbitrate_label("author"))
        self.assertTrue(source_can_arbitrate_label("local_author_data"))

    def test_discovers_local_emoart_inventory_from_minimal_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Annotation.json").write_text(
                json.dumps(
                    [
                        {
                            "image_path": "Images\\Chinese Painting\\example.jpg",
                            "request_id": "Chinese Painting_request-1",
                            "description": {},
                        },
                        {
                            "image_path": "Images\\Ukiyo-e\\example.jpg",
                            "request_id": "Ukiyo-e_request-1",
                            "description": {},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            (root / "Chinese Painting.tar.gz").write_bytes(b"stub")
            (root / "Ukiyo-e.tar.gz").write_bytes(b"stub")

            inventory = discover_local_emoart_inventory(root)

        self.assertEqual(inventory["annotation_rows"], 2)
        self.assertEqual(inventory["tar_count"], 2)
        self.assertEqual(inventory["styles"], ["Chinese Painting", "Ukiyo-e"])


class Track2V22ScoringTests(unittest.TestCase):
    def test_build_style_emotion_prior_parses_description_dict_and_json(self) -> None:
        rows = [
            {
                "image_path": "Images\\Chinese Painting\\a.jpg",
                "description": {
                    "third_section": {
                        "dominant_emotion": "Calm",
                        "emotional_valence": "Positive",
                        "emotional_arousal_level": "Low",
                    }
                },
            },
            {
                "image_path": "Images\\Chinese Painting\\b.jpg",
                "description": json.dumps(
                    {
                        "third_section": {
                            "dominant_emotion": "Contentment",
                            "emotional_valence": "Positive",
                            "emotional_arousal_level": "Low",
                        }
                    }
                ),
            },
            {
                "image_path": "Images\\Ukiyo-e\\c.jpg",
                "description": {
                    "third_section": {
                        "dominant_emotion": "Calm",
                        "emotional_valence": "Positive",
                        "emotional_arousal_level": "Low",
                    }
                },
            },
        ]

        prior = build_style_emotion_prior(rows)

        chinese = [row for row in prior if row["style"] == "Chinese Painting"]
        self.assertEqual({row["emotion"] for row in chinese}, {"calm", "content"})
        self.assertEqual(sum(row["count"] for row in chinese), 2)

    def test_score_calmshift_uses_official_verified_direction(self) -> None:
        score = score_calmshift_change(
            _evidence_row(),
            official_delta=OfficialScore(
                submission_id="785979",
                overall=0.842559,
                classification=0.740034,
                description=0.945083,
                emotion_accuracy=0.66,
                emotion_macro_f1=0.320642,
            ),
        )

        self.assertEqual(score["decision"], "accept_candidate")
        self.assertEqual(score["tier"], "official_reference")
        self.assertGreater(score["score"], 4.0)
        self.assertIn("official_v21_content_to_calm_net_gain", score["reasons"])

    def test_score_blocks_non_calmshift_transition(self) -> None:
        score = score_calmshift_change(_evidence_row(current="calm", proposed="content"))

        self.assertEqual(score["decision"], "block")
        self.assertIn("not_content_to_calm", score["reasons"])

    def test_select_v22_calmshift_changes_respects_target_and_existing_base(self) -> None:
        rows = [
            _evidence_row(sample_id=f"track2_{index:04d}", support=3.0 - index * 0.01)
            for index in range(12)
        ]
        base_distribution = Counter({"content": 12, "calm": 100})

        selected = select_v22_calmshift_changes(rows, target_count=5, base_distribution=base_distribution)

        self.assertEqual(len(selected), 5)
        self.assertEqual([row["sample_id"] for row in selected], [f"track2_{index:04d}" for index in range(5)])
        self.assertTrue(all(row["transition"] == "content->calm" for row in selected))

    def test_build_v22_candidate_suite_writes_side_path_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_json = root / "base.json"
            evidence_json = root / "evidence.json"
            out_dir = root / "out"
            submissions_dir = root / "submissions"
            base_json.write_text(
                json.dumps([_base_row(f"track2_{index:04d}") for index in range(6)]),
                encoding="utf-8",
            )
            evidence_json.write_text(
                json.dumps([_evidence_row(sample_id=f"track2_{index:04d}") for index in range(6)]),
                encoding="utf-8",
            )

            suite = build_v22_candidate_suite(
                base_json=base_json,
                evidence_json=evidence_json,
                out_dir=out_dir,
                submissions_dir=submissions_dir,
                ladder_counts=(3, 5),
                write_upload_copy=True,
            )

            self.assertFalse((submissions_dir / "track2_submission.zip").exists())
            self.assertTrue((submissions_dir / "v22_final_upload" / "track2_submission.zip").exists())
            self.assertEqual(suite["recommended_profile"], "calmshift5")
            self.assertEqual(suite["candidates"]["calmshift5"]["accepted_label_changes"], 5)
            with zipfile.ZipFile(submissions_dir / "v22_final_upload" / "track2_submission.zip") as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
