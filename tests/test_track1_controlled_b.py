from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from affectiveart.track1_controlled_b import (
    ControlledBCandidate,
    filter_current_identical_candidates,
    select_controlled_ladder,
)


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path, "JPEG")
    return path


class Track1ControlledBTest(unittest.TestCase):
    def test_select_controlled_ladder_prefers_high_priority_duplicate(self) -> None:
        candidates = [
            ControlledBCandidate(
                sample_id="track1_0001",
                candidate_image="low.jpg",
                source="low",
                priority=10,
                score=0.9,
            ),
            ControlledBCandidate(
                sample_id="track1_0001",
                candidate_image="high.jpg",
                source="high",
                priority=100,
                score=0.1,
            ),
            ControlledBCandidate(
                sample_id="track1_0002",
                candidate_image="other.jpg",
                source="other",
                priority=50,
                score=0.5,
            ),
        ]

        selected = select_controlled_ladder(candidates, target_count=2)

        self.assertEqual([item.sample_id for item in selected], ["track1_0001", "track1_0002"])
        self.assertEqual(selected[0].candidate_image, "high.jpg")

    def test_select_controlled_ladder_fails_closed_when_pool_is_too_small(self) -> None:
        candidates = [
            ControlledBCandidate(
                sample_id="track1_0001",
                candidate_image="one.jpg",
                source="one",
                priority=1,
                score=0.0,
            )
        ]

        with self.assertRaisesRegex(ValueError, "not enough unique candidates"):
            select_controlled_ladder(candidates, target_count=2)

    def test_filter_current_identical_candidates_blocks_exact_old_current_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_dir = root / "current"
            same_candidate = _image(root / "same.jpg", (255, 0, 0))
            different_candidate = _image(root / "different.jpg", (0, 255, 0))
            _image(current_dir / "track1_0001.jpg", (255, 0, 0))
            _image(current_dir / "track1_0002.jpg", (0, 0, 255))
            candidates = [
                ControlledBCandidate(
                    sample_id="track1_0001",
                    candidate_image=str(same_candidate),
                    source="same",
                    priority=1,
                    score=0.0,
                ),
                ControlledBCandidate(
                    sample_id="track1_0002",
                    candidate_image=str(different_candidate),
                    source="different",
                    priority=1,
                    score=0.0,
                ),
            ]

            filtered, blocked = filter_current_identical_candidates(candidates, current_dir)

            self.assertEqual([item.sample_id for item in filtered], ["track1_0002"])
            self.assertEqual([item.sample_id for item in blocked], ["track1_0001"])


if __name__ == "__main__":
    unittest.main()
