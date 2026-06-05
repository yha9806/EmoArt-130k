import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from affectiveart.track1_official_reference_bank import build_official_reference_bank


def _jpg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_style_tar(root: Path, style: str, filenames: list[str]) -> None:
    with tarfile.open(root / f"{style}.tar.gz", "w:gz") as archive:
        for index, filename in enumerate(filenames):
            payload = _jpg_bytes((40 + index, 80, 120))
            info = tarfile.TarInfo(f"{style}/{filename}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _annotation(style: str, filename: str, description: str) -> dict:
    return {
        "request_id": f"{style}_request-{filename}",
        "image_path": f"Images\\{style}\\{filename}",
        "description": {
            "first_section": {"description": description},
            "second_section": {
                "visual_attributes": {
                    "brushstroke": "bold brushwork",
                    "color": "red and blue palette",
                    "composition": "dynamic composition",
                    "light_and_shadow": "dramatic lighting",
                    "line_quality": "strong lines",
                },
                "emotional_impact": "heroic mood",
            },
        },
    }


class Track1OfficialReferenceBankTest(unittest.TestCase):
    def test_builds_route_references_from_official_style_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emoart = root / "emoart"
            emoart.mkdir()
            _write_style_tar(
                emoart,
                "Abstract Art",
                [
                    "0000001_RedCircle.jpg",
                    "0000002_BlueTriangle.jpg",
                    "0000003_GreenSquare.jpg",
                ],
            )
            (emoart / "Annotation.json").write_text(
                json.dumps(
                    [
                        _annotation("Abstract Art", "0000001_RedCircle.jpg", "red circular abstract geometry"),
                        _annotation("Abstract Art", "0000002_BlueTriangle.jpg", "blue triangle abstraction"),
                        _annotation("Abstract Art", "0000003_GreenSquare.jpg", "green square minimal form"),
                    ]
                ),
                encoding="utf-8",
            )
            out_dir = root / "bank"

            result = build_official_reference_bank(
                [{"sample_id": "track1_0001", "caption": "An abstract artwork with red circular geometry."}],
                emoart_root=emoart,
                out_dir=out_dir,
                max_candidates_per_route=2,
            )

            refs = result["index"]["route_references"]["track1_0001"]
            self.assertEqual(len(refs), 2)
            self.assertIn("RedCircle", refs[0]["file"])
            self.assertIn("official EmoArt-130k retrieval", refs[0]["note"])
            self.assertTrue((out_dir / "reference_assets" / refs[0]["file"]).exists())
            self.assertEqual(result["summary"]["official_candidate_pool_by_style"]["Abstract Art"], 3)

    def test_socialist_realism_poster_caption_prefers_poster_like_official_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emoart = root / "emoart"
            emoart.mkdir()
            _write_style_tar(
                emoart,
                "Socialist Realism",
                [
                    "0000001_QuietLandscape.jpg",
                    "0000002_VictoryPoster.jpg",
                    "0000003_WorkerPortrait.jpg",
                ],
            )
            (emoart / "Annotation.json").write_text(
                json.dumps(
                    [
                        _annotation("Socialist Realism", "0000001_QuietLandscape.jpg", "quiet countryside painting"),
                        _annotation("Socialist Realism", "0000002_VictoryPoster.jpg", "victory propaganda poster with red flags"),
                        _annotation("Socialist Realism", "0000003_WorkerPortrait.jpg", "worker portrait painting"),
                    ]
                ),
                encoding="utf-8",
            )

            result = build_official_reference_bank(
                [
                    {
                        "sample_id": "track1_0077",
                        "caption": "A Socialist Realism Soviet propaganda poster with red flags and bold Cyrillic typography.",
                    }
                ],
                emoart_root=emoart,
                out_dir=root / "bank",
                max_candidates_per_route=2,
            )

        refs = result["index"]["route_references"]["track1_0077"]
        self.assertIn("VictoryPoster", refs[0]["file"])
        self.assertIn("candidate_pool=3", refs[0]["note"])

    def test_official_style_matching_covers_parenthetical_and_diacritic_styles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emoart = root / "emoart"
            emoart.mkdir()
            _write_style_tar(emoart, "Art Nouveau (Modern)", ["0000001_OrnateFloral.jpg"])
            _write_style_tar(emoart, "Naïve Art (Primitivism)", ["0000002_FolkVillage.jpg"])
            (emoart / "Annotation.json").write_text(
                json.dumps(
                    [
                        _annotation("Art Nouveau (Modern)", "0000001_OrnateFloral.jpg", "ornate floral design"),
                        _annotation("Naïve Art (Primitivism)", "0000002_FolkVillage.jpg", "folk village scene"),
                    ]
                ),
                encoding="utf-8",
            )

            result = build_official_reference_bank(
                [
                    {"sample_id": "track1_0101", "caption": "An Art Nouveau ornamental poster with flowers."},
                    {"sample_id": "track1_0102", "caption": "A Naive Art village scene with flattened perspective."},
                ],
                emoart_root=emoart,
                out_dir=root / "bank",
                max_candidates_per_route=1,
            )

        first_ref = result["index"]["route_references"]["track1_0101"][0]
        second_ref = result["index"]["route_references"]["track1_0102"][0]
        self.assertIn("Art_Nouveau_Modern", first_ref["file"])
        self.assertIn("style=Art Nouveau (Modern)", first_ref["note"])
        self.assertIn("Naive_Art_Primitivism", second_ref["file"])
        self.assertIn("style=Naïve Art (Primitivism)", second_ref["note"])

    def test_annotation_subpath_cannot_escape_reference_asset_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emoart = root / "emoart"
            emoart.mkdir()
            _write_style_tar(emoart, "Abstract Art", ["outside.jpg"])
            (emoart / "Annotation.json").write_text(
                json.dumps(
                    [
                        {
                            **_annotation("Abstract Art", "outside.jpg", "abstract red geometry"),
                            "image_path": "Images\\Abstract Art\\..\\outside.jpg",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result = build_official_reference_bank(
                [{"sample_id": "track1_0001", "caption": "An Abstract Art image with red geometry."}],
                emoart_root=emoart,
                out_dir=root / "bank",
                max_candidates_per_route=1,
            )

            ref_file = result["index"]["route_references"]["track1_0001"][0]["file"]
            self.assertNotIn("..", Path(ref_file).parts)
            self.assertEqual(Path(ref_file).name, ref_file)
            self.assertTrue((root / "bank" / "reference_assets" / ref_file).exists())
            self.assertFalse((root / "outside.jpg").exists())

    def test_selection_is_deterministic_when_annotation_order_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emoart = root / "emoart"
            emoart.mkdir()
            filenames = ["0000001_Alpha.jpg", "0000002_Beta.jpg", "0000003_Gamma.jpg"]
            _write_style_tar(emoart, "Abstract Art", filenames)

            def build_with_order(order: list[str], out_name: str) -> list[str]:
                (emoart / "Annotation.json").write_text(
                    json.dumps([_annotation("Abstract Art", filename, "abstract geometry") for filename in order]),
                    encoding="utf-8",
                )
                result = build_official_reference_bank(
                    [{"sample_id": "track1_0001", "caption": "An Abstract Art image with abstract geometry."}],
                    emoart_root=emoart,
                    out_dir=root / out_name,
                    max_candidates_per_route=3,
                )
                return [row["file"] for row in result["index"]["route_references"]["track1_0001"]]

            first = build_with_order(filenames, "bank1")
            second = build_with_order(list(reversed(filenames)), "bank2")

        self.assertEqual(first, second)

    def test_sanitized_reference_asset_name_collisions_are_avoided(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emoart = root / "emoart"
            emoart.mkdir()
            _write_style_tar(emoart, "Abstract Art", ["0000001_A-B.jpg", "0000002_A B.jpg"])
            (emoart / "Annotation.json").write_text(
                json.dumps(
                    [
                        _annotation("Abstract Art", "0000001_A-B.jpg", "abstract geometry"),
                        _annotation("Abstract Art", "0000002_A B.jpg", "abstract geometry"),
                    ]
                ),
                encoding="utf-8",
            )

            result = build_official_reference_bank(
                [{"sample_id": "track1_0001", "caption": "An Abstract Art image with abstract geometry."}],
                emoart_root=emoart,
                out_dir=root / "bank",
                max_candidates_per_route=2,
            )

            files = [row["file"] for row in result["index"]["route_references"]["track1_0001"]]
            self.assertEqual(len(files), 2)
            self.assertEqual(len(set(files)), 2)
            for file in files:
                self.assertTrue((root / "bank" / "reference_assets" / file).exists())


if __name__ == "__main__":
    unittest.main()
