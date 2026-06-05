import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from affectiveart.track2_description_only_merge import build_description_only_outputs, build_description_only_rows


def _row(sample_id: str, emotion: str, caption: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": "Positive",
        "emotional_arousal_level": "Low",
        "overall_caption": caption,
        "brushstroke": f"{caption} brushstroke detail.",
        "composition": f"{caption} composition detail.",
        "color": f"{caption} color detail.",
        "line": f"{caption} line detail.",
        "light": f"{caption} light detail.",
    }


class Track2DescriptionOnlyMergeTest(unittest.TestCase):
    def test_preserves_labels_while_copying_text_fields(self):
        labels = [_row("track2_0001", "calm", "old")]
        text = [_row("track2_0001", "content", "new")]

        rows, report = build_description_only_rows(labels, text)

        self.assertEqual(rows[0]["emotion"], "calm")
        self.assertEqual(rows[0]["overall_caption"], "new")
        self.assertEqual(report["classification_label_changes"], 0)
        self.assertEqual(report["text_changed_rows"], 1)

    def test_writes_side_path_candidate_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels_json = root / "labels.json"
            text_json = root / "text.json"
            out_json = root / "track2_submission_v12_stable_probe_candidate.json"
            out_zip = root / "track2_submission_v12_stable_probe_candidate.zip"
            report_json = root / "report.json"
            report_md = root / "report.md"
            labels_json.write_text(json.dumps([_row("track2_0001", "calm", "old")]), encoding="utf-8")
            text_json.write_text(json.dumps([_row("track2_0001", "content", "new")]), encoding="utf-8")

            report = build_description_only_outputs(
                label_source_json=labels_json,
                text_source_json=text_json,
                out_json=out_json,
                out_zip=out_zip,
                report_json=report_json,
                report_md=report_md,
            )

            self.assertFalse(report["formal_submission_overwritten"])
            self.assertEqual(report["classification_label_changes"], 0)
            with zipfile.ZipFile(out_zip) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
