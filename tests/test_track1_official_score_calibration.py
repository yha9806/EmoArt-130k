import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track1_official_score_calibration import (
    calibrate_track1_packages,
    fit_fid_score_model,
    load_local_fid_rows,
    load_official_score_rows,
    official_overall,
    predict_fid_score,
    write_calibration_reports,
)


def _official_rows():
    return [
        {
            "participant": "emosuis",
            "submission_id": "top1",
            "official_overall": "0.80",
            "official_fid": "66.36",
            "official_fid_score": "0.60",
            "official_aas": "0.99",
        },
        {
            "participant": "edaich",
            "submission_id": "rank3",
            "official_overall": "0.77",
            "official_fid": "78.59",
            "official_fid_score": "0.56",
            "official_aas": "0.99",
        },
        {
            "participant": "vulcaart",
            "submission_id": "782831",
            "file_name": "track1_submit_v3_gate7_20260606.zip",
            "local_package": "v3_gate7",
            "official_overall": "0.77",
            "official_fid": "80.92",
            "official_fid_score": "0.55",
            "official_aas": "0.98",
            "local_fid_like": "63.010429",
        },
        {
            "participant": "EmoVCC",
            "submission_id": "744030",
            "official_overall": "0.66",
            "official_fid": "130.53",
            "official_fid_score": "0.43",
            "official_aas": "0.89",
        },
    ]


class Track1OfficialScoreCalibrationTest(unittest.TestCase):
    def test_fit_fid_score_model_is_monotonic_lower_fid_better(self):
        model = fit_fid_score_model(_official_rows())

        self.assertLess(model["slope"], 0.0)
        self.assertGreater(predict_fid_score(66.36, model), predict_fid_score(130.53, model))

    def test_official_overall_uses_equal_weights(self):
        self.assertAlmostEqual(official_overall(0.55, 0.98), 0.765)

    def test_calibration_ranks_better_local_fid_like_above_anchor(self):
        local_rows = [
            {"package": "current", "fid_like": 58.526207},
            {"package": "v3_gate7", "fid_like": 63.010429},
            {"package": "full1000", "fid_like": 63.098811},
        ]

        report = calibrate_track1_packages(_official_rows(), local_rows, anchor_package="v3_gate7")

        self.assertEqual(report["calibration_confidence"]["level"], "low")
        self.assertEqual(report["packages"][0]["package"], "current")
        current = next(row for row in report["packages"] if row["package"] == "current")
        anchor = next(row for row in report["packages"] if row["package"] == "v3_gate7")
        self.assertGreater(current["overall_expected"], anchor["overall_expected"])
        self.assertIn("LOCAL SHADOW SCORE ONLY", report["warning"])

    def test_extreme_local_fid_like_values_are_bounded(self):
        local_rows = [
            {"package": "unrealistically_good", "fid_like": -1000.0},
            {"package": "v3_gate7", "fid_like": 63.010429},
            {"package": "unrealistically_bad", "fid_like": 1000.0},
        ]

        report = calibrate_track1_packages(_official_rows(), local_rows, anchor_package="v3_gate7")

        for row in report["packages"]:
            self.assertGreaterEqual(row["overall_lower"], 0.0)
            self.assertLessEqual(row["overall_upper"], 1.0)
            for scenario in row["scenarios"].values():
                self.assertGreaterEqual(scenario["projected_fid_score"], 0.0)
                self.assertLessEqual(scenario["projected_fid_score"], 1.0)

    def test_loaders_and_cli_write_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "anchors.csv"
            csv_path.write_text(
                "participant,submission_id,official_overall,official_fid,official_fid_score,official_aas,local_package,local_fid_like\n"
                "emosuis,top1,0.80,66.36,0.60,0.99,,\n"
                "vulcaart,782831,0.77,80.92,0.55,0.98,v3_gate7,63.010429\n"
                "EmoVCC,744030,0.66,130.53,0.43,0.89,,\n",
                encoding="utf-8",
            )
            fid_path = root / "fid.json"
            fid_path.write_text(
                json.dumps(
                    {
                        "packages": [
                            {"package": "current", "fid_like": 58.526207},
                            {"package": "v3_gate7", "fid_like": 63.010429},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = calibrate_track1_packages(
                load_official_score_rows(csv_path),
                load_local_fid_rows(fid_path),
                anchor_package="v3_gate7",
            )
            out_json = root / "report.json"
            out_md = root / "report.md"
            write_calibration_reports(report, out_json, out_md)

            script = Path(__file__).resolve().parents[1] / "scripts" / "track1_official_score_calibration.py"
            spec = importlib.util.spec_from_file_location("track1_official_score_calibration_cli", script)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            code = module.main(
                [
                    "--official-anchors-csv",
                    str(csv_path),
                    "--local-fid-json",
                    str(fid_path),
                    "--anchor-package",
                    "v3_gate7",
                    "--out-json",
                    str(root / "cli.json"),
                    "--out-md",
                    str(root / "cli.md"),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue(out_json.exists())
            self.assertIn("Track1 Local Shadow Score Calibration", out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
