import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track1_official_score_calibration import (
    build_scorer_reproducibility_audit,
    calibrate_track1_packages,
    fit_fid_score_model,
    load_local_fid_rows,
    load_official_score_rows,
    merge_local_anchor_metadata,
    official_overall,
    predict_fid_score,
    write_calibration_reports,
    write_scorer_reproducibility_audit,
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

    def test_two_own_anchors_detect_anti_correlated_local_proxy(self):
        official_rows = _official_rows() + [
            {
                "participant": "vulcaart",
                "submission_id": "784403",
                "file_name": "hybrid_probe_redteam_fid_pass4.zip",
                "local_package": "hybrid_redteam_fid_pass4",
                "official_overall": "0.74",
                "official_fid": "105.66",
                "official_fid_score": "0.49",
                "official_aas": "0.99",
                "local_fid_like": "58.348904",
            }
        ]
        local_rows = [
            {"package": "current", "fid_like": 58.526207},
            {"package": "v3_gate7", "fid_like": 63.010429},
            {"package": "hybrid_redteam_fid_pass4", "fid_like": 58.348904},
            {"package": "full1000_no_fallback", "fid_like": 63.098811},
        ]

        report = calibrate_track1_packages(official_rows, local_rows, anchor_package="v3_gate7")

        self.assertEqual(report["local_to_official_fid_model"]["proxy_direction"], "anti_correlated")
        self.assertEqual(report["packages"][0]["package"], "full1000_no_fallback")
        current = next(row for row in report["packages"] if row["package"] == "current")
        hybrid = next(row for row in report["packages"] if row["package"] == "hybrid_redteam_fid_pass4")
        self.assertEqual(hybrid["projection_method"], "observed_official")
        self.assertLess(current["overall_expected"], report["anchor"]["official_overall"])

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

    def test_load_official_score_rows_normalizes_codabench_api_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            api_json = Path(tmp) / "codabench.json"
            api_json.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "id": 784403,
                                "owner": "vulcaart",
                                "filename": "hybrid_probe.zip",
                                "created_when": "2026-06-07T14:42:38Z",
                                "scores": [
                                    {"column_key": "track1_overall", "score": "0.7396402011"},
                                    {"column_key": "fid", "score": "105.6638160234"},
                                    {"column_key": "fid_score", "score": "0.4862304023"},
                                    {"column_key": "aas", "score": "0.9930500000"},
                                    {"column_key": "content_alignment", "score": "0.9920000000"},
                                    {"column_key": "style_alignment", "score": "0.9939000000"},
                                    {"column_key": "attribute_alignment", "score": "0.9932500000"},
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows = load_official_score_rows(api_json)

        self.assertEqual(rows[0]["participant"], "vulcaart")
        self.assertEqual(rows[0]["submission_id"], "784403")
        self.assertEqual(rows[0]["file_name"], "hybrid_probe.zip")
        self.assertEqual(rows[0]["official_fid"], 105.6638160234)
        self.assertEqual(rows[0]["official_fid_score"], 0.4862304023)
        self.assertEqual(rows[0]["official_aas"], 0.99305)
        self.assertEqual(rows[0]["content_alignment"], 0.992)

    def test_reproducibility_audit_reports_formula_fit_and_local_proxy_readiness(self):
        official_rows = [
            {
                "participant": "emosuis",
                "submission_id": "778783",
                "official_overall": "0.7969073906",
                "official_fid": "66.3621921022",
                "official_fid_score": "0.6010981145",
                "official_aas": "0.9927166667",
            },
            {
                "participant": "edaich",
                "submission_id": "740721",
                "official_overall": "0.7748755042",
                "official_fid": "78.5870523547",
                "official_fid_score": "0.5599510085",
                "official_aas": "0.9898000000",
            },
            {
                "participant": "vulcaart",
                "submission_id": "782831",
                "file_name": "track1_submit_v3_gate7_20260606.zip",
                "local_package": "v3_gate7",
                "official_overall": "0.7650000000",
                "official_fid": "80.9200000000",
                "official_fid_score": "0.5500000000",
                "official_aas": "0.9800000000",
                "local_fid_like": "63.010429",
            },
            {
                "participant": "vulcaart",
                "submission_id": "784403",
                "file_name": "hybrid_probe_redteam_fid_pass4.zip",
                "local_package": "hybrid_redteam_fid_pass4",
                "official_overall": "0.7396402011",
                "official_fid": "105.6638160234",
                "official_fid_score": "0.4862304023",
                "official_aas": "0.9930500000",
                "local_fid_like": "58.348904",
            },
        ]

        audit = build_scorer_reproducibility_audit(official_rows)

        self.assertLess(audit["official_formula_reproduction"]["max_abs_error"], 1e-9)
        self.assertLess(audit["fid_score_reproduction"]["rmse"], 0.02)
        self.assertEqual(audit["local_proxy_reproduction"]["own_anchor_count"], 2)
        self.assertEqual(audit["local_proxy_reproduction"]["readiness"], "insufficient_own_anchors")
        self.assertEqual(audit["local_proxy_reproduction"]["proxy_direction"], "anti_correlated")

    def test_write_scorer_reproducibility_audit_outputs_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = build_scorer_reproducibility_audit(
                [
                    {
                        "participant": "emosuis",
                        "submission_id": "778783",
                        "official_overall": "0.7969073906",
                        "official_fid": "66.3621921022",
                        "official_fid_score": "0.6010981145",
                        "official_aas": "0.9927166667",
                    },
                    {
                        "participant": "vulcaart",
                        "submission_id": "782831",
                        "local_package": "v3_gate7",
                        "official_overall": "0.7650000000",
                        "official_fid": "80.9200000000",
                        "official_fid_score": "0.5500000000",
                        "official_aas": "0.9800000000",
                        "local_fid_like": "63.010429",
                    },
                ]
            )

            write_scorer_reproducibility_audit(audit, root / "audit.json", root / "audit.md")

            self.assertEqual(json.loads((root / "audit.json").read_text(encoding="utf-8"))["method"], audit["method"])
            md = (root / "audit.md").read_text(encoding="utf-8")
            self.assertIn("Track1 Scorer Reproducibility Audit", md)
            self.assertIn("insufficient_own_anchors", md)

    def test_merge_local_anchor_metadata_preserves_exact_api_scores(self):
        api_rows = [
            {
                "participant": "vulcaart",
                "submission_id": "784403",
                "file_name": "hybrid_probe.zip",
                "official_overall": 0.7396402011,
                "official_fid": 105.6638160234,
                "official_fid_score": 0.4862304023,
                "official_aas": 0.99305,
            }
        ]
        anchor_rows = [
            {
                "submission_id": "784403",
                "local_package": "hybrid_redteam_fid_pass4",
                "local_fid_like": "58.348904",
                "official_overall": "0.74",
                "official_fid": "105.66",
            },
            {
                "submission_id": "782831",
                "local_package": "v3_gate7",
                "local_fid_like": "63.010429",
                "official_overall": "0.765",
                "official_fid": "80.92",
                "official_fid_score": "0.55",
                "official_aas": "0.98",
            },
        ]

        rows = merge_local_anchor_metadata(api_rows, anchor_rows)

        hybrid = next(row for row in rows if row["submission_id"] == "784403")
        self.assertEqual(hybrid["official_fid"], 105.6638160234)
        self.assertEqual(hybrid["local_package"], "hybrid_redteam_fid_pass4")
        self.assertEqual(hybrid["local_fid_like"], "58.348904")
        self.assertIn("782831", {row["submission_id"] for row in rows})

    def test_reproducibility_audit_cli_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api_json = root / "api.json"
            api_json.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "id": 784403,
                                "owner": "vulcaart",
                                "filename": "hybrid_probe.zip",
                                "scores": [
                                    {"column_key": "track1_overall", "score": "0.7396402011"},
                                    {"column_key": "fid", "score": "105.6638160234"},
                                    {"column_key": "fid_score", "score": "0.4862304023"},
                                    {"column_key": "aas", "score": "0.9930500000"},
                                ],
                            },
                            {
                                "id": 778783,
                                "owner": "emosuis",
                                "filename": "submission.zip",
                                "scores": [
                                    {"column_key": "track1_overall", "score": "0.7969073906"},
                                    {"column_key": "fid", "score": "66.3621921022"},
                                    {"column_key": "fid_score", "score": "0.6010981145"},
                                    {"column_key": "aas", "score": "0.9927166667"},
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            anchors_csv = root / "anchors.csv"
            anchors_csv.write_text(
                "participant,submission_id,local_package,local_fid_like,official_overall,official_fid,official_fid_score,official_aas\n"
                "vulcaart,784403,hybrid_redteam_fid_pass4,58.348904,0.74,105.66,0.49,0.99\n",
                encoding="utf-8",
            )
            script = Path(__file__).resolve().parents[1] / "scripts" / "track1_scorer_reproducibility_audit.py"
            spec = importlib.util.spec_from_file_location("track1_scorer_reproducibility_audit_cli", script)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)

            code = module.main(
                [
                    "--official-scores-json",
                    str(api_json),
                    "--local-anchors-csv",
                    str(anchors_csv),
                    "--out-json",
                    str(root / "audit.json"),
                    "--out-md",
                    str(root / "audit.md"),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((root / "audit.json").exists())
            self.assertIn("Track1 Scorer Reproducibility Audit", (root / "audit.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
