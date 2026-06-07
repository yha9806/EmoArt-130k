import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track1_distribution_shortlist import (
    build_distribution_shortlist,
    parse_report_spec,
    render_shortlist_html,
    render_shortlist_markdown,
    write_shortlist_reports,
)


def _side(distribution, perceptual, official_like=0.5):
    return {
        "aas_proxy": 0.0,
        "distribution_proxy": distribution,
        "perceptual_proxy": perceptual,
        "surface_gate": 1.0,
        "official_like_score": official_like,
        "hard_reject_reasons": [],
    }


def _row(sample_id, delta, source_path="candidate.jpg", *, changed=True):
    return {
        "sample_id": sample_id,
        "changed": changed,
        "delta": delta,
        "recommendation": "keep_current",
        "current": _side(0.5, 0.7),
        "candidate": _side(0.5 + delta, 0.72, official_like=0.5 + delta),
        "current_image": "current.jpg",
        "candidate_image": source_path,
    }


class Track1DistributionShortlistTest(unittest.TestCase):
    def test_parse_report_spec_requires_label_and_path(self):
        self.assertEqual(parse_report_spec("full=metric.json"), ("full", Path("metric.json")))
        with self.assertRaises(ValueError):
            parse_report_spec("metric.json")

    def test_build_shortlist_deduplicates_by_best_delta_without_accepting(self):
        report = build_distribution_shortlist(
            {
                "full1000": {"rows": [_row("track1_0001", 0.04, "full.jpg")]},
                "partial757": {"rows": [_row("track1_0001", 0.06, "partial.jpg"), _row("track1_0002", 0.02)]},
            },
            captions={"track1_0001": "caption one"},
            top_n=10,
            min_delta=0.03,
            high_delta=0.05,
        )

        queue = report["review_queue"]
        self.assertEqual([row["sample_id"] for row in queue], ["track1_0001"])
        self.assertEqual(queue[0]["source"], "partial757")
        self.assertEqual(queue[0]["review_tier"], "high_delta_review")
        self.assertEqual(queue[0]["acceptance_status"], "review_needed_not_accepted")
        self.assertIn("candidate_aas_vlm_review_missing", queue[0]["blocking_gates"])
        self.assertEqual(queue[0]["duplicate_candidate_count"], 2)
        self.assertEqual(queue[0]["alternatives"][0]["source"], "full1000")
        self.assertEqual(report["summary"]["watchlist_samples"], 1)

    def test_markdown_and_html_render_chinese_review_labels_and_escape_values(self):
        report = build_distribution_shortlist(
            {"full": {"rows": [_row("track1_<x>", 0.05)]}},
            captions={"track1_<x>": "<caption>"},
            top_n=1,
            min_delta=0.03,
            high_delta=0.05,
        )

        md = render_shortlist_markdown(report)
        html = render_shortlist_html(report, out_html=Path("/tmp/review.html"))

        self.assertIn("不是 accepted replacement manifest", md)
        self.assertIn("current：当前冠军包", html)
        self.assertIn("candidate：本地 FID/proxy 候选", html)
        self.assertIn("&lt;caption&gt;", html)

    def test_write_reports_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metric = root / "metric.json"
            metric.write_text(json.dumps({"rows": [_row("track1_0001", 0.04)]}), encoding="utf-8")
            out_json = root / "shortlist.json"
            out_csv = root / "shortlist.csv"
            out_md = root / "shortlist.md"
            out_html = root / "shortlist.html"
            report = build_distribution_shortlist({"full": json.loads(metric.read_text())}, top_n=1)
            write_shortlist_reports(report, json_path=out_json, csv_path=out_csv, md_path=out_md, html_path=out_html)

            self.assertTrue(out_json.exists())
            self.assertIn("track1_0001", out_csv.read_text(encoding="utf-8"))
            self.assertIn("Track1 V4 Distribution Shortlist", out_md.read_text(encoding="utf-8"))
            self.assertIn("Track1 V4 FID 候选审查队列", out_html.read_text(encoding="utf-8"))

            script = Path(__file__).resolve().parents[1] / "scripts" / "track1_distribution_shortlist.py"
            spec = importlib.util.spec_from_file_location("track1_distribution_shortlist_cli", script)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)

            code = module.main(
                [
                    "--report",
                    f"full={metric}",
                    "--track1-zip",
                    str(Path(__file__).resolve().parents[1] / "data/raw/Track1_testset.zip"),
                    "--top-n",
                    "1",
                    "--out-json",
                    str(root / "cli.json"),
                    "--out-csv",
                    str(root / "cli.csv"),
                    "--out-md",
                    str(root / "cli.md"),
                    "--out-html",
                    str(root / "cli.html"),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((root / "cli.html").exists())


if __name__ == "__main__":
    unittest.main()
