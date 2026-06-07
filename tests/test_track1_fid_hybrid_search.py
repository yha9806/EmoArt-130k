import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from affectiveart.track1_fid_hybrid_search import (
    greedy_hybrid_fid_search,
    load_feature_cache,
    load_shortlist_sample_ids,
    render_hybrid_search_markdown,
    write_hybrid_search_reports,
)


class Track1FidHybridSearchTest(unittest.TestCase):
    def test_greedy_search_selects_only_package_level_improvement(self):
        current_ids = ["track1_0001", "track1_0002", "track1_0003"]
        current = np.array([[5.0, 0.0], [5.0, 0.5], [4.5, -0.5]], dtype=np.float64)
        candidate = np.array([[0.1, 0.0], [8.0, 8.0], [4.5, -0.5]], dtype=np.float64)
        reference = np.array([[0.0, 0.0], [0.2, 0.1], [-0.1, 0.0]], dtype=np.float64)

        report = greedy_hybrid_fid_search(
            current_ids=current_ids,
            current_features=current,
            candidate_ids=current_ids,
            candidate_features=candidate,
            reference_features=reference,
            candidate_pool=["track1_0001", "track1_0002"],
            max_replacements=2,
        )

        self.assertEqual([row["sample_id"] for row in report["selected_replacements"]], ["track1_0001"])
        self.assertGreater(report["summary"]["total_improvement"], 0.0)
        self.assertEqual(report["stop_conditions"][0]["reason"], "no_remaining_candidate_improves_package_fid_like")

    def test_greedy_search_selects_none_when_no_candidate_improves(self):
        ids = ["track1_0001", "track1_0002", "track1_0003"]
        current = np.array([[0.0, 0.0], [0.2, 0.1], [-0.1, 0.0]], dtype=np.float64)
        candidate = np.array([[6.0, 6.0], [7.0, 7.0], [8.0, 8.0]], dtype=np.float64)
        reference = current.copy()

        report = greedy_hybrid_fid_search(
            current_ids=ids,
            current_features=current,
            candidate_ids=ids,
            candidate_features=candidate,
            reference_features=reference,
            candidate_pool=ids,
            max_replacements=3,
        )

        self.assertEqual(report["selected_replacements"], [])
        self.assertEqual(report["summary"]["selected_replacement_count"], 0)
        self.assertLessEqual(report["summary"]["total_improvement"], 0.0)

    def test_feature_cache_shortlist_reports_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = np.array(["track1_0001", "track1_0002", "track1_0003"])
            current = np.array([[5.0, 0.0], [5.0, 0.5], [4.5, -0.5]], dtype=np.float32)
            candidate = np.array([[0.1, 0.0], [8.0, 8.0], [4.5, -0.5]], dtype=np.float32)
            reference = np.array([[0.0, 0.0], [0.2, 0.1], [-0.1, 0.0]], dtype=np.float32)
            np.savez(root / "current.npz", ids=ids, features=current)
            np.savez(root / "candidate.npz", ids=ids, features=candidate)
            np.savez(root / "reference.npz", ids=np.array(["r1", "r2", "r3"]), features=reference)
            shortlist = root / "shortlist.json"
            shortlist.write_text(
                json.dumps({"review_queue": [{"sample_id": "track1_0001"}, {"sample_id": "track1_0002"}]}),
                encoding="utf-8",
            )

            loaded = load_feature_cache(root / "current.npz")
            self.assertEqual(loaded["ids"][0], "track1_0001")
            self.assertEqual(load_shortlist_sample_ids(shortlist, max_candidates=1), ["track1_0001"])

            report = greedy_hybrid_fid_search(
                current_ids=ids.tolist(),
                current_features=current,
                candidate_ids=ids.tolist(),
                candidate_features=candidate,
                reference_features=reference,
                candidate_pool=["track1_0001", "track1_0002"],
            )
            out_json = root / "search.json"
            out_md = root / "search.md"
            manifest = root / "manifest.json"
            write_hybrid_search_reports(
                report,
                json_path=out_json,
                md_path=out_md,
                replacement_manifest_path=manifest,
                candidate_image_dir=root / "images",
            )

            self.assertIn("Track1 Greedy Inception Hybrid FID Search", render_hybrid_search_markdown(report))
            self.assertTrue(out_json.exists())
            self.assertTrue(manifest.exists())

            script = Path(__file__).resolve().parents[1] / "scripts" / "track1_fid_hybrid_search.py"
            spec = importlib.util.spec_from_file_location("track1_fid_hybrid_search_cli", script)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            code = module.main(
                [
                    "--current-features",
                    str(root / "current.npz"),
                    "--candidate-features",
                    str(root / "candidate.npz"),
                    "--reference-features",
                    str(root / "reference.npz"),
                    "--shortlist-json",
                    str(shortlist),
                    "--candidate-label",
                    "candidate",
                    "--candidate-image-dir",
                    str(root / "images"),
                    "--out-json",
                    str(root / "cli.json"),
                    "--out-md",
                    str(root / "cli.md"),
                    "--replacement-manifest",
                    str(root / "cli_manifest.json"),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((root / "cli_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
