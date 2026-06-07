from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_v14_public_resource_agsr import (
    DEFAULT_OUT_DIR,
    build_emotion_boundary_report,
    build_historical_correlation_report,
    build_threshold_ablation_report,
    build_v14_outputs,
)


DEFAULT_CURRENT_JSON = "submissions/track2_submission_moe_v2_accept5_candidate.json"
DEFAULT_SUBMISSION_DIR = "submissions"
DEFAULT_PUBLIC_AUDIT_JSONS = [
    "experiments/track2_emoart130k_clip/deep_duplicate_audit_20260511/track2_deep_duplicate_top10_audit.json",
    "experiments/track2_emoart130k_clip/overlap_reference/ge095_all/ge095_public_neighbor_audit.json",
    "experiments/track2_public_style_distillation_20260603/public_leak_exclusion_manifest.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Track2 v14 public-resource AGSR candidates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--current-json", type=Path, default=Path(DEFAULT_CURRENT_JSON))
    build.add_argument("--public-audit-json", type=Path, action="append", default=[])
    build.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    build.add_argument("--submission-dir", type=Path, default=Path(DEFAULT_SUBMISSION_DIR))
    build.add_argument("--image-dir", type=Path, default=None)
    build.add_argument("--no-shadow", action="store_true")

    threshold = subparsers.add_parser("threshold-ablation")
    threshold.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    threshold.add_argument("--evidence-matrix-json", type=Path, default=None)

    boundary = subparsers.add_parser("emotion-boundary-report")
    boundary.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    boundary.add_argument("--evidence-matrix-json", type=Path, default=None)

    historical = subparsers.add_parser("historical-correlation")
    historical.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    args = parser.parse_args()
    if args.command == "build":
        public_jsons = args.public_audit_json or [Path(path) for path in DEFAULT_PUBLIC_AUDIT_JSONS]
        report = build_v14_outputs(
            current_json=args.current_json,
            public_audit_jsons=public_jsons,
            out_dir=args.out_dir,
            submission_dir=args.submission_dir,
            image_dir=args.image_dir,
            run_shadow=not args.no_shadow,
        )
        print(json.dumps(report["candidates"], indent=2, ensure_ascii=False))
        return 0
    if args.command == "threshold-ablation":
        report = build_threshold_ablation_report(evidence_matrix_json=args.evidence_matrix_json, out_dir=args.out_dir)
        print(json.dumps(report["rows"], indent=2, ensure_ascii=False))
        return 0
    if args.command == "emotion-boundary-report":
        report = build_emotion_boundary_report(evidence_matrix_json=args.evidence_matrix_json, out_dir=args.out_dir)
        print(f"accepted_count={report['accepted_count']}")
        print(f"blocked_count={report['blocked_count']}")
        print(f"wrote {args.out_dir / 'v14_emotion_boundary_report.md'}")
        return 0
    if args.command == "historical-correlation":
        report = build_historical_correlation_report(out_dir=args.out_dir)
        print(f"official_best={report['official_best']}")
        print(f"shadow_must_not_prefer={report['shadow_must_not_prefer']}")
        print(f"historical_alignment={report['historical_alignment']}")
        print(f"spearman_proxy={report['spearman_proxy']}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
