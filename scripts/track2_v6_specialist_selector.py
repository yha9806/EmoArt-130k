#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_v6_specialist_selector import write_v6_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Track2 V6 specialist selector."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Write Track2 V6 selector outputs and side-path candidate submissions.",
    )
    run_parser.add_argument("--baseline-json", type=Path, required=True)
    run_parser.add_argument("--evidence-matrix", type=Path, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--submission-dir", type=Path, default=Path("submissions"))
    run_parser.add_argument("--expected-row-count", type=int, default=1000)
    run_parser.add_argument(
        "--allow-missing-emotions-for-smoke",
        action="store_true",
        help="Allow smoke fixtures that do not contain every Track2 emotion.",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            summary = write_v6_outputs(
                baseline_json=args.baseline_json,
                evidence_matrix=args.evidence_matrix,
                out_dir=args.out_dir,
                submission_dir=args.submission_dir,
                expected_row_count=args.expected_row_count,
                require_all_emotions=(
                    False if args.allow_missing_emotions_for_smoke else None
                ),
            )
            shadow_report = json.loads(
                Path(summary["shadow_report_json"]).read_text(encoding="utf-8")
            )
            top_candidate = shadow_report["ranking"][0]
            print(
                "track2 v6 selector "
                f"decision={summary['decision']} "
                f"top_candidate={top_candidate['candidate_name']} "
                f"overall_expected={float(top_candidate['overall_expected']):.6f} "
                f"overall_lower={float(top_candidate['overall_lower']):.6f}"
            )
            return 0
    except (OSError, ValueError, json.JSONDecodeError, KeyError, IndexError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
