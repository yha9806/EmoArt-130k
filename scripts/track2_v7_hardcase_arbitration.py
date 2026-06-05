#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_v7_hardcase_arbitration import write_v7_outputs


def _format_cli_summary(summary: dict, shadow_report: dict) -> str:
    ranking = shadow_report.get("ranking", [])
    ranking = ranking if isinstance(ranking, list) else []
    top_candidate = ranking[0] if ranking and isinstance(ranking[0], dict) else {}
    return (
        "track2 v7 hardcase arbitration "
        f"decision={summary.get('decision') or 'invalid'} "
        f"top_candidate={top_candidate.get('candidate_name') or 'none'} "
        f"overall_expected={_float_or_zero(top_candidate.get('overall_expected')):.6f} "
        f"overall_lower={_float_or_zero(top_candidate.get('overall_lower')):.6f} "
        f"strict_accepts={int(summary.get('strict_accept_count') or 0)} "
        f"borderline_accepts={int(summary.get('borderline_accept_count') or 0)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Track2 V7 hard-case label arbitration pass."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Write Track2 V7 hard-case arbitration reports and side-path candidates.",
    )
    run_parser.add_argument("--baseline-json", type=Path, required=True)
    run_parser.add_argument("--evidence-matrix", type=Path, required=True)
    run_parser.add_argument("--arbitration-json", type=Path, required=True)
    run_parser.add_argument("--text-override-json", type=Path, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--submission-dir", type=Path, default=Path("submissions"))
    run_parser.add_argument(
        "--shadow-baseline-json",
        type=Path,
        default=None,
        help="Optional baseline for absolute proxy scoring, usually the submitted accept5 anchor.",
    )
    run_parser.add_argument("--expected-row-count", type=int, default=1000)
    run_parser.add_argument(
        "--allow-missing-emotions-for-smoke",
        action="store_true",
        help="Allow smoke fixtures that do not contain every Track2 emotion.",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            summary = write_v7_outputs(
                baseline_json=args.baseline_json,
                evidence_matrix=args.evidence_matrix,
                arbitration_json=args.arbitration_json,
                text_override_json=args.text_override_json,
                out_dir=args.out_dir,
                submission_dir=args.submission_dir,
                shadow_baseline_json=args.shadow_baseline_json,
                expected_row_count=args.expected_row_count,
                require_all_emotions=(
                    False if args.allow_missing_emotions_for_smoke else None
                ),
            )
            shadow_report = _load_shadow_report(summary)
            print(_format_cli_summary(summary, shadow_report))
            return 0
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


def _load_shadow_report(summary: dict) -> dict:
    path = summary.get("shadow_report_json")
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
