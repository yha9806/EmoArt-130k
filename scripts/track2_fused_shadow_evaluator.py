from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_fused_shadow_evaluator import write_fused_shadow_evaluator_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Track2 fused local shadow evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score and rank Track2 candidates with VULCA fusion")
    score_parser.add_argument("--baseline-json", type=Path, required=True)
    score_parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME=PATH candidate JSON mapping. Repeat to compare multiple candidates.",
    )
    score_parser.add_argument("--out-dir", type=Path, required=True)
    score_parser.add_argument("--expected-row-count", type=int, default=1000)
    score_parser.add_argument(
        "--no-require-all-emotions",
        action="store_true",
        help="Disable the full 12-emotion coverage gate, intended for small tests only.",
    )

    args = parser.parse_args()
    if args.command == "score":
        candidates = [_parse_candidate(value) for value in args.candidate]
        report = write_fused_shadow_evaluator_outputs(
            baseline_json=args.baseline_json,
            candidates=candidates,
            out_dir=args.out_dir,
            expected_row_count=args.expected_row_count,
            require_all_emotions=False if args.no_require_all_emotions else None,
        )
        top = report["ranking"][0]
        print(report["warning"])
        print(f"top_candidate={top['candidate_name']}")
        print(f"top_decision={top['decision']}")
        print(f"top_overall_expected={float(top['overall_expected']):.6f}")
        print(f"top_overall_lower={float(top['overall_lower']):.6f}")
        print(f"top_vulca_description_delta={float(top['vulca_description_delta']):.6f}")
        print(f"wrote {args.out_dir / 'fused_shadow_score_report.md'}")
        return 0
    return 1


def _parse_candidate(value: str) -> dict[str, Path | str]:
    if "=" not in value:
        raise SystemExit(f"--candidate must be NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    if not name.strip():
        raise SystemExit("--candidate name must not be empty")
    if not path.strip():
        raise SystemExit("--candidate path must not be empty")
    return {"name": name.strip(), "json": Path(path.strip())}


if __name__ == "__main__":
    raise SystemExit(main())
