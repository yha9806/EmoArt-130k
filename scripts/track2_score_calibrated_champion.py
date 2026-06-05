#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_score_calibrated_champion import (
    load_candidate_source,
    load_optional_csv,
    load_optional_json,
    write_score_calibrated_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Track2 score-calibrated champion candidate ladder."
    )
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="name=family=path candidate JSON source",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/track2_score_calibrated_champion_20260605"),
    )
    parser.add_argument("--submission-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--public-style-report", type=Path, default=None)
    parser.add_argument("--human-gate-csv", type=Path, default=None)
    parser.add_argument("--rollback-report", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        sources = [load_candidate_source(spec) for spec in args.candidate]
        summary = write_score_calibrated_outputs(
            baseline_json=args.baseline_json,
            sources=sources,
            out_dir=args.out_dir,
            submission_dir=args.submission_dir,
            public_style_report=load_optional_json(args.public_style_report),
            human_gate_rows=load_optional_csv(args.human_gate_csv),
            rollback_report=load_optional_json(args.rollback_report),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    printable = {
        "out_dir": summary["out_dir"],
        "evidence_row_count": summary["evidence_row_count"],
        "candidates": {
            tier: {
                "changed_rows": payload["changed_rows"],
                "json": payload["json"],
                "zip": payload["zip"],
            }
            for tier, payload in summary["candidates"].items()
        },
        "html_review": summary["html_review"],
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
