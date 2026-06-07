#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_official_score_calibration import (  # noqa: E402
    calibrate_track1_packages,
    load_local_fid_rows,
    load_official_score_rows,
    write_calibration_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Track1 local shadow score from Codabench component-score anchors."
    )
    parser.add_argument("--official-anchors-csv", type=Path, required=True)
    parser.add_argument("--local-fid-json", type=Path, required=True)
    parser.add_argument("--anchor-package", default="v3_gate7")
    parser.add_argument("--aas-assumption", type=float, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = calibrate_track1_packages(
            load_official_score_rows(args.official_anchors_csv),
            load_local_fid_rows(args.local_fid_json),
            anchor_package=args.anchor_package,
            aas_assumption=args.aas_assumption,
        )
        write_calibration_reports(report, args.out_json, args.out_md)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "confidence": report["calibration_confidence"],
                "top_package": report["packages"][0]["package"] if report["packages"] else "",
                "top_expected_overall": report["packages"][0]["overall_expected"] if report["packages"] else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
