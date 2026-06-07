#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.challenge import _read_track1_samples  # noqa: E402
from affectiveart.track1_distribution_shortlist import (  # noqa: E402
    build_distribution_shortlist,
    load_metric_report,
    parse_report_spec,
    write_shortlist_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Track1 distribution/FID proxy review shortlist.")
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Metric proxy report as SOURCE_LABEL=PATH.",
    )
    parser.add_argument("--track1-zip", type=Path, default=Path("data/raw/Track1_testset.zip"))
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=0.03)
    parser.add_argument("--high-delta", type=float, default=0.05)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--out-html", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = {}
    for spec in args.report:
        label, path = parse_report_spec(spec)
        reports[label] = load_metric_report(path if path.is_absolute() else ROOT / path)
    captions = _load_captions(args.track1_zip if args.track1_zip.is_absolute() else ROOT / args.track1_zip)
    report = build_distribution_shortlist(
        reports,
        captions=captions,
        top_n=args.top_n,
        min_delta=args.min_delta,
        high_delta=args.high_delta,
    )
    write_shortlist_reports(
        report,
        json_path=args.out_json if args.out_json.is_absolute() else ROOT / args.out_json,
        csv_path=args.out_csv if args.out_csv.is_absolute() else ROOT / args.out_csv,
        md_path=args.out_md if args.out_md.is_absolute() else ROOT / args.out_md,
        html_path=args.out_html if args.out_html.is_absolute() else ROOT / args.out_html,
    )
    print(args.out_json)
    print(args.out_csv)
    print(args.out_md)
    print(args.out_html)
    return 0


def _load_captions(track1_zip: Path) -> dict[str, str]:
    _, rows = _read_track1_samples(track1_zip)
    return {str(row["sample_id"]): str(row.get("caption") or "") for row in rows}


if __name__ == "__main__":
    raise SystemExit(main())
