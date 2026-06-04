#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_reference_family_bank import (  # noqa: E402
    build_reference_family_bank,
    load_reference_rows,
    write_reference_family_reports,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Track1 reference family bank reports.")
    parser.add_argument("--references-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = load_reference_rows(args.references_json)
    bank = build_reference_family_bank(rows)
    write_reference_family_reports(bank, json_path=args.out_json, csv_path=args.out_csv, md_path=args.out_md)
    print(json.dumps(bank["summary"], indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
