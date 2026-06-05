#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_official_reference_bank import write_official_reference_bank  # noqa: E402
from affectiveart.track1_reference_asset_bindings import load_route_rows  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Track1 official EmoArt-130k route reference bank.")
    parser.add_argument("--routes-json", required=True, type=Path)
    parser.add_argument("--emoart-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-index-json", required=True, type=Path)
    parser.add_argument("--out-summary-json", required=True, type=Path)
    parser.add_argument("--max-candidates-per-route", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = write_official_reference_bank(
            load_route_rows(args.routes_json),
            emoart_root=args.emoart_root,
            out_dir=args.out_dir,
            index_json=args.out_index_json,
            summary_json=args.out_summary_json,
            max_candidates_per_route=args.max_candidates_per_route,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
