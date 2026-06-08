#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.challenge_anchor_sync import (  # noqa: E402
    build_unified_anchor_registry,
    load_csv_rows,
    load_json_payload,
    write_anchor_registry_outputs,
)
from affectiveart.track1_official_score_calibration import load_official_score_rows  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize Track1/Track2 official anchors and public author/method resources."
    )
    parser.add_argument("--track1-api-json", type=Path)
    parser.add_argument("--track1-anchors-csv", type=Path)
    parser.add_argument("--track2-exact-csv", action="append", type=Path, default=[])
    parser.add_argument("--track2-scoreboard-csv", type=Path)
    parser.add_argument("--source-index-csv", type=Path)
    parser.add_argument("--public-resource-scan-json", type=Path)
    parser.add_argument("--method-cards-json", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        track1_api_rows = _load_track1_api_rows(args.track1_api_json)
        track1_anchor_rows = _load_csv_if_present(args.track1_anchors_csv)
        track2_exact_rows = []
        for path in args.track2_exact_csv:
            track2_exact_rows.extend(_load_csv_if_present(path))
        registry = build_unified_anchor_registry(
            track1_api_rows=track1_api_rows,
            track1_anchor_rows=track1_anchor_rows,
            track2_exact_rows=track2_exact_rows,
            track2_scoreboard_rows=_load_csv_if_present(args.track2_scoreboard_csv),
            source_index_rows=_load_csv_if_present(args.source_index_csv),
            public_resource_scan=_load_json_if_present(args.public_resource_scan_json),
            method_cards=_load_json_if_present(args.method_cards_json),
        )
        write_anchor_registry_outputs(
            registry,
            json_path=args.out_json,
            csv_path=args.out_csv,
            md_path=args.out_md,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_csv": str(args.out_csv),
                "out_md": str(args.out_md),
                "anchors": registry["summary"]["anchor_count"],
                "authors": registry["summary"]["author_count"],
                "blocking_issues": registry["consistency"]["blocking_issue_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _load_track1_api_rows(path: Path | None) -> list[dict[str, object]]:
    if not path or not path.exists():
        return []
    return load_official_score_rows(path)


def _load_csv_if_present(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    return load_csv_rows(path)


def _load_json_if_present(path: Path | None) -> dict[str, object]:
    if not path or not path.exists():
        return {}
    payload = load_json_payload(path)
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
