#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_reference_role_gate import load_routes_json  # noqa: E402
from affectiveart.track1_reference_role_repair import (  # noqa: E402
    repair_reference_routes,
    write_reference_role_repair_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair Track1 reference routes by required reference roles.")
    parser.add_argument("--routes-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--role", action="append", default=None)
    parser.add_argument("--max-assets-per-route", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = repair_reference_routes(
            load_routes_json(args.routes_json),
            repair_roles=args.role or ["poster_print"],
            max_assets_per_route=args.max_assets_per_route,
        )
        paths = write_reference_role_repair_artifacts(result, out_dir=args.out_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(paths, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
