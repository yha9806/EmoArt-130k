#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_reference_asset_bindings import (  # noqa: E402
    attach_reference_assets_to_routes,
    load_reference_asset_index,
    load_route_rows,
    write_reference_asset_route_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attach Track1 reference image assets to distribution routes.")
    parser.add_argument("--routes-json", required=True, type=Path)
    parser.add_argument("--reference-assets-index-json", required=True, type=Path)
    parser.add_argument("--reference-assets-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--max-assets-per-route", type=int, default=4)
    parser.add_argument(
        "--selection-mode",
        choices=["stable", "diversity_balanced"],
        default="stable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        routes = attach_reference_assets_to_routes(
            load_route_rows(args.routes_json),
            load_reference_asset_index(args.reference_assets_index_json),
            asset_root=args.reference_assets_dir,
            max_assets_per_route=args.max_assets_per_route,
            selection_mode=args.selection_mode,
        )
        write_reference_asset_route_reports(
            routes,
            json_path=args.out_json,
            csv_path=args.out_csv,
            md_path=args.out_md,
            repo_root=ROOT,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    unique_assets = {
        Path(asset).name
        for route in routes
        for asset in route.get("reference_assets", [])
    }
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "total": len(routes),
                "selection_mode": args.selection_mode,
                "unique_reference_assets": len(unique_assets),
                "diversity_limited_routes": sum(1 for route in routes if route.get("reference_diversity_limited")),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
