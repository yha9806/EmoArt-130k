#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_distribution_router import (  # noqa: E402
    build_distribution_routes,
    load_contract_rows,
    load_reference_family_bank,
    write_distribution_route_reports,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Track1 distribution router reports.")
    parser.add_argument("--contracts-json", type=Path, required=True)
    parser.add_argument("--reference-family-bank-json", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        contracts = load_contract_rows(args.contracts_json)
        reference_family_bank = load_reference_family_bank(args.reference_family_bank_json)
        routes = build_distribution_routes(contracts, reference_family_bank=reference_family_bank)
        write_distribution_route_reports(
            routes,
            json_path=args.out_json,
            csv_path=args.out_csv,
            md_path=args.out_md,
            repo_root=ROOT,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    summary = {
        "total": len(routes),
        "families": sorted({route["family_id"] for route in routes}),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
