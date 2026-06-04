#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_moe_prompt_packets import (  # noqa: E402
    build_moe_prompt_packets,
    load_contracts,
    load_distribution_routes,
    load_reference_text_packets,
    load_routes,
    write_moe_prompt_packet_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build clean Track1 MoE provider prompt packets.")
    parser.add_argument("--routes-json", required=True, type=Path)
    parser.add_argument("--contract-json", required=True, type=Path)
    parser.add_argument("--distribution-routes-json", type=Path)
    parser.add_argument("--reference-text-bank-jsonl", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=28)
    parser.add_argument("--max-strategies-per-sample", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = build_moe_prompt_packets(
        load_routes(args.routes_json),
        contracts=load_contracts(args.contract_json),
        reference_text_bank=load_reference_text_packets(args.reference_text_bank_jsonl),
        out_dir=args.out_dir,
        limit=args.limit,
        distribution_routes=(
            load_distribution_routes(args.distribution_routes_json)
            if args.distribution_routes_json
            else None
        ),
        max_strategies_per_sample=args.max_strategies_per_sample,
    )
    write_moe_prompt_packet_reports(rows, out_dir=args.out_dir)
    print(json.dumps({"out_dir": str(args.out_dir), "total": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
