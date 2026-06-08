#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_v5_fid_breakthrough import (  # noqa: E402
    build_v5_plan,
    load_payload_rows,
    write_v5_reports,
)


DEFAULT_CONTRACTS = Path(
    "experiments/track1_reference_conditioned_pilot_20260603/final_contract_20260603/track1_domain_contract_1000.json"
)
DEFAULT_EXPERT_ROUTES = Path(
    "experiments/track1_reference_conditioned_pilot_20260603/expert_router_20260604/track1_expert_routes_1000.json"
)
DEFAULT_OFFICIAL_ROUTES = Path(
    "experiments/track1_reference_conditioned_pilot_20260603/reference_asset_routes_official_v7_20260605/track1_distribution_routes_with_reference_assets.json"
)
DEFAULT_OUT = Path("experiments/track1_v5_fid_breakthrough_20260608/offline_plan")
DEFAULT_CURRENT_IMAGES = Path("submissions/track1_candidate_v3_gate7_20260606/images")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Track1 v5 FID-breakthrough offline route/prompt plan.")
    parser.add_argument("--contract-json", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--expert-routes-json", type=Path, default=DEFAULT_EXPERT_ROUTES)
    parser.add_argument("--official-routes-json", type=Path, default=DEFAULT_OFFICIAL_ROUTES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--current-image-dir", type=Path, default=DEFAULT_CURRENT_IMAGES)
    parser.add_argument("--html-limit", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contracts = load_payload_rows(_abs(args.contract_json), keys=("rows",))
    expert_routes = load_payload_rows(_abs(args.expert_routes_json), keys=("rows", "routes"))
    official_routes = load_payload_rows(_abs(args.official_routes_json), keys=("routes", "rows"))
    rows = build_v5_plan(contracts, expert_routes, official_routes)
    result = write_v5_reports(
        rows,
        out_dir=_abs(args.out_dir),
        current_image_dir=_abs(args.current_image_dir),
        html_limit=args.html_limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
