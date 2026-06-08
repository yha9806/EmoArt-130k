#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_v5_smoke_queue import (  # noqa: E402
    DEFAULT_SMOKE_ANCHORS,
    load_v5_plan,
    select_smoke_rows,
    write_smoke_queue,
)


DEFAULT_PLAN = Path(
    "experiments/track1_v5_fid_breakthrough_20260608/offline_plan/track1_v5_fid_breakthrough_plan.json"
)
DEFAULT_OUT = Path("experiments/track1_v5_smoke_20260608/queue")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a stratified Track1 v5 smoke generation queue.")
    parser.add_argument("--v5-plan-json", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-count", type=int, default=48)
    parser.add_argument("--anchor", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_v5_plan(_abs(args.v5_plan_json))
    anchors = args.anchor if args.anchor else DEFAULT_SMOKE_ANCHORS
    selected = select_smoke_rows(rows, target_count=args.target_count, anchors=anchors)
    outputs = write_smoke_queue(selected, out_dir=_abs(args.out_dir))
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
