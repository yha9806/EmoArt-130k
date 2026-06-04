#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_prompt_lint import (  # noqa: E402
    lint_prompt_batch,
    load_prompt_rows,
    write_prompt_lint_reports,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lint Track1 provider prompts for batch-level template collapse.")
    parser.add_argument("--packets-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--allow-fail", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        rows = load_prompt_rows(args.packets_json)
        report = lint_prompt_batch(rows, threshold=args.threshold)
        write_prompt_lint_reports(
            report,
            json_path=args.out_json,
            md_path=args.out_md,
            repo_root=ROOT,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False, sort_keys=True))
    if report["summary"]["status"] == "fail" and not args.allow_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
