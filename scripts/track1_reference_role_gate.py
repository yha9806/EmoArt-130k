#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_reference_role_gate import (  # noqa: E402
    build_reference_role_gate,
    load_candidate_manifest,
    load_routes_json,
    write_reference_role_gate_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Track1 reference role quality-gate report.")
    parser.add_argument("--routes-json", required=True, type=Path)
    parser.add_argument("--candidate-manifest-json", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_reference_role_gate(
            load_routes_json(args.routes_json),
            candidate_manifest=load_candidate_manifest(args.candidate_manifest_json),
        )
        paths = write_reference_role_gate_artifacts(report, out_dir=args.out_dir, repo_root=ROOT)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(paths, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
