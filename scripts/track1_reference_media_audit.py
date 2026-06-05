#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_reference_family_bank import _safe_output_paths  # noqa: E402
from affectiveart.track1_reference_media_audit import (  # noqa: E402
    build_reference_media_audit,
    load_routes_json,
    write_reference_media_audit_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Track1 reference media audit visualizations.")
    parser.add_argument("--old-routes-json", required=True, type=Path)
    parser.add_argument("--new-routes-json", required=True, type=Path)
    parser.add_argument("--partial-candidate-image-dir", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        (out_dir,) = _safe_output_paths([args.out_dir], repo_root=ROOT)
        report = build_reference_media_audit(
            load_routes_json(args.old_routes_json),
            load_routes_json(args.new_routes_json),
            partial_candidate_image_dir=args.partial_candidate_image_dir,
        )
        paths = write_reference_media_audit_artifacts(report, out_dir=out_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(paths, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
