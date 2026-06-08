#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from affectiveart.track1_v5_smoke_queue import write_generated_review


DEFAULT_QUEUE_JSON = Path(
    "experiments/track1_v5_smoke_20260608/queue/track1_v5_smoke_queue.json"
)
DEFAULT_MANIFEST_JSON = Path(
    "experiments/track1_v5_smoke_20260608/generated/candidate_manifest.json"
)
DEFAULT_CURRENT_IMAGES_DIR = Path("submissions/track1/images")
DEFAULT_OUT_DIR = Path("experiments/track1_v5_smoke_20260608/review")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Track1 v5 smoke generated review HTML.")
    parser.add_argument("--queue-json", type=Path, default=DEFAULT_QUEUE_JSON)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--current-images-dir", type=Path, default=DEFAULT_CURRENT_IMAGES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = write_generated_review(
        queue_json=args.queue_json,
        manifest_json=args.manifest_json,
        current_images_dir=args.current_images_dir,
        out_dir=args.out_dir,
    )
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
