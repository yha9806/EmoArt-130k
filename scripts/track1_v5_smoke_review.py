from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affectiveart.track1_v5_smoke_review import build_review_rows, write_review_reports  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Chinese Track1 v5 smoke review page from generation manifests.")
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--current-image-dir", type=Path, default=Path("submissions/track1/images"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/track1_v5_smoke_20260608/review"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = build_review_rows(args.manifest, current_image_dir=args.current_image_dir)
    outputs = write_review_reports(rows, out_dir=args.out_dir)
    print(outputs["html"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
