#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from affectiveart.track2_v26_public_knn_probe import (
    DEFAULT_BASE_JSON,
    DEFAULT_NEAREST_JSON,
    DEFAULT_OUT_DIR,
    DEFAULT_THRESHOLDS,
    build_public_knn_probe_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track2 public nearest-neighbor label-copy probes.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    parser.add_argument("--nearest-json", type=Path, default=DEFAULT_NEAREST_JSON)
    parser.add_argument("--thresholds", nargs="*", type=float, default=list(DEFAULT_THRESHOLDS))
    args = parser.parse_args()
    report = build_public_knn_probe_outputs(
        out_dir=args.out_dir,
        base_json=args.base_json,
        nearest_json=args.nearest_json,
        thresholds=tuple(args.thresholds),
    )
    best = report.get("best") or {}
    print(
        "decision={decision} best_threshold={threshold} overall={overall:.6f}".format(
            decision=report.get("decision", ""),
            threshold=best.get("threshold", ""),
            overall=float(best.get("overall_expected", 0.0) or 0.0),
        )
    )
    print(args.out_dir / "v26_public_knn_probe_zh.md")


if __name__ == "__main__":
    main()
