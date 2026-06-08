#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from affectiveart.track2_v25_signal_audit import (
    DEFAULT_BASE_JSON,
    DEFAULT_CANDIDATE_GLOB,
    DEFAULT_MODEL_PREDICTIONS,
    DEFAULT_OUT_DIR,
    build_v25_signal_audit_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Track2 v25 candidate and model signal frontier.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    parser.add_argument("--candidate-glob", default=DEFAULT_CANDIDATE_GLOB)
    parser.add_argument("--target-overall", type=float, default=0.89)
    args = parser.parse_args()
    report = build_v25_signal_audit_outputs(
        out_dir=args.out_dir,
        base_json=args.base_json,
        model_prediction_paths=DEFAULT_MODEL_PREDICTIONS,
        candidate_glob=args.candidate_glob,
        target_overall=args.target_overall,
    )
    decision = report["decision"]
    best = report["candidate_pool"].get("best") or {}
    print(
        "decision={decision} best={best} overall={overall:.6f} above_target={above_target}".format(
            decision=decision["decision"],
            best=best.get("candidate_name", ""),
            overall=float(best.get("overall_expected", 0.0) or 0.0),
            above_target=report["candidate_pool"].get("above_target", 0),
        )
    )
    print(args.out_dir / "v25_signal_audit_zh.md")


if __name__ == "__main__":
    main()
