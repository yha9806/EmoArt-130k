#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from affectiveart.track2_v27_gold_like_ledger import (
    DEFAULT_BASE_JSON,
    DEFAULT_EVIDENCE_JSON,
    DEFAULT_OUT_DIR,
    DEFAULT_SUBMISSIONS_DIR,
    build_v27_candidate_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track2 v27 gold-like ledger candidates.")
    parser.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    parser.add_argument("--evidence-json", type=Path, default=DEFAULT_EVIDENCE_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--submissions-dir", type=Path, default=DEFAULT_SUBMISSIONS_DIR)
    args = parser.parse_args()
    report = build_v27_candidate_suite(
        base_json=args.base_json,
        evidence_json=args.evidence_json,
        out_dir=args.out_dir,
        submissions_dir=args.submissions_dir,
    )
    best = report.get("best") or {}
    print(
        "decision={decision} best_profile={profile} overall={overall:.6f} class={classification:.6f} desc={description:.6f}".format(
            decision=report.get("decision", ""),
            profile=report.get("best_profile", ""),
            overall=float(best.get("overall_expected", 0.0) or 0.0),
            classification=float(best.get("classification_expected", 0.0) or 0.0),
            description=float(best.get("description_expected", 0.0) or 0.0),
        )
    )
    print(args.out_dir / "v27_candidate_report_zh.md")


if __name__ == "__main__":
    main()
