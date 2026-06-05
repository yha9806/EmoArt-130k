#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_v9_transition_ablation import build_v9_transition_ablation_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Track2 v9 transition-family ablation candidates.")
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--v9-json", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path, default=None)
    parser.add_argument("--human-gate-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, default=Path("submissions"))
    args = parser.parse_args(argv)

    try:
        summary = build_v9_transition_ablation_outputs(
            baseline_json=args.baseline_json,
            v9_json=args.v9_json,
            evidence_json=args.evidence_json,
            human_gate_csv=args.human_gate_csv,
            out_dir=args.out_dir,
            submission_dir=args.submission_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    printable = {
        "change_count": summary["change_count"],
        "transition_counts": summary["transition_counts"],
        "candidates": {
            kind: {
                "changed_rows": payload["changed_rows"],
                "zip": payload["zip"],
            }
            for kind, payload in summary["candidates"].items()
        },
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
