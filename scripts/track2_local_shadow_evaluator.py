#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_local_shadow_evaluator import (
    append_calibration_entry,
    load_calibration_summary,
    write_shadow_evaluator_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Track2 local shadow evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score and rank Track2 candidate JSON files")
    score_parser.add_argument("--baseline-json", type=Path, required=True)
    score_parser.add_argument("--candidate", action="append", required=True, help="NAME=PATH candidate JSON mapping")
    score_parser.add_argument("--out-dir", type=Path, required=True)
    score_parser.add_argument("--expected-row-count", type=int, default=1000)

    ledger_parser = subparsers.add_parser("ledger-add", help="Append one official feedback calibration entry")
    ledger_parser.add_argument("--ledger", type=Path, required=True)
    ledger_parser.add_argument("--submission-id", required=True)
    ledger_parser.add_argument("--file-name", required=True)
    ledger_parser.add_argument("--shadow-overall-expected", type=float, required=True)
    ledger_parser.add_argument("--shadow-classification-expected", type=float, required=True)
    ledger_parser.add_argument("--shadow-description-expected", type=float, required=True)
    ledger_parser.add_argument("--official-overall", type=float, required=True)
    ledger_parser.add_argument("--official-classification", type=float, required=True)
    ledger_parser.add_argument("--official-description", type=float, required=True)
    ledger_parser.add_argument("--notes", default="")

    args = parser.parse_args()
    if args.command == "score":
        candidates = [_parse_candidate(value) for value in args.candidate]
        report = write_shadow_evaluator_outputs(
            baseline_json=args.baseline_json,
            candidates=candidates,
            out_dir=args.out_dir,
            expected_row_count=args.expected_row_count,
        )
        top = report["ranking"][0]
        print(report["warning"])
        print(f"top_candidate={top['candidate_name']}")
        print(f"top_decision={top['decision']}")
        print(f"top_overall_lower={float(top['overall_lower']):.6f}")
        print(f"wrote {args.out_dir / 'shadow_score_report.md'}")
        return

    if args.command == "ledger-add":
        append_calibration_entry(
            ledger_path=args.ledger,
            entry={
                "submission_id": args.submission_id,
                "file_name": args.file_name,
                "shadow_overall_expected": args.shadow_overall_expected,
                "shadow_classification_expected": args.shadow_classification_expected,
                "shadow_description_expected": args.shadow_description_expected,
                "official_overall": args.official_overall,
                "official_classification": args.official_classification,
                "official_description": args.official_description,
                "notes": args.notes,
            },
        )
        summary = load_calibration_summary(args.ledger)
        print(f"entry_count={summary['entry_count']}")
        print(f"overall_mae={summary['overall_mae']:.6f}")
        return


def _parse_candidate(value: str) -> dict[str, Path | str]:
    if "=" not in value:
        raise SystemExit("--candidate must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise SystemExit("--candidate must include non-empty NAME and PATH")
    return {"name": name.strip(), "json": Path(path.strip())}


if __name__ == "__main__":
    main()
