#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_compiler_gate import summarize_decisions


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a Track1 challenger A/B decision report.")
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=Path("experiments/track1_130k_compiler_gate_v1/challengers/run_summary.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("experiments/track1_130k_compiler_gate_v1/decision_report.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("experiments/track1_130k_compiler_gate_v1/decision_report.json"),
    )
    args = parser.parse_args()

    runs = json.loads(args.run_summary.read_text(encoding="utf-8"))
    decisions = []
    for row in runs:
        sample_id = row["sample_id"]
        if row.get("returncode", 0) != 0:
            decisions.append({"sample_id": sample_id, "decision": "reject", "reason": "generation failed"})
        else:
            decisions.append({"sample_id": sample_id, "decision": "hold", "reason": "manual A/B review required"})

    summary = summarize_decisions(decisions)
    payload = {"summary": summary, "decisions": decisions}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(args.out_json)
    print(args.out_md)
    return 0


def render_md(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    decisions = payload["decisions"]
    lines = [
        "# Track1 130k Compiler Gate Challenger Decision Report",
        "",
        "The final Track1 submission is not modified by this report.",
        "",
        "## Summary",
        "",
        f"- total: {summary['total']}",
        f"- accepted: {summary['accepted']}",
        f"- rejected: {summary['rejected']}",
        f"- held_for_manual_review: {summary['held']}",
        "",
        "## Decisions",
        "",
        "| sample_id | decision | reason |",
        "|---|---|---|",
    ]
    for row in decisions:
        lines.append(f"| {row['sample_id']} | {row['decision']} | {row['reason']} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
