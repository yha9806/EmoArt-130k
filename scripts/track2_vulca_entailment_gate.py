#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_vulca_entailment_gate import write_vulca_entailment_gate_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Track2 VULCA/Judge++ entailment-gated text candidate."
    )
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-zip", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--shadow-out-dir", type=Path, default=None)
    parser.add_argument("--expected-row-count", type=int, default=1000)
    args = parser.parse_args()

    report = write_vulca_entailment_gate_outputs(
        source_json=args.source_json,
        out_json=args.out_json,
        out_zip=args.out_zip,
        report_json=args.report_json,
        report_md=args.report_md,
        baseline_json=args.baseline_json,
        shadow_out_dir=args.shadow_out_dir,
        expected_row_count=args.expected_row_count,
    )
    before = report["risk_summary_before"]
    after = report["risk_summary_after"]
    print(f"changed_rows={report['changed_rows']}")
    print(f"changed_fields={report['changed_fields']}")
    print(f"classification_label_changes={report['classification_label_changes']}")
    print(f"risk_issues_before={before['total_issues']}")
    print(f"risk_issues_after={after['total_issues']}")
    if report.get("shadow_top"):
        top = report["shadow_top"]
        print(f"shadow_decision={top['decision']}")
        print(f"shadow_overall_expected={float(top['overall_expected']):.6f}")
        print(f"shadow_overall_lower={float(top['overall_lower']):.6f}")
    print(f"wrote_report={args.report_md}")


if __name__ == "__main__":
    main()
