from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from affectiveart.track1_body_logic_gate import (  # noqa: E402
    build_body_logic_risk_report,
    write_body_logic_risk_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Track1 body/physical-logic risk recall reports from a P1-light report."
    )
    parser.add_argument("--p1-report-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    p1_report = json.loads(Path(args.p1_report_json).read_text(encoding="utf-8"))
    report = build_body_logic_risk_report(p1_report)
    write_body_logic_risk_reports(report, args.out_json, args.out_md)
    print(args.out_md)


if __name__ == "__main__":
    main()
