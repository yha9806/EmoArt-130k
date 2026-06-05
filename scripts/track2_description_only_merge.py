#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_description_only_merge import build_description_only_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Track2 candidate that preserves labels and copies text fields.")
    parser.add_argument("--label-source-json", type=Path, required=True)
    parser.add_argument("--text-source-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-zip", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        report = build_description_only_outputs(
            label_source_json=args.label_source_json,
            text_source_json=args.text_source_json,
            out_json=args.out_json,
            out_zip=args.out_zip,
            report_json=args.report_json,
            report_md=args.report_md,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "classification_label_changes": report["classification_label_changes"],
                "text_changed_rows": report["text_changed_rows"],
                "json": report["out_json"],
                "zip": report["out_zip"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
