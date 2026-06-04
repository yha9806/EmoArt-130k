#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_moe_specialist_ensemble import (
    build_dry_run_report,
    normalize_expert_entries,
    write_dry_run_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Track2 MoE specialist ensemble utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run_parser = subparsers.add_parser(
        "dry-run",
        help="Build Track2 MoE specialist dry-run review artifacts.",
    )
    dry_run_parser.add_argument("--current-json", type=Path, required=True)
    dry_run_parser.add_argument(
        "--expert-source",
        action="append",
        default=[],
        help="name=role=path",
    )
    dry_run_parser.add_argument("--queue-sample-id", action="append", default=[])
    dry_run_parser.add_argument("--queue-csv", type=Path, default=None)
    dry_run_parser.add_argument("--image-zip", type=Path, required=True)
    dry_run_parser.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "dry-run":
            summary = _run_dry_run(args)
            print(json.dumps(summary, sort_keys=True))
            return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


def _run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_dry_run_out_dir(args.out_dir)
    current_rows = _read_json_list(args.current_json)
    expert_rows: list[dict[str, Any]] = []
    for spec in args.expert_source:
        source, role, source_path = _parse_source_spec(spec)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        expert_rows.extend(
            normalize_expert_entries(payload, source=source, role=role)
        )

    queue_ids = _queue_sample_ids(
        explicit_sample_ids=args.queue_sample_id,
        queue_csv=args.queue_csv,
        expert_rows=expert_rows,
    )
    report = build_dry_run_report(
        current_rows,
        expert_rows,
        queue_sample_ids=queue_ids,
    )
    outputs = write_dry_run_outputs(
        report,
        image_zip=args.image_zip,
        out_dir=args.out_dir,
    )
    return {"row_count": report["row_count"], "outputs": outputs}


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return payload


def _parse_source_spec(spec: str) -> tuple[str, str, Path]:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise ValueError(f"expected name=role=path expert source: {spec}")
    source, role, path = (part.strip() for part in parts)
    if not source or not role or not path:
        raise ValueError(f"expected name=role=path expert source: {spec}")
    return source, role, Path(path)


def _queue_sample_ids(
    *,
    explicit_sample_ids: list[str],
    queue_csv: Path | None,
    expert_rows: list[dict[str, Any]],
) -> list[str]:
    if explicit_sample_ids:
        return explicit_sample_ids
    if queue_csv is not None:
        with queue_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
                raise ValueError(f"queue CSV must contain sample_id column: {queue_csv}")
            return [row.get("sample_id", "") for row in reader]
    return sorted(
        {
            str(row.get("sample_id", "")).strip()
            for row in expert_rows
            if str(row.get("sample_id", "")).strip()
        }
    )


def _validate_dry_run_out_dir(out_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    submissions_dir = (repo_root / "submissions").resolve(strict=False)
    resolved_out_dir = out_dir.expanduser().resolve(strict=False)
    try:
        resolved_out_dir.relative_to(submissions_dir)
    except ValueError:
        return
    raise ValueError("dry-run output must not be under submissions")


if __name__ == "__main__":
    raise SystemExit(main())
