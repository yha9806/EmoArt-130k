#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_official_score_calibration import (  # noqa: E402
    build_scorer_reproducibility_audit,
    load_official_score_rows,
    merge_local_anchor_metadata,
    write_scorer_reproducibility_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit whether the Track1 local scorer can reproduce Codabench score components."
    )
    parser.add_argument("--official-scores-json", type=Path, required=True)
    parser.add_argument("--local-anchors-csv", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        official_rows = load_official_score_rows(args.official_scores_json)
        if args.local_anchors_csv:
            anchor_rows = [
                row
                for row in load_official_score_rows(args.local_anchors_csv)
                if _has_reproducible_anchor_data(row)
            ]
            _fill_component_overall_for_anchor_only_rows(anchor_rows, official_rows)
            official_rows = merge_local_anchor_metadata(official_rows, anchor_rows)
        report = build_scorer_reproducibility_audit(official_rows)
        write_scorer_reproducibility_audit(report, args.out_json, args.out_md)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "formula_max_abs_error": report["official_formula_reproduction"]["max_abs_error"],
                "fid_score_rmse": report["fid_score_reproduction"]["rmse"],
                "local_proxy_readiness": report["local_proxy_reproduction"]["readiness"],
                "own_anchor_count": report["local_proxy_reproduction"]["own_anchor_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _has_reproducible_anchor_data(row: dict[str, object]) -> bool:
    if str(row.get("local_fid_like") or "").strip():
        return True
    return _float_or_none(row.get("official_fid_score")) is not None and _float_or_none(row.get("official_aas")) is not None


def _fill_component_overall_for_anchor_only_rows(anchor_rows: list[dict[str, object]], official_rows: list[dict[str, object]]) -> None:
    official_ids = {str(row.get("submission_id") or row.get("id") or "") for row in official_rows}
    for row in anchor_rows:
        submission_id = str(row.get("submission_id") or row.get("id") or "")
        if submission_id in official_ids:
            continue
        fid_score = _float_or_none(row.get("official_fid_score"))
        aas = _float_or_none(row.get("official_aas"))
        if fid_score is None or aas is None:
            continue
        row["official_overall"] = f"{0.5 * (fid_score + aas):.10f}"
        row["notes"] = (
            f"{str(row.get('notes') or '').strip()} "
            "Component-derived exact overall for scorer reproducibility audit."
        ).strip()


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
