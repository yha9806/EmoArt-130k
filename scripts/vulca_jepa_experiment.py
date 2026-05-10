#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.jepa_backbones import BACKBONES
from affectiveart.vulca_jepa_audit import build_disagreement_report, select_vulca_jepa_review_samples


DEFAULT_PREDICTIONS = Path("experiments/track2_ensemble_strict/predictions.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local Vulca JEPA experiment reports.")
    parser.add_argument("--predictions-json", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-json", type=Path, default=Path("experiments/vulca_jepa_audit/report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/vulca_jepa_experiment_report.md"))
    parser.add_argument("--review-limit", type=int, default=40)
    args = parser.parse_args()

    rows = load_prediction_entries(args.predictions_json)
    report = build_disagreement_report(rows)
    selected = select_vulca_jepa_review_samples(rows, limit=args.review_limit)
    report["selected_review_samples"] = selected
    report["backbones"] = [backbone.__dict__ for backbone in BACKBONES]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown_report(report), encoding="utf-8")
    print(args.out_json)
    print(args.out_md)


def load_prediction_entries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("prediction file must be a list or an object with entries")
    return [dict(row) for row in rows if isinstance(row, dict)]


def render_markdown_report(report: dict) -> str:
    lines = [
        "# Vulca JEPA Experiment Report",
        "",
        "## Summary",
        "",
        f"- Row count: {report['row_count']}",
        f"- Model/current disagreements: {report['model_current_disagreements']}",
        f"- Content-to-calm candidates: {report['content_to_calm_candidates']}",
        f"- Model/KNN disagreements: {report['model_knn_disagreements']}",
        "",
        "## Review Samples",
        "",
    ]
    for row in report.get("selected_review_samples", []):
        lines.append(
            f"- `{row.get('sample_id')}` priority={row.get('review_priority')} "
            f"{row.get('current', '')} -> {row.get('emotion', '')}"
        )
    lines.extend(["", "## Backbone Registry", ""])
    for backbone in report.get("backbones", []):
        lines.append(
            f"- `{backbone['name']}`: {backbone['model_id']} "
            f"({backbone['modality']}, {backbone['family']}) - {backbone['local_notes']}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
