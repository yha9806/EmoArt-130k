from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_DIR = Path("experiments/track2_v19_aggregate_calibration_20260607")
DEFAULT_LEADERBOARD_CSV = Path("experiments/track2_official_results_20260606/track2_public_leaderboard_20260606.csv")

AGGREGATE_FIELDS = {
    "overall": "Overall Score",
    "classification": "Classification Score",
    "description": "Description Score",
    "emotion_accuracy": "Emotion Accuracy",
    "emotion_macro_f1": "Emotion Macro F1",
    "valence_accuracy": "Valence Accuracy",
    "valence_macro_f1": "Valence Macro F1",
    "arousal_accuracy": "Arousal Accuracy",
    "arousal_macro_f1": "Arousal Macro F1",
}


def load_leaderboard_aggregates(path: str | Path) -> list[dict[str, Any]]:
    """Load visible Codabench leaderboard aggregates without inferring hidden labels."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {"participant": str(raw.get("Participant", "")).strip()}
            for key, column in AGGREGATE_FIELDS.items():
                row[key] = _safe_float(raw.get(column))
            rows.append(row)
    return rows


def derive_aggregate_targets(rows: list[dict[str, Any]], participant: str = "vulcaart") -> dict[str, Any]:
    """Compare our visible aggregate row with the public frontier row."""
    if not rows:
        raise ValueError("leaderboard rows are empty")
    our_row = next((row for row in rows if str(row.get("participant")) == participant), None)
    if our_row is None:
        raise ValueError(f"participant not found in leaderboard rows: {participant}")
    frontier = max(rows, key=lambda row: float(row.get("overall", 0.0)))
    gaps = {
        key: round(float(frontier.get(key, 0.0)) - float(our_row.get(key, 0.0)), 6)
        for key in AGGREGATE_FIELDS
    }
    target_bands = {
        "emotion_accuracy_min_delta": 0.05,
        "emotion_macro_f1_min_delta": -0.005,
        "classification_lower_floor": 0.723150,
        "description_lower_max_drop": 0.003,
    }
    return {
        "method": "track2_v19_aggregate_targets_v1",
        "warning": "Visible leaderboard aggregates are not hidden labels and are only used for local risk control.",
        "participant": participant,
        "our": dict(our_row),
        "frontier": dict(frontier),
        "gaps": gaps,
        "target_bands": target_bands,
    }


def write_aggregate_target_outputs(*, leaderboard_csv: str | Path, out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = derive_aggregate_targets(load_leaderboard_aggregates(leaderboard_csv))
    (out_dir / "aggregate_targets.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "aggregate_targets.md").write_text(_render_aggregate_targets_md(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Track2 v19 aggregate calibration artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    targets = subparsers.add_parser("targets", help="Write aggregate target reports only.")
    targets.add_argument("--leaderboard-csv", default=str(DEFAULT_LEADERBOARD_CSV))
    targets.add_argument("--out-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    args = parser.parse_args(argv)
    if args.command == "targets":
        report = write_aggregate_target_outputs(leaderboard_csv=args.leaderboard_csv, out_dir=args.out_dir)
        print(json.dumps({"aggregate_targets": str(Path(args.out_dir) / "aggregate_targets.json"), "frontier": report["frontier"]["participant"]}))


def _safe_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _render_aggregate_targets_md(report: dict[str, Any]) -> str:
    our = report["our"]
    frontier = report["frontier"]
    gaps = report["gaps"]
    return "\n".join(
        [
            "# Track2 v19 Aggregate Targets",
            "",
            f"- Participant: `{report['participant']}`",
            f"- Frontier: `{frontier['participant']}`",
            f"- Our overall/class/desc: `{our['overall']:.6f}` / `{our['classification']:.6f}` / `{our['description']:.6f}`",
            f"- Frontier overall/class/desc: `{frontier['overall']:.6f}` / `{frontier['classification']:.6f}` / `{frontier['description']:.6f}`",
            f"- Emotion accuracy gap: `{gaps['emotion_accuracy']:.6f}`",
            f"- Emotion macro-F1 gap: `{gaps['emotion_macro_f1']:.6f}`",
            "",
            "These aggregates are not hidden labels. They only define local risk-control targets.",
            "",
        ]
    )
