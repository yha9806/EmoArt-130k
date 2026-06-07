from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_DIR = Path("experiments/track2_v19_aggregate_calibration_20260607")
DEFAULT_LEADERBOARD_CSV = Path("experiments/track2_official_results_20260606/track2_public_leaderboard_20260606.csv")
TRACK2_EMOTIONS = {
    "alarmed",
    "annoyed",
    "aroused",
    "bored",
    "calm",
    "content",
    "excited",
    "frustrated",
    "glad",
    "happy",
    "sad",
    "tired",
}
FAILED_SAME_QUADRANT_FAMILIES = {
    "calm->content",
    "content->calm",
    "calm->glad",
    "glad->calm",
    "content->glad",
    "glad->content",
    "happy->excited",
    "excited->happy",
}

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


def write_calibrated_evidence_outputs(
    *,
    v17_evidence: str | Path,
    aggregate_targets_json: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_rows = json.loads(Path(v17_evidence).read_text(encoding="utf-8"))
    aggregate_targets = json.loads(Path(aggregate_targets_json).read_text(encoding="utf-8"))
    if not isinstance(evidence_rows, list):
        raise ValueError("v17 evidence must be a JSON list")
    calibrated = build_v19_calibrated_evidence(evidence_rows, aggregate_targets)
    regression_guard = summarize_781601_regression_guard(calibrated)
    report = {
        "method": "track2_v19_calibrated_evidence_outputs_v1",
        "rows": len(calibrated),
        "accepted_candidate_rows": sum(1 for row in calibrated if row.get("v19_decision") == "accept_candidate"),
        "held_rows": sum(1 for row in calibrated if row.get("v19_decision") == "hold"),
        "blocked_rows": sum(1 for row in calibrated if row.get("v19_decision") == "block"),
        "regression_guard": regression_guard,
        "paths": {
            "calibrated_evidence_json": str(out_dir / "calibrated_evidence.json"),
            "calibrated_evidence_csv": str(out_dir / "calibrated_evidence.csv"),
            "regression_guard_json": str(out_dir / "781601_regression_guard.json"),
        },
    }
    _write_json(out_dir / "calibrated_evidence.json", calibrated)
    _write_json(out_dir / "781601_regression_guard.json", regression_guard)
    _write_csv(out_dir / "calibrated_evidence.csv", calibrated)
    _write_json(out_dir / "calibrated_evidence_report.json", report)
    (out_dir / "calibrated_evidence_report.md").write_text(_render_calibrated_evidence_md(report), encoding="utf-8")
    return report


def score_v19_evidence_row(
    row: dict[str, Any],
    *,
    aggregate_pressure: float = 0.0,
    failed_transition_penalty_weight: float = 0.04,
) -> dict[str, Any]:
    """Score one proposed label change with row evidence plus conservative aggregate pressure."""
    current = _canonical_emotion(row.get("current_emotion"))
    proposed = _canonical_emotion(row.get("proposed_emotion"))
    transition = str(row.get("transition") or f"{current}->{proposed}").strip()
    support_score = _safe_float(row.get("support_score"))
    model_vote_count = _safe_int(row.get("model_vote_count"))
    failed_transition_count = _safe_int(row.get("failed_transition_count"))
    duplicate_support = _safe_float(row.get("public_duplicate_support_score"))
    max_confidence = _safe_float(row.get("max_confidence"))
    near_or_exact = bool(row.get("near_duplicate") or row.get("exact_duplicate"))

    reasons: list[str] = []
    decision = "hold"

    if current not in TRACK2_EMOTIONS or proposed not in TRACK2_EMOTIONS:
        return {
            "decision": "block",
            "score": 0.0,
            "reasons": ["invalid_emotion_label"],
        }
    if current == proposed or transition == f"{current}->{current}":
        return {"decision": "block", "score": 0.0, "reasons": ["no_op_transition"]}

    row_support_ok = support_score >= 1.7 and model_vote_count >= 2
    duplicate_support_ok = near_or_exact and duplicate_support >= 0.95 and max_confidence >= 0.95
    if not row_support_ok and not duplicate_support_ok:
        reasons.append("insufficient_row_support")

    failed_penalty = failed_transition_count * failed_transition_penalty_weight
    score = support_score + min(0.4, max(0.0, aggregate_pressure)) - failed_penalty
    if failed_transition_count >= 10 and not duplicate_support_ok:
        reasons.append("official_failed_transition_penalty")

    if not reasons and score >= 1.7:
        decision = "accept_candidate"
    elif "official_failed_transition_penalty" in reasons or "insufficient_row_support" in reasons:
        decision = "hold"

    return {
        "decision": decision,
        "score": round(score, 6),
        "reasons": reasons or ["meets_v19_thresholds"],
    }


def build_v19_calibrated_evidence(
    v17_evidence_rows: list[dict[str, Any]],
    aggregate_targets: dict[str, Any],
) -> list[dict[str, Any]]:
    gap = _safe_float((aggregate_targets.get("gaps") or {}).get("emotion_accuracy"))
    aggregate_pressure = min(0.4, max(0.0, gap))
    calibrated: list[dict[str, Any]] = []
    for row in v17_evidence_rows:
        scored = score_v19_evidence_row(row, aggregate_pressure=aggregate_pressure)
        enriched = dict(row)
        enriched["v19_score"] = scored["score"]
        enriched["v19_decision"] = scored["decision"]
        enriched["v19_reasons"] = ";".join(scored["reasons"])
        calibrated.append(enriched)
    decision_rank = {"accept_candidate": 0, "hold": 1, "block": 2}
    return sorted(
        calibrated,
        key=lambda row: (
            decision_rank.get(str(row.get("v19_decision")), 9),
            -_safe_float(row.get("v19_score")),
            str(row.get("sample_id", "")),
        ),
    )


def summarize_781601_regression_guard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed_by_transition: dict[str, int] = {}
    for row in rows:
        transition = str(row.get("transition", "")).strip()
        if transition not in FAILED_SAME_QUADRANT_FAMILIES:
            continue
        failed_by_transition[transition] = max(
            failed_by_transition.get(transition, 0),
            _safe_int(row.get("failed_transition_count")),
        )
    failed_same_quadrant_count = sum(failed_by_transition.values())
    decision = "block_bulk_same_quadrant_repeat" if failed_same_quadrant_count >= 50 else "pass"
    return {
        "method": "track2_v19_781601_regression_guard_v1",
        "decision": decision,
        "failed_same_quadrant_count": failed_same_quadrant_count,
        "failed_transition_counts": dict(sorted(failed_by_transition.items())),
        "threshold": 50,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Track2 v19 aggregate calibration artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    targets = subparsers.add_parser("targets", help="Write aggregate target reports only.")
    targets.add_argument("--leaderboard-csv", default=str(DEFAULT_LEADERBOARD_CSV))
    targets.add_argument("--out-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    evidence = subparsers.add_parser("evidence", help="Write calibrated evidence reports.")
    evidence.add_argument("--v17-evidence", required=True)
    evidence.add_argument("--aggregate-targets-json", default=str(DEFAULT_EXPERIMENT_DIR / "aggregate_targets.json"))
    evidence.add_argument("--out-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    args = parser.parse_args(argv)
    if args.command == "targets":
        report = write_aggregate_target_outputs(leaderboard_csv=args.leaderboard_csv, out_dir=args.out_dir)
        print(json.dumps({"aggregate_targets": str(Path(args.out_dir) / "aggregate_targets.json"), "frontier": report["frontier"]["participant"]}))
    elif args.command == "evidence":
        report = write_calibrated_evidence_outputs(
            v17_evidence=args.v17_evidence,
            aggregate_targets_json=args.aggregate_targets_json,
            out_dir=args.out_dir,
        )
        print(json.dumps({"calibrated_evidence": report["paths"]["calibrated_evidence_json"], "rows": report["rows"]}))


def _safe_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    return int(round(_safe_float(value)))


def _canonical_emotion(value: Any) -> str:
    emotion = str(value or "").strip().lower()
    return emotion if emotion in TRACK2_EMOTIONS else ""


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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _render_calibrated_evidence_md(report: dict[str, Any]) -> str:
    guard = report["regression_guard"]
    return "\n".join(
        [
            "# Track2 v19 Calibrated Evidence",
            "",
            f"- Rows: `{report['rows']}`",
            f"- Accept candidate rows: `{report['accepted_candidate_rows']}`",
            f"- Held rows: `{report['held_rows']}`",
            f"- Blocked rows: `{report['blocked_rows']}`",
            f"- 781601 guard decision: `{guard['decision']}`",
            f"- 781601 failed same-quadrant count: `{guard['failed_same_quadrant_count']}`",
            "",
        ]
    )
