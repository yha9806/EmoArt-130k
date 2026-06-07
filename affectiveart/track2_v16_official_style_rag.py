from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OfficialSubmissionScore:
    submission_id: str
    overall: float
    classification: float
    description: float


def official_delta_summary(
    *,
    anchor: OfficialSubmissionScore,
    candidate: OfficialSubmissionScore,
    changed_rows: int,
) -> dict[str, Any]:
    changed = max(1, int(changed_rows))
    classification_delta = float(candidate.classification) - float(anchor.classification)
    return {
        "anchor_submission_id": anchor.submission_id,
        "candidate_submission_id": candidate.submission_id,
        "classification_delta": classification_delta,
        "classification_delta_per_changed_row": classification_delta / changed,
        "overall_delta": float(candidate.overall) - float(anchor.overall),
        "description_delta": float(candidate.description) - float(anchor.description),
        "changed_rows": int(changed_rows),
        "batch_verdict": "negative_official_evidence" if classification_delta < 0 else "nonnegative_official_evidence",
    }


def classify_transition_risk(
    *,
    transition: str,
    evidence_sources: set[str],
    official_failed_transition_count: int = 0,
) -> str:
    del transition
    if official_failed_transition_count >= 10 and evidence_sources <= {"same_quadrant_batch"}:
        return "blocked_by_failed_official_batch"
    if "exact_public_duplicate" in evidence_sources:
        return "allow_exact_duplicate"
    if {"rag_teacher", "multibackbone_consensus"} <= evidence_sources:
        return "allow_strong_consensus"
    return "hold_needs_more_evidence"


def load_official_scores(path: str | Path) -> dict[str, OfficialSubmissionScore]:
    scores: dict[str, OfficialSubmissionScore] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            submission_id = str(row.get("submission_id", "")).strip()
            if not submission_id:
                continue
            scores[submission_id] = OfficialSubmissionScore(
                submission_id=submission_id,
                overall=_safe_float(row.get("official_overall")),
                classification=_safe_float(row.get("official_classification")),
                description=_safe_float(row.get("official_description")),
            )
    return scores


def load_pairwise_diff_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "candidate_a": str(row.get("candidate_a", "")).strip(),
                    "candidate_b": str(row.get("candidate_b", "")).strip(),
                    "candidate_a_submission_id": _extract_submission_id(row.get("candidate_a")),
                    "candidate_b_submission_id": _extract_submission_id(row.get("candidate_b")),
                    "emotion_label_changes": int(_safe_float(row.get("emotion_label_changes"))),
                    "top_emotion_transitions": str(row.get("top_emotion_transitions", "")).strip(),
                    "transition_counts": parse_transition_counts(row.get("top_emotion_transitions", "")),
                }
            )
    return rows


def parse_transition_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in str(value or "").split(";"):
        item = part.strip()
        if not item or ":" not in item:
            continue
        transition, count_text = item.rsplit(":", 1)
        transition = transition.strip()
        if "->" not in transition:
            continue
        counts[transition] = int(_safe_float(count_text))
    return counts


def build_official_counterfactual_report(
    *,
    official_scores: dict[str, OfficialSubmissionScore],
    pairwise_diff_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not official_scores:
        raise ValueError("official scores are empty")
    anchor = max(official_scores.values(), key=lambda item: (item.classification, item.overall, item.submission_id))
    failed_batches: list[dict[str, Any]] = []
    failed_transition_counts: Counter[str] = Counter()

    for row in pairwise_diff_rows:
        candidate_id = str(row.get("candidate_a_submission_id", ""))
        anchor_id = str(row.get("candidate_b_submission_id", ""))
        if candidate_id not in official_scores or anchor_id not in official_scores:
            continue
        summary = official_delta_summary(
            anchor=official_scores[anchor_id],
            candidate=official_scores[candidate_id],
            changed_rows=int(row.get("emotion_label_changes") or 0),
        )
        summary["candidate_a"] = row.get("candidate_a", "")
        summary["candidate_b"] = row.get("candidate_b", "")
        summary["transition_counts"] = row.get("transition_counts", {})
        if summary["batch_verdict"] == "negative_official_evidence":
            failed_batches.append(summary)
            for transition, count in dict(row.get("transition_counts", {})).items():
                failed_transition_counts[str(transition)] += int(count)

    return {
        "method": "track2_v16_official_counterfactual_calibration_v1",
        "anchor_submission_id": anchor.submission_id,
        "anchor_classification": anchor.classification,
        "official_score_count": len(official_scores),
        "pairwise_diff_count": len(pairwise_diff_rows),
        "known_failed_batches": failed_batches,
        "failed_transition_counts": dict(sorted(failed_transition_counts.items())),
        "policy": (
            "Official aggregate results show that the 781601 same-quadrant batch reduced classification. "
            "v16 therefore blocks failed high-count boundary transitions unless stronger per-sample evidence is present."
        ),
    }


def write_official_counterfactual_outputs(
    *,
    official_scores_csv: str | Path,
    pairwise_diffs_csv: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_official_counterfactual_report(
        official_scores=load_official_scores(official_scores_csv),
        pairwise_diff_rows=load_pairwise_diff_rows(pairwise_diffs_csv),
    )
    (out_dir / "official_counterfactual_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "official_counterfactual_report.md").write_text(
        render_official_counterfactual_markdown(report),
        encoding="utf-8",
    )
    return report


def render_official_counterfactual_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v16 Official Counterfactual Calibration",
        "",
        f"- Method: `{report['method']}`",
        f"- Anchor submission: `{report['anchor_submission_id']}`",
        f"- Anchor classification: {float(report['anchor_classification']):.6f}",
        f"- Official score rows: {report['official_score_count']}",
        f"- Pairwise diff rows: {report['pairwise_diff_count']}",
        "",
        "## Policy",
        "",
        str(report["policy"]),
        "",
        "## Known Failed Batches",
        "",
    ]
    failed = list(report.get("known_failed_batches", []))
    if not failed:
        lines.append("- none")
    for batch in failed:
        lines.append(
            "- "
            f"{batch['candidate_submission_id']} vs {batch['anchor_submission_id']}: "
            f"class delta {float(batch['classification_delta']):.6f}, "
            f"changed rows {int(batch['changed_rows'])}, "
            f"per-row {float(batch['classification_delta_per_changed_row']):.8f}"
        )
    lines.extend(["", "## Failed Transition Counts", ""])
    counts = dict(report.get("failed_transition_counts", {}))
    if not counts:
        lines.append("- none")
    for transition, count in counts.items():
        lines.append(f"- {transition}: {count}")
    return "\n".join(lines) + "\n"


def _extract_submission_id(value: Any) -> str:
    match = re.match(r"^(\d{6,})", str(value or "").strip())
    return match.group(1) if match else ""


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Track2 v16 official-style RAG classification tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser("calibrate", help="Build official counterfactual calibration report")
    calibrate.add_argument("--official-scores", type=Path, required=True)
    calibrate.add_argument("--pairwise-diffs", type=Path, required=True)
    calibrate.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        report = write_official_counterfactual_outputs(
            official_scores_csv=args.official_scores,
            pairwise_diffs_csv=args.pairwise_diffs,
            out_dir=args.out_dir,
        )
        print(
            json.dumps(
                {
                    "anchor_submission_id": report["anchor_submission_id"],
                    "failed_batch_count": len(report["known_failed_batches"]),
                    "failed_transition_count": len(report["failed_transition_counts"]),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
