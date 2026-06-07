from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_DIR = Path("experiments/track2_v20_championship_scorer_20260607")
DEFAULT_BASE_JSON = Path("submissions/track2_submission_v15_desc_expand300_candidate.json")
DEFAULT_LEADERBOARD_CSV = Path("experiments/track2_official_results_20260606/track2_public_leaderboard_20260606.csv")
DEFAULT_PAIRWISE_DIFFS = Path(
    "experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv"
)
DEFAULT_V17_EVIDENCE = Path("experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json")
FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}
TRACK2_SUBMISSION_KEYS = (
    "sample_id",
    "emotion",
    "emotional_valence",
    "emotional_arousal_level",
    "overall_caption",
    "brushstroke",
    "composition",
    "color",
    "line",
    "light",
)
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
NEGATIVE_EMOTIONS = {"alarmed", "annoyed", "bored", "frustrated", "sad", "tired"}
HIGH_AROUSAL_EMOTIONS = {"alarmed", "annoyed", "aroused", "excited", "frustrated", "happy"}
PROFILE_CONFIG = {
    "precision": {"total_cap": 18, "caution_cap": 18, "min_score": 3.35},
    "champion_probe": {"total_cap": 56, "caution_cap": 48, "min_score": 3.05},
    "last_shot": {"total_cap": 96, "caution_cap": 80, "min_score": 2.85},
}


def required_classification_for_overall(target_overall: float, description_score: float) -> float:
    return round(2.0 * float(target_overall) - float(description_score), 6)


def load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_track2_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Track2 JSON must be a list: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Track2 row {index} must be an object")
        row = {key: item.get(key, "") for key in TRACK2_SUBMISSION_KEYS}
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"blank sample_id at row {index}")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        rows.append(row)
    return rows


def build_frontier_requirement_report(
    rows: list[dict[str, Any]],
    *,
    participant: str = "vulcaart",
    target_overall: float = 0.89,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("leaderboard rows are empty")
    frontier = max(rows, key=lambda row: _safe_float(row.get("Overall Score")))
    current = next((row for row in rows if str(row.get("Participant", "")).strip() == participant), None)
    if current is None:
        raise ValueError(f"participant not found: {participant}")
    frontier_class = _safe_float(frontier.get("Classification Score"))
    current_class = _safe_float(current.get("Classification Score"))
    frontier_desc = _safe_float(frontier.get("Description Score"))
    current_desc = _safe_float(current.get("Description Score"))
    frontier_emotion_acc = _safe_float(frontier.get("Emotion Accuracy"))
    current_emotion_acc = _safe_float(current.get("Emotion Accuracy"))
    frontier_macro = _safe_float(frontier.get("Emotion Macro F1"))
    current_macro = _safe_float(current.get("Emotion Macro F1"))
    return {
        "method": "track2_v20_frontier_requirement_v1",
        "participant": participant,
        "frontier_participant": str(frontier.get("Participant", "")).strip(),
        "target_overall": float(target_overall),
        "current_overall": _safe_float(current.get("Overall Score")),
        "frontier_overall": _safe_float(frontier.get("Overall Score")),
        "current_classification": current_class,
        "frontier_classification": frontier_class,
        "current_description": current_desc,
        "frontier_description": frontier_desc,
        "classification_gap": round(frontier_class - current_class, 6),
        "description_gap": round(frontier_desc - current_desc, 6),
        "emotion_accuracy_gap": round(frontier_emotion_acc - current_emotion_acc, 6),
        "macro_f1_gap": round(frontier_macro - current_macro, 6),
        "required_classification_if_description_1.00": required_classification_for_overall(target_overall, 1.0),
        "required_classification_if_description_0.98": required_classification_for_overall(target_overall, 0.98),
        "warning": "This is an aggregate frontier requirement, not hidden-label reconstruction.",
    }


def build_directional_risk_map(pairwise_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in pairwise_rows:
        if not _is_failed_batch_direction(row):
            continue
        counts.update(_parse_transition_counts(row.get("top_emotion_transitions", "")))
    risk: dict[str, dict[str, Any]] = {}
    for transition, count in sorted(counts.items()):
        reverse = _reverse_transition(transition)
        reverse_count = counts.get(reverse, 0)
        level = "caution" if count >= 5 else "observe"
        if count >= 20 and count >= max(1, 2 * reverse_count):
            level = "danger"
        elif count >= 10:
            level = "caution"
        risk[transition] = {
            "transition": transition,
            "reverse_transition": reverse,
            "count": int(count),
            "reverse_count": int(reverse_count),
            "risk": level,
            "interpretation": "directional_failed_batch_inference",
        }
        if reverse_count and reverse not in risk and level == "danger":
            risk[reverse] = {
                "transition": reverse,
                "reverse_transition": transition,
                "count": int(reverse_count),
                "reverse_count": int(count),
                "risk": "caution",
                "interpretation": "reverse_of_dominant_failed_direction",
            }
    return dict(sorted(risk.items()))


def score_v20_evidence_row(
    row: dict[str, Any],
    *,
    directional_risk: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    current = _canonical_emotion(row.get("current_emotion"))
    proposed = _canonical_emotion(row.get("proposed_emotion"))
    transition = str(row.get("transition") or f"{current}->{proposed}").strip()
    support = _safe_float(row.get("support_score"))
    votes = _safe_int(row.get("model_vote_count"))
    duplicate_support = _safe_float(row.get("public_duplicate_support_score"))
    confidence = _safe_float(row.get("max_confidence"))
    exact_duplicate = _safe_bool(row.get("exact_duplicate"))
    near_duplicate = _safe_bool(row.get("near_duplicate"))
    risk_payload = directional_risk.get(transition, {"risk": "observe"})
    risk = str(risk_payload.get("risk", "observe"))
    reasons: list[str] = []
    role = "observe"

    if current not in TRACK2_EMOTIONS or proposed not in TRACK2_EMOTIONS:
        return _score_result("block", 0.0, "invalid", ["invalid_emotion_label"], risk)
    if current == proposed:
        return _score_result("block", 0.0, "invalid", ["no_op_transition"], risk)

    exact_override = exact_duplicate and duplicate_support >= 0.98 and confidence >= 0.98
    same_quadrant = _valence(current) == _valence(proposed) and _arousal(current) == _arousal(proposed)
    if not same_quadrant and not exact_override:
        return _score_result(
            "block",
            _raw_score(support, votes, duplicate_support, confidence, risk),
            "cross_quadrant",
            ["cross_quadrant_requires_exact_duplicate"],
            risk,
        )
    if risk == "danger" and not exact_override:
        return _score_result(
            "block",
            _raw_score(support, votes, duplicate_support, confidence, risk),
            "danger_block",
            ["danger_direction_requires_exact_duplicate"],
            risk,
        )

    high_evidence = votes >= 3 and support >= 3.2 and (
        (near_duplicate or exact_duplicate) and duplicate_support >= 0.95 or confidence >= 0.94
    )
    consensus_evidence = votes >= 3 and support >= 2.95 and _has_public_style_consensus(row)
    medium_evidence = votes >= 2 and support >= 2.2 and confidence >= 0.86

    if risk == "caution":
        if high_evidence or exact_override:
            role = "championship_caution_accept"
            return _score_result(
                "accept_candidate",
                _raw_score(support, votes, duplicate_support, confidence, risk),
                role,
                ["strong_row_evidence_overrides_caution_direction"],
                risk,
            )
        if consensus_evidence:
            role = "championship_consensus_accept"
            return _score_result(
                "accept_candidate",
                _raw_score(support, votes, duplicate_support, confidence, risk),
                role,
                ["model_and_public_style_consensus_overrides_caution_direction"],
                risk,
            )
        reasons.append("caution_direction_needs_high_evidence")
    elif high_evidence or medium_evidence or exact_override:
        role = "championship_supported_accept" if high_evidence or exact_override else "supported_accept"
        return _score_result(
            "accept_candidate",
            _raw_score(support, votes, duplicate_support, confidence, risk),
            role,
            ["meets_v20_thresholds"],
            risk,
        )
    else:
        reasons.append("insufficient_row_support")

    return _score_result(
        "hold",
        _raw_score(support, votes, duplicate_support, confidence, risk),
        "hold",
        reasons,
        risk,
    )


def build_v20_championship_evidence(
    v17_rows: list[dict[str, Any]],
    directional_risk: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in v17_rows:
        scored = score_v20_evidence_row(row, directional_risk=directional_risk)
        enriched = dict(row)
        enriched["v20_decision"] = scored["decision"]
        enriched["v20_score"] = scored["score"]
        enriched["v20_role"] = scored["role"]
        enriched["v20_risk"] = scored["risk"]
        enriched["v20_reasons"] = ";".join(scored["reasons"])
        output.append(enriched)
    rank = {"accept_candidate": 0, "hold": 1, "block": 2}
    return sorted(
        output,
        key=lambda row: (
            rank.get(str(row.get("v20_decision")), 9),
            -_safe_float(row.get("v20_score")),
            str(row.get("sample_id", "")),
            str(row.get("proposed_emotion", "")),
        ),
    )


def select_v20_changes(
    evidence_rows: list[dict[str, Any]],
    *,
    profile: str,
    base_distribution: Counter[str] | dict[str, int],
) -> list[dict[str, Any]]:
    config = _profile_config(profile)
    selected: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    caution_count = 0
    projected = Counter({str(key): int(value) for key, value in dict(base_distribution).items()})
    for row in sorted(evidence_rows, key=lambda item: (-_safe_float(item.get("v20_score")), str(item.get("sample_id", "")))):
        if len(selected) >= int(config["total_cap"]):
            break
        if row.get("v20_decision") != "accept_candidate":
            continue
        if _safe_float(row.get("v20_score")) < float(config["min_score"]):
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen_samples:
            continue
        current = _canonical_emotion(row.get("current_emotion"))
        proposed = _canonical_emotion(row.get("proposed_emotion"))
        if current not in TRACK2_EMOTIONS or proposed not in TRACK2_EMOTIONS or current == proposed:
            continue
        if projected[current] <= 3:
            continue
        risk = str(row.get("v20_risk", "observe"))
        if risk == "caution":
            if caution_count >= int(config["caution_cap"]):
                continue
            caution_count += 1
        selected_row = dict(row)
        selected_row["current_emotion"] = current
        selected_row["proposed_emotion"] = proposed
        selected_row["transition"] = str(row.get("transition") or f"{current}->{proposed}")
        selected.append(selected_row)
        seen_samples.add(sample_id)
        projected[current] -= 1
        projected[proposed] += 1
    return selected


def write_v20_candidate_outputs(
    *,
    base_rows: list[dict[str, Any]],
    selected_changes: list[dict[str, Any]],
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
    profile: str,
) -> dict[str, Any]:
    out_json = Path(out_json)
    out_zip = Path(out_zip)
    report_json = Path(report_json)
    report_md = Path(report_md)
    for path in (out_json, out_zip, report_json, report_md):
        _reject_formal_submission_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    candidate_rows, apply_report = _apply_changes(base_rows, selected_changes)
    report = {
        "method": "track2_v20_championship_candidate_v1",
        "profile": profile,
        "row_count": len(candidate_rows),
        "accepted_label_changes": apply_report["accepted_label_changes"],
        "transition_counts": apply_report["transition_counts"],
        "distribution": apply_report["distribution"],
        "missing_emotions": apply_report["missing_emotions"],
        "top_emotion": apply_report["top_emotion"],
        "top_emotion_share": apply_report["top_emotion_share"],
        "label_consistency_issue_count": apply_report["label_consistency_issue_count"],
        "accepted_changes": apply_report["accepted_changes"],
        "description_anchor_preserved": True,
        "formal_submission_overwritten": False,
        "paths": {
            "out_json": str(out_json),
            "out_zip": str(out_zip),
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
    }
    _write_json(out_json, candidate_rows)
    _write_json(report_json, report)
    report_md.write_text(_render_candidate_md(report), encoding="utf-8")
    _write_zip_payload(out_zip, candidate_rows)
    return report


def write_v20_candidate_ladder_outputs(
    *,
    base_json: str | Path,
    championship_evidence_json: str | Path,
    out_dir: str | Path,
    submissions_dir: str | Path = "submissions",
) -> dict[str, Any]:
    base_rows = load_track2_rows(base_json)
    evidence_rows = json.loads(Path(championship_evidence_json).read_text(encoding="utf-8"))
    if not isinstance(evidence_rows, list):
        raise ValueError("championship evidence must be a JSON list")
    report_dir = Path(out_dir) / "candidate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    base_distribution = Counter(str(row.get("emotion", "")) for row in base_rows)
    candidates: dict[str, Any] = {}
    for profile in ("precision", "champion_probe", "last_shot"):
        selected = select_v20_changes(evidence_rows, profile=profile, base_distribution=base_distribution)
        stem = f"track2_submission_v20_{profile}_candidate"
        candidate_report = write_v20_candidate_outputs(
            base_rows=base_rows,
            selected_changes=selected,
            out_json=Path(submissions_dir) / f"{stem}.json",
            out_zip=Path(submissions_dir) / f"{stem}.zip",
            report_json=report_dir / f"{stem}_report.json",
            report_md=report_dir / f"{stem}_report.md",
            profile=profile,
        )
        candidates[profile] = {
            key: candidate_report[key]
            for key in (
                "accepted_label_changes",
                "transition_counts",
                "distribution",
                "missing_emotions",
                "top_emotion",
                "top_emotion_share",
                "label_consistency_issue_count",
                "description_anchor_preserved",
                "paths",
            )
        }
    summary = {
        "method": "track2_v20_candidate_ladder_v1",
        "base_json": str(base_json),
        "championship_evidence_json": str(championship_evidence_json),
        "candidates": candidates,
        "formal_submission_overwritten": False,
    }
    _write_json(Path(out_dir) / "candidate_ladder_summary.json", summary)
    (Path(out_dir) / "candidate_ladder_summary.md").write_text(_render_ladder_md(summary), encoding="utf-8")
    return summary


def choose_v20_final_gate(
    *,
    candidate_name: str,
    accepted_label_changes: int,
    description_anchor_preserved: bool,
    label_consistency_issue_count: int,
    missing_emotions: list[str],
    frontier_classification_gap: float,
    profile: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if label_consistency_issue_count:
        reasons.append("label_consistency_issues")
    if missing_emotions:
        reasons.append("missing_emotions")
    if not description_anchor_preserved:
        reasons.append("description_anchor_not_preserved")
    min_changes = 12 if profile == "precision" else 28
    if accepted_label_changes < min_changes and frontier_classification_gap >= 0.04:
        reasons.append("insufficient_championship_classification_lift")
    if profile == "last_shot" and accepted_label_changes < 48:
        reasons.append("last_shot_not_large_enough")
    if reasons:
        decision = "hold"
    elif profile == "precision":
        decision = "recommend_precision_probe"
    else:
        decision = "recommend_high_variance_probe"
    return {
        "method": "track2_v20_final_gate_v1",
        "candidate_name": candidate_name,
        "profile": profile,
        "decision": decision,
        "reasons": reasons,
        "accepted_label_changes": int(accepted_label_changes),
        "description_anchor_preserved": bool(description_anchor_preserved),
        "label_consistency_issue_count": int(label_consistency_issue_count),
        "missing_emotions": list(missing_emotions),
        "frontier_classification_gap": float(frontier_classification_gap),
        "no_auto_submit": True,
        "caveat": "Local high-variance gate only; this is not the official hidden scorer.",
    }


def write_v20_final_gate_report(
    *,
    ladder_summary: dict[str, Any],
    frontier_report: dict[str, Any],
    out_json: str | Path,
    out_md: str | Path,
) -> dict[str, Any]:
    candidates = ladder_summary.get("candidates") or {}
    gate_reports: dict[str, Any] = {}
    gap = _safe_float(frontier_report.get("classification_gap"))
    for profile, row in sorted(candidates.items()):
        gate_reports[profile] = choose_v20_final_gate(
            candidate_name=f"v20_{profile}",
            accepted_label_changes=_safe_int(row.get("accepted_label_changes")),
            description_anchor_preserved=_safe_bool(row.get("description_anchor_preserved")),
            label_consistency_issue_count=_safe_int(row.get("label_consistency_issue_count")),
            missing_emotions=list(row.get("missing_emotions") or []),
            frontier_classification_gap=gap,
            profile=profile,
        )
    priority = {"recommend_high_variance_probe": 0, "recommend_precision_probe": 1, "hold": 2}
    best_profile, best_gate = min(
        gate_reports.items(),
        key=lambda item: (priority.get(str(item[1].get("decision")), 9), -_safe_int(item[1].get("accepted_label_changes")), item[0]),
    )
    report = {
        "method": "track2_v20_final_gate_report_v1",
        "decision": best_gate["decision"],
        "selected_profile": best_profile,
        "selected_candidate": candidates.get(best_profile, {}),
        "gate_reports": gate_reports,
        "frontier_report": frontier_report,
        "no_auto_submit": True,
    }
    _reject_formal_submission_path(Path(out_json))
    _reject_formal_submission_path(Path(out_md))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    _write_json(Path(out_json), report)
    Path(out_md).write_text(_render_final_gate_md(report), encoding="utf-8")
    return report


def write_v20_run_outputs(
    *,
    base_json: str | Path = DEFAULT_BASE_JSON,
    leaderboard_csv: str | Path = DEFAULT_LEADERBOARD_CSV,
    pairwise_diffs: str | Path = DEFAULT_PAIRWISE_DIFFS,
    v17_evidence: str | Path = DEFAULT_V17_EVIDENCE,
    out_dir: str | Path = DEFAULT_EXPERIMENT_DIR,
    submissions_dir: str | Path = "submissions",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_rows = load_csv_rows(leaderboard_csv)
    pairwise_rows = load_csv_rows(pairwise_diffs)
    v17_rows = json.loads(Path(v17_evidence).read_text(encoding="utf-8"))
    if not isinstance(v17_rows, list):
        raise ValueError("v17 evidence must be a JSON list")
    frontier = build_frontier_requirement_report(leaderboard_rows)
    risk = build_directional_risk_map(pairwise_rows)
    evidence = build_v20_championship_evidence(v17_rows, risk)
    _write_json(out_dir / "frontier_requirements.json", frontier)
    (out_dir / "frontier_requirements.md").write_text(_render_frontier_md(frontier), encoding="utf-8")
    _write_json(out_dir / "directional_risk_map.json", risk)
    (out_dir / "directional_risk_map.md").write_text(_render_risk_md(risk), encoding="utf-8")
    _write_json(out_dir / "championship_evidence.json", evidence)
    _write_csv(out_dir / "championship_evidence.csv", evidence)
    ladder = write_v20_candidate_ladder_outputs(
        base_json=base_json,
        championship_evidence_json=out_dir / "championship_evidence.json",
        out_dir=out_dir,
        submissions_dir=submissions_dir,
    )
    final_gate = write_v20_final_gate_report(
        ladder_summary=ladder,
        frontier_report=frontier,
        out_json=out_dir / "final_gate_report.json",
        out_md=out_dir / "final_gate_report.md",
    )
    return {
        "method": "track2_v20_run_outputs_v1",
        "frontier_requirements": str(out_dir / "frontier_requirements.json"),
        "directional_risk_map": str(out_dir / "directional_risk_map.json"),
        "championship_evidence": str(out_dir / "championship_evidence.json"),
        "candidate_ladder_summary": str(out_dir / "candidate_ladder_summary.json"),
        "final_gate_report": str(out_dir / "final_gate_report.json"),
        "decision": final_gate["decision"],
        "selected_profile": final_gate["selected_profile"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Track2 v20 championship scorer artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Write full v20 artifacts and side-path candidates.")
    run.add_argument("--base-json", default=str(DEFAULT_BASE_JSON))
    run.add_argument("--leaderboard-csv", default=str(DEFAULT_LEADERBOARD_CSV))
    run.add_argument("--pairwise-diffs", default=str(DEFAULT_PAIRWISE_DIFFS))
    run.add_argument("--v17-evidence", default=str(DEFAULT_V17_EVIDENCE))
    run.add_argument("--out-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    run.add_argument("--submissions-dir", default="submissions")
    args = parser.parse_args(argv)
    if args.command == "run":
        report = write_v20_run_outputs(
            base_json=args.base_json,
            leaderboard_csv=args.leaderboard_csv,
            pairwise_diffs=args.pairwise_diffs,
            v17_evidence=args.v17_evidence,
            out_dir=args.out_dir,
            submissions_dir=args.submissions_dir,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _apply_changes(
    base_rows: list[dict[str, Any]],
    selected_changes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    changes_by_id = {
        str(row.get("sample_id", "")).strip(): row
        for row in selected_changes
        if str(row.get("sample_id", "")).strip()
    }
    candidate_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for base in base_rows:
        row = dict(base)
        sample_id = str(row.get("sample_id", "")).strip()
        change = changes_by_id.get(sample_id)
        if change:
            current = _canonical_emotion(row.get("emotion"))
            expected = _canonical_emotion(change.get("current_emotion"))
            proposed = _canonical_emotion(change.get("proposed_emotion"))
            if proposed and proposed != current and (not expected or expected == current):
                before = _label_triplet(row)
                row["emotion"] = proposed
                row["emotional_valence"] = _valence(proposed)
                row["emotional_arousal_level"] = _arousal(proposed)
                after = _label_triplet(row)
                accepted.append(
                    {
                        "sample_id": sample_id,
                        "transition": str(change.get("transition") or f"{before[0]}->{after[0]}"),
                        "before": {"emotion": before[0], "valence": before[1], "arousal": before[2]},
                        "after": {"emotion": after[0], "valence": after[1], "arousal": after[2]},
                        "v20_score": _safe_float(change.get("v20_score")),
                        "v20_role": str(change.get("v20_role", "")),
                    }
                )
        candidate_rows.append({key: row.get(key, "") for key in TRACK2_SUBMISSION_KEYS})
    distribution = Counter(str(row.get("emotion", "")) for row in candidate_rows)
    label_issues = _label_issues(candidate_rows)
    return candidate_rows, {
        "accepted_label_changes": len(accepted),
        "transition_counts": dict(sorted(Counter(item["transition"] for item in accepted).items())),
        "distribution": dict(sorted(distribution.items())),
        "missing_emotions": sorted(TRACK2_EMOTIONS - set(distribution)),
        "top_emotion": distribution.most_common(1)[0][0] if distribution else "",
        "top_emotion_share": round(distribution.most_common(1)[0][1] / max(1, len(candidate_rows)), 6)
        if distribution
        else 0.0,
        "label_consistency_issue_count": len(label_issues),
        "label_consistency_issues": label_issues[:80],
        "accepted_changes": accepted,
    }


def _label_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        emotion = _canonical_emotion(row.get("emotion"))
        if emotion not in TRACK2_EMOTIONS:
            issues.append({"sample_id": sample_id, "field": "emotion", "actual": row.get("emotion", "")})
            continue
        expected_valence = _valence(emotion)
        expected_arousal = _arousal(emotion)
        if row.get("emotional_valence") != expected_valence:
            issues.append(
                {
                    "sample_id": sample_id,
                    "field": "emotional_valence",
                    "actual": row.get("emotional_valence", ""),
                    "expected": expected_valence,
                }
            )
        if row.get("emotional_arousal_level") != expected_arousal:
            issues.append(
                {
                    "sample_id": sample_id,
                    "field": "emotional_arousal_level",
                    "actual": row.get("emotional_arousal_level", ""),
                    "expected": expected_arousal,
                }
            )
    return issues


def _raw_score(support: float, votes: int, duplicate_support: float, confidence: float, risk: str) -> float:
    penalty = {"danger": 1.25, "caution": 0.15}.get(risk, 0.0)
    return round(support + votes * 0.12 + duplicate_support * 0.28 + confidence * 0.10 - penalty, 6)


def _score_result(decision: str, score: float, role: str, reasons: list[str], risk: str) -> dict[str, Any]:
    return {"decision": decision, "score": round(score, 6), "role": role, "reasons": reasons, "risk": risk}


def _parse_transition_counts(text: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for part in str(text or "").split(";"):
        if ":" not in part:
            continue
        transition, raw_count = part.split(":", maxsplit=1)
        transition = transition.strip()
        if "->" not in transition:
            continue
        counts[transition] += _safe_int(raw_count)
    return counts


def _is_failed_batch_direction(row: dict[str, Any]) -> bool:
    candidate_a = str(row.get("candidate_a", "")).lower()
    candidate_b = str(row.get("candidate_b", "")).lower()
    direction = str(row.get("direction", "")).strip().lower()
    if not candidate_a and not candidate_b:
        return True
    if "781601" in candidate_a and "candidate_b -> candidate_a" in direction:
        return True
    if "781601" in candidate_b and "candidate_a -> candidate_b" in direction:
        return True
    return False


def _has_public_style_consensus(row: dict[str, Any]) -> bool:
    public_style_support = _safe_float(row.get("public_style_support_score"))
    sources = {part.strip() for part in str(row.get("all_sources", "")).split(",") if part.strip()}
    return public_style_support >= 1.0 or {"public_clean", "public_inclusive"}.issubset(sources)


def _reverse_transition(transition: str) -> str:
    parts = [part.strip() for part in str(transition).split("->")]
    if len(parts) != 2:
        return ""
    return f"{parts[1]}->{parts[0]}"


def _canonical_emotion(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in TRACK2_EMOTIONS else ""


def _valence(emotion: str) -> str:
    return "Negative" if emotion in NEGATIVE_EMOTIONS else "Positive"


def _arousal(emotion: str) -> str:
    return "High" if emotion in HIGH_AROUSAL_EMOTIONS else "Low"


def _label_triplet(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("emotion", "")).strip(),
        str(row.get("emotional_valence", "")).strip(),
        str(row.get("emotional_arousal_level", "")).strip(),
    )


def _profile_config(profile: str) -> dict[str, float | int]:
    if profile not in PROFILE_CONFIG:
        raise ValueError(f"unknown v20 profile: {profile}")
    return PROFILE_CONFIG[profile]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0", ""}:
        return False
    return False


def _reject_formal_submission_path(path: Path) -> None:
    if path.name.casefold() in {name.casefold() for name in FORMAL_SUBMISSION_NAMES}:
        raise ValueError(f"refusing to write formal submission path: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_zip_payload(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
    info = zipfile.ZipInfo("submission.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, payload)


def _render_frontier_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Track2 v20 Frontier Requirements",
            "",
            f"- Frontier: `{report['frontier_participant']}`",
            f"- Current overall/class/desc: `{report['current_overall']}` / `{report['current_classification']}` / `{report['current_description']}`",
            f"- Frontier overall/class/desc: `{report['frontier_overall']}` / `{report['frontier_classification']}` / `{report['frontier_description']}`",
            f"- Emotion accuracy gap: `{report['emotion_accuracy_gap']}`",
            f"- Macro-F1 gap: `{report['macro_f1_gap']}`",
            f"- Required class if desc=1.00: `{report['required_classification_if_description_1.00']}`",
            f"- Required class if desc=0.98: `{report['required_classification_if_description_0.98']}`",
            "",
            report["warning"],
            "",
        ]
    )


def _render_risk_md(risk: dict[str, dict[str, Any]]) -> str:
    lines = ["# Track2 v20 Directional Risk Map", ""]
    for transition, row in sorted(risk.items()):
        lines.append(f"- `{transition}`: {row['risk']} count={row['count']} reverse={row['reverse_count']}")
    return "\n".join(lines) + "\n"


def _render_candidate_md(report: dict[str, Any]) -> str:
    lines = [
        f"# Track2 v20 {report['profile']} Candidate",
        "",
        f"- Accepted label changes: `{report['accepted_label_changes']}`",
        f"- Top emotion: `{report['top_emotion']}` ({float(report['top_emotion_share']):.1%})",
        f"- Missing emotions: `{', '.join(report['missing_emotions']) or 'none'}`",
        f"- Label consistency issues: `{report['label_consistency_issue_count']}`",
        "",
        "## Transition Counts",
        "",
    ]
    if report.get("transition_counts"):
        for transition, count in sorted(report["transition_counts"].items()):
            lines.append(f"- `{transition}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Accepted Changes", ""])
    for item in report.get("accepted_changes", [])[:100]:
        lines.append(f"- `{item['sample_id']}`: {item['transition']} score={float(item['v20_score']):.3f}")
    if not report.get("accepted_changes"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _render_ladder_md(summary: dict[str, Any]) -> str:
    lines = ["# Track2 v20 Candidate Ladder", ""]
    for profile, row in summary["candidates"].items():
        lines.extend(
            [
                f"## {profile}",
                "",
                f"- Accepted label changes: `{row['accepted_label_changes']}`",
                f"- Top emotion: `{row['top_emotion']}` ({float(row['top_emotion_share']):.1%})",
                f"- Missing emotions: `{', '.join(row['missing_emotions']) or 'none'}`",
                f"- Candidate ZIP: `{row['paths']['out_zip']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _render_final_gate_md(report: dict[str, Any]) -> str:
    selected = report.get("selected_candidate") or {}
    lines = [
        "# Track2 v20 Final Gate",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Selected profile: `{report['selected_profile']}`",
        f"- Candidate ZIP: `{(selected.get('paths') or {}).get('out_zip', '')}`",
        f"- Accepted label changes: `{selected.get('accepted_label_changes', '')}`",
        f"- Top emotion: `{selected.get('top_emotion', '')}` ({float(selected.get('top_emotion_share') or 0.0):.1%})",
        "",
        "## Gate Reports",
        "",
    ]
    for profile, gate in sorted(report["gate_reports"].items()):
        lines.append(
            f"- `{profile}`: {gate['decision']} changes={gate['accepted_label_changes']} "
            f"reasons={', '.join(gate['reasons']) or 'none'}"
        )
    lines.extend(["", "No automatic Codabench submission is allowed from this report.", ""])
    return "\n".join(lines)
