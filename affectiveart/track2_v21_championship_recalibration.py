from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_DIR = Path("experiments/track2_v21_championship_recalibration_20260607")
DEFAULT_BASE_JSON = Path("submissions/track2_submission_v15_desc_expand300_candidate.json")
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
    "precision80": {
        "total_cap": 80,
        "cross_cap": 12,
        "transition_cap": 40,
        "min_score": 1.55,
        "top_emotion_cap": 0.58,
        "class_floor": 4,
    },
    "champion120": {
        "total_cap": 120,
        "cross_cap": 24,
        "transition_cap": 55,
        "min_score": 1.35,
        "top_emotion_cap": 0.58,
        "class_floor": 4,
    },
    "lastshot160": {
        "total_cap": 160,
        "cross_cap": 36,
        "transition_cap": 70,
        "min_score": 1.15,
        "top_emotion_cap": 0.58,
        "class_floor": 4,
    },
    "calmshift90": {
        "total_cap": 90,
        "cross_cap": 0,
        "transition_cap": 90,
        "min_score": 1.0,
        "top_emotion_cap": 0.58,
        "class_floor": 4,
        "allowed_transitions": ("content->calm",),
    },
}
TIER_RANK = {"reference": 0, "consensus": 1, "expansion": 2, "hold": 8, "block": 9}


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


def score_v21_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    current = _canonical_emotion(row.get("current_emotion"))
    proposed = _canonical_emotion(row.get("proposed_emotion"))
    support = _safe_float(row.get("support_score"))
    votes = _safe_int(row.get("model_vote_count"))
    confidence = _safe_float(row.get("max_confidence"))
    duplicate_support = _safe_float(row.get("public_duplicate_support_score"))
    public_style = _safe_float(row.get("public_style_support_score"))
    exact_duplicate = _safe_bool(row.get("exact_duplicate"))
    near_duplicate = _safe_bool(row.get("near_duplicate"))
    source_count = len(_source_set(row))
    same_quadrant = _same_quadrant(current, proposed)
    reasons: list[str] = []

    if current not in TRACK2_EMOTIONS or proposed not in TRACK2_EMOTIONS:
        return _score_result("block", "invalid", 0.0, ["invalid_emotion_label"])
    if current == proposed:
        return _score_result("block", "invalid", 0.0, ["no_op_transition"])

    reference = (exact_duplicate or near_duplicate) and duplicate_support >= 0.95 and confidence >= 0.90 and votes >= 2
    consensus = votes >= 3 and support >= 2.0 and _has_public_style_consensus(row)
    expansion = (votes >= 2 and support >= 1.45 and confidence >= 0.78) or (
        source_count >= 4 and votes >= 2 and support >= 1.65
    )

    if reference:
        return _score_result(
            "accept_candidate",
            "reference",
            _raw_score(support, votes, duplicate_support, confidence, public_style, "reference"),
            ["public_duplicate_or_near_duplicate_reference"],
        )
    if consensus:
        if not same_quadrant and support < 2.3:
            return _score_result(
                "hold",
                "hold",
                _raw_score(support, votes, duplicate_support, confidence, public_style, "consensus"),
                ["cross_quadrant_consensus_needs_stronger_support"],
            )
        return _score_result(
            "accept_candidate",
            "consensus",
            _raw_score(support, votes, duplicate_support, confidence, public_style, "consensus"),
            ["model_family_and_public_style_consensus"],
        )
    if expansion:
        if not same_quadrant:
            return _score_result(
                "hold",
                "hold",
                _raw_score(support, votes, duplicate_support, confidence, public_style, "expansion"),
                ["cross_quadrant_expansion_blocked"],
            )
        return _score_result(
            "accept_candidate",
            "expansion",
            _raw_score(support, votes, duplicate_support, confidence, public_style, "expansion"),
            ["two_model_family_high_confidence_expansion"],
        )

    reasons.append("insufficient_v21_evidence")
    return _score_result(
        "hold",
        "hold",
        _raw_score(support, votes, duplicate_support, confidence, public_style, "hold"),
        reasons,
    )


def build_v21_championship_evidence(v17_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in v17_rows:
        scored = score_v21_evidence_row(row)
        current = _canonical_emotion(row.get("current_emotion"))
        proposed = _canonical_emotion(row.get("proposed_emotion"))
        transition = f"{current}->{proposed}" if current and proposed else str(row.get("transition") or "")
        enriched = dict(row)
        enriched["current_emotion"] = current
        enriched["proposed_emotion"] = proposed
        enriched["transition"] = transition
        enriched["same_quadrant"] = _same_quadrant(current, proposed)
        enriched["v21_decision"] = scored["decision"]
        enriched["v21_score"] = scored["score"]
        enriched["v21_tier"] = scored["tier"]
        enriched["v21_reasons"] = ";".join(scored["reasons"])
        output.append(enriched)
    return sorted(output, key=_evidence_sort_key)


def select_v21_changes(
    evidence_rows: list[dict[str, Any]],
    *,
    profile: str,
    base_distribution: Counter[str] | dict[str, int],
) -> list[dict[str, Any]]:
    config = _profile_config(profile)
    selected: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    transition_counts: Counter[str] = Counter()
    cross_count = 0
    projected = Counter({str(key): int(value) for key, value in dict(base_distribution).items()})
    total_rows = max(1, sum(projected.values()))
    for row in sorted(evidence_rows, key=_evidence_sort_key):
        if len(selected) >= int(config["total_cap"]):
            break
        if row.get("v21_decision") != "accept_candidate":
            continue
        if _safe_float(row.get("v21_score")) < float(config["min_score"]):
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen_samples:
            continue
        current = _canonical_emotion(row.get("current_emotion"))
        proposed = _canonical_emotion(row.get("proposed_emotion"))
        if current not in TRACK2_EMOTIONS or proposed not in TRACK2_EMOTIONS or current == proposed:
            continue
        transition = f"{current}->{proposed}"
        allowed_transitions = set(config.get("allowed_transitions") or ())
        if allowed_transitions and transition not in allowed_transitions:
            continue
        if transition_counts[transition] >= int(config["transition_cap"]):
            continue
        if projected[current] - 1 < int(config["class_floor"]):
            continue
        if (projected[proposed] + 1) / total_rows > float(config["top_emotion_cap"]):
            continue
        same_quadrant = _same_quadrant(current, proposed)
        if not same_quadrant and cross_count >= int(config["cross_cap"]):
            continue
        selected_row = dict(row)
        selected_row["current_emotion"] = current
        selected_row["proposed_emotion"] = proposed
        selected_row["transition"] = transition
        selected_row["same_quadrant"] = same_quadrant
        selected.append(selected_row)
        seen_samples.add(sample_id)
        transition_counts[transition] += 1
        if not same_quadrant:
            cross_count += 1
        projected[current] -= 1
        projected[proposed] += 1
    return selected


def write_v21_candidate_outputs(
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
        "method": "track2_v21_championship_scale_recalibration_v1",
        "profile": profile,
        "row_count": len(candidate_rows),
        "accepted_label_changes": apply_report["accepted_label_changes"],
        "transition_counts": apply_report["transition_counts"],
        "tier_counts": apply_report["tier_counts"],
        "cross_quadrant_change_count": apply_report["cross_quadrant_change_count"],
        "distribution": apply_report["distribution"],
        "missing_emotions": apply_report["missing_emotions"],
        "top_emotion": apply_report["top_emotion"],
        "top_emotion_share": apply_report["top_emotion_share"],
        "label_consistency_issue_count": apply_report["label_consistency_issue_count"],
        "label_consistency_issues": apply_report["label_consistency_issues"],
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


def write_v21_candidate_ladder_outputs(
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
    for profile in ("precision80", "champion120", "lastshot160", "calmshift90"):
        selected = select_v21_changes(evidence_rows, profile=profile, base_distribution=base_distribution)
        stem = f"track2_submission_v21_{profile}_candidate"
        candidate_report = write_v21_candidate_outputs(
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
                "tier_counts",
                "cross_quadrant_change_count",
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
        "method": "track2_v21_candidate_ladder_v1",
        "base_json": str(base_json),
        "championship_evidence_json": str(championship_evidence_json),
        "candidates": candidates,
        "formal_submission_overwritten": False,
    }
    _write_json(Path(out_dir) / "candidate_ladder_summary.json", summary)
    (Path(out_dir) / "candidate_ladder_summary.md").write_text(_render_ladder_md(summary), encoding="utf-8")
    return summary


def choose_v21_final_gate(
    *,
    candidate_name: str,
    profile: str,
    accepted_label_changes: int,
    label_consistency_issue_count: int,
    missing_emotions: list[str],
    top_emotion_share: float,
    description_anchor_preserved: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if accepted_label_changes < 80:
        reasons.append("accepted_changes_below_80")
    if label_consistency_issue_count:
        reasons.append("label_consistency_issues")
    if missing_emotions:
        reasons.append("missing_emotions")
    if top_emotion_share > 0.58:
        reasons.append("top_emotion_share_above_0.58")
    if not description_anchor_preserved:
        reasons.append("description_anchor_not_preserved")
    if reasons:
        decision = "hold"
    elif profile == "champion120" and accepted_label_changes >= 100:
        decision = "recommend_high_variance_second_last_submission"
    elif profile == "precision80":
        decision = "recommend_precision_second_last_submission"
    elif profile == "calmshift90" and accepted_label_changes >= 70:
        decision = "recommend_directional_second_last_submission"
    elif profile == "lastshot160":
        decision = "hold_for_final_submission_only"
    else:
        decision = "hold"
        reasons.append("profile_does_not_meet_submission_gate")
    return {
        "method": "track2_v21_final_gate_v1",
        "candidate_name": candidate_name,
        "profile": profile,
        "decision": decision,
        "reasons": reasons,
        "accepted_label_changes": int(accepted_label_changes),
        "label_consistency_issue_count": int(label_consistency_issue_count),
        "missing_emotions": list(missing_emotions),
        "top_emotion_share": round(float(top_emotion_share), 6),
        "description_anchor_preserved": bool(description_anchor_preserved),
        "no_auto_submit": True,
        "caveat": "High-variance local competition gate; this is not an official hidden scorer reconstruction.",
    }


def write_v21_final_gate_report(
    *,
    ladder_summary: dict[str, Any],
    out_json: str | Path,
    out_md: str | Path,
) -> dict[str, Any]:
    candidates = ladder_summary.get("candidates") or {}
    gate_reports: dict[str, Any] = {}
    for profile, row in sorted(candidates.items()):
        gate_reports[profile] = choose_v21_final_gate(
            candidate_name=f"v21_{profile}",
            profile=profile,
            accepted_label_changes=_safe_int(row.get("accepted_label_changes")),
            label_consistency_issue_count=_safe_int(row.get("label_consistency_issue_count")),
            missing_emotions=list(row.get("missing_emotions") or []),
            top_emotion_share=_safe_float(row.get("top_emotion_share")),
            description_anchor_preserved=_safe_bool(row.get("description_anchor_preserved")),
        )
    priority = {
        "recommend_high_variance_second_last_submission": 0,
        "recommend_directional_second_last_submission": 1,
        "recommend_precision_second_last_submission": 2,
        "hold_for_final_submission_only": 3,
        "hold": 4,
    }
    best_profile, best_gate = min(
        gate_reports.items(),
        key=lambda item: (
            priority.get(str(item[1].get("decision")), 9),
            -_safe_int(item[1].get("accepted_label_changes")),
            item[0],
        ),
    )
    report = {
        "method": "track2_v21_final_gate_report_v1",
        "decision": best_gate["decision"],
        "selected_profile": best_profile,
        "selected_candidate": candidates.get(best_profile, {}),
        "gate_reports": gate_reports,
        "no_auto_submit": True,
    }
    _reject_formal_submission_path(Path(out_json))
    _reject_formal_submission_path(Path(out_md))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    _write_json(Path(out_json), report)
    Path(out_md).write_text(_render_final_gate_md(report), encoding="utf-8")
    return report


def write_v21_run_outputs(
    *,
    base_json: str | Path = DEFAULT_BASE_JSON,
    v17_evidence: str | Path = DEFAULT_V17_EVIDENCE,
    out_dir: str | Path = DEFAULT_EXPERIMENT_DIR,
    submissions_dir: str | Path = "submissions",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v17_rows = json.loads(Path(v17_evidence).read_text(encoding="utf-8"))
    if not isinstance(v17_rows, list):
        raise ValueError("v17 evidence must be a JSON list")
    evidence = build_v21_championship_evidence(v17_rows)
    _write_json(out_dir / "championship_evidence.json", evidence)
    _write_csv(out_dir / "championship_evidence.csv", evidence)
    ladder = write_v21_candidate_ladder_outputs(
        base_json=base_json,
        championship_evidence_json=out_dir / "championship_evidence.json",
        out_dir=out_dir,
        submissions_dir=submissions_dir,
    )
    final_gate = write_v21_final_gate_report(
        ladder_summary=ladder,
        out_json=out_dir / "final_gate_report.json",
        out_md=out_dir / "final_gate_report.md",
    )
    return {
        "method": "track2_v21_run_outputs_v1",
        "championship_evidence": str(out_dir / "championship_evidence.json"),
        "candidate_ladder_summary": str(out_dir / "candidate_ladder_summary.json"),
        "final_gate_report": str(out_dir / "final_gate_report.json"),
        "decision": final_gate["decision"],
        "selected_profile": final_gate["selected_profile"],
        "selected_zip": (final_gate.get("selected_candidate") or {}).get("paths", {}).get("out_zip", ""),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Track2 v21 championship-scale recalibration artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Write full v21 artifacts and side-path candidates.")
    run.add_argument("--base-json", default=str(DEFAULT_BASE_JSON))
    run.add_argument("--v17-evidence", default=str(DEFAULT_V17_EVIDENCE))
    run.add_argument("--out-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    run.add_argument("--submissions-dir", default="submissions")
    args = parser.parse_args(argv)
    if args.command == "run":
        report = write_v21_run_outputs(
            base_json=args.base_json,
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
                        "v21_score": _safe_float(change.get("v21_score")),
                        "v21_tier": str(change.get("v21_tier", "")),
                        "same_quadrant": _safe_bool(change.get("same_quadrant")),
                    }
                )
        candidate_rows.append({key: row.get(key, "") for key in TRACK2_SUBMISSION_KEYS})
    distribution = Counter(str(row.get("emotion", "")) for row in candidate_rows)
    label_issues = _label_issues(candidate_rows)
    return candidate_rows, {
        "accepted_label_changes": len(accepted),
        "transition_counts": dict(sorted(Counter(item["transition"] for item in accepted).items())),
        "tier_counts": dict(sorted(Counter(item["v21_tier"] for item in accepted).items())),
        "cross_quadrant_change_count": sum(1 for item in accepted if not item["same_quadrant"]),
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


def _evidence_sort_key(row: dict[str, Any]) -> tuple[int, float, str, str]:
    return (
        TIER_RANK.get(str(row.get("v21_tier", "hold")), 7),
        -_safe_float(row.get("v21_score")),
        str(row.get("sample_id", "")),
        str(row.get("proposed_emotion", "")),
    )


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


def _raw_score(
    support: float,
    votes: int,
    duplicate_support: float,
    confidence: float,
    public_style: float,
    tier: str,
) -> float:
    tier_bonus = {"reference": 0.55, "consensus": 0.25, "expansion": 0.0, "hold": -0.2}.get(tier, 0.0)
    return round(support + votes * 0.12 + duplicate_support * 0.22 + confidence * 0.08 + public_style * 0.07 + tier_bonus, 6)


def _score_result(decision: str, tier: str, score: float, reasons: list[str]) -> dict[str, Any]:
    return {"decision": decision, "tier": tier, "score": round(score, 6), "reasons": reasons}


def _profile_config(profile: str) -> dict[str, float | int]:
    if profile not in PROFILE_CONFIG:
        raise ValueError(f"unknown v21 profile: {profile}")
    return PROFILE_CONFIG[profile]


def _has_public_style_consensus(row: dict[str, Any]) -> bool:
    sources = _source_set(row)
    return _safe_float(row.get("public_style_support_score")) >= 0.8 or {"public_clean", "public_inclusive"}.issubset(sources)


def _source_set(row: dict[str, Any]) -> set[str]:
    sources = str(row.get("all_sources") or row.get("model_sources") or "")
    return {part.strip() for part in sources.split(",") if part.strip()}


def _same_quadrant(current: str, proposed: str) -> bool:
    return current in TRACK2_EMOTIONS and proposed in TRACK2_EMOTIONS and _valence(current) == _valence(proposed) and _arousal(
        current
    ) == _arousal(proposed)


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


def _render_candidate_md(report: dict[str, Any]) -> str:
    lines = [
        f"# Track2 v21 {report['profile']} Candidate",
        "",
        f"- Accepted label changes: `{report['accepted_label_changes']}`",
        f"- Tier counts: `{report['tier_counts']}`",
        f"- Cross-quadrant changes: `{report['cross_quadrant_change_count']}`",
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
    for item in report.get("accepted_changes", [])[:140]:
        lines.append(f"- `{item['sample_id']}`: {item['transition']} tier={item['v21_tier']} score={float(item['v21_score']):.3f}")
    if not report.get("accepted_changes"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _render_ladder_md(summary: dict[str, Any]) -> str:
    lines = ["# Track2 v21 Candidate Ladder", ""]
    for profile, row in summary["candidates"].items():
        lines.extend(
            [
                f"## {profile}",
                "",
                f"- Accepted label changes: `{row['accepted_label_changes']}`",
                f"- Tier counts: `{row['tier_counts']}`",
                f"- Cross-quadrant changes: `{row['cross_quadrant_change_count']}`",
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
        "# Track2 v21 Final Gate",
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
