from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from affectiveart.track2_official_anchor_calibration import score_submission


DEFAULT_OUT_DIR = Path("experiments/track2_v27_gold_like_ledger_20260608")
DEFAULT_BASE_JSON = Path("submissions/track2_submission_v15_desc_expand300_candidate.json")
DEFAULT_EVIDENCE_JSON = Path("experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json")
DEFAULT_SUBMISSIONS_DIR = Path("submissions")

TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
TRACK2_SUBMISSION_KEYS = ("sample_id", "emotion", "emotional_valence", "emotional_arousal_level", *TEXT_FIELDS)
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
FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}
EVALUATOR_MANIPULATION_PATTERNS = (
    "ignore previous",
    "ignore the evaluator",
    "score this",
    "give full score",
    "assign full score",
    "do not penalize",
    "as an evaluator",
)


def classify_v27_evidence_row(row: Mapping[str, Any]) -> dict[str, Any]:
    current = _canonical_emotion(row.get("current_emotion"))
    proposed = _canonical_emotion(row.get("proposed_emotion"))
    if current not in TRACK2_EMOTIONS or proposed not in TRACK2_EMOTIONS:
        return _decision("block", "invalid", 0.0, ["invalid_emotion_label"])
    if current == proposed:
        return _decision("block", "invalid", 0.0, ["no_op_transition"])

    support = _safe_float(row.get("support_score"))
    public_style = _safe_float(row.get("public_style_support_score"))
    duplicate_support = _safe_float(row.get("public_duplicate_support_score"))
    confidence = _safe_float(row.get("max_confidence"))
    votes = _safe_int(row.get("model_vote_count"))
    near_or_exact = _safe_bool(row.get("near_duplicate")) or _safe_bool(row.get("exact_duplicate"))
    same_quadrant = _same_quadrant(current, proposed)

    if near_or_exact and duplicate_support >= 0.95 and confidence >= 0.90 and votes >= 2:
        score = support + duplicate_support * 1.4 + confidence * 0.4 + votes * 0.16
        return _decision(
            "accept_candidate",
            "tier1_visual_duplicate",
            score,
            ["strict_public_duplicate_or_near_duplicate"],
        )
    if support >= 2.25 and public_style >= 0.50 and votes >= 3 and same_quadrant:
        score = support + public_style * 0.55 + confidence * 0.18 + votes * 0.10
        return _decision(
            "accept_candidate",
            "tier2_same_series",
            score,
            ["same_series_same_quadrant_high_agreement"],
        )
    return _decision("hold", "tier3_weak_model_or_style", support, ["weak_model_or_style_only"])


def build_v27_ledger(v17_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for row in v17_rows:
        current = _canonical_emotion(row.get("current_emotion"))
        proposed = _canonical_emotion(row.get("proposed_emotion"))
        scored = classify_v27_evidence_row(row)
        enriched = dict(row)
        enriched.update(
            {
                "current_emotion": current,
                "proposed_emotion": proposed,
                "transition": f"{current}->{proposed}" if current and proposed else str(row.get("transition", "")),
                "same_quadrant": _same_quadrant(current, proposed),
                "v27_decision": scored["decision"],
                "v27_tier": scored["tier"],
                "v27_score": scored["score"],
                "v27_reasons": ";".join(scored["reasons"]),
            }
        )
        ledger.append(enriched)
    return sorted(ledger, key=_ledger_sort_key)


def select_v27_changes(
    rows: list[Mapping[str, Any]],
    *,
    base_distribution: Counter[str] | Mapping[str, int],
    total_cap: int = 72,
    tier2_cap: int = 24,
    top_emotion_cap: float = 0.58,
    class_floor: int = 4,
) -> list[dict[str, Any]]:
    projected = Counter({str(key): int(value) for key, value in dict(base_distribution).items()})
    total_rows = max(1, sum(projected.values()))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    tier2_count = 0
    for row in sorted(rows, key=_selection_sort_key):
        if len(selected) >= total_cap:
            break
        if row.get("v27_decision") != "accept_candidate":
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen:
            continue
        current = _canonical_emotion(row.get("current_emotion"))
        proposed = _canonical_emotion(row.get("proposed_emotion"))
        if current not in TRACK2_EMOTIONS or proposed not in TRACK2_EMOTIONS or current == proposed:
            continue
        if projected[current] - 1 < class_floor:
            continue
        if (projected[proposed] + 1) / total_rows > top_emotion_cap:
            continue
        tier = str(row.get("v27_tier", ""))
        if tier == "tier2_same_series":
            if tier2_count >= tier2_cap:
                continue
            tier2_count += 1
        selected_row = dict(row)
        selected_row["current_emotion"] = current
        selected_row["proposed_emotion"] = proposed
        selected_row["transition"] = f"{current}->{proposed}"
        selected.append(selected_row)
        seen.add(sample_id)
        projected[current] -= 1
        projected[proposed] += 1
    return selected


def choose_v27_final_gate(
    *,
    projected_overall: float,
    projected_classification: float,
    projected_description: float,
    label_consistency_issue_count: int,
    missing_emotions: list[str],
    unsafe_text_count: int,
    top_emotion_share: float,
    cross_quadrant_change_count: int,
    cross_quadrant_risk_report_count: int,
    target_overall: float = 0.89,
    max_top_emotion_share: float = 0.58,
) -> dict[str, Any]:
    reasons: list[str] = []
    if projected_overall < target_overall:
        reasons.append("below_089_target")
    if projected_classification < 0.78:
        reasons.append("classification_below_public_first_place_reference")
    if projected_description < 0.94:
        reasons.append("description_below_safe_floor")
    if label_consistency_issue_count:
        reasons.append("label_consistency_issues")
    if missing_emotions:
        reasons.append("missing_emotions")
    if unsafe_text_count:
        reasons.append("unsafe_description_text")
    if top_emotion_share > max_top_emotion_share:
        reasons.append("top_emotion_collapse_risk")
    if cross_quadrant_risk_report_count != cross_quadrant_change_count:
        reasons.append("cross_quadrant_risk_not_fully_reported")
    return {
        "decision": "hold_no_submit" if reasons else "recommend_final_submit",
        "target_overall": round(target_overall, 6),
        "projected_overall": round(projected_overall, 6),
        "projected_classification": round(projected_classification, 6),
        "projected_description": round(projected_description, 6),
        "reasons": reasons,
    }


def load_track2_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Track2 JSON must be a list: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Track2 row {index} must be an object")
        row = {key: str(item.get(key, "")).strip() for key in TRACK2_SUBMISSION_KEYS}
        sample_id = row["sample_id"]
        if not sample_id:
            raise ValueError(f"blank sample_id at row {index}")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        rows.append(row)
    return rows


def write_v27_candidate_outputs(
    *,
    base_rows: list[Mapping[str, Any]],
    selected_changes: list[Mapping[str, Any]],
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
        "method": "track2_v27_gold_like_ledger_v1",
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
        "unsafe_text_count": _unsafe_text_count(candidate_rows),
        "accepted_changes": apply_report["accepted_changes"],
        "formal_submission_overwritten": False,
        "paths": {
            "out_json": str(out_json),
            "out_zip": str(out_zip),
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
    }
    _write_json(out_json, candidate_rows)
    _write_zip_payload(out_zip, candidate_rows)
    _write_json(report_json, report)
    report_md.write_text(_render_candidate_md(report), encoding="utf-8")
    return report


def build_v27_candidate_suite(
    *,
    base_json: str | Path = DEFAULT_BASE_JSON,
    evidence_json: str | Path = DEFAULT_EVIDENCE_JSON,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    submissions_dir: str | Path = DEFAULT_SUBMISSIONS_DIR,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    submissions_dir = Path(submissions_dir)
    report_dir = out_dir / "candidate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    base_rows = load_track2_rows(base_json)
    evidence_rows = _load_evidence_rows(evidence_json)
    ledger = build_v27_ledger(evidence_rows)
    _write_json(out_dir / "v27_gold_like_ledger.json", ledger)
    _write_csv(out_dir / "v27_gold_like_ledger.csv", ledger)

    profiles = {
        "strict": {"total_cap": 36, "tier2_cap": 12},
        "frontier": {"total_cap": 96, "tier2_cap": 36},
    }
    base_distribution = Counter(str(row.get("emotion", "")) for row in base_rows)
    profile_rows: list[dict[str, Any]] = []
    profile_reports: dict[str, dict[str, Any]] = {}
    for profile, config in profiles.items():
        selected = select_v27_changes(
            ledger,
            base_distribution=base_distribution,
            total_cap=int(config["total_cap"]),
            tier2_cap=int(config["tier2_cap"]),
        )
        stem = f"track2_submission_v27_gold_like_{profile}_candidate"
        report = write_v27_candidate_outputs(
            base_rows=base_rows,
            selected_changes=selected,
            out_json=submissions_dir / f"{stem}.json",
            out_zip=submissions_dir / f"{stem}.zip",
            report_json=report_dir / f"{stem}_report.json",
            report_md=report_dir / f"{stem}_report.md",
            profile=profile,
        )
        score = score_submission(report["paths"]["out_json"], candidate_name=stem)
        cross_risk_count = len(
            [item for item in report["accepted_changes"] if not _same_quadrant(item["before"]["emotion"], item["after"]["emotion"])]
        )
        gate = choose_v27_final_gate(
            projected_overall=score.overall_expected,
            projected_classification=score.classification_expected,
            projected_description=score.description_expected,
            label_consistency_issue_count=int(report["label_consistency_issue_count"]),
            missing_emotions=list(report["missing_emotions"]),
            unsafe_text_count=int(report["unsafe_text_count"]),
            top_emotion_share=float(report["top_emotion_share"]),
            cross_quadrant_change_count=int(report["cross_quadrant_change_count"]),
            cross_quadrant_risk_report_count=cross_risk_count,
        )
        row = {
            "profile": profile,
            "candidate_name": stem,
            "json_path": report["paths"]["out_json"],
            "zip_path": report["paths"]["out_zip"],
            "accepted_label_changes": report["accepted_label_changes"],
            "transition_counts": "; ".join(f"{key}:{value}" for key, value in report["transition_counts"].items()),
            "tier_counts": "; ".join(f"{key}:{value}" for key, value in report["tier_counts"].items()),
            "cross_quadrant_change_count": report["cross_quadrant_change_count"],
            "top_emotion": report["top_emotion"],
            "top_emotion_share": report["top_emotion_share"],
            "label_consistency_issue_count": report["label_consistency_issue_count"],
            "unsafe_text_count": report["unsafe_text_count"],
            "overall_expected": score.overall_expected,
            "classification_expected": score.classification_expected,
            "description_expected": score.description_expected,
            "score_warnings": ";".join(score.warnings),
            "gate_decision": gate["decision"],
            "gate_reasons": ";".join(gate["reasons"]),
        }
        profile_rows.append(row)
        profile_reports[profile] = {**report, "score": row, "gate": gate}
    ranked = sorted(profile_rows, key=lambda row: (-float(row["overall_expected"]), str(row["profile"])))
    passing = [row for row in ranked if row["gate_decision"] == "recommend_final_submit"]
    best = passing[0] if passing else (ranked[0] if ranked else {})
    final_decision = "recommend_final_submit" if passing else "hold_no_submit"
    summary = {
        "method": "track2_v27_gold_like_ledger_suite_v1",
        "base_json": str(base_json),
        "evidence_json": str(evidence_json),
        "target_overall": 0.89,
        "decision": final_decision,
        "best_profile": best.get("profile", ""),
        "best": best,
        "profiles": profile_rows,
        "profile_reports": profile_reports,
        "ledger_counts": dict(sorted(Counter(str(row.get("v27_tier", "")) for row in ledger).items())),
        "accepted_ledger_count": sum(1 for row in ledger if row.get("v27_decision") == "accept_candidate"),
        "formal_submission_overwritten": False,
        "interpretation": "Use the best ZIP only if decision is recommend_final_submit; otherwise preserve final Codabench attempt.",
    }
    _write_json(out_dir / "v27_profile_scoreboard.json", summary)
    _write_csv(out_dir / "v27_profile_scoreboard.csv", profile_rows)
    (out_dir / "v27_candidate_report_zh.md").write_text(_render_suite_md(summary), encoding="utf-8")
    return summary


def _apply_changes(
    base_rows: list[Mapping[str, Any]],
    selected_changes: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    changes_by_id = {
        str(row.get("sample_id", "")).strip(): row
        for row in selected_changes
        if str(row.get("sample_id", "")).strip()
    }
    candidate_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for base in base_rows:
        row = {key: str(base.get(key, "")).strip() for key in TRACK2_SUBMISSION_KEYS}
        sample_id = row["sample_id"]
        change = changes_by_id.get(sample_id)
        if change:
            current = _canonical_emotion(row.get("emotion"))
            expected = _canonical_emotion(change.get("current_emotion"))
            proposed = _canonical_emotion(change.get("proposed_emotion"))
            if current == expected and proposed in TRACK2_EMOTIONS and proposed != current:
                before = _label_triplet(row)
                row["emotion"] = proposed
                row["emotional_valence"] = _valence(proposed)
                row["emotional_arousal_level"] = _arousal(proposed)
                after = _label_triplet(row)
                accepted.append(
                    {
                        "sample_id": sample_id,
                        "transition": f"{current}->{proposed}",
                        "before": {"emotion": before[0], "valence": before[1], "arousal": before[2]},
                        "after": {"emotion": after[0], "valence": after[1], "arousal": after[2]},
                        "v27_score": _safe_float(change.get("v27_score")),
                        "v27_tier": str(change.get("v27_tier", "")),
                        "v27_reasons": str(change.get("v27_reasons", "")),
                    }
                )
        candidate_rows.append({key: row.get(key, "") for key in TRACK2_SUBMISSION_KEYS})
    distribution = Counter(str(row.get("emotion", "")) for row in candidate_rows)
    issues = _label_issues(candidate_rows)
    return candidate_rows, {
        "accepted_label_changes": len(accepted),
        "transition_counts": dict(sorted(Counter(item["transition"] for item in accepted).items())),
        "tier_counts": dict(sorted(Counter(item["v27_tier"] for item in accepted).items())),
        "cross_quadrant_change_count": sum(
            1 for item in accepted if not _same_quadrant(item["before"]["emotion"], item["after"]["emotion"])
        ),
        "distribution": dict(sorted(distribution.items())),
        "missing_emotions": sorted(TRACK2_EMOTIONS - set(distribution)),
        "top_emotion": distribution.most_common(1)[0][0] if distribution else "",
        "top_emotion_share": round(distribution.most_common(1)[0][1] / max(1, len(candidate_rows)), 6)
        if distribution
        else 0.0,
        "label_consistency_issue_count": len(issues),
        "label_consistency_issues": issues[:80],
        "accepted_changes": accepted,
    }


def _decision(decision: str, tier: str, score: float, reasons: list[str]) -> dict[str, Any]:
    return {"decision": decision, "tier": tier, "score": round(float(score), 6), "reasons": reasons}


def _ledger_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    tier_rank = {"tier1_visual_duplicate": 0, "tier2_same_series": 1, "tier3_weak_model_or_style": 6, "invalid": 9}
    return (
        0 if row.get("v27_decision") == "accept_candidate" else 1,
        tier_rank.get(str(row.get("v27_tier", "")), 8),
        -_safe_float(row.get("v27_score")),
        str(row.get("sample_id", "")),
    )


def _selection_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    tier_rank = {"tier1_visual_duplicate": 0, "tier2_same_series": 1, "tier3_weak_model_or_style": 6, "invalid": 9}
    return (
        0 if row.get("v27_decision") == "accept_candidate" else 1,
        -_safe_float(row.get("v27_score")),
        tier_rank.get(str(row.get("v27_tier", "")), 8),
        str(row.get("sample_id", "")),
    )


def _label_issues(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        emotion = _canonical_emotion(row.get("emotion"))
        if emotion not in TRACK2_EMOTIONS:
            issues.append({"sample_id": sample_id, "field": "emotion", "actual": row.get("emotion", "")})
            continue
        valence = str(row.get("emotional_valence", "")).strip()
        arousal = str(row.get("emotional_arousal_level", "")).strip()
        if valence != _valence(emotion):
            issues.append({"sample_id": sample_id, "field": "emotional_valence", "actual": valence, "expected": _valence(emotion)})
        if arousal != _arousal(emotion):
            issues.append({"sample_id": sample_id, "field": "emotional_arousal_level", "actual": arousal, "expected": _arousal(emotion)})
    return issues


def _unsafe_text_count(rows: list[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        text = " ".join(str(row.get(field, "")) for field in TEXT_FIELDS).lower()
        if any(pattern in text for pattern in EVALUATOR_MANIPULATION_PATTERNS):
            count += 1
    return count


def _label_triplet(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("emotion", "")).strip(),
        str(row.get("emotional_valence", "")).strip(),
        str(row.get("emotional_arousal_level", "")).strip(),
    )


def _canonical_emotion(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "contentment": "content",
        "relaxed": "calm",
        "joy": "glad",
        "joyful": "glad",
    }
    text = aliases.get(text, text)
    return text if text in TRACK2_EMOTIONS else ""


def _same_quadrant(first: str, second: str) -> bool:
    return _valence(first) == _valence(second) and _arousal(first) == _arousal(second)


def _valence(emotion: str) -> str:
    return "Negative" if emotion in NEGATIVE_EMOTIONS else "Positive"


def _arousal(emotion: str) -> str:
    return "High" if emotion in HIGH_AROUSAL_EMOTIONS else "Low"


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


def _load_evidence_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"evidence JSON must be a list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _reject_formal_submission_path(path: Path) -> None:
    if path.name.casefold() in {name.casefold() for name in FORMAL_SUBMISSION_NAMES} and path.parent.name == "submissions":
        raise ValueError(f"refusing to write root formal submission path: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_zip_payload(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
    info = zipfile.ZipInfo("submission.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, payload)


def _render_candidate_md(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Track2 v27 {report['profile']} Candidate",
        "",
        f"- accepted_label_changes: `{report['accepted_label_changes']}`",
        f"- transition_counts: `{report['transition_counts']}`",
        f"- tier_counts: `{report['tier_counts']}`",
        f"- cross_quadrant_change_count: `{report['cross_quadrant_change_count']}`",
        f"- top_emotion: `{report['top_emotion']}` ({float(report['top_emotion_share']):.1%})",
        f"- missing_emotions: `{', '.join(report['missing_emotions']) or 'none'}`",
        f"- label_consistency_issue_count: `{report['label_consistency_issue_count']}`",
        f"- unsafe_text_count: `{report['unsafe_text_count']}`",
        "",
        "## Accepted Changes",
        "",
    ]
    for item in report.get("accepted_changes", [])[:160]:
        lines.append(f"- `{item['sample_id']}`: {item['transition']} `{item['v27_tier']}` score={float(item['v27_score']):.3f}")
    lines.append("")
    return "\n".join(lines)


def _render_suite_md(summary: Mapping[str, Any]) -> str:
    best = summary.get("best") or {}
    lines = [
        "# Track2 v27 gold-like ledger report",
        "",
        "## 结论",
        "",
        f"- decision: `{summary.get('decision', '')}`",
        f"- best_profile: `{summary.get('best_profile', '')}`",
        f"- best_overall_expected: `{float(best.get('overall_expected', 0.0) or 0.0):.6f}`",
        f"- best_classification_expected: `{float(best.get('classification_expected', 0.0) or 0.0):.6f}`",
        f"- best_description_expected: `{float(best.get('description_expected', 0.0) or 0.0):.6f}`",
        f"- accepted_ledger_count: `{summary.get('accepted_ledger_count', 0)}`",
        f"- ledger_counts: `{summary.get('ledger_counts', {})}`",
        "",
        "## Profile Scoreboard",
        "",
        "| profile | decision | overall | class | desc | changes | top | top share | transitions | reasons |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for row in sorted(summary.get("profiles", []), key=lambda item: str(item.get("profile", ""))):
        lines.append(
            "| {profile} | {decision} | {overall:.6f} | {classification:.6f} | {description:.6f} | "
            "{changes} | {top} | {share:.3f} | {transitions} | {reasons} |".format(
                profile=row.get("profile", ""),
                decision=row.get("gate_decision", ""),
                overall=float(row.get("overall_expected", 0.0) or 0.0),
                classification=float(row.get("classification_expected", 0.0) or 0.0),
                description=float(row.get("description_expected", 0.0) or 0.0),
                changes=row.get("accepted_label_changes", 0),
                top=row.get("top_emotion", ""),
                share=float(row.get("top_emotion_share", 0.0) or 0.0),
                transitions=row.get("transition_counts", ""),
                reasons=row.get("gate_reasons", ""),
            )
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- `hold_no_submit` 表示本地 0.89 gate 未过，不建议消耗最后一次 Codabench 提交。",
            "- 这个报告使用 v23 anchor-calibrated proxy；未提交候选仍不是 hidden gold reconstruction。",
            "",
        ]
    )
    return "\n".join(lines)
