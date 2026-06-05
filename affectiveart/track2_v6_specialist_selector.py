from __future__ import annotations

import csv
import html
import json
import zipfile
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_EMOTIONS, TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_audit import compute_track2_distribution, strict_track2_label_issues
from affectiveart.track2_local_shadow_evaluator import write_shadow_evaluator_outputs
from affectiveart.track2_moe_specialist_ensemble import expected_label_for_emotion


ACCEPT_SAFE = "accept_safe"
ACCEPT_CROSS_MICRO = "accept_cross_micro"
HOLD_REVIEW = "hold_review"
BLOCK = "block"

V6_LADDERS = ("v6_safe_sameq", "v6_cross_micro", "v6_desc_plus")
DEFAULT_OUTPUT_NAMES = {
    "v6_safe_sameq": "track2_submission_v6_safe_sameq_candidate",
    "v6_cross_micro": "track2_submission_v6_cross_micro_candidate",
    "v6_desc_plus": "track2_submission_v6_desc_plus_candidate",
}
FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}


@dataclass(frozen=True)
class V6Delta:
    sample_id: str
    current_emotion: str
    proposed_emotion: str
    current_valence: str = ""
    current_arousal: str = ""
    proposed_valence: str = ""
    proposed_arousal: str = ""
    transition: str = ""
    same_quadrant: bool = False
    cross_quadrant_risk: bool = False
    supporting_source_count: int = 0
    supporting_family_count: int = 0
    supporting_sources: tuple[str, ...] = ()
    supporting_families: tuple[str, ...] = ()
    evidence_score: float = 0.0
    gemini35_prefers_proposed: bool = False
    gemini35_fit_margin: float = 0.0
    gemini35_objection: bool = False
    vulca_objection: bool = False
    public_reference_support: bool = False
    public_reference_contradiction: bool = False
    human_confirmed: bool = False
    hard96_net_gain: int = 0
    hard96_net_loss: int = 0
    malformed: bool = False
    malformed_reasons: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectorThresholds:
    sameq_min_score: float = 8.0
    sameq_min_sources: int = 3
    sameq_min_families: int = 2
    cross_min_score: float = 9.0
    cross_min_sources: int = 4
    cross_min_families: int = 3
    cross_min_fit_margin: float = 0.30
    cross_min_hard96_gain: int = 1
    max_cross_micro: int = 3


@dataclass(frozen=True)
class SelectorDecision:
    sample_id: str
    proposed_emotion: str
    transition: str
    decision: str
    ladder: str
    reason_codes: tuple[str, ...]
    evidence_score: float
    same_quadrant: bool
    cross_quadrant_risk: bool
    proposed_valence: str
    proposed_arousal: str
    source_count: int
    family_count: int


def load_evidence_matrix(path: str | Path) -> list[V6Delta]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [_delta_from_row(row) for row in csv.DictReader(handle)]


def select_v6_deltas(
    deltas: list[V6Delta],
    thresholds: SelectorThresholds | None = None,
) -> list[SelectorDecision]:
    thresholds = thresholds or SelectorThresholds()
    decisions = [_select_one(delta, thresholds) for delta in deltas]

    accepted_by_sample: dict[str, list[int]] = {}
    for index, decision in enumerate(decisions):
        if decision.decision in {ACCEPT_SAFE, ACCEPT_CROSS_MICRO}:
            accepted_by_sample.setdefault(decision.sample_id, []).append(index)

    for indexes in accepted_by_sample.values():
        if len(indexes) <= 1:
            continue
        for index in indexes:
            decision = decisions[index]
            reasons = _append_reason(decision.reason_codes, "proposal_conflict")
            decisions[index] = replace(
                decision,
                decision=HOLD_REVIEW,
                ladder="",
                reason_codes=reasons,
            )

    return decisions


def build_candidate_rows(
    baseline_rows: list[dict[str, Any]],
    decisions: list[SelectorDecision],
    *,
    ladder: str,
    thresholds: SelectorThresholds | None = None,
) -> list[dict[str, Any]]:
    selected = _selected_decisions_for_ladder(decisions, ladder, thresholds or SelectorThresholds())
    output: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        row = dict(baseline)
        decision = selected.get(str(row.get("sample_id", "")).strip())
        if decision is not None:
            row["emotion"] = decision.proposed_emotion
            row["emotional_valence"] = decision.proposed_valence
            row["emotional_arousal_level"] = decision.proposed_arousal
        output.append({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})
    return output


def write_v6_outputs(
    *,
    baseline_json: str | Path,
    evidence_matrix: str | Path,
    out_dir: str | Path,
    submission_dir: str | Path,
    output_names: dict[str, str] | None = None,
    expected_row_count: int = 1000,
    require_all_emotions: bool | None = None,
    thresholds: SelectorThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SelectorThresholds()
    baseline_json = Path(baseline_json)
    evidence_matrix = Path(evidence_matrix)
    out_dir = Path(out_dir)
    submission_dir = Path(submission_dir)
    names = {**DEFAULT_OUTPUT_NAMES, **(output_names or {})}
    submission_root = submission_dir.resolve()
    candidate_paths = {
        ladder: _candidate_paths_for_name(names[ladder], submission_dir, submission_root)
        for ladder in V6_LADDERS
    }
    for json_path, zip_path in candidate_paths.values():
        _assert_safe_candidate_path(json_path)
        _assert_safe_candidate_path(zip_path)

    baseline_rows = _load_json_rows(baseline_json)
    deltas = load_evidence_matrix(evidence_matrix)
    decisions = select_v6_deltas(deltas, thresholds)

    out_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    html_dir = out_dir / "html_review"
    html_dir.mkdir(parents=True, exist_ok=True)

    normalized_rows = [_delta_to_dict(delta) for delta in deltas]
    decision_rows = [_decision_to_dict(decision) for decision in decisions]
    normalized_evidence_json = out_dir / "normalized_evidence.json"
    normalized_evidence_csv = out_dir / "normalized_evidence.csv"
    selector_decisions_json = out_dir / "selector_decisions.json"
    selector_decisions_csv = out_dir / "selector_decisions.csv"
    _write_json(normalized_evidence_json, normalized_rows)
    _write_csv(normalized_evidence_csv, normalized_rows)
    _write_json(selector_decisions_json, decision_rows)
    _write_csv(selector_decisions_csv, decision_rows)

    candidates: dict[str, Any] = {}
    shadow_candidates: list[dict[str, Any]] = []
    for ladder in V6_LADDERS:
        json_path, zip_path = candidate_paths[ladder]
        rows = build_candidate_rows(baseline_rows, decisions, ladder=ladder, thresholds=thresholds)
        _write_json(json_path, rows)
        _write_zip(zip_path, json_path)
        candidate_report = _candidate_report(
            ladder=ladder,
            json_path=json_path,
            zip_path=zip_path,
            rows=rows,
            baseline_rows=baseline_rows,
            decisions=decisions,
            thresholds=thresholds,
        )
        _write_json(out_dir / f"candidate_report_{ladder}.json", candidate_report)
        (out_dir / f"candidate_report_{ladder}.md").write_text(
            _render_candidate_report_markdown(candidate_report),
            encoding="utf-8",
        )
        candidates[ladder] = candidate_report
        shadow_candidates.append({"name": ladder, "json": json_path})

    stability_report_json = out_dir / "hard96_stability_report.json"
    stability = _build_stability_report(decisions, thresholds)
    _write_json(stability_report_json, stability)
    (out_dir / "hard96_stability_report.md").write_text(
        _render_stability_markdown(stability),
        encoding="utf-8",
    )

    shadow_report = write_shadow_evaluator_outputs(
        baseline_json=baseline_json,
        candidates=shadow_candidates,
        out_dir=out_dir / "shadow_eval",
        expected_row_count=expected_row_count,
        require_all_emotions=require_all_emotions,
    )
    decision = _top_level_decision(stability, shadow_report)
    html_review_path = html_dir / "track2_v6_specialist_selector_review.html"
    summary = {
        "method": "track2_v6_risk_calibrated_specialist_selector",
        "decision": decision,
        "baseline_json": str(baseline_json),
        "evidence_matrix": str(evidence_matrix),
        "out_dir": str(out_dir),
        "evidence_row_count": len(deltas),
        "selector_decision_count": len(decisions),
        "candidates": candidates,
        "normalized_evidence_json": str(normalized_evidence_json),
        "normalized_evidence_csv": str(normalized_evidence_csv),
        "selector_decisions_json": str(selector_decisions_json),
        "selector_decisions_csv": str(selector_decisions_csv),
        "stability_report_json": str(stability_report_json),
        "shadow_report_json": str(out_dir / "shadow_eval" / "shadow_score_report.json"),
        "html_review_path": str(html_review_path),
        "formal_submission_overwritten": False,
    }
    _write_json(out_dir / "v6_summary.json", summary)
    (out_dir / "v6_summary.md").write_text(
        _render_summary_markdown(summary, shadow_report),
        encoding="utf-8",
    )
    html_review_path.write_text(_render_html_review(summary, decision_rows), encoding="utf-8")
    return summary


def _delta_from_row(row: dict[str, Any]) -> V6Delta:
    raw = dict(row)
    sample_id = _text(row.get("sample_id"))
    current_emotion = _emotion(row.get("current_emotion"))
    proposed_emotion = _emotion(row.get("proposed_emotion"))
    supporting_source_count = _int(row.get("supporting_source_count"))
    supporting_family_count = _int(row.get("supporting_family_count"))
    supporting_sources = _split_list(row.get("supporting_sources"))
    supporting_families = _split_list(row.get("supporting_families"))
    malformed_reasons = _malformed_reasons(
        sample_id=sample_id,
        current_emotion=current_emotion,
        proposed_emotion=proposed_emotion,
        supporting_source_count=supporting_source_count,
        supporting_family_count=supporting_family_count,
        supporting_sources=supporting_sources,
        supporting_families=supporting_families,
    )

    current_valence, current_arousal = _labels_for(
        current_emotion,
        row.get("current_valence"),
        row.get("current_arousal"),
    )
    proposed_valence, proposed_arousal = _labels_for(
        proposed_emotion,
        row.get("proposed_valence"),
        row.get("proposed_arousal"),
    )
    same_quadrant = _optional_bool(row.get("same_quadrant"))
    if same_quadrant is None:
        same_quadrant = bool(
            current_valence
            and current_arousal
            and current_valence == proposed_valence
            and current_arousal == proposed_arousal
        )

    transition = ""
    if current_emotion or proposed_emotion:
        transition = f"{current_emotion}->{proposed_emotion}"

    return V6Delta(
        sample_id=sample_id,
        current_emotion=current_emotion,
        current_valence=current_valence,
        current_arousal=current_arousal,
        proposed_emotion=proposed_emotion,
        proposed_valence=proposed_valence,
        proposed_arousal=proposed_arousal,
        transition=transition,
        same_quadrant=same_quadrant,
        cross_quadrant_risk=not same_quadrant,
        supporting_source_count=supporting_source_count,
        supporting_family_count=supporting_family_count,
        supporting_sources=supporting_sources,
        supporting_families=supporting_families,
        evidence_score=_float(row.get("evidence_score")),
        gemini35_prefers_proposed=_bool(row.get("gemini35_prefers_proposed")),
        gemini35_fit_margin=_float(row.get("gemini35_fit_margin")),
        gemini35_objection=_bool(row.get("gemini35_objection")),
        vulca_objection=_bool(row.get("vulca_objection")),
        public_reference_support=_bool(row.get("public_reference_support")),
        public_reference_contradiction=_bool(row.get("public_reference_contradiction")),
        human_confirmed=_bool(row.get("human_confirmed")),
        hard96_net_gain=_int(row.get("hard96_net_gain")),
        hard96_net_loss=_int(row.get("hard96_net_loss")),
        malformed=bool(malformed_reasons),
        malformed_reasons=malformed_reasons,
        raw=raw,
    )


def _select_one(delta: V6Delta, thresholds: SelectorThresholds) -> SelectorDecision:
    reasons: list[str] = []
    decision = HOLD_REVIEW
    ladder = ""

    if delta.malformed:
        reasons.extend(delta.malformed_reasons or ("missing_required_delta_field",))
    if delta.public_reference_contradiction:
        reasons.append("public_reference_contradiction")
    if delta.gemini35_objection:
        reasons.append("gemini35_objection")
    if delta.vulca_objection:
        reasons.append("vulca_objection")
    if delta.hard96_net_loss > 0:
        reasons.append("hard96_net_loss")
    if _dangerous_positive_low_to_negative(delta):
        reasons.append("dangerous_positive_low_to_negative")

    if not reasons:
        if delta.same_quadrant:
            reasons.extend(_sameq_threshold_reasons(delta, thresholds))
            if not reasons:
                decision = ACCEPT_SAFE
                ladder = "v6_safe_sameq"
                reasons.append("same_quadrant_multi_source_support")
        else:
            reasons.extend(_cross_threshold_reasons(delta, thresholds))
            if not reasons:
                decision = ACCEPT_CROSS_MICRO
                ladder = "v6_cross_micro"
                reasons.append("cross_quadrant_arbitrated_micro_gain")
    elif not delta.same_quadrant and not delta.malformed:
        for reason in _cross_threshold_reasons(delta, thresholds):
            if reason not in reasons:
                reasons.append(reason)

    return SelectorDecision(
        sample_id=delta.sample_id,
        proposed_emotion=delta.proposed_emotion,
        transition=delta.transition,
        decision=decision,
        ladder=ladder,
        reason_codes=tuple(reasons or ["insufficient_evidence"]),
        evidence_score=delta.evidence_score,
        same_quadrant=delta.same_quadrant,
        cross_quadrant_risk=delta.cross_quadrant_risk,
        proposed_valence=delta.proposed_valence,
        proposed_arousal=delta.proposed_arousal,
        source_count=delta.supporting_source_count,
        family_count=delta.supporting_family_count,
    )


def _sameq_threshold_reasons(delta: V6Delta, thresholds: SelectorThresholds) -> list[str]:
    reasons: list[str] = []
    if delta.evidence_score < thresholds.sameq_min_score:
        reasons.append("low_evidence_score")
    if delta.supporting_source_count < thresholds.sameq_min_sources:
        reasons.append("insufficient_source_count")
    if delta.supporting_family_count < thresholds.sameq_min_families:
        reasons.append("insufficient_family_count")
    return reasons


def _cross_threshold_reasons(delta: V6Delta, thresholds: SelectorThresholds) -> list[str]:
    reasons: list[str] = []
    if not delta.gemini35_prefers_proposed or delta.gemini35_fit_margin < thresholds.cross_min_fit_margin:
        reasons.append("missing_cross_arbitration")
    if delta.hard96_net_gain < thresholds.cross_min_hard96_gain:
        reasons.append("missing_hard96_gain")
    if delta.evidence_score < thresholds.cross_min_score:
        reasons.append("low_evidence_score")
    if delta.supporting_source_count < thresholds.cross_min_sources:
        reasons.append("insufficient_source_count")
    if delta.supporting_family_count < thresholds.cross_min_families:
        reasons.append("insufficient_family_count")
    return reasons


def _dangerous_positive_low_to_negative(delta: V6Delta) -> bool:
    return (
        delta.current_emotion in {"calm", "content", "glad"}
        and delta.current_valence == "Positive"
        and delta.current_arousal == "Low"
        and delta.proposed_valence == "Negative"
    )


def _selected_decisions_for_ladder(
    decisions: list[SelectorDecision],
    ladder: str,
    thresholds: SelectorThresholds,
) -> dict[str, SelectorDecision]:
    if ladder not in V6_LADDERS:
        raise ValueError(f"unknown v6 ladder: {ladder}")

    safe_decisions = [
        decision
        for decision in decisions
        if decision.decision == ACCEPT_SAFE and decision.ladder == "v6_safe_sameq"
    ]
    if ladder == "v6_safe_sameq":
        return {decision.sample_id: decision for decision in safe_decisions}

    cross_decisions = _capped_cross_micro_decisions(decisions, thresholds.max_cross_micro)
    return {
        decision.sample_id: decision
        for decision in (*safe_decisions, *cross_decisions)
    }


def _capped_cross_micro_decisions(
    decisions: list[SelectorDecision],
    limit: int,
) -> list[SelectorDecision]:
    accepted = [
        decision
        for decision in decisions
        if decision.decision == ACCEPT_CROSS_MICRO and decision.ladder == "v6_cross_micro"
    ]
    return sorted(
        accepted,
        key=lambda decision: (
            -decision.evidence_score,
            decision.sample_id,
            decision.proposed_emotion,
        ),
    )[: max(0, limit)]


def _candidate_report(
    *,
    ladder: str,
    json_path: Path,
    zip_path: Path,
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    decisions: list[SelectorDecision],
    thresholds: SelectorThresholds,
) -> dict[str, Any]:
    selected = _selected_decisions_for_ladder(decisions, ladder, thresholds)
    changed = [
        {
            "sample_id": decision.sample_id,
            "transition": decision.transition,
            "decision": decision.decision,
            "evidence_score": decision.evidence_score,
            "reason_codes": list(decision.reason_codes),
        }
        for decision in selected.values()
    ]
    distribution = compute_track2_distribution(rows)
    label_issue_count = sum(len(strict_track2_label_issues(row)) for row in rows)
    return {
        "ladder": ladder,
        "json": str(json_path),
        "zip": str(zip_path),
        "changed_rows": len(_changed_rows(baseline_rows, rows)),
        "selected_changes": changed,
        "distribution": distribution,
        "label_consistency_issue_count": label_issue_count,
        "missing_emotions": distribution.get("missing_emotions", []),
        "formal_submission_overwritten": False,
    }


def _build_stability_report(
    decisions: list[SelectorDecision],
    thresholds: SelectorThresholds,
) -> dict[str, Any]:
    accepted = [item for item in decisions if item.decision in {ACCEPT_SAFE, ACCEPT_CROSS_MICRO}]
    safe_sameq = [item for item in accepted if item.decision == ACCEPT_SAFE]
    raw_cross = [item for item in accepted if item.decision == ACCEPT_CROSS_MICRO]
    applied_cross = _capped_cross_micro_decisions(decisions, thresholds.max_cross_micro)
    applied = [*safe_sameq, *applied_cross]
    raw_transition_counts = Counter(item.transition for item in accepted)
    applied_transition_counts = Counter(item.transition for item in applied)
    issues: list[dict[str, Any]] = []
    info: list[dict[str, Any]] = []
    if len(applied_cross) > thresholds.max_cross_micro:
        issues.append({"code": "too_many_cross_micro", "count": len(applied_cross)})
    if len(raw_cross) > len(applied_cross):
        info.append(
            {
                "code": "raw_cross_micro_capped",
                "raw_count": len(raw_cross),
                "applied_count": len(applied_cross),
                "cap": thresholds.max_cross_micro,
            }
        )
    if applied:
        top_transition, top_count = applied_transition_counts.most_common(1)[0]
        if top_count >= 5 and top_count / len(applied) > 0.55:
            issues.append(
                {
                    "code": "transition_concentration",
                    "transition": top_transition,
                    "count": top_count,
                }
            )
    return {
        "method": "track2_v6_rule_stability_summary",
        "passed": not issues,
        "accepted_count": len(applied),
        "raw_accepted_count": len(accepted),
        "applied_accepted_count": len(applied),
        "safe_sameq_count": len(safe_sameq),
        "cross_micro_count": len(applied_cross),
        "raw_cross_micro_count": len(raw_cross),
        "applied_cross_micro_count": len(applied_cross),
        "hold_count": sum(1 for item in decisions if item.decision == HOLD_REVIEW),
        "block_count": sum(1 for item in decisions if item.decision == BLOCK),
        "transition_counts": dict(sorted(applied_transition_counts.items())),
        "raw_transition_counts": dict(sorted(raw_transition_counts.items())),
        "applied_transition_counts": dict(sorted(applied_transition_counts.items())),
        "issues": issues,
        "info": info,
    }


def _top_level_decision(stability: dict[str, Any], shadow_report: dict[str, Any]) -> str:
    if not stability.get("passed", False):
        return "recommend_hold"
    ranking = shadow_report.get("ranking", [])
    if not ranking:
        return "invalid"
    top = ranking[0]
    if top.get("decision") != "recommend_submit":
        return "recommend_hold"
    if float(top.get("overall_expected", 0.0)) >= 0.8639085 and float(top.get("overall_lower", 0.0)) >= 0.796000:
        return "recommend_submit"
    return "recommend_hold"


def _changed_rows(
    baseline_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_id = {str(row.get("sample_id", "")): row for row in baseline_rows}
    changed: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        before = baseline_by_id.get(sample_id)
        if not before:
            continue
        before_labels = (
            before.get("emotion"),
            before.get("emotional_valence"),
            before.get("emotional_arousal_level"),
        )
        after_labels = (
            row.get("emotion"),
            row.get("emotional_valence"),
            row.get("emotional_arousal_level"),
        )
        if before_labels != after_labels:
            changed.append({"sample_id": sample_id, "before": before, "after": row})
    return changed


def _delta_to_dict(delta: V6Delta) -> dict[str, Any]:
    return {
        "sample_id": delta.sample_id,
        "current_emotion": delta.current_emotion,
        "current_valence": delta.current_valence,
        "current_arousal": delta.current_arousal,
        "proposed_emotion": delta.proposed_emotion,
        "proposed_valence": delta.proposed_valence,
        "proposed_arousal": delta.proposed_arousal,
        "transition": delta.transition,
        "same_quadrant": delta.same_quadrant,
        "cross_quadrant_risk": delta.cross_quadrant_risk,
        "supporting_source_count": delta.supporting_source_count,
        "supporting_family_count": delta.supporting_family_count,
        "supporting_sources": ";".join(delta.supporting_sources),
        "supporting_families": ";".join(delta.supporting_families),
        "evidence_score": delta.evidence_score,
        "gemini35_prefers_proposed": delta.gemini35_prefers_proposed,
        "gemini35_fit_margin": delta.gemini35_fit_margin,
        "gemini35_objection": delta.gemini35_objection,
        "vulca_objection": delta.vulca_objection,
        "public_reference_support": delta.public_reference_support,
        "public_reference_contradiction": delta.public_reference_contradiction,
        "human_confirmed": delta.human_confirmed,
        "hard96_net_gain": delta.hard96_net_gain,
        "hard96_net_loss": delta.hard96_net_loss,
        "malformed": delta.malformed,
        "malformed_reasons": ";".join(delta.malformed_reasons),
    }


def _decision_to_dict(decision: SelectorDecision) -> dict[str, Any]:
    return {
        "sample_id": decision.sample_id,
        "proposed_emotion": decision.proposed_emotion,
        "transition": decision.transition,
        "decision": decision.decision,
        "ladder": decision.ladder,
        "reason_codes": ";".join(decision.reason_codes),
        "evidence_score": decision.evidence_score,
        "same_quadrant": decision.same_quadrant,
        "cross_quadrant_risk": decision.cross_quadrant_risk,
        "proposed_valence": decision.proposed_valence,
        "proposed_arousal": decision.proposed_arousal,
        "source_count": decision.source_count,
        "family_count": decision.family_count,
    }


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected Track2 JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _candidate_paths_for_name(
    name: str,
    submission_dir: Path,
    submission_root: Path,
) -> tuple[Path, Path]:
    _assert_safe_output_name(name)
    json_path = submission_dir / f"{name}.json"
    zip_path = submission_dir / f"{name}.zip"
    _assert_under_directory(json_path, submission_root)
    _assert_under_directory(zip_path, submission_root)
    return json_path, zip_path


def _assert_safe_output_name(name: str) -> None:
    text = str(name)
    if not text:
        raise ValueError("candidate output name must be non-empty")
    if Path(text).is_absolute() or "/" in text or "\\" in text or ".." in text:
        raise ValueError(f"candidate output name must be a safe basename: {name}")


def _assert_under_directory(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"candidate output path escapes submission directory: {path}") from exc


def _assert_safe_candidate_path(path: Path) -> None:
    if path.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError(f"refusing to write formal Track2 submission path: {path}")
    if path.suffix not in {".json", ".zip"}:
        raise ValueError(f"candidate output must be JSON or ZIP: {path}")
    if not path.stem.startswith("track2_submission_v6_") or not path.stem.endswith("_candidate"):
        raise ValueError(f"candidate output must be a v6 side-path candidate: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _write_zip(zip_path: Path, json_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, "submission.json")


def _render_candidate_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Track2 {report['ladder']} Candidate Report",
        "",
        f"- JSON: `{report['json']}`",
        f"- ZIP: `{report['zip']}`",
        f"- Changed rows: {report['changed_rows']}",
        f"- Label consistency issues: {report['label_consistency_issue_count']}",
        f"- Missing emotions: {', '.join(report.get('missing_emotions') or []) or 'none'}",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        "",
        "## Selected Changes",
        "",
    ]
    for item in report.get("selected_changes", []):
        lines.append(
            f"- {item['sample_id']}: {item['transition']}; "
            f"score={item['evidence_score']}; reasons={','.join(item['reason_codes'])}"
        )
    if not report.get("selected_changes"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _render_stability_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 V6 Stability Report",
        "",
        f"- Passed: {report['passed']}",
        f"- Accepted count: {report['accepted_count']}",
        f"- Safe same-quadrant count: {report['safe_sameq_count']}",
        f"- Cross micro count: {report['cross_micro_count']}",
        f"- Hold count: {report['hold_count']}",
        f"- Block count: {report['block_count']}",
        "",
        "## Issues",
        "",
    ]
    for issue in report.get("issues", []):
        lines.append(f"- {issue['code']}: `{json.dumps(issue, ensure_ascii=False, sort_keys=True)}`")
    if not report.get("issues"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _render_summary_markdown(summary: dict[str, Any], shadow_report: dict[str, Any]) -> str:
    lines = [
        "# Track2 V6 Specialist Selector Summary",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Baseline JSON: `{summary['baseline_json']}`",
        f"- Evidence rows: {summary['evidence_row_count']}",
        f"- Selector decisions: {summary['selector_decision_count']}",
        f"- Formal submission overwritten: {summary['formal_submission_overwritten']}",
        "",
        "## Shadow Ranking",
        "",
        "| rank | candidate | decision | overall lower | overall expected | changes | cross |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(shadow_report.get("ranking", []), start=1):
        lines.append(
            f"| {index} | {item['candidate_name']} | {item['decision']} | "
            f"{float(item['overall_lower']):.6f} | {float(item['overall_expected']):.6f} | "
            f"{int(item['changed_rows'])} | {int(item['cross_quadrant_changes'])} |"
        )
    return "\n".join(lines) + "\n"


def _render_html_review(summary: dict[str, Any], decisions: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item['sample_id']))}</td>"
        f"<td>{html.escape(str(item['transition']))}</td>"
        f"<td>{html.escape(str(item['decision']))}</td>"
        f"<td>{html.escape(str(item['evidence_score']))}</td>"
        f"<td>{html.escape(str(item['reason_codes']))}</td>"
        "</tr>"
        for item in decisions[:500]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Track2 V6 Specialist Selector</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 7px; text-align: left; }}
    th {{ background: #eef2f6; }}
    .decision {{ padding: 12px; background: #eef7ee; border: 1px solid #b9d8b9; }}
  </style>
</head>
<body>
  <h1>Track2 V6 Specialist Selector</h1>
  <p class="decision">Decision: <code>{html.escape(str(summary['decision']))}</code></p>
  <p>Baseline: <code>{html.escape(str(summary['baseline_json']))}</code></p>
  <h2>Selector Decisions</h2>
  <table>
    <thead><tr><th>sample</th><th>transition</th><th>decision</th><th>score</th><th>reasons</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def _malformed_reasons(
    *,
    sample_id: str,
    current_emotion: str,
    proposed_emotion: str,
    supporting_source_count: int,
    supporting_family_count: int,
    supporting_sources: tuple[str, ...],
    supporting_families: tuple[str, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not sample_id or not current_emotion or not proposed_emotion:
        reasons.append("missing_required_delta_field")
    if current_emotion and current_emotion not in TRACK2_JSON_EMOTIONS:
        reasons.append("invalid_track2_emotion")
    if proposed_emotion and proposed_emotion not in TRACK2_JSON_EMOTIONS:
        reasons.append("invalid_track2_emotion")
    if (supporting_source_count > 0 and not supporting_sources) or (
        supporting_family_count > 0 and not supporting_families
    ):
        reasons.append("missing_support_traceability")
    if supporting_source_count > len(supporting_sources) or supporting_family_count > len(
        supporting_families
    ):
        reasons.append("support_count_traceability_mismatch")
    return tuple(dict.fromkeys(reasons))


def _labels_for(emotion: str, valence: Any, arousal: Any) -> tuple[str, str]:
    clean_valence = _text(valence)
    clean_arousal = _text(arousal)
    if clean_valence and clean_arousal:
        return clean_valence, clean_arousal
    if not emotion:
        return clean_valence, clean_arousal
    try:
        expected_valence, expected_arousal = expected_label_for_emotion(emotion)
    except ValueError:
        return clean_valence, clean_arousal
    return clean_valence or expected_valence, clean_arousal or expected_arousal


def _text(value: Any) -> str:
    return str(value or "").strip()


def _emotion(value: Any) -> str:
    return _text(value).lower()


def _optional_bool(value: Any) -> bool | None:
    text = _text(value).lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _bool(value: Any) -> bool:
    return _optional_bool(value) is True


def _int(value: Any) -> int:
    try:
        return int(float(_text(value)))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(_text(value))
    except (TypeError, ValueError):
        return 0.0


def _split_list(value: Any) -> tuple[str, ...]:
    text = _text(value)
    if not text:
        return ()
    return tuple(item.strip() for item in text.replace(",", ";").split(";") if item.strip())


def _append_reason(reasons: tuple[str, ...], reason: str) -> tuple[str, ...]:
    if reason in reasons:
        return reasons
    return (*reasons, reason)
