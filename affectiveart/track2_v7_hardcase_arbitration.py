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


ACCEPT_STRICT = "accept_strict"
ACCEPT_BORDERLINE = "accept_borderline_diagnostic"
HOLD = "hold"

V7_LADDERS = ("v7_cross1", "v7_cross3", "v7_hardcase_push")
DEFAULT_OUTPUT_NAMES = {
    "v7_cross1": "track2_submission_v7_cross1_candidate",
    "v7_cross3": "track2_submission_v7_cross3_candidate",
    "v7_hardcase_push": "track2_submission_v7_hardcase_push_candidate",
}
FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}
TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")


@dataclass(frozen=True)
class V7Thresholds:
    strict_min_confidence: float = 0.75
    strict_min_fit_margin: float = 0.30
    strict_min_evidence_score: float = 9.0
    strict_min_sources: int = 4
    strict_min_families: int = 3
    borderline_min_confidence: float = 0.70
    borderline_min_fit_margin: float = 0.18
    borderline_min_evidence_score: float = 7.0
    borderline_min_sources: int = 6
    borderline_min_families: int = 3
    cross1_cap: int = 1
    cross3_cap: int = 3
    diagnostic_cap: int = 5


@dataclass(frozen=True)
class V7Decision:
    sample_id: str
    transition: str
    current_emotion: str
    proposed_emotion: str
    proposed_valence: str
    proposed_arousal: str
    decision: str
    ladder: str
    reason_codes: tuple[str, ...]
    confidence: float
    fit_margin: float
    evidence_score: float
    supporting_source_count: int
    supporting_family_count: int
    text_override: dict[str, Any] = field(default_factory=dict)


def load_evidence_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"expected evidence JSON list: {path}")
        return [dict(row) for row in payload if isinstance(row, dict)]
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_arbitration_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected arbitration JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def select_v7_arbitrations(
    *,
    baseline_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    arbitration_rows: list[dict[str, Any]],
    text_override_rows: list[dict[str, Any]] | None = None,
    thresholds: V7Thresholds | None = None,
) -> list[V7Decision]:
    thresholds = thresholds or V7Thresholds()
    baseline_by_id = _index_rows(baseline_rows)
    evidence_by_key = _evidence_index(evidence_rows)
    text_by_id = _index_rows(text_override_rows or [])

    decisions = [
        _decision_from_arbitration(
            row,
            baseline_by_id=baseline_by_id,
            evidence_by_key=evidence_by_key,
            text_by_id=text_by_id,
            thresholds=thresholds,
        )
        for row in arbitration_rows
    ]
    return _hold_conflicting_accepts(decisions)


def build_candidate_rows(
    baseline_rows: list[dict[str, Any]],
    decisions: list[V7Decision],
    *,
    ladder: str,
    thresholds: V7Thresholds | None = None,
) -> list[dict[str, Any]]:
    selected = _selected_decisions(decisions, ladder, thresholds or V7Thresholds())
    output: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        row = dict(baseline)
        decision = selected.get(str(row.get("sample_id", "")))
        if decision is not None:
            row["emotion"] = decision.proposed_emotion
            row["emotional_valence"] = decision.proposed_valence
            row["emotional_arousal_level"] = decision.proposed_arousal
            for field_name in TEXT_FIELDS:
                if field_name in decision.text_override:
                    row[field_name] = decision.text_override[field_name]
        output.append({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})
    return output


def write_v7_outputs(
    *,
    baseline_json: str | Path,
    evidence_matrix: str | Path,
    arbitration_json: str | Path,
    text_override_json: str | Path,
    out_dir: str | Path,
    submission_dir: str | Path,
    shadow_baseline_json: str | Path | None = None,
    output_names: dict[str, str] | None = None,
    expected_row_count: int = 1000,
    require_all_emotions: bool | None = None,
    thresholds: V7Thresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or V7Thresholds()
    baseline_json = Path(baseline_json)
    out_dir = Path(out_dir)
    submission_dir = Path(submission_dir)
    shadow_baseline_json = Path(shadow_baseline_json) if shadow_baseline_json else baseline_json
    output_names = {**DEFAULT_OUTPUT_NAMES, **(output_names or {})}
    submission_root = submission_dir.resolve()

    baseline_rows = _load_json_rows(baseline_json)
    evidence_rows = load_evidence_rows(evidence_matrix)
    arbitration_rows = load_arbitration_rows(arbitration_json)
    text_override_rows = _load_json_rows(Path(text_override_json))
    decisions = select_v7_arbitrations(
        baseline_rows=baseline_rows,
        evidence_rows=evidence_rows,
        arbitration_rows=arbitration_rows,
        text_override_rows=text_override_rows,
        thresholds=thresholds,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    html_dir = out_dir / "html_review"
    html_dir.mkdir(parents=True, exist_ok=True)

    decision_rows = [_decision_to_dict(decision) for decision in decisions]
    decisions_json = out_dir / "v7_arbitration_decisions.json"
    decisions_csv = out_dir / "v7_arbitration_decisions.csv"
    _write_json(decisions_json, decision_rows)
    _write_csv(decisions_csv, decision_rows)

    candidates: dict[str, Any] = {}
    shadow_candidates: list[dict[str, Any]] = []
    for ladder in V7_LADDERS:
        json_path, zip_path = _candidate_paths_for_name(
            output_names[ladder],
            submission_dir,
            submission_root,
        )
        rows = build_candidate_rows(baseline_rows, decisions, ladder=ladder, thresholds=thresholds)
        _write_json(json_path, rows)
        _write_zip(zip_path, json_path)
        report = _candidate_report(
            ladder=ladder,
            json_path=json_path,
            zip_path=zip_path,
            baseline_rows=baseline_rows,
            rows=rows,
            decisions=decisions,
            thresholds=thresholds,
        )
        _write_json(out_dir / f"candidate_report_{ladder}.json", report)
        (out_dir / f"candidate_report_{ladder}.md").write_text(
            _render_candidate_markdown(report),
            encoding="utf-8",
        )
        candidates[ladder] = report
        shadow_candidates.append({"name": ladder, "json": json_path})

    shadow_report = write_shadow_evaluator_outputs(
        baseline_json=shadow_baseline_json,
        candidates=shadow_candidates,
        out_dir=out_dir / "shadow_eval",
        expected_row_count=expected_row_count,
        require_all_emotions=require_all_emotions,
    )
    html_review_path = html_dir / "track2_v7_hardcase_arbitration_review.html"
    summary = {
        "method": "track2_v7_hardcase_arbitration",
        "decision": _top_level_decision(candidates),
        "baseline_json": str(baseline_json),
        "shadow_baseline_json": str(shadow_baseline_json),
        "evidence_matrix": str(evidence_matrix),
        "arbitration_json": str(arbitration_json),
        "text_override_json": str(text_override_json),
        "out_dir": str(out_dir),
        "decision_count": len(decisions),
        "strict_accept_count": sum(1 for item in decisions if item.decision == ACCEPT_STRICT),
        "borderline_accept_count": sum(1 for item in decisions if item.decision == ACCEPT_BORDERLINE),
        "hold_count": sum(1 for item in decisions if item.decision == HOLD),
        "decisions_json": str(decisions_json),
        "decisions_csv": str(decisions_csv),
        "candidates": candidates,
        "shadow_report_json": str(out_dir / "shadow_eval" / "shadow_score_report.json"),
        "html_review_path": str(html_review_path),
        "formal_submission_overwritten": False,
    }
    _write_json(out_dir / "v7_summary.json", summary)
    (out_dir / "v7_summary.md").write_text(
        _render_summary_markdown(summary, shadow_report),
        encoding="utf-8",
    )
    html_review_path.write_text(_render_html_review(summary, decision_rows), encoding="utf-8")
    return summary


def _decision_from_arbitration(
    arbitration: dict[str, Any],
    *,
    baseline_by_id: dict[str, dict[str, Any]],
    evidence_by_key: dict[tuple[str, str], dict[str, Any]],
    text_by_id: dict[str, dict[str, Any]],
    thresholds: V7Thresholds,
) -> V7Decision:
    sample_id = _text(arbitration.get("sample_id"))
    transition = _text(arbitration.get("transition"))
    evidence = evidence_by_key.get((sample_id, transition), {})
    current_emotion, proposed_emotion = _transition_parts(transition)
    current_emotion = _emotion(evidence.get("current_emotion")) or current_emotion
    proposed_emotion = _emotion(evidence.get("proposed_emotion")) or proposed_emotion
    proposed_valence, proposed_arousal = _labels_for(
        proposed_emotion,
        evidence.get("proposed_valence"),
        evidence.get("proposed_arousal"),
    )
    current_valence, current_arousal = _labels_for(
        current_emotion,
        evidence.get("current_valence"),
        evidence.get("current_arousal"),
    )
    baseline = baseline_by_id.get(sample_id, {})
    confidence = _float(arbitration.get("confidence"))
    fit_margin = _float(arbitration.get("fit_margin"))
    evidence_score = _float(evidence.get("evidence_score", arbitration.get("evidence_score")))
    source_count = _int(evidence.get("supporting_source_count"))
    family_count = _int(evidence.get("supporting_family_count"))
    strict_gate = _bool(arbitration.get("enter_strict_candidate"))
    prefers_proposed = _text(arbitration.get("preferred_option")).lower() == "proposed"
    text_override = text_by_id.get(sample_id, {})

    reasons: list[str] = []
    if not sample_id or not current_emotion or not proposed_emotion:
        reasons.append("missing_required_arbitration_field")
    if proposed_emotion and proposed_emotion not in TRACK2_JSON_EMOTIONS:
        reasons.append("invalid_proposed_emotion")
    if proposed_emotion and not _emotion_matches_labels(proposed_emotion, proposed_valence, proposed_arousal):
        reasons.append("proposed_va_label_mismatch")
    if current_emotion and not _emotion_matches_labels(current_emotion, current_valence, current_arousal):
        reasons.append("current_va_label_mismatch")
    if baseline and _emotion(baseline.get("emotion")) != current_emotion:
        reasons.append("baseline_current_label_mismatch")
    if not evidence:
        reasons.append("missing_evidence_row")
    if not prefers_proposed:
        reasons.append("arbitration_prefers_current")
    if _bool(evidence.get("gemini35_objection")):
        reasons.append("gemini35_objection")
    if _bool(evidence.get("public_reference_contradiction")):
        reasons.append("public_reference_contradiction")
    if not _has_label_aligned_text_override(text_override, proposed_emotion, proposed_valence, proposed_arousal):
        reasons.append("missing_label_aligned_text_override")

    strict_reasons = _strict_threshold_reasons(
        confidence=confidence,
        fit_margin=fit_margin,
        evidence_score=evidence_score,
        source_count=source_count,
        family_count=family_count,
        strict_gate=strict_gate,
        thresholds=thresholds,
    )
    dangerous_shift = _dangerous_positive_low_shift(
        current_emotion=current_emotion,
        current_valence=current_valence,
        current_arousal=current_arousal,
        proposed_valence=proposed_valence,
        proposed_arousal=proposed_arousal,
    )
    if dangerous_shift and strict_reasons:
        reasons.append("dangerous_positive_low_shift_without_strict_gate")

    decision = HOLD
    ladder = ""
    if not reasons and not strict_reasons:
        decision = ACCEPT_STRICT
        ladder = "v7_strict"
        reasons.append("strict_multimodal_arbitration_support")
    elif not reasons:
        borderline_reasons = _borderline_threshold_reasons(
            confidence=confidence,
            fit_margin=fit_margin,
            evidence_score=evidence_score,
            source_count=source_count,
            family_count=family_count,
            thresholds=thresholds,
        )
        if dangerous_shift:
            borderline_reasons.append("dangerous_positive_low_shift_without_strict_gate")
        if not borderline_reasons:
            decision = ACCEPT_BORDERLINE
            ladder = "v7_diagnostic"
            reasons.append("borderline_proposed_diagnostic_support")
        else:
            reasons.extend(borderline_reasons)
    else:
        for reason in strict_reasons:
            if reason not in reasons:
                reasons.append(reason)

    return V7Decision(
        sample_id=sample_id,
        transition=transition or f"{current_emotion}->{proposed_emotion}",
        current_emotion=current_emotion,
        proposed_emotion=proposed_emotion,
        proposed_valence=proposed_valence,
        proposed_arousal=proposed_arousal,
        decision=decision,
        ladder=ladder,
        reason_codes=tuple(dict.fromkeys(reasons or ["insufficient_arbitration_support"])),
        confidence=confidence,
        fit_margin=fit_margin,
        evidence_score=evidence_score,
        supporting_source_count=source_count,
        supporting_family_count=family_count,
        text_override=text_override if decision in {ACCEPT_STRICT, ACCEPT_BORDERLINE} else {},
    )


def _strict_threshold_reasons(
    *,
    confidence: float,
    fit_margin: float,
    evidence_score: float,
    source_count: int,
    family_count: int,
    strict_gate: bool,
    thresholds: V7Thresholds,
) -> list[str]:
    reasons: list[str] = []
    if not strict_gate:
        reasons.append("not_enter_strict_candidate")
    if confidence < thresholds.strict_min_confidence:
        reasons.append("low_strict_confidence")
    if fit_margin < thresholds.strict_min_fit_margin:
        reasons.append("low_strict_fit_margin")
    if evidence_score < thresholds.strict_min_evidence_score:
        reasons.append("low_strict_evidence_score")
    if source_count < thresholds.strict_min_sources:
        reasons.append("insufficient_strict_source_count")
    if family_count < thresholds.strict_min_families:
        reasons.append("insufficient_strict_family_count")
    return reasons


def _borderline_threshold_reasons(
    *,
    confidence: float,
    fit_margin: float,
    evidence_score: float,
    source_count: int,
    family_count: int,
    thresholds: V7Thresholds,
) -> list[str]:
    reasons: list[str] = []
    if confidence < thresholds.borderline_min_confidence:
        reasons.append("low_borderline_confidence")
    if fit_margin < thresholds.borderline_min_fit_margin:
        reasons.append("low_borderline_fit_margin")
    if evidence_score < thresholds.borderline_min_evidence_score:
        reasons.append("low_borderline_evidence_score")
    if source_count < thresholds.borderline_min_sources:
        reasons.append("insufficient_borderline_source_count")
    if family_count < thresholds.borderline_min_families:
        reasons.append("insufficient_borderline_family_count")
    return reasons


def _hold_conflicting_accepts(decisions: list[V7Decision]) -> list[V7Decision]:
    accepted_by_sample: dict[str, list[int]] = {}
    for index, decision in enumerate(decisions):
        if decision.decision in {ACCEPT_STRICT, ACCEPT_BORDERLINE}:
            accepted_by_sample.setdefault(decision.sample_id, []).append(index)
    output = list(decisions)
    for indexes in accepted_by_sample.values():
        proposed = {output[index].proposed_emotion for index in indexes}
        if len(indexes) <= 1 or len(proposed) <= 1:
            continue
        for index in indexes:
            output[index] = replace(
                output[index],
                decision=HOLD,
                ladder="",
                reason_codes=_append_reason(output[index].reason_codes, "proposal_conflict"),
                text_override={},
            )
    return output


def _selected_decisions(
    decisions: list[V7Decision],
    ladder: str,
    thresholds: V7Thresholds,
) -> dict[str, V7Decision]:
    if ladder not in V7_LADDERS:
        raise ValueError(f"unknown v7 ladder: {ladder}")
    strict = _sorted_accepts(decisions, ACCEPT_STRICT)
    if ladder == "v7_cross1":
        selected = strict[: thresholds.cross1_cap]
    elif ladder == "v7_cross3":
        selected = strict[: thresholds.cross3_cap]
    else:
        selected = [*strict[: thresholds.cross3_cap], *_sorted_accepts(decisions, ACCEPT_BORDERLINE)]
        selected = selected[: thresholds.diagnostic_cap]
    return {decision.sample_id: decision for decision in selected}


def _sorted_accepts(decisions: list[V7Decision], decision_name: str) -> list[V7Decision]:
    return sorted(
        [decision for decision in decisions if decision.decision == decision_name],
        key=lambda decision: (
            -decision.evidence_score,
            -decision.fit_margin,
            -decision.confidence,
            decision.sample_id,
            decision.proposed_emotion,
        ),
    )


def _candidate_report(
    *,
    ladder: str,
    json_path: Path,
    zip_path: Path,
    baseline_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    decisions: list[V7Decision],
    thresholds: V7Thresholds,
) -> dict[str, Any]:
    selected = _selected_decisions(decisions, ladder, thresholds)
    changed = _changed_rows(baseline_rows, rows)
    distribution = compute_track2_distribution(rows)
    label_issue_count = sum(len(strict_track2_label_issues(row)) for row in rows)
    return {
        "ladder": ladder,
        "json": str(json_path),
        "zip": str(zip_path),
        "changed_rows": len(changed),
        "selected_changes": [
            {
                "sample_id": decision.sample_id,
                "transition": decision.transition,
                "decision": decision.decision,
                "confidence": decision.confidence,
                "fit_margin": decision.fit_margin,
                "evidence_score": decision.evidence_score,
                "reason_codes": list(decision.reason_codes),
                "text_override_applied": bool(decision.text_override),
            }
            for decision in selected.values()
        ],
        "distribution": distribution,
        "label_consistency_issue_count": label_issue_count,
        "missing_emotions": distribution.get("missing_emotions", []),
        "formal_submission_overwritten": False,
    }


def _top_level_decision(candidates: dict[str, Any]) -> str:
    cross1 = candidates.get("v7_cross1", {})
    if (
        int(cross1.get("changed_rows", 0)) == 1
        and int(cross1.get("label_consistency_issue_count", 1)) == 0
        and not cross1.get("missing_emotions")
    ):
        return "recommend_manual_review_cross1"
    return "recommend_hold"


def _render_candidate_markdown(report: dict[str, Any]) -> str:
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
            f"- {item['sample_id']}: {item['transition']}; confidence={item['confidence']:.2f}; "
            f"margin={item['fit_margin']:.2f}; score={item['evidence_score']:.1f}; "
            f"text_override={item['text_override_applied']}; reasons={','.join(item['reason_codes'])}"
        )
    if not report.get("selected_changes"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _render_summary_markdown(summary: dict[str, Any], shadow_report: dict[str, Any]) -> str:
    lines = [
        "# Track2 V7 Hard-Case Arbitration Summary",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Baseline JSON: `{summary['baseline_json']}`",
        f"- Shadow baseline JSON: `{summary['shadow_baseline_json']}`",
        f"- Strict accepts: {summary['strict_accept_count']}",
        f"- Borderline diagnostic accepts: {summary['borderline_accept_count']}",
        f"- Holds: {summary['hold_count']}",
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
        f"<td>{html.escape(str(item['confidence']))}</td>"
        f"<td>{html.escape(str(item['fit_margin']))}</td>"
        f"<td>{html.escape(str(item['reason_codes']))}</td>"
        "</tr>"
        for item in decisions[:500]
    )
    cards = "\n".join(
        f"<section><h2>{html.escape(name)}</h2><p>Changed rows: {payload['changed_rows']}</p>"
        f"<p><code>{html.escape(payload['zip'])}</code></p></section>"
        for name, payload in summary.get("candidates", {}).items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Track2 V7 Hard-Case Arbitration</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 7px; text-align: left; }}
    th {{ background: #eef2f6; }}
    section {{ border: 1px solid #d7dde5; padding: 12px; margin: 12px 0; }}
    .decision {{ padding: 12px; background: #fff4d6; border: 1px solid #e5c66a; }}
  </style>
</head>
<body>
  <h1>Track2 V7 Hard-Case Arbitration</h1>
  <p class="decision">Decision: <code>{html.escape(str(summary['decision']))}</code></p>
  <p>Baseline: <code>{html.escape(str(summary['baseline_json']))}</code></p>
  {cards}
  <h2>Arbitration Decisions</h2>
  <table>
    <thead><tr><th>sample</th><th>transition</th><th>decision</th><th>confidence</th><th>margin</th><th>reasons</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def _decision_to_dict(decision: V7Decision) -> dict[str, Any]:
    return {
        "sample_id": decision.sample_id,
        "transition": decision.transition,
        "current_emotion": decision.current_emotion,
        "proposed_emotion": decision.proposed_emotion,
        "proposed_valence": decision.proposed_valence,
        "proposed_arousal": decision.proposed_arousal,
        "decision": decision.decision,
        "ladder": decision.ladder,
        "reason_codes": ";".join(decision.reason_codes),
        "confidence": decision.confidence,
        "fit_margin": decision.fit_margin,
        "evidence_score": decision.evidence_score,
        "supporting_source_count": decision.supporting_source_count,
        "supporting_family_count": decision.supporting_family_count,
        "text_override_applied": bool(decision.text_override),
    }


def _evidence_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        sample_id = _text(row.get("sample_id"))
        current = _emotion(row.get("current_emotion"))
        proposed = _emotion(row.get("proposed_emotion"))
        transition = _text(row.get("transition")) or f"{current}->{proposed}"
        if sample_id and transition:
            index[(sample_id, transition)] = dict(row)
    return index


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id", "")): dict(row) for row in rows if str(row.get("sample_id", ""))}


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected Track2 JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _candidate_paths_for_name(name: str, submission_dir: Path, submission_root: Path) -> tuple[Path, Path]:
    _assert_safe_output_name(name)
    json_path = submission_dir / f"{name}.json"
    zip_path = submission_dir / f"{name}.zip"
    _assert_under_directory(json_path, submission_root)
    _assert_under_directory(zip_path, submission_root)
    _assert_safe_candidate_path(json_path)
    _assert_safe_candidate_path(zip_path)
    return json_path, zip_path


def _assert_safe_output_name(name: str) -> None:
    text = str(name)
    if not text or Path(text).is_absolute() or "/" in text or "\\" in text or ".." in text:
        raise ValueError(f"candidate output name must be a safe basename: {name}")


def _assert_under_directory(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"candidate output path escapes submission directory: {path}") from exc


def _assert_safe_candidate_path(path: Path) -> None:
    if path.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError(f"refusing to write formal Track2 submission path: {path}")
    if path.suffix not in {".json", ".zip"}:
        raise ValueError(f"candidate output must be JSON or ZIP: {path}")
    if not path.stem.startswith("track2_submission_v7_") or not path.stem.endswith("_candidate"):
        raise ValueError(f"candidate output must be a v7 side-path candidate: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_zip(zip_path: Path, json_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, "submission.json")


def _changed_rows(baseline_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_id = _index_rows(baseline_rows)
    changed: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        before = baseline_by_id.get(sample_id)
        if not before:
            continue
        if _label_triplet(before) != _label_triplet(row):
            changed.append({"sample_id": sample_id, "before": before, "after": row})
    return changed


def _label_triplet(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("emotion", "")),
        str(row.get("emotional_valence", "")),
        str(row.get("emotional_arousal_level", "")),
    )


def _has_label_aligned_text_override(
    row: dict[str, Any],
    proposed_emotion: str,
    proposed_valence: str,
    proposed_arousal: str,
) -> bool:
    if not row:
        return False
    return (
        _emotion(row.get("emotion")) == proposed_emotion
        and _text(row.get("emotional_valence")) == proposed_valence
        and _text(row.get("emotional_arousal_level")) == proposed_arousal
        and all(_text(row.get(field_name)) for field_name in TEXT_FIELDS)
    )


def _emotion_matches_labels(emotion: str, valence: str, arousal: str) -> bool:
    try:
        expected_valence, expected_arousal = expected_label_for_emotion(emotion)
    except ValueError:
        return False
    return valence == expected_valence and arousal == expected_arousal


def _dangerous_positive_low_shift(
    *,
    current_emotion: str,
    current_valence: str,
    current_arousal: str,
    proposed_valence: str,
    proposed_arousal: str,
) -> bool:
    return (
        current_emotion in {"calm", "content", "glad"}
        and current_valence == "Positive"
        and current_arousal == "Low"
        and (proposed_valence != "Positive" or proposed_arousal != "Low")
    )


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


def _transition_parts(transition: str) -> tuple[str, str]:
    if "->" not in transition:
        return "", ""
    current, proposed = transition.split("->", 1)
    return _emotion(current), _emotion(proposed)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _emotion(value: Any) -> str:
    return _text(value).lower()


def _bool(value: Any) -> bool:
    return _text(value).lower() in {"true", "t", "1", "yes", "y"}


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


def _append_reason(reasons: tuple[str, ...], reason: str) -> tuple[str, ...]:
    if reason in reasons:
        return reasons
    return (*reasons, reason)
