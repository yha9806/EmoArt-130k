from __future__ import annotations

import csv
import html
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from affectiveart.track2_local_shadow_evaluator import (
    OFFICIAL_ANCHOR,
    ScoreBand,
    ShadowScoreResult,
    load_json_rows,
    score_candidate_rows,
)
from affectiveart.track2_vulca_entailment_gate import audit_vulca_entailment_row


@dataclass(frozen=True)
class FusedShadowScoreResult:
    candidate_name: str
    candidate_json: str
    decision: str
    base_decision: str
    changed_rows: int
    same_quadrant_changes: int
    cross_quadrant_changes: int
    classification: ScoreBand
    description: ScoreBand
    overall: ScoreBand
    base_result: ShadowScoreResult
    vulca_baseline_summary: dict[str, Any]
    vulca_candidate_summary: dict[str, Any]
    vulca_description_delta: float


def score_fused_candidate_rows(
    *,
    candidate_name: str,
    candidate_json: Path,
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    expected_row_count: int = 1000,
    require_all_emotions: bool | None = None,
) -> FusedShadowScoreResult:
    base = score_candidate_rows(
        candidate_name=candidate_name,
        candidate_json=candidate_json,
        rows=rows,
        baseline_rows=baseline_rows,
        expected_row_count=expected_row_count,
        require_all_emotions=require_all_emotions,
    )
    baseline_vulca = summarize_vulca_entailment_risk(baseline_rows)
    candidate_vulca = summarize_vulca_entailment_risk(rows)
    vulca_delta = _vulca_description_delta(baseline_vulca, candidate_vulca)

    if not base.safety.passed:
        description = base.description
        overall = base.overall
        decision = base.decision
    else:
        description = _fused_description_band(base.description, vulca_delta)
        overall = ScoreBand(
            expected=0.5 * base.classification.expected + 0.5 * description.expected,
            lower=0.5 * base.classification.lower + 0.5 * description.lower,
            upper=0.5 * base.classification.upper + 0.5 * description.upper,
        ).clamped()
        decision = _fused_decision(base=base, overall=overall, description=description)

    return FusedShadowScoreResult(
        candidate_name=candidate_name,
        candidate_json=str(candidate_json),
        decision=decision,
        base_decision=base.decision,
        changed_rows=base.changed_rows,
        same_quadrant_changes=base.same_quadrant_changes,
        cross_quadrant_changes=base.cross_quadrant_changes,
        classification=base.classification,
        description=description,
        overall=overall,
        base_result=base,
        vulca_baseline_summary=baseline_vulca,
        vulca_candidate_summary=candidate_vulca,
        vulca_description_delta=vulca_delta,
    )


def rank_fused_shadow_candidates(results: list[FusedShadowScoreResult]) -> list[FusedShadowScoreResult]:
    return sorted(
        results,
        key=lambda item: (
            item.decision != "recommend_submit",
            -item.overall.lower,
            -item.description.lower,
            item.cross_quadrant_changes,
            item.vulca_candidate_summary["total_issues"],
            item.changed_rows,
            -item.overall.expected,
            item.candidate_name,
        ),
    )


def summarize_vulca_entailment_risk(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [audit_vulca_entailment_row(row) for row in rows]
    total_issues = sum(int(item.get("issue_count", 0)) for item in audits)
    risk_total = sum(float(item.get("pseudo_understanding_risk", 0.0)) for item in audits)
    counts = Counter(code for item in audits for code, count in item.get("issue_counts", {}).items() for _ in range(int(count)))
    return {
        "row_count": len(rows),
        "rows_with_issues": sum(1 for item in audits if int(item.get("issue_count", 0)) > 0),
        "total_issues": total_issues,
        "issue_counts": dict(sorted(counts.items())),
        "mean_pseudo_understanding_risk": risk_total / max(1, len(rows)),
    }


def write_fused_shadow_evaluator_outputs(
    *,
    baseline_json: str | Path,
    candidates: list[dict[str, Any]],
    out_dir: str | Path,
    expected_row_count: int = 1000,
    require_all_emotions: bool | None = None,
) -> dict[str, Any]:
    baseline_json = Path(baseline_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = load_json_rows(baseline_json)
    results: list[FusedShadowScoreResult] = []
    for candidate in candidates:
        candidate_json = Path(candidate["json"])
        results.append(
            score_fused_candidate_rows(
                candidate_name=str(candidate["name"]),
                candidate_json=candidate_json,
                rows=load_json_rows(candidate_json),
                baseline_rows=baseline_rows,
                expected_row_count=expected_row_count,
                require_all_emotions=require_all_emotions,
            )
        )
    ranked = rank_fused_shadow_candidates(results)
    ranking_rows = [_result_to_dict(item) for item in ranked]
    report = {
        "method": "track2_fused_shadow_score_v1_formula_plus_vulca_entailment",
        "warning": (
            "This is a local fused shadow score. It is not the official Codabench score "
            "and must not be treated as hidden-test ground truth."
        ),
        "policy": (
            "Classification comes from the calibrated local shadow evaluator. Description is "
            "conservatively adjusted by VULCA/Judge++ entailment risk delta with a small capped "
            "effect, so text-risk improvements can break ties but cannot override classification "
            "hold decisions."
        ),
        "baseline_json": str(baseline_json),
        "official_anchor": OFFICIAL_ANCHOR.__dict__,
        "ranking": ranking_rows,
        "formal_submission_overwritten": False,
    }
    _write_json(out_dir / "fused_shadow_score_report.json", report)
    _write_json(out_dir / "candidate_ranking.json", ranking_rows)
    _write_csv(out_dir / "candidate_ranking.csv", ranking_rows)
    (out_dir / "fused_shadow_score_report.md").write_text(_render_markdown(report), encoding="utf-8")
    html_dir = out_dir / "html_review"
    html_dir.mkdir(parents=True, exist_ok=True)
    (html_dir / "track2_fused_shadow_evaluator_review.html").write_text(_render_html(report), encoding="utf-8")
    return report


def _vulca_description_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> float:
    row_count = max(1, int(candidate.get("row_count") or baseline.get("row_count") or 1))
    effective_count = max(1000, row_count)
    issue_delta_rate = (float(baseline["total_issues"]) - float(candidate["total_issues"])) / effective_count
    risk_delta = (
        float(baseline["mean_pseudo_understanding_risk"]) - float(candidate["mean_pseudo_understanding_risk"])
    ) * (row_count / effective_count)
    raw = 0.16 * issue_delta_rate + 0.35 * risk_delta

    baseline_conflicts = int(baseline.get("issue_counts", {}).get("emotion_text_conflict", 0))
    candidate_conflicts = int(candidate.get("issue_counts", {}).get("emotion_text_conflict", 0))
    if candidate_conflicts > baseline_conflicts:
        raw -= min(0.010, 0.004 * (candidate_conflicts - baseline_conflicts))

    return max(-0.012, min(0.008, raw))


def _fused_description_band(base_description: ScoreBand, vulca_delta: float) -> ScoreBand:
    if vulca_delta >= 0:
        expected_delta = vulca_delta
        lower_delta = 0.70 * vulca_delta
        upper_delta = 0.35 * vulca_delta
    else:
        expected_delta = vulca_delta
        lower_delta = vulca_delta
        upper_delta = 0.50 * vulca_delta
    expected_cap = min(1.0, OFFICIAL_ANCHOR.description + 0.008)
    return ScoreBand(
        expected=min(expected_cap, base_description.expected + expected_delta),
        lower=base_description.lower + lower_delta,
        upper=min(1.0, base_description.upper + upper_delta),
    ).clamped()


def _fused_decision(*, base: ShadowScoreResult, overall: ScoreBand, description: ScoreBand) -> str:
    if base.decision != "recommend_submit":
        return base.decision
    if base.cross_quadrant_changes and overall.lower < OFFICIAL_ANCHOR.overall:
        return "recommend_hold"
    if description.lower < OFFICIAL_ANCHOR.description - 0.040:
        return "recommend_hold"
    if overall.lower >= OFFICIAL_ANCHOR.overall - 0.040:
        return "recommend_submit"
    return "recommend_hold"


def _result_to_dict(result: FusedShadowScoreResult) -> dict[str, Any]:
    return {
        "candidate_name": result.candidate_name,
        "candidate_json": result.candidate_json,
        "decision": result.decision,
        "base_decision": result.base_decision,
        "changed_rows": result.changed_rows,
        "same_quadrant_changes": result.same_quadrant_changes,
        "cross_quadrant_changes": result.cross_quadrant_changes,
        "classification_expected": result.classification.expected,
        "classification_lower": result.classification.lower,
        "classification_upper": result.classification.upper,
        "description_expected": result.description.expected,
        "description_lower": result.description.lower,
        "description_upper": result.description.upper,
        "overall_expected": result.overall.expected,
        "overall_lower": result.overall.lower,
        "overall_upper": result.overall.upper,
        "base_overall_expected": result.base_result.overall.expected,
        "base_overall_lower": result.base_result.overall.lower,
        "vulca_description_delta": result.vulca_description_delta,
        "vulca_baseline_total_issues": result.vulca_baseline_summary["total_issues"],
        "vulca_candidate_total_issues": result.vulca_candidate_summary["total_issues"],
        "vulca_baseline_mean_risk": result.vulca_baseline_summary["mean_pseudo_understanding_risk"],
        "vulca_candidate_mean_risk": result.vulca_candidate_summary["mean_pseudo_understanding_risk"],
        "safety_passed": result.base_result.safety.passed,
        "safety_issue_codes": ",".join(result.base_result.safety.issue_codes),
        "description_issue_count": result.base_result.safety.description_issue_count,
        "top_emotion": result.base_result.safety.distribution.get("top_emotion", ""),
        "top_emotion_share": result.base_result.safety.distribution.get("top_emotion_share", 0.0),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 Fused Shadow Evaluator Report",
        "",
        f"> {report['warning']}",
        "",
        f"- Method: `{report['method']}`",
        f"- Baseline JSON: `{report['baseline_json']}`",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        "",
        "## Policy",
        "",
        report["policy"],
        "",
        "## Candidate Ranking",
        "",
        "| rank | candidate | decision | overall lower | overall expected | class expected | desc expected | vulca delta | vulca issues | changes | cross quadrant |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(report["ranking"], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    str(item["candidate_name"]),
                    str(item["decision"]),
                    f"{float(item['overall_lower']):.6f}",
                    f"{float(item['overall_expected']):.6f}",
                    f"{float(item['classification_expected']):.6f}",
                    f"{float(item['description_expected']):.6f}",
                    f"{float(item['vulca_description_delta']):.6f}",
                    str(item["vulca_candidate_total_issues"]),
                    str(item["changed_rows"]),
                    str(item["cross_quadrant_changes"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_html(report: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item['candidate_name']))}</td>"
        f"<td>{html.escape(str(item['decision']))}</td>"
        f"<td>{float(item['overall_lower']):.6f}</td>"
        f"<td>{float(item['overall_expected']):.6f}</td>"
        f"<td>{float(item['description_expected']):.6f}</td>"
        f"<td>{float(item['vulca_description_delta']):.6f}</td>"
        f"<td>{int(item['vulca_candidate_total_issues'])}</td>"
        f"<td>{int(item['changed_rows'])}</td>"
        f"<td>{int(item['cross_quadrant_changes'])}</td>"
        "</tr>"
        for item in report["ranking"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Track2 Fused Shadow Evaluator</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px; text-align: left; }}
    th {{ background: #eef2f6; }}
    .warning {{ padding: 12px; background: #fff4d6; border: 1px solid #e5c66a; }}
  </style>
</head>
<body>
  <h1>Track2 Fused Shadow Evaluator</h1>
  <p class="warning">{html.escape(str(report['warning']))}</p>
  <p>{html.escape(str(report['policy']))}</p>
  <table>
    <thead><tr><th>candidate</th><th>decision</th><th>overall lower</th><th>overall expected</th><th>description expected</th><th>VULCA delta</th><th>VULCA issues</th><th>changes</th><th>cross quadrant</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
