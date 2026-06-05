from __future__ import annotations

import csv
import html
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_audit import compute_track2_distribution, strict_track2_label_issues
from affectiveart.track2_moe_specialist_ensemble import expected_label_for_emotion


FORMAL_SUBMISSION_PATHS = {
    Path("submissions/track2_submission.json"),
    Path("submissions/track2_submission.zip"),
}
TIER_ORDER = ("v3_safe_plus", "v3_mid", "v3_push")
DEFAULT_OUTPUT_NAMES = {
    "v3_safe_plus": "track2_submission_v3_safe_plus_candidate",
    "v3_mid": "track2_submission_v3_mid_candidate",
    "v3_push": "track2_submission_v3_push_candidate",
}
TIER_LIMITS = {
    "v3_safe_plus": 45,
    "v3_mid": 85,
    "v3_push": 125,
}


@dataclass(frozen=True)
class CandidateSource:
    name: str
    family: str
    rows: list[dict[str, Any]]
    path: str | None = None


def build_evidence_matrix(
    baseline_rows: list[dict[str, Any]],
    sources: list[CandidateSource],
    *,
    public_style_report: dict[str, Any] | None = None,
    human_gate_rows: list[dict[str, Any]] | None = None,
    rollback_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    baseline_by_id = _index_rows(baseline_rows)
    human_accepts = _human_accept_index(human_gate_rows or [])
    public_style_ids = _public_style_review_ids(public_style_report)
    rollback_ids = set(str(sample_id) for sample_id in (rollback_report or {}).get("rollback_ids", []))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for source in sources:
        for row in source.rows:
            sample_id = str(row.get("sample_id", ""))
            baseline = baseline_by_id.get(sample_id)
            if not baseline:
                continue
            current_emotion = str(baseline.get("emotion", "")).strip().lower()
            proposed_emotion = str(row.get("emotion", "")).strip().lower()
            if not proposed_emotion or proposed_emotion == current_emotion:
                continue
            grouped[(sample_id, proposed_emotion)].append(
                {
                    "source": source.name,
                    "family": source.family,
                    "path": source.path or "",
                    "candidate_valence": str(row.get("emotional_valence", "")),
                    "candidate_arousal": str(row.get("emotional_arousal_level", "")),
                }
            )

    matrix: list[dict[str, Any]] = []
    for (sample_id, proposed_emotion), support_rows in sorted(grouped.items()):
        baseline = baseline_by_id[sample_id]
        current_emotion = str(baseline.get("emotion", "")).strip().lower()
        current_valence = str(baseline.get("emotional_valence", ""))
        current_arousal = str(baseline.get("emotional_arousal_level", ""))
        proposed_valence, proposed_arousal = expected_label_for_emotion(proposed_emotion)
        sources_for_row = sorted({row["source"] for row in support_rows})
        families_for_row = sorted({row["family"] for row in support_rows})
        same_quadrant = current_valence == proposed_valence and current_arousal == proposed_arousal
        public_style_support = (
            sample_id in public_style_ids
            or any("public_style" in value for value in sources_for_row + families_for_row)
        )
        human_accept = human_accepts.get((sample_id, proposed_emotion), False)
        gemini_objection = sample_id in rollback_ids
        score = _evidence_score(
            supporting_source_count=len(sources_for_row),
            supporting_family_count=len(families_for_row),
            same_quadrant=same_quadrant,
            public_style_support=public_style_support,
            human_accept=human_accept,
            gemini_objection=gemini_objection,
        )
        decision = _decision_for_evidence(
            score=score,
            same_quadrant=same_quadrant,
            supporting_source_count=len(sources_for_row),
            supporting_family_count=len(families_for_row),
            gemini_objection=gemini_objection,
        )
        matrix.append(
            {
                "sample_id": sample_id,
                "current_emotion": current_emotion,
                "current_valence": current_valence,
                "current_arousal": current_arousal,
                "proposed_emotion": proposed_emotion,
                "proposed_valence": proposed_valence,
                "proposed_arousal": proposed_arousal,
                "transition": f"{current_emotion}->{proposed_emotion}",
                "supporting_source_count": len(sources_for_row),
                "supporting_family_count": len(families_for_row),
                "supporting_sources": ";".join(sources_for_row),
                "supporting_families": ";".join(families_for_row),
                "same_quadrant": same_quadrant,
                "cross_quadrant_risk": not same_quadrant,
                "public_style_agreement": public_style_support,
                "human_accept_support": human_accept,
                "gemini35_objection": gemini_objection,
                "evidence_score": score,
                "decision": decision,
            }
        )
    return sorted(
        matrix,
        key=lambda row: (
            -int(row["evidence_score"]),
            bool(row["cross_quadrant_risk"]),
            str(row["sample_id"]),
            str(row["proposed_emotion"]),
        ),
    )


def build_score_calibrated_candidates(
    baseline_rows: list[dict[str, Any]],
    evidence_matrix: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected_by_tier = {
        tier: _select_rows_for_tier(evidence_matrix, tier=tier, limit=TIER_LIMITS[tier])
        for tier in TIER_ORDER
    }
    candidates: dict[str, dict[str, Any]] = {}
    for tier in TIER_ORDER:
        selected = selected_by_tier[tier]
        rows = _apply_selected_changes(baseline_rows, selected)
        distribution = compute_track2_distribution(rows)
        label_issue_count = sum(len(strict_track2_label_issues(row)) for row in rows)
        candidates[tier] = {
            "tier": tier,
            "rows": rows,
            "changed_rows": len(selected),
            "selected_changes": selected,
            "distribution": distribution,
            "label_consistency_issue_count": label_issue_count,
            "missing_emotions": distribution.get("missing_emotions", []),
            "formal_submission_overwritten": False,
        }
    return candidates


def write_score_calibrated_outputs(
    *,
    baseline_json: str | Path,
    sources: list[CandidateSource],
    out_dir: str | Path,
    submission_dir: str | Path,
    public_style_report: dict[str, Any] | None = None,
    human_gate_rows: list[dict[str, Any]] | None = None,
    rollback_report: dict[str, Any] | None = None,
    output_names: dict[str, str] | None = None,
    formal_submission_paths: set[Path] | None = None,
) -> dict[str, Any]:
    baseline_path = Path(baseline_json)
    baseline_rows = _load_json_list(baseline_path)
    out_dir = Path(out_dir)
    submission_dir = Path(submission_dir)
    output_names = {**DEFAULT_OUTPUT_NAMES, **(output_names or {})}
    formal_paths = formal_submission_paths or FORMAL_SUBMISSION_PATHS
    matrix = build_evidence_matrix(
        baseline_rows,
        sources,
        public_style_report=public_style_report,
        human_gate_rows=human_gate_rows,
        rollback_report=rollback_report,
    )
    candidates = build_score_calibrated_candidates(baseline_rows, matrix)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "html_review").mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "evidence_matrix.json", matrix)
    _write_csv(out_dir / "evidence_matrix.csv", matrix)

    candidate_outputs: dict[str, Any] = {}
    for tier in TIER_ORDER:
        base_name = output_names[tier]
        json_path = submission_dir / f"{base_name}.json"
        zip_path = submission_dir / f"{base_name}.zip"
        _assert_safe_candidate_path(json_path, formal_paths)
        _assert_safe_candidate_path(zip_path, formal_paths)
        candidate = candidates[tier]
        _write_json(json_path, candidate["rows"])
        _write_zip(zip_path, json_path)
        report_md = out_dir / f"candidate_report_{tier}.md"
        report_json = out_dir / f"candidate_report_{tier}.json"
        report_payload = {key: value for key, value in candidate.items() if key != "rows"}
        report_payload["json"] = str(json_path)
        report_payload["zip"] = str(zip_path)
        _write_json(report_json, report_payload)
        report_md.write_text(_render_candidate_markdown(report_payload), encoding="utf-8")
        candidate_outputs[tier] = {
            "json": str(json_path),
            "zip": str(zip_path),
            "report_md": str(report_md),
            "report_json": str(report_json),
            "changed_rows": candidate["changed_rows"],
            "label_consistency_issue_count": candidate["label_consistency_issue_count"],
            "missing_emotions": candidate["missing_emotions"],
        }

    html_path = out_dir / "html_review" / "track2_score_calibrated_v3_review.html"
    html_path.write_text(_render_html(matrix, candidate_outputs), encoding="utf-8")
    summary = {
        "method": "track2_score_calibrated_champion_v1",
        "baseline_json": str(baseline_path),
        "out_dir": str(out_dir),
        "source_count": len(sources),
        "evidence_row_count": len(matrix),
        "candidates": candidate_outputs,
        "html_review": str(html_path),
        "formal_submission_overwritten": False,
    }
    _write_json(out_dir / "score_calibrated_summary.json", summary)
    return summary


def load_candidate_source(spec: str) -> CandidateSource:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise ValueError(f"expected name=family=path: {spec}")
    name, family, path = (part.strip() for part in parts)
    if not name or not family or not path:
        raise ValueError(f"expected non-empty name=family=path: {spec}")
    return CandidateSource(name=name, family=family, rows=_load_json_list(Path(path)), path=path)


def load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {resolved}")
    return payload


def load_optional_csv(path: str | Path | None) -> list[dict[str, str]]:
    if path is None or not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id", "")): row for row in rows if str(row.get("sample_id", ""))}


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return payload


def _public_style_review_ids(public_style_report: dict[str, Any] | None) -> set[str]:
    ids: set[str] = set()
    for row in (public_style_report or {}).get("review_allow_rows", []):
        if isinstance(row, dict) and row.get("sample_id"):
            ids.add(str(row["sample_id"]))
    return ids


def _human_accept_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], bool]:
    accepted: dict[tuple[str, str], bool] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            continue
        decision = " ".join(str(row.get(key, "")) for key in ("decision", "gate", "recommendation", "status")).lower()
        if "accept" not in decision and "allow" not in decision:
            continue
        emotion = _first_present(row, "proposed_emotion", "accepted_emotion", "emotion", "final_emotion")
        if emotion:
            accepted[(sample_id, emotion.lower())] = True
    return accepted


def _first_present(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _evidence_score(
    *,
    supporting_source_count: int,
    supporting_family_count: int,
    same_quadrant: bool,
    public_style_support: bool,
    human_accept: bool,
    gemini_objection: bool,
) -> int:
    score = supporting_source_count + supporting_family_count
    if same_quadrant:
        score += 1
    else:
        score -= 4
    if public_style_support:
        score += 2
    if human_accept:
        score += 2
    if gemini_objection:
        score -= 5
    return score


def _decision_for_evidence(
    *,
    score: int,
    same_quadrant: bool,
    supporting_source_count: int,
    supporting_family_count: int,
    gemini_objection: bool,
) -> str:
    if gemini_objection:
        return "hold"
    if same_quadrant and score >= 4 and supporting_source_count >= 2:
        return "accept_safe"
    if same_quadrant and score >= 3:
        return "accept_mid"
    if not same_quadrant and score >= 5 and supporting_family_count >= 3:
        return "accept_push_cross_quadrant"
    if score >= 2:
        return "diagnostic_push"
    return "hold"


def _select_rows_for_tier(
    evidence_matrix: list[dict[str, Any]],
    *,
    tier: str,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_sample_ids: set[str] = set()
    for row in sorted(
        evidence_matrix,
        key=lambda item: (
            -int(item["evidence_score"]),
            bool(item["cross_quadrant_risk"]),
            str(item["sample_id"]),
        ),
    ):
        sample_id = str(row["sample_id"])
        if sample_id in used_sample_ids:
            continue
        if _tier_allows_row(row, tier):
            selected.append(row)
            used_sample_ids.add(sample_id)
        if len(selected) >= limit:
            break
    return selected


def _tier_allows_row(row: dict[str, Any], tier: str) -> bool:
    decision = str(row.get("decision", ""))
    if tier == "v3_safe_plus":
        return decision == "accept_safe" and bool(row.get("same_quadrant"))
    if tier == "v3_mid":
        return decision in {"accept_safe", "accept_mid"} and bool(row.get("same_quadrant"))
    if tier == "v3_push":
        return decision in {"accept_safe", "accept_mid", "accept_push_cross_quadrant", "diagnostic_push"} and not bool(
            row.get("gemini35_objection")
        )
    raise ValueError(f"unknown candidate tier: {tier}")


def _apply_selected_changes(
    baseline_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_id = {str(row["sample_id"]): row for row in selected_rows}
    output: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        row = dict(baseline)
        selected = selected_by_id.get(str(row.get("sample_id", "")))
        if selected:
            row["emotion"] = selected["proposed_emotion"]
            row["emotional_valence"] = selected["proposed_valence"]
            row["emotional_arousal_level"] = selected["proposed_arousal"]
        output.append({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})
    return output


def _assert_safe_candidate_path(path: Path, formal_paths: set[Path]) -> None:
    normalized = Path(path)
    formal_names = {Path(item).name for item in formal_paths}
    if normalized.name in formal_names:
        raise ValueError(f"refusing to write formal Track2 submission path: {path}")
    if not normalized.name.startswith("track2_submission_v3_") or "_candidate." not in normalized.name:
        raise ValueError(f"candidate output must be a v3 side-path candidate: {path}")


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
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, "submission.json")


def _render_candidate_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Track2 {report['tier']} Candidate Report",
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
    for row in report.get("selected_changes", [])[:160]:
        lines.append(
            "- {sample_id}: {transition}; score={evidence_score}; sources={supporting_sources}".format(**row)
        )
    if not report.get("selected_changes"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _render_html(matrix: list[dict[str, Any]], candidates: dict[str, Any]) -> str:
    rows = []
    for item in matrix[:300]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['sample_id']))}</td>"
            f"<td>{html.escape(str(item['transition']))}</td>"
            f"<td>{html.escape(str(item['evidence_score']))}</td>"
            f"<td>{html.escape(str(item['decision']))}</td>"
            f"<td>{html.escape(str(item['supporting_sources']))}</td>"
            f"<td>{html.escape(str(item['same_quadrant']))}</td>"
            "</tr>"
        )
    cards = []
    for tier, payload in candidates.items():
        cards.append(
            f"<section><h2>{html.escape(tier)}</h2>"
            f"<p>changed rows: {payload['changed_rows']}</p>"
            f"<p><code>{html.escape(payload['zip'])}</code></p></section>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Track2 Score-Calibrated V3 Review</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;}"
        "table{border-collapse:collapse;width:100%;font-size:13px;}"
        "td,th{border:1px solid #ddd;padding:6px;text-align:left;}"
        "th{background:#f4f4f4;}section{margin:12px 0;padding:12px;border:1px solid #ddd;}</style>"
        "</head><body><h1>Track2 Score-Calibrated V3 Review</h1>"
        + "".join(cards)
        + "<h2>Evidence Matrix</h2><table><thead><tr>"
        "<th>sample_id</th><th>transition</th><th>score</th><th>decision</th><th>sources</th><th>same quadrant</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )
