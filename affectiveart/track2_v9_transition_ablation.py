from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from affectiveart.challenge import TRACK2_JSON_SUBMISSION_KEYS


ABLATION_KINDS = (
    "content_to_calm_only",
    "no_calm_to_content",
    "public_duplicate_only",
    "human_high_confidence_only",
    "multisource_consensus_only",
)
FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}


def extract_v9_changes(
    baseline_rows: list[dict[str, Any]],
    v9_rows: list[dict[str, Any]],
    *,
    evidence_rows: list[dict[str, Any]] | None = None,
    human_gate_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    baseline_by_id = _index_rows(baseline_rows)
    evidence_by_key = _evidence_index(evidence_rows or [])
    human_by_key = _human_gate_index(human_gate_rows or [])
    changes: list[dict[str, Any]] = []
    for row in v9_rows:
        sample_id = str(row.get("sample_id", ""))
        baseline = baseline_by_id.get(sample_id)
        if not baseline:
            continue
        current = _label_triplet(baseline)
        proposed = _label_triplet(row)
        if current == proposed:
            continue
        key = (sample_id, proposed["emotion"].lower())
        evidence = evidence_by_key.get(key, {})
        human = human_by_key.get(key, {})
        transition = f"{current['emotion'].lower()}->{proposed['emotion'].lower()}"
        changes.append(
            {
                "sample_id": sample_id,
                "transition": transition,
                "current_emotion": current["emotion"].lower(),
                "current_valence": current["valence"],
                "current_arousal": current["arousal"],
                "proposed_emotion": proposed["emotion"].lower(),
                "proposed_valence": proposed["valence"],
                "proposed_arousal": proposed["arousal"],
                "same_quadrant": current["valence"] == proposed["valence"] and current["arousal"] == proposed["arousal"],
                "supporting_source_count": _safe_int(evidence.get("supporting_source_count")),
                "supporting_family_count": _safe_int(evidence.get("supporting_family_count")),
                "supporting_sources": str(evidence.get("supporting_sources", "")),
                "supporting_families": str(evidence.get("supporting_families", "")),
                "public_style_agreement": _truthy(evidence.get("public_style_agreement")),
                "evidence_score": _safe_int(evidence.get("evidence_score")),
                "evidence_decision": str(evidence.get("decision", "")),
                "human_decision": str(human.get("human_decision", "")),
                "human_confidence": _safe_int(human.get("reviewer_confidence_1_5")),
                "human_gate_recommendation": str(human.get("gate_recommendation", "")),
                "human_high_confidence_accept": _is_high_confidence_human_accept(human),
            }
        )
    return sorted(changes, key=lambda item: str(item["sample_id"]))


def select_ablation_changes(changes: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    selectors: dict[str, Callable[[dict[str, Any]], bool]] = {
        "content_to_calm_only": lambda row: row["transition"] == "content->calm",
        "no_calm_to_content": lambda row: row["transition"] != "calm->content",
        "public_duplicate_only": lambda row: bool(row.get("public_style_agreement")),
        "human_high_confidence_only": lambda row: bool(row.get("human_high_confidence_accept")),
        "multisource_consensus_only": lambda row: int(row.get("supporting_source_count") or 0) >= 8
        and int(row.get("supporting_family_count") or 0) >= 5,
    }
    if kind not in selectors:
        raise ValueError(f"unknown v9 ablation kind: {kind}")
    return [row for row in changes if selectors[kind](row)]


def build_ablation_candidate_rows(
    baseline_rows: list[dict[str, Any]],
    selected_changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_id = {str(row["sample_id"]): row for row in selected_changes}
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


def build_v9_transition_ablation_outputs(
    *,
    baseline_json: str | Path,
    v9_json: str | Path,
    evidence_json: str | Path | None,
    human_gate_csv: str | Path | None,
    out_dir: str | Path,
    submission_dir: str | Path,
    ablation_kinds: tuple[str, ...] = ABLATION_KINDS,
) -> dict[str, Any]:
    baseline_path = Path(baseline_json)
    v9_path = Path(v9_json)
    out_path = Path(out_dir)
    submission_path = Path(submission_dir)
    baseline_rows = _load_json_list(baseline_path)
    v9_rows = _load_json_list(v9_path)
    evidence_rows = _load_json_list(Path(evidence_json)) if evidence_json else []
    human_rows = _load_csv_rows(Path(human_gate_csv)) if human_gate_csv else []
    changes = extract_v9_changes(
        baseline_rows,
        v9_rows,
        evidence_rows=evidence_rows,
        human_gate_rows=human_rows,
    )

    out_path.mkdir(parents=True, exist_ok=True)
    submission_path.mkdir(parents=True, exist_ok=True)
    _write_json(out_path / "v9_transition_changes.json", changes)
    _write_csv(out_path / "v9_transition_changes.csv", changes)

    candidate_reports: dict[str, dict[str, Any]] = {}
    for kind in ablation_kinds:
        selected = select_ablation_changes(changes, kind)
        candidate_rows = build_ablation_candidate_rows(baseline_rows, selected)
        json_path = submission_path / f"track2_submission_v12_{kind}_candidate.json"
        zip_path = submission_path / f"track2_submission_v12_{kind}_candidate.zip"
        _assert_side_path(json_path)
        _assert_side_path(zip_path)
        _write_json(json_path, candidate_rows)
        _write_zip(zip_path, json_path)
        report = {
            "kind": kind,
            "json": str(json_path),
            "zip": str(zip_path),
            "changed_rows": len(selected),
            "transition_counts": dict(Counter(str(row["transition"]) for row in selected)),
            "selected_changes": selected,
            "formal_submission_overwritten": False,
        }
        _write_json(out_path / f"{kind}_report.json", report)
        (out_path / f"{kind}_report.md").write_text(_render_candidate_markdown(report), encoding="utf-8")
        candidate_reports[kind] = {key: value for key, value in report.items() if key != "selected_changes"}

    summary = {
        "method": "track2_v9_transition_ablation_v1",
        "baseline_json": str(baseline_path),
        "v9_json": str(v9_path),
        "change_count": len(changes),
        "transition_counts": dict(Counter(str(row["transition"]) for row in changes)),
        "candidates": candidate_reports,
        "formal_submission_overwritten": False,
    }
    _write_json(out_path / "v9_transition_ablation_summary.json", summary)
    (out_path / "v9_transition_ablation_summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
    return summary


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id", "")): row for row in rows if str(row.get("sample_id", ""))}


def _evidence_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("sample_id", "")), str(row.get("proposed_emotion", "")).lower()): row
        for row in rows
        if row.get("sample_id") and row.get("proposed_emotion")
    }


def _human_gate_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("sample_id", "")), str(row.get("proposed_emotion", "")).lower()): row
        for row in rows
        if row.get("sample_id") and row.get("proposed_emotion")
    }


def _label_triplet(row: dict[str, Any]) -> dict[str, str]:
    return {
        "emotion": str(row.get("emotion", "")),
        "valence": str(row.get("emotional_valence", "")),
        "arousal": str(row.get("emotional_arousal_level", "")),
    }


def _is_high_confidence_human_accept(row: dict[str, Any]) -> bool:
    decision = str(row.get("human_decision", "")).lower()
    gate = str(row.get("gate_recommendation", "")).lower()
    return decision == "accept_proposed" and _safe_int(row.get("reviewer_confidence_1_5")) >= 4 and "allow" in gate


def _safe_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _assert_side_path(path: Path) -> None:
    if path.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError(f"refusing to write formal Track2 submission path: {path}")
    if not path.name.startswith("track2_submission_v12_") or "_candidate." not in path.name:
        raise ValueError(f"v9 ablation output must be a v12 side-path candidate: {path}")


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
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, "submission.json")


def _render_candidate_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Track2 v12 {report['kind']} Candidate",
        "",
        f"- JSON: `{report['json']}`",
        f"- ZIP: `{report['zip']}`",
        f"- Changed rows: {report['changed_rows']}",
        f"- Transition counts: `{json.dumps(report['transition_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        "",
        "## Selected Changes",
        "",
    ]
    for row in report.get("selected_changes", []):
        lines.append(
            "- {sample_id}: {transition}; evidence={evidence_score}; sources={supporting_source_count}; families={supporting_family_count}; human={human_gate_recommendation}".format(
                **row
            )
        )
    if not report.get("selected_changes"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Track2 v9 Transition Ablation Summary",
        "",
        f"- Method: `{summary['method']}`",
        f"- Baseline JSON: `{summary['baseline_json']}`",
        f"- v9 JSON: `{summary['v9_json']}`",
        f"- v9 changed rows: {summary['change_count']}",
        f"- v9 transition counts: `{json.dumps(summary['transition_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Formal submission overwritten: {summary['formal_submission_overwritten']}",
        "",
        "## Candidates",
        "",
        "| candidate | changed rows | transitions | ZIP |",
        "| --- | ---: | --- | --- |",
    ]
    for kind, report in summary["candidates"].items():
        lines.append(
            f"| {kind} | {report['changed_rows']} | `{json.dumps(report['transition_counts'], ensure_ascii=False, sort_keys=True)}` | `{report['zip']}` |"
        )
    return "\n".join(lines) + "\n"
