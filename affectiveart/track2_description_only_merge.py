from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_SUBMISSION_KEYS, TRACK2_JSON_TEXT_FIELDS


FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}


def build_description_only_rows(
    label_source_rows: list[dict[str, Any]],
    text_source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text_by_id = {str(row.get("sample_id", "")): row for row in text_source_rows if row.get("sample_id")}
    output: list[dict[str, Any]] = []
    text_changes: list[dict[str, Any]] = []
    for label_row in label_source_rows:
        sample_id = str(label_row.get("sample_id", ""))
        row = dict(label_row)
        text_row = text_by_id.get(sample_id)
        changed_fields: list[str] = []
        if text_row:
            for field in TRACK2_JSON_TEXT_FIELDS:
                incoming = str(text_row.get(field, "")).strip()
                if incoming and incoming != str(row.get(field, "")):
                    row[field] = incoming
                    changed_fields.append(field)
        if changed_fields:
            text_changes.append({"sample_id": sample_id, "changed_fields": changed_fields})
        output.append({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})

    report = {
        "method": "track2_description_only_merge_v1",
        "row_count": len(output),
        "text_changed_rows": len(text_changes),
        "text_changed_fields": sum(len(item["changed_fields"]) for item in text_changes),
        "classification_label_changes": _classification_label_changes(label_source_rows, output),
        "changes": text_changes,
        "formal_submission_overwritten": False,
    }
    return output, report


def build_description_only_outputs(
    *,
    label_source_json: str | Path,
    text_source_json: str | Path,
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
) -> dict[str, Any]:
    label_source_json = Path(label_source_json)
    text_source_json = Path(text_source_json)
    out_json = Path(out_json)
    out_zip = Path(out_zip)
    _assert_side_path(out_json)
    _assert_side_path(out_zip)
    label_rows = _load_json_list(label_source_json)
    text_rows = _load_json_list(text_source_json)
    merged_rows, report = build_description_only_rows(label_rows, text_rows)
    report["label_source_json"] = str(label_source_json)
    report["text_source_json"] = str(text_source_json)
    report["out_json"] = str(out_json)
    report["out_zip"] = str(out_zip)
    _write_json(out_json, merged_rows)
    _write_zip(out_zip, out_json)
    _write_json(Path(report_json), report)
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text(render_description_only_markdown(report), encoding="utf-8")
    return report


def render_description_only_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 Description-Only Merge Report",
        "",
        f"- Method: `{report['method']}`",
        f"- Label source: `{report.get('label_source_json', '')}`",
        f"- Text source: `{report.get('text_source_json', '')}`",
        f"- JSON: `{report.get('out_json', '')}`",
        f"- ZIP: `{report.get('out_zip', '')}`",
        f"- Rows: {report['row_count']}",
        f"- Text changed rows: {report['text_changed_rows']}",
        f"- Text changed fields: {report['text_changed_fields']}",
        f"- Classification label changes: {report['classification_label_changes']}",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        "",
        "## Text Changes",
        "",
    ]
    for item in report.get("changes", [])[:160]:
        lines.append(f"- {item['sample_id']}: {', '.join(item['changed_fields'])}")
    if not report.get("changes"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _classification_label_changes(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> int:
    before_by_id = {str(row.get("sample_id", "")): row for row in before}
    count = 0
    for row in after:
        baseline = before_by_id.get(str(row.get("sample_id", "")))
        if not baseline:
            count += 1
            continue
        if (
            str(baseline.get("emotion", "")) != str(row.get("emotion", ""))
            or str(baseline.get("emotional_valence", "")) != str(row.get("emotional_valence", ""))
            or str(baseline.get("emotional_arousal_level", "")) != str(row.get("emotional_arousal_level", ""))
        ):
            count += 1
    return count


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _assert_side_path(path: Path) -> None:
    if path.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError(f"refusing to write formal Track2 submission path: {path}")
    if not path.name.startswith("track2_submission_v12_") or "_candidate." not in path.name:
        raise ValueError(f"description-only output must be a v12 side-path candidate: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_zip(zip_path: Path, json_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, "submission.json")
