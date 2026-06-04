from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from affectiveart.track1_reference_family_bank import _safe_output_paths


PROMPT_LINT_VERSION = "track1_prompt_lint_v1"
DEFAULT_PHRASES = (
    "front-facing",
    "portrait poster canvas",
    "flat printed poster",
    "medium-distance figures",
    "fewer, larger",
    "fewer, larger text blocks",
    "fewer/larger text blocks",
    "internal poster margins",
    "graphic poster composition",
)
POSTER_SUPPORT_PHRASES = {
    "portrait poster canvas",
    "flat printed poster",
    "internal poster margins",
    "graphic poster composition",
}


def lint_prompt_batch(
    rows: Iterable[dict[str, Any]],
    threshold: float = 0.6,
    phrases: Iterable[str] = DEFAULT_PHRASES,
) -> dict[str, Any]:
    items = [_validated_prompt_row(row, index) for index, row in enumerate(rows)]
    _require_prompt_rows(items)
    total = len(items)
    failed: list[str] = []
    warned: list[str] = []
    justified: list[str] = []
    phrase_report: dict[str, dict[str, Any]] = {}

    for phrase in phrases:
        matching_rows = [
            row
            for row in items
            if phrase.lower() in _prompt_text(row).lower()
        ]
        sample_ids = [str(row["sample_id"]) for row in matching_rows]
        ratio = len(matching_rows) / total if total else 0.0
        justification_by_sample = {
            str(row["sample_id"]): _justification_reasons(row, phrase)
            for row in matching_rows
        }
        justified_sample_ids = [
            sample_id
            for sample_id, reasons in justification_by_sample.items()
            if reasons
        ]
        justified_count = len(justified_sample_ids)
        justification_ratio = justified_count / len(matching_rows) if matching_rows else 0.0
        unjustified_sample_ids = [
            sample_id
            for sample_id in sample_ids
            if sample_id not in set(justified_sample_ids)
        ]
        phrase_status = "pass"

        if total and ratio > threshold:
            if matching_rows and not unjustified_sample_ids:
                phrase_status = "warn"
                warned.append(phrase)
                justified.append(phrase)
            else:
                phrase_status = "fail"
                failed.append(phrase)

        phrase_report[phrase] = {
            "status": phrase_status,
            "count": len(matching_rows),
            "ratio": round(ratio, 6),
            "sample_ids": sample_ids,
            "justified_count": justified_count,
            "justified_ratio": round(justification_ratio, 6),
            "justified_sample_ids": justified_sample_ids,
            "unjustified_sample_ids": unjustified_sample_ids,
            "justifications": sorted({reason for reasons in justification_by_sample.values() for reason in reasons}),
        }

    status = "fail" if failed else "warn" if warned else "pass"
    return {
        "version": PROMPT_LINT_VERSION,
        "summary": {
            "version": PROMPT_LINT_VERSION,
            "total": total,
            "threshold": threshold,
            "status": status,
            "failed_phrases": failed,
            "warned_phrases": warned,
            "justified_phrases": justified,
        },
        "phrases": phrase_report,
    }


def write_prompt_lint_reports(
    report: dict[str, Any],
    json_path: str | Path,
    md_path: str | Path,
    repo_root: str | Path | None = None,
) -> None:
    json_path, md_path = _safe_output_paths([json_path, md_path], repo_root=repo_root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")


def load_prompt_rows(path_or_dir: str | Path) -> list[dict[str, Any]]:
    source = Path(path_or_dir)
    if source.is_dir():
        rows = [
            {
                "sample_id": txt_path.stem,
                "provider_prompt": txt_path.read_text(encoding="utf-8").strip(),
            }
            for txt_path in sorted(source.glob("*.txt"))
        ]
        _require_prompt_rows(rows)
        return [_validated_prompt_row(row, index) for index, row in enumerate(rows)]

    if source.suffix.lower() == ".jsonl":
        rows: list[Any] = []
        for line_index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"prompt row {line_index} is malformed JSONL: {exc}") from exc
        _require_prompt_rows(rows)
        return [_validated_prompt_row(row, index) for index, row in enumerate(rows)]

    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("packets"), list):
        rows = payload["packets"]
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise ValueError("prompt input must be a list, an object with packets/rows, JSONL, or a directory")
    _require_prompt_rows(rows)
    return [_validated_prompt_row(row, index) for index, row in enumerate(rows)]


def _validated_prompt_row(row: Any, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"prompt row {index} must be an object")
    if not row.get("sample_id"):
        raise ValueError(f"prompt row {index} missing required sample_id")
    if not isinstance(_prompt_text(row), str) or not _prompt_text(row).strip():
        raise ValueError(f"prompt row {index} missing required provider_prompt or prompt")
    return dict(row)


def _require_prompt_rows(rows: list[Any]) -> None:
    if not rows:
        raise ValueError("no prompt rows found")


def _prompt_text(row: dict[str, Any]) -> str:
    return str(row.get("provider_prompt") or row.get("prompt") or "")


def _justification_reasons(row: dict[str, Any], phrase: str) -> list[str]:
    reasons: list[str] = []
    if _explicitly_justified(row, phrase):
        reasons.append("justified_template_phrases")
    if phrase in POSTER_SUPPORT_PHRASES:
        reasons.extend(_poster_support_reasons(row))
    if phrase == "fewer, larger text blocks" and _text_contract_required(row):
        reasons.append("text_contract_required")
    return sorted(set(reasons))


def _explicitly_justified(row: dict[str, Any], phrase: str) -> bool:
    raw = row.get("justified_template_phrases") or row.get("template_phrase_justifications")
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, dict):
        values = list(raw)
    elif isinstance(raw, list | tuple | set):
        values = list(raw)
    else:
        values = []
    lowered = {str(value).lower() for value in values}
    return phrase.lower() in lowered or "*" in lowered


def _poster_support_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    caption = str(row.get("caption") or "").lower()
    if "poster" in caption:
        reasons.append("caption_mentions_poster")
    surface_contract = row.get("surface_contract")
    if isinstance(surface_contract, dict):
        searchable = json.dumps(surface_contract, sort_keys=True).lower()
        if surface_contract.get("required") and "poster" in searchable:
            reasons.append("surface_contract_requires_poster")
    elif isinstance(surface_contract, str) and "poster" in surface_contract.lower():
        reasons.append("surface_contract_requires_poster")
    support_contract = row.get("support_contract")
    if isinstance(support_contract, dict):
        searchable = json.dumps(support_contract, sort_keys=True).lower()
        if support_contract.get("required") and "poster" in searchable:
            reasons.append("support_contract_requires_poster")
    return reasons


def _text_contract_required(row: dict[str, Any]) -> bool:
    contract = row.get("text_contract")
    if isinstance(contract, dict):
        return bool(contract.get("required"))
    return False


def _render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Track1 Prompt Template Lint",
        "",
        f"- Status: `{summary['status']}`",
        f"- Total prompts: {summary['total']}",
        f"- Threshold: {summary['threshold']}",
        f"- Failed phrases: {summary['failed_phrases']}",
        f"- Warned phrases: {summary['warned_phrases']}",
        f"- Justified phrases: {summary['justified_phrases']}",
        "",
        "## Phrase Counts",
        "",
        "| Phrase | Status | Count | Ratio | Justified | Sample IDs |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for phrase, data in report["phrases"].items():
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} |".format(
                phrase,
                data["status"],
                data["count"],
                data["ratio"],
                data["justified_count"],
                ", ".join(data["sample_ids"][:10]),
            )
        )
    return "\n".join(lines).rstrip() + "\n"
