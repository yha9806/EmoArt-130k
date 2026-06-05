from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageStat

from affectiveart.track1_prompt_lint import lint_prompt_batch
from affectiveart.track1_reference_family_bank import classify_aspect, _safe_output_paths


AUDIT_VERSION = "track1_distribution_audit_v1"


def audit_candidate_distribution(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = [_validated_manifest_row(row, index) for index, row in enumerate(rows)]
    _require_manifest_rows(items)

    audit_rows = [_audit_row(row) for row in items]
    aspect_counts = Counter(row["aspect"]["label"] for row in audit_rows)
    strategy_counts = Counter(
        str(row.get("candidate_strategy") or "unspecified")
        for row in items
    )
    family_counts = Counter(
        family_id
        for row in items
        for family_id in [_family_id_from_row(row)]
        if family_id
    )
    style_counts = Counter(
        style_id
        for row in items
        for style_id in [_style_family_from_row(row)]
        if style_id
    )
    prompt_lint = lint_prompt_batch(items)
    missing_image_count = sum(1 for row in audit_rows if row["aspect"]["label"] == "missing")
    unreadable_image_count = sum(1 for row in audit_rows if row["aspect"]["label"] == "unreadable")

    return {
        "version": AUDIT_VERSION,
        "summary": {
            "version": AUDIT_VERSION,
            "total": len(items),
            "existing_image_count": sum(1 for row in audit_rows if row["exists"]),
            "missing_image_count": missing_image_count,
            "unreadable_image_count": unreadable_image_count,
            "invalid_image_count": missing_image_count + unreadable_image_count,
            "aspect_labels": dict(sorted(aspect_counts.items())),
            "strategy_counts": dict(sorted(strategy_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "style_counts": dict(sorted(style_counts.items())),
            "prompt_lint_status": prompt_lint["summary"]["status"],
        },
        "prompt_lint": prompt_lint,
        "rows": audit_rows,
    }


def write_distribution_audit_reports(
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


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    elif isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        rows = payload["candidates"]
    elif isinstance(payload, dict) and isinstance(payload.get("packets"), list):
        rows = payload["packets"]
    else:
        raise ValueError("manifest JSON must be a list or an object with rows, candidates, or packets")

    _require_manifest_rows(rows)
    return [_validated_manifest_row(row, index, source.parent) for index, row in enumerate(rows)]


def _validated_manifest_row(
    row: Any,
    index: int,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"manifest row {index} must be an object")
    if not row.get("sample_id"):
        raise ValueError(f"manifest row {index} missing required sample_id")
    if not _prompt_text(row):
        raise ValueError(f"manifest row {index} missing required provider_prompt or prompt")

    validated = dict(row)
    if base_dir is not None and validated.get("image_path"):
        image_path = Path(str(validated["image_path"]))
        if not image_path.is_absolute():
            image_path = base_dir / image_path
        validated["image_path"] = str(image_path)
    return validated


def _require_manifest_rows(rows: list[Any]) -> None:
    if not rows:
        raise ValueError("no manifest rows found")


def _prompt_text(row: dict[str, Any]) -> str:
    provider_prompt = row.get("provider_prompt")
    if isinstance(provider_prompt, str) and provider_prompt.strip():
        return provider_prompt
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return ""


def _family_id_from_row(row: dict[str, Any]) -> str:
    direct = row.get("family_id")
    if direct:
        return str(direct)
    review_metadata = row.get("review_metadata")
    if isinstance(review_metadata, dict) and review_metadata.get("family_id"):
        return str(review_metadata["family_id"])
    return ""


def _style_family_from_row(row: dict[str, Any]) -> str:
    for key in ("style_family", "reference_style", "style", "artistic_style"):
        value = row.get(key)
        if value:
            return str(value)
    review_metadata = row.get("review_metadata")
    if isinstance(review_metadata, dict):
        for key in ("style_family", "reference_style", "style", "artistic_style"):
            value = review_metadata.get(key)
            if value:
                return str(value)
    return ""


def _audit_row(row: dict[str, Any]) -> dict[str, Any]:
    image_stats = _image_stats(row.get("image_path"))
    return {
        "sample_id": str(row["sample_id"]),
        "image_path": str(row.get("image_path") or ""),
        "exists": image_stats["exists"],
        "width": image_stats["width"],
        "height": image_stats["height"],
        "mean_luma": image_stats["mean_luma"],
        "luma_stddev": image_stats["luma_stddev"],
        "aspect": image_stats["aspect"],
        "image_error": str(image_stats.get("image_error") or ""),
        "candidate_strategy": str(row.get("candidate_strategy") or "unspecified"),
        "family_id": _family_id_from_row(row),
        "style_family": _style_family_from_row(row),
    }


def _image_stats(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return _missing_image_stats()
    image_path = Path(str(path_value))
    if not image_path.exists():
        return _missing_image_stats()

    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            luma = rgb.convert("L")
            stat = ImageStat.Stat(luma)
            return {
                "exists": True,
                "width": width,
                "height": height,
                "mean_luma": round(float(stat.mean[0]), 6),
                "luma_stddev": round(float(stat.stddev[0]), 6),
                "aspect": classify_aspect(width, height),
            }
    except Exception as exc:  # pragma: no cover - defensive for corrupt local assets
        stats = _missing_image_stats(label="unreadable")
        stats["image_error"] = str(exc)
        return stats


def _missing_image_stats(label: str = "missing") -> dict[str, Any]:
    return {
        "exists": False,
        "width": None,
        "height": None,
        "mean_luma": None,
        "luma_stddev": None,
        "aspect": {"width": None, "height": None, "ratio": None, "label": label},
    }


def _render_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Track1 Distribution Audit",
        "",
        f"- Version: {report.get('version', AUDIT_VERSION)}",
        f"- Total candidates: {summary.get('total', 0)}",
        f"- Existing images: {summary.get('existing_image_count', 0)}",
        f"- Missing images: {summary.get('missing_image_count', 0)}",
        f"- Unreadable images: {summary.get('unreadable_image_count', 0)}",
        f"- Invalid images: {summary.get('invalid_image_count', 0)}",
        f"- Prompt lint status: `{summary.get('prompt_lint_status', '')}`",
        "",
        "## Aspect Labels",
        "",
    ]
    _append_counts(lines, summary.get("aspect_labels", {}))
    lines.extend(["", "## Strategies", ""])
    _append_counts(lines, summary.get("strategy_counts", {}))
    lines.extend(["", "## Reference Families", ""])
    _append_counts(lines, summary.get("family_counts", {}))
    lines.extend(["", "## Style Families", ""])
    _append_counts(lines, summary.get("style_counts", {}))
    lines.extend(
        [
            "",
            "## Prompt Phrase Counts",
            "",
            "| Phrase | Status | Count | Ratio |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for phrase, phrase_report in report.get("prompt_lint", {}).get("phrases", {}).items():
        lines.append(
            f"| {phrase} | {phrase_report.get('status', '')} | "
            f"{phrase_report.get('count', 0)} | {phrase_report.get('ratio', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Sample | Exists | Aspect | Size | Mean luma | Luma stddev | Image error | Strategy | Family | Style |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in report.get("rows", []):
        size = f"{row.get('width')}x{row.get('height')}" if row.get("width") and row.get("height") else ""
        lines.append(
            f"| {row.get('sample_id', '')} | {row.get('exists', False)} | "
            f"{row.get('aspect', {}).get('label', '')} | {size} | "
            f"{_format_optional_float(row.get('mean_luma'))} | "
            f"{_format_optional_float(row.get('luma_stddev'))} | "
            f"{row.get('image_error', '')} | "
            f"{row.get('candidate_strategy', '')} | {row.get('family_id', '')} | "
            f"{row.get('style_family', '')} |"
        )
    return "\n".join(lines).strip() + "\n"


def _append_counts(lines: list[str], counts: dict[str, Any]) -> None:
    if not counts:
        lines.append("- none: 0")
        return
    for label, count in counts.items():
        lines.append(f"- {label}: {count}")


def _format_optional_float(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.6f}"
    return ""
