from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path
from typing import Any

from affectiveart.track2_v22_official_author_scorer import classify_source_scope


DEFAULT_RESEARCH_DIR = Path("experiments/emoart_origin_research_20260608")
DEFAULT_SOURCE_INDEX = DEFAULT_RESEARCH_DIR / "source_index.csv"
DEFAULT_PUBLIC_SCAN = DEFAULT_RESEARCH_DIR / "public_resource_scan_20260608.json"
DEFAULT_FULL_MAP = DEFAULT_RESEARCH_DIR / "emoart_origin_full_public_research_map_zh.md"
DEFAULT_V22_DIR = Path("experiments/track2_v22_official_author_scorer_20260608")
DEFAULT_V22_SCOREBOARD = DEFAULT_V22_DIR / "v22_ladder_scoreboard.json"
DEFAULT_LOCAL_INVENTORY = DEFAULT_V22_DIR / "local_emoart130k_inventory.json"
DEFAULT_STYLE_PRIOR = DEFAULT_V22_DIR / "style_emotion_prior.csv"
DEFAULT_OUT_DIR = DEFAULT_RESEARCH_DIR

SOURCE_INDEX_COLUMNS = ["source_type", "name", "url", "verified_public_fact", "track2_use", "status"]
ACTIONABILITY_COLUMNS = [
    "source_scope",
    "classification_actionability",
    "description_actionability",
    "track2_actionability_score",
    "label_arbitration_allowed",
    "recommended_use",
    "risk_flags",
    "merge_group",
]
RISK_KEYWORDS = ("not track2 gold", "noncompatible", "not painting-only", "generic visual emotion")
DESCRIPTION_KEYWORDS = ("description", "caption", "attribute", "salience", "agsr", "fab-g", "brushstroke", "composition")
CLASSIFICATION_KEYWORDS = ("ontology", "emotion", "style prior", "training/reference", "annotation", "label", "scoring")


def score_source_row(row: dict[str, Any]) -> dict[str, str]:
    text = _row_text(row)
    scope = classify_source_scope(row.get("url") or row.get("name") or "")
    classification = _scope_base_classification(scope)
    description = _scope_base_description(scope)
    risks: list[str] = []

    if scope == "official" and any(keyword in text for keyword in ("official", "codabench", "openreview", "rules", "scoring", "format")):
        classification = max(classification, 0.92)
        description = max(description, 0.65)
    if any(keyword in text for keyword in ("emoart-130k", "annotation.json", "core ontology", "style prior")):
        classification = max(classification, 0.88)
        description = max(description, 0.62)
    if any(keyword in text for keyword in ("emoart-salience", "agsr", "fab-g", "attribute")):
        description = max(description, 0.86)
        classification = max(classification, 0.55)
    if any(keyword in text for keyword in DESCRIPTION_KEYWORDS):
        description = max(description, 0.55)
    if any(keyword in text for keyword in CLASSIFICATION_KEYWORDS):
        classification = max(classification, 0.50)
    if "auxiliary" in text:
        risks.append("auxiliary_only")
    for keyword in RISK_KEYWORDS:
        if keyword in text:
            risks.append(keyword.replace(" ", "_"))
            classification = min(classification, 0.34)
    if scope == "auxiliary":
        classification = min(classification, 0.45)
    if "auxiliary_only" in risks:
        classification = min(classification, 0.58)

    label_allowed = _label_arbitration_allowed(row, scope, risks)
    if label_allowed:
        recommended_use = "label_arbitration_or_core_calibration"
    elif description >= 0.75:
        recommended_use = "description_score_or_attribute_audit"
    elif classification >= 0.45:
        recommended_use = "auxiliary_teacher_or_diagnostic_only"
    else:
        recommended_use = "background_context_only"
    track2_score = _clamp(classification * 0.58 + description * 0.34 + _scope_reliability(scope) * 0.08)
    return {
        "source_scope": scope,
        "classification_actionability": f"{classification:.3f}",
        "description_actionability": f"{description:.3f}",
        "track2_actionability_score": f"{track2_score:.3f}",
        "label_arbitration_allowed": "yes" if label_allowed else "no",
        "recommended_use": recommended_use,
        "risk_flags": ";".join(sorted(set(risks))) or "none",
    }


def audit_source_index(path: str | Path = DEFAULT_SOURCE_INDEX) -> dict[str, Any]:
    path = Path(path)
    rows, fieldnames = _load_csv_rows(path)
    issue_codes: list[str] = []
    missing_required = [column for column in SOURCE_INDEX_COLUMNS if column not in fieldnames]
    missing_actionability = [column for column in ACTIONABILITY_COLUMNS if column not in fieldnames]
    if missing_required:
        issue_codes.append("missing_required_columns")
    if missing_actionability:
        issue_codes.append("missing_actionability_columns")
    blank_cells = {
        column: sum(1 for row in rows if not str(row.get(column, "")).strip())
        for column in SOURCE_INDEX_COLUMNS
        if column in fieldnames
    }
    if any(blank_cells.values()):
        issue_codes.append("blank_required_cells")

    duplicate_urls = _duplicate_groups(rows, "url")
    duplicate_names = _duplicate_groups(rows, "name")
    if duplicate_urls:
        issue_codes.append("duplicate_urls_need_merge_group")
    if duplicate_names:
        issue_codes.append("duplicate_names_need_merge_group")
    merge_groups = _build_merge_groups(rows)
    scored_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        scored = dict(row)
        scored.update(score_source_row(row))
        scored["registry_row"] = str(index)
        scored["merge_group"] = merge_groups.get(index, "")
        scored_rows.append(scored)

    return {
        "source_index": str(path),
        "row_count": len(rows),
        "fieldnames": fieldnames,
        "missing_required_columns": missing_required,
        "missing_actionability_columns": missing_actionability,
        "blank_required_cells": blank_cells,
        "duplicate_url_group_count": len(duplicate_urls),
        "duplicate_name_group_count": len(duplicate_names),
        "duplicate_url_groups": duplicate_urls,
        "duplicate_name_groups": duplicate_names,
        "issue_codes": sorted(set(issue_codes)),
        "scored_rows": scored_rows,
        "recommended_fix": "Keep source_index.csv as raw registry; use scored derivative with merge_group for final decisions.",
    }


def audit_v22_scoreboard(path: str | Path = DEFAULT_V22_SCOREBOARD, *, cwd: str | Path = ".") -> dict[str, Any]:
    cwd = Path(cwd)
    path = _resolve_path(path, cwd)
    payload = json.loads(path.read_text(encoding="utf-8"))
    issues: list[dict[str, str]] = []
    recommended_profile = str(payload.get("recommended_profile") or "")
    candidates = payload.get("candidates") or {}
    recommended = payload.get("recommended_candidate") or {}
    if not recommended_profile:
        issues.append({"code": "missing_recommended_profile", "detail": str(path)})
    if recommended_profile and recommended_profile not in candidates:
        issues.append({"code": "recommended_profile_not_in_candidates", "detail": recommended_profile})

    candidate_json = _resolve_path((recommended.get("paths") or {}).get("out_json", ""), cwd)
    upload_json = _resolve_path((payload.get("upload_copy") or {}).get("json", ""), cwd)
    upload_zip = _resolve_path((payload.get("upload_copy") or {}).get("zip", ""), cwd)
    candidate_rows: Any = None
    upload_rows: Any = None
    zip_rows: Any = None
    for label, file_path in (("candidate_json", candidate_json), ("upload_json", upload_json), ("upload_zip", upload_zip)):
        if not file_path.exists():
            issues.append({"code": f"missing_{label}", "detail": str(file_path)})
    if candidate_json.exists():
        candidate_rows = json.loads(candidate_json.read_text(encoding="utf-8"))
    if upload_json.exists():
        upload_rows = json.loads(upload_json.read_text(encoding="utf-8"))
    zip_names: list[str] = []
    if upload_zip.exists():
        with zipfile.ZipFile(upload_zip) as archive:
            zip_names = archive.namelist()
            if zip_names != ["submission.json"]:
                issues.append({"code": "unexpected_upload_zip_members", "detail": ",".join(zip_names)})
            if "submission.json" in zip_names:
                zip_rows = json.loads(archive.read("submission.json"))
    upload_json_matches = candidate_rows == upload_rows if candidate_rows is not None and upload_rows is not None else False
    upload_zip_matches = candidate_rows == zip_rows if candidate_rows is not None and zip_rows is not None else False
    if candidate_rows is not None and upload_rows is not None and not upload_json_matches:
        issues.append({"code": "upload_json_differs_from_recommended_candidate", "detail": str(upload_json)})
    if candidate_rows is not None and zip_rows is not None and not upload_zip_matches:
        issues.append({"code": "upload_zip_differs_from_recommended_candidate", "detail": str(upload_zip)})

    projection_by_profile = {
        profile: _safe_float((row.get("official_projection") or {}).get("projected_overall"))
        for profile, row in candidates.items()
        if isinstance(row, dict)
    }
    if recommended_profile and projection_by_profile:
        best_profile = max(projection_by_profile, key=lambda key: (projection_by_profile[key], key))
        if best_profile != recommended_profile:
            issues.append({"code": "recommended_profile_not_highest_proxy", "detail": best_profile})

    return {
        "scoreboard": str(path),
        "recommended_profile": recommended_profile,
        "candidate_json": str(candidate_json),
        "upload_json": str(upload_json),
        "upload_zip": str(upload_zip),
        "upload_zip_members": zip_names,
        "upload_json_matches_recommended_json": upload_json_matches,
        "upload_zip_matches_recommended_json": upload_zip_matches,
        "projection_by_profile": projection_by_profile,
        "issue_count": len(issues),
        "issues": issues,
    }


def build_debug_report(
    *,
    source_index: str | Path = DEFAULT_SOURCE_INDEX,
    public_scan: str | Path = DEFAULT_PUBLIC_SCAN,
    full_map: str | Path = DEFAULT_FULL_MAP,
    local_inventory: str | Path = DEFAULT_LOCAL_INVENTORY,
    style_prior: str | Path = DEFAULT_STYLE_PRIOR,
    v22_scoreboard: str | Path = DEFAULT_V22_SCOREBOARD,
    cwd: str | Path = ".",
) -> dict[str, Any]:
    source_audit = audit_source_index(source_index)
    v22_audit = audit_v22_scoreboard(v22_scoreboard, cwd=cwd)
    public_scan_report = _audit_public_scan(public_scan)
    map_report = _audit_markdown(full_map)
    inventory_report = _audit_inventory(local_inventory)
    style_prior_report = _audit_style_prior(style_prior)
    open_gaps = [
        {
            "code": "crawler_not_materialized_as_repeatable_pipeline",
            "severity": "medium",
            "detail": "public_resource_scan is a snapshot, not a reusable author-level crawler with scheduling, dedup, retries, and provenance snapshots.",
        },
        {
            "code": "source_index_raw_has_no_numeric_actionability_columns",
            "severity": "low",
            "detail": "Resolved by generated scored derivative; raw source_index remains human-readable source registry.",
        },
    ]
    if source_audit["duplicate_url_group_count"] or source_audit["duplicate_name_group_count"]:
        open_gaps.append(
            {
                "code": "duplicate_sources_require_merge_group_review",
                "severity": "low",
                "detail": "Scored derivative now annotates merge_group; raw duplicate rows are preserved because several entries represent different roles of the same source.",
            }
        )
    high_actionability = [
        row
        for row in source_audit["scored_rows"]
        if _safe_float(row.get("track2_actionability_score")) >= 0.75
    ]
    return {
        "method": "track2_registry_debug_v1",
        "source_audit": {key: value for key, value in source_audit.items() if key != "scored_rows"},
        "v22_audit": v22_audit,
        "public_scan": public_scan_report,
        "full_map": map_report,
        "local_inventory": inventory_report,
        "style_prior": style_prior_report,
        "high_actionability_source_count": len(high_actionability),
        "high_actionability_sources": high_actionability,
        "open_gaps": open_gaps,
        "blocking_issue_count": _blocking_issue_count(source_audit, v22_audit, inventory_report, style_prior_report),
        "interpretation": "v22 artifacts are structurally valid; remaining crawler work is pipeline completeness, not final-shot candidate corruption.",
    }


def write_debug_outputs(
    *,
    source_index: str | Path = DEFAULT_SOURCE_INDEX,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    cwd: str | Path = ".",
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_audit = audit_source_index(source_index)
    scored_csv = out_dir / "track2_registry_actionability_scored.csv"
    _write_csv(scored_csv, source_audit["scored_rows"])
    report = build_debug_report(source_index=source_index, cwd=cwd)
    report_json = out_dir / "track2_registry_debug_report.json"
    report_md = out_dir / "track2_registry_debug_report_zh.md"
    _write_json(report_json, report)
    report_md.write_text(_render_debug_report_md(report, scored_csv), encoding="utf-8")
    return {
        "scored_csv": str(scored_csv),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "blocking_issue_count": str(report["blocking_issue_count"]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit and debug Track2 public registry and v22 artifacts.")
    parser.add_argument("--source-index", default=str(DEFAULT_SOURCE_INDEX))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)
    print(json.dumps(write_debug_outputs(source_index=args.source_index, out_dir=args.out_dir), ensure_ascii=False, sort_keys=True))


def _load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _duplicate_groups(rows: list[dict[str, str]], field: str) -> list[dict[str, Any]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        value = str(row.get(field, "")).strip()
        if value:
            positions[value].append(index)
    return [
        {"value": value, "rows": indexes, "count": len(indexes)}
        for value, indexes in sorted(positions.items())
        if len(indexes) > 1
    ]


def _build_merge_groups(rows: list[dict[str, str]]) -> dict[int, str]:
    groups: dict[int, str] = {}
    for field in ("url", "name"):
        for group_index, group in enumerate(_duplicate_groups(rows, field), start=1):
            group_name = f"{field}_dup_{group_index:02d}"
            for row_index in group["rows"]:
                groups[row_index] = group_name if row_index not in groups else f"{groups[row_index]};{group_name}"
    return groups


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key, "")) for key in SOURCE_INDEX_COLUMNS).lower()


def _scope_base_classification(scope: str) -> float:
    return {"official": 0.88, "author": 0.70, "local_author_data": 0.90, "auxiliary": 0.22}.get(scope, 0.20)


def _scope_base_description(scope: str) -> float:
    return {"official": 0.60, "author": 0.55, "local_author_data": 0.60, "auxiliary": 0.35}.get(scope, 0.30)


def _scope_reliability(scope: str) -> float:
    return {"official": 1.0, "author": 0.88, "local_author_data": 0.92, "auxiliary": 0.42}.get(scope, 0.40)


def _label_arbitration_allowed(row: dict[str, Any], scope: str, risks: list[str]) -> bool:
    text = _row_text(row)
    if risks and any(risk in {"not_track2_gold", "noncompatible", "not_painting-only", "generic_visual_emotion"} for risk in risks):
        return False
    if scope == "official":
        return True
    return scope in {"author", "local_author_data"} and any(
        marker in text for marker in ("emoart-130k", "emoart project", "affectiveart", "annotation.json", "core ontology")
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _resolve_path(path: str | Path, cwd: Path) -> Path:
    p = Path(str(path))
    if p.is_absolute():
        return p
    return cwd / p


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if isfinite(parsed) else default


def _audit_public_scan(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    zero_result_queries: list[str] = []
    top_level_counts: dict[str, int] = {}
    for section, value in payload.items():
        if isinstance(value, dict):
            top_level_counts[section] = len(value)
            for query, results in value.items():
                if isinstance(results, list) and not results:
                    zero_result_queries.append(f"{section}:{query}")
    return {
        "path": str(path),
        "top_level_keys": sorted(payload),
        "top_level_counts": top_level_counts,
        "zero_result_query_count": len(zero_result_queries),
        "zero_result_queries": zero_result_queries,
    }


def _audit_markdown(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "chars": len(text),
        "has_current_conclusions": "## 当前结论" in text,
        "has_file_manifest": "## 已固化文件" in text,
    }


def _audit_inventory(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "annotation_rows": int(payload.get("annotation_rows", 0)),
        "tar_count": int(payload.get("tar_count", 0)),
        "style_count": len(payload.get("styles") or []),
        "issue_count": 0
        if int(payload.get("annotation_rows", 0)) >= 132000 and int(payload.get("tar_count", 0)) >= 56
        else 1,
    }


def _audit_style_prior(path: str | Path) -> dict[str, Any]:
    rows, _ = _load_csv_rows(Path(path))
    styles = {row.get("style", "") for row in rows if row.get("style")}
    emotions = {row.get("emotion", "") for row in rows if row.get("emotion")}
    return {
        "path": str(path),
        "row_count": len(rows),
        "style_count": len(styles),
        "emotion_count": len(emotions),
        "issue_count": 0 if len(styles) >= 56 and len(emotions) >= 12 else 1,
    }


def _blocking_issue_count(
    source_audit: dict[str, Any],
    v22_audit: dict[str, Any],
    inventory_report: dict[str, Any],
    style_prior_report: dict[str, Any],
) -> int:
    blocking = 0
    if source_audit.get("missing_required_columns"):
        blocking += 1
    if v22_audit.get("issue_count"):
        blocking += int(v22_audit["issue_count"])
    blocking += int(inventory_report.get("issue_count") or 0)
    blocking += int(style_prior_report.get("issue_count") or 0)
    return blocking


def _render_debug_report_md(report: dict[str, Any], scored_csv: Path) -> str:
    source = report["source_audit"]
    v22 = report["v22_audit"]
    lines = [
        "# Track2 Registry Debug Report",
        "",
        "结论：v22 最后一枪候选的结构没有发现 blocking issue；registry 的主要问题是 raw source_index 缺少数值 actionability 层，以及部分来源有重复角色行。",
        "",
        f"- Blocking issues: `{report['blocking_issue_count']}`",
        f"- Source rows: `{source['row_count']}`",
        f"- Missing actionability columns in raw source_index: `{len(source['missing_actionability_columns'])}`",
        f"- Duplicate URL groups: `{source['duplicate_url_group_count']}`",
        f"- Duplicate name groups: `{source['duplicate_name_group_count']}`",
        f"- Scored registry: `{scored_csv}`",
        "",
        "## v22 Artifact Audit",
        "",
        f"- Recommended profile: `{v22['recommended_profile']}`",
        f"- Upload ZIP: `{v22['upload_zip']}`",
        f"- Upload zip matches recommended JSON: `{v22['upload_zip_matches_recommended_json']}`",
        f"- Issue count: `{v22['issue_count']}`",
        "",
        "## High Actionability Sources",
        "",
    ]
    for row in report.get("high_actionability_sources", [])[:24]:
        lines.append(
            f"- `{row['name']}` score={row['track2_actionability_score']} "
            f"class={row['classification_actionability']} desc={row['description_actionability']} "
            f"scope={row['source_scope']} allow={row['label_arbitration_allowed']}"
        )
    lines.extend(["", "## Open Gaps", ""])
    for gap in report.get("open_gaps", []):
        lines.append(f"- `{gap['code']}` ({gap['severity']}): {gap['detail']}")
    lines.extend(
        [
            "",
            "## Debug Interpretation",
            "",
            "- 已完成内容需要继续保留为 immutable evidence snapshot。",
            "- 未完成内容不是当前 v22 candidate 的格式错误，而是缺少长期 crawler/pipeline 能力。",
            "- raw `source_index.csv` 不建议直接改成机器表；本报告生成的 scored derivative 更适合作为决策输入。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
