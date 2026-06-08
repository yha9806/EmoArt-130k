from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BODY_LOGIC_GATE_VERSION = "track1_body_logic_gate_v1"

RISK_PATTERNS: tuple[tuple[str, str], ...] = (
    ("人体/肢体", r"\b(men|man|woman|women|figures?|people|person|children|child|muscular|body|bodies|hands?|arms?|legs?|face|heads?)\b"),
    ("动作/姿态", r"\b(grappl\w*|fight\w*|flee\w*|march\w*|salut\w*|lean\w*|holding|hoist\w*|carrying|pulling|pushing|climb\w*|dance|pose|gesture|running|kneeling|standing|sitting|riding)\b"),
    ("军人/群体", r"\b(soldiers?|sailors?|guards?|warriors?|civilians?|riders?|crowd|procession)\b"),
    ("动物/骑乘", r"\b(horse|horses|mounted|cavalry|rider|camel|elephant|ox|cattle)\b"),
    ("交通/载具", r"\b(tank|train|window|glass|vehicle|car|truck|aircraft|airplane|plane|ship|boat|warship|cart)\b"),
    ("武器/旗帜/绳索", r"\b(cannon|gun|rifle|weapon|sword|shell|artillery|flag|banner|rope|mast|pole)\b"),
    ("空间关系", r"\b(around|beneath|before|behind|beside|leaning|marching|hoisting|protect|protecting|escort|pursuing|fleeing|battle|conflict|grappling)\b"),
)

P0_PATTERN = re.compile(
    r"\b(grappl\w*|window|glass|horse|mounted|cavalry|tank|train|vehicle|aircraft|plane|ship|"
    r"cannon|gun|rifle|weapon|sword|flag|banner|rope|mast|pole|flee\w*|protect\w*|escort\w*|pursu\w*|"
    r"battle|conflict|hands?|arms?|legs?)\b",
    re.IGNORECASE,
)


def classify_body_logic_risk(sample_id: str, caption: str) -> dict[str, Any]:
    categories = [
        label
        for label, pattern in RISK_PATTERNS
        if re.search(pattern, caption, re.IGNORECASE)
    ]
    if not categories:
        return {
            "sample_id": sample_id,
            "priority": "",
            "categories": [],
            "needs_body_logic_review": False,
        }
    return {
        "sample_id": sample_id,
        "priority": "P0" if P0_PATTERN.search(caption) else "P1",
        "categories": categories,
        "needs_body_logic_review": True,
    }


def build_body_logic_risk_report(p1_report: dict[str, Any]) -> dict[str, Any]:
    sample_rows: dict[str, dict[str, Any]] = {}
    for package in p1_report.get("packages", []):
        package_name = str(package.get("package") or "")
        for row in package.get("rows", []):
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                continue
            caption = str(row.get("caption") or "")
            if sample_id not in sample_rows:
                sample_rows[sample_id] = {
                    "sample_id": sample_id,
                    "caption": caption,
                    "packages": {},
                }
            elif not sample_rows[sample_id].get("caption") and caption:
                sample_rows[sample_id]["caption"] = caption
            sample_rows[sample_id]["packages"][package_name] = {
                "actual_hash_changed": bool(row.get("actual_hash_changed", row.get("changed", False))),
                "manifest_changed": bool(row.get("manifest_changed", row.get("changed", False))),
                "manifest_actual_gap": bool(row.get("manifest_actual_gap", False)),
                "known_placeholder_type": str(row.get("known_placeholder_type") or ""),
                "sha256": str(row.get("sha256") or ""),
            }

    risk_rows = []
    for sample_id, sample in sorted(sample_rows.items()):
        classification = classify_body_logic_risk(sample_id, str(sample.get("caption") or ""))
        if not classification["needs_body_logic_review"]:
            continue
        package_statuses = sample.get("packages") or {}
        risk_rows.append(
            {
                "sample_id": sample_id,
                "caption": sample.get("caption") or "",
                "priority": classification["priority"],
                "categories": classification["categories"],
                "manifest_gap": any(
                    bool(status.get("manifest_actual_gap")) for status in package_statuses.values()
                ),
                "actual_changed_any": any(
                    bool(status.get("actual_hash_changed")) for status in package_statuses.values()
                ),
                "packages": package_statuses,
            }
        )
    risk_rows.sort(key=_risk_sort_key)
    summary = _summary(risk_rows)
    return {
        "version": BODY_LOGIC_GATE_VERSION,
        "source_version": p1_report.get("version", ""),
        "summary": summary,
        "rows": risk_rows,
    }


def write_body_logic_risk_reports(report: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")


def _risk_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        0 if row.get("priority") == "P0" else 1,
        0 if row.get("manifest_gap") else 1,
        str(row.get("sample_id") or ""),
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(category for row in rows for category in row.get("categories", []))
    return {
        "total_risk_recall_samples": len(rows),
        "p0_samples": sum(1 for row in rows if row.get("priority") == "P0"),
        "p1_samples": sum(1 for row in rows if row.get("priority") == "P1"),
        "manifest_gap_samples": sum(1 for row in rows if row.get("manifest_gap")),
        "actual_changed_samples": sum(1 for row in rows if row.get("actual_changed_any")),
        "category_counts": dict(category_counts.most_common()),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Track1 Body/Physical Logic Gate",
        "",
        "这是人体、动作、接触点和空间关系的风险召回 gate；它不替代人工/VLM 视觉裁决。",
        "",
        "## Summary",
        "",
        f"- Total risk recall samples: `{summary.get('total_risk_recall_samples', 0)}`",
        f"- P0 samples: `{summary.get('p0_samples', 0)}`",
        f"- P1 samples: `{summary.get('p1_samples', 0)}`",
        f"- Manifest gap samples: `{summary.get('manifest_gap_samples', 0)}`",
        f"- Actual changed samples: `{summary.get('actual_changed_samples', 0)}`",
        "",
        "## Category Counts",
        "",
    ]
    for category, count in (summary.get("category_counts") or {}).items():
        lines.append(f"- `{category}`: `{count}`")
    lines.extend(["", "## P0 Samples", ""])
    for row in report.get("rows", []):
        if row.get("priority") != "P0":
            continue
        gap = " gap" if row.get("manifest_gap") else ""
        categories = ", ".join(row.get("categories") or [])
        lines.append(f"- `{row.get('sample_id')}`{gap}: {categories}")
    return "\n".join(lines).rstrip() + "\n"
