from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


V5_VERSION = "track1_v5_fid_breakthrough_offline_plan_v1"

CAPTION_FAITHFUL_ROUTE = "caption_faithful_guard"
REFERENCE_FAMILY_ROUTE = "reference_family_primary"
ANTI_TEMPLATE_ROUTE = "anti_template_diversifier"

HARD_REFERENCE_CATEGORIES = {
    "aircraft",
    "document",
    "flags",
    "landmark",
    "medal",
    "uniform",
    "vehicle",
    "weapon",
}

TEXT_RISK_MODES = {"calligraphy", "cyrillic", "document", "seal"}
DEFAULT_TRACK1_IMAGE_MODEL = "gemini-3-pro-image"


def load_payload_rows(path: str | Path, *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = []
        for key in keys:
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError(f"expected rows in {path}")
    return [dict(row) for row in rows if isinstance(row, dict) and row.get("sample_id")]


def build_v5_plan(
    contracts: Iterable[dict[str, Any]],
    expert_routes: Iterable[dict[str, Any]],
    official_routes: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    contract_rows = [dict(row) for row in contracts]
    expert_by_id = {str(row["sample_id"]): dict(row) for row in expert_routes}
    official_by_id = {str(row["sample_id"]): dict(row) for row in official_routes}
    reference_reuse_counts = count_reference_reuse(official_by_id.values())
    rows: list[dict[str, Any]] = []
    for contract in sorted(contract_rows, key=lambda row: str(row.get("sample_id") or "")):
        sample_id = str(contract.get("sample_id") or "")
        if not sample_id:
            continue
        expert = expert_by_id.get(sample_id, {})
        official = official_by_id.get(sample_id, {})
        route = classify_v5_route(
            contract,
            expert,
            official,
            reference_reuse_counts=reference_reuse_counts,
        )
        row = {
            "version": V5_VERSION,
            "sample_id": sample_id,
            "caption": str(contract.get("caption") or official.get("caption") or expert.get("caption") or ""),
            "v5_route": route["v5_route"],
            "secondary_routes": route["secondary_routes"],
            "route_reasons": route["route_reasons"],
            "risk_tags": route["risk_tags"],
            "generation_strategy": route["generation_strategy"],
            "candidate_budget": route["candidate_budget"],
            "v7_family_id": str(official.get("family_id") or ""),
            "style_family": str(contract.get("style_family") or ""),
            "surface_contract": str(contract.get("surface_contract") or ""),
            "aspect_plan": _aspect_plan(contract, official),
            "primary_expert": str(expert.get("primary_expert") or ""),
            "support_experts": list(expert.get("support_experts") or []),
            "recommended_model": _recommended_model(route["v5_route"], expert),
            "reference_assets": _clean_reference_assets(official.get("reference_assets")),
            "reference_asset_notes": [str(item) for item in official.get("reference_asset_notes") or []],
            "overused_reference_assets": route["overused_reference_assets"],
            "hard_constraints": list(official.get("hard_constraints") or []),
            "medium_options": list(official.get("medium_options") or []),
            "composition_hints": list(official.get("composition_hints") or []),
            "style_freedom": list(official.get("style_freedom") or []),
            "fid_risk": list(contract.get("fid_risk") or []),
            "text_contract": dict(contract.get("text_contract") or {}),
            "reference_contract": dict(contract.get("reference_contract") or {}),
            "relation_contract": dict(contract.get("relation_contract") or {}),
            "v7_candidate_strategies": [
                str(item.get("strategy") if isinstance(item, dict) else item)
                for item in official.get("candidate_strategies") or []
            ],
        }
        row["provider_prompt"] = build_v5_provider_prompt(row)
        rows.append(row)
    return rows


def classify_v5_route(
    contract: dict[str, Any],
    expert_route: dict[str, Any] | None = None,
    official_route: dict[str, Any] | None = None,
    *,
    reference_reuse_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    expert_route = expert_route or {}
    official_route = official_route or {}
    reference_reuse_counts = reference_reuse_counts or {}

    caption = str(contract.get("caption") or official_route.get("caption") or "").lower()
    text_contract = dict(contract.get("text_contract") or {})
    reference_contract = dict(contract.get("reference_contract") or {})
    relation_contract = dict(contract.get("relation_contract") or {})
    aspect = _aspect_plan(contract, official_route)
    fid_risk = {str(item) for item in contract.get("fid_risk") or []}
    reference_categories = {str(item) for item in reference_contract.get("categories") or []}
    text_modes = {str(item) for item in text_contract.get("modes") or []}
    style_risks = {str(item) for item in contract.get("style_risk_tags") or []}
    reference_assets = _clean_reference_assets(official_route.get("reference_assets"))
    overused_assets = [
        asset for asset in reference_assets if reference_reuse_counts.get(Path(asset).name, 0) >= 20
    ]

    hard_aas = False
    risk_tags: list[str] = []
    if bool(text_contract.get("required")) or text_modes & TEXT_RISK_MODES:
        hard_aas = True
        risk_tags.append("text_or_symbol_guard")
    if reference_categories & HARD_REFERENCE_CATEGORIES:
        hard_aas = True
        risk_tags.append("real_world_reference_guard")
    if bool(relation_contract.get("required")) or relation_contract.get("checks"):
        hard_aas = True
        risk_tags.append("relation_logic_guard")
    if "scroll" in str(contract.get("surface_contract") or "") or aspect.get("label") in {
        "vertical_scroll",
        "horizontal_scroll",
        "album_spread",
        "panel_story",
    }:
        hard_aas = True
        risk_tags.append("support_surface_guard")

    template_sensitive = (
        "poster_distribution_sensitive" in fid_risk
        or "poster" in caption
        or "propaganda" in caption
        or aspect.get("label") == "portrait_poster"
        or bool(overused_assets)
    )
    if template_sensitive:
        risk_tags.append("template_distribution_guard")
    if overused_assets:
        risk_tags.append("overused_reference_guard")

    if hard_aas:
        route = CAPTION_FAITHFUL_ROUTE
        secondary = [ANTI_TEMPLATE_ROUTE] if template_sensitive else [REFERENCE_FAMILY_ROUTE]
        generation_strategy = "guarded_fidelity"
        candidate_budget = 1
        route_reasons = ["hard AAS risk requires caption-faithful guard"]
    elif template_sensitive or style_risks & {"poster_surface_boundary", "text_or_calligraphy"}:
        route = ANTI_TEMPLATE_ROUTE
        secondary = [REFERENCE_FAMILY_ROUTE]
        generation_strategy = "anti_template_reference_variation"
        candidate_budget = 1
        route_reasons = ["distribution/template risk without hard AAS lock"]
    else:
        route = REFERENCE_FAMILY_ROUTE
        secondary = [ANTI_TEMPLATE_ROUTE]
        generation_strategy = "official_reference_family_match"
        candidate_budget = 1
        route_reasons = ["low hard-risk sample can prioritize official reference family"]

    if overused_assets:
        route_reasons.append(f"{len(overused_assets)} overused reference assets need diversification")

    return {
        "v5_route": route,
        "secondary_routes": secondary,
        "route_reasons": route_reasons,
        "risk_tags": _unique(risk_tags),
        "generation_strategy": generation_strategy,
        "candidate_budget": candidate_budget,
        "overused_reference_assets": overused_assets,
    }


def build_v5_provider_prompt(row: dict[str, Any]) -> str:
    caption = str(row.get("caption") or "").strip()
    route = str(row.get("v5_route") or "")
    sample_id = str(row.get("sample_id") or "")
    aspect = dict(row.get("aspect_plan") or {})
    reference_notes = _reference_note_summary(row.get("reference_asset_notes") or [])
    medium_options = _humanize_list(row.get("medium_options") or [])
    composition_hints = _humanize_list(row.get("composition_hints") or [])
    hard_constraints = _humanize_list(row.get("hard_constraints") or [])

    lines = [
        "Create one finished artwork image for an art-generation challenge.",
        "",
        "OFFICIAL CAPTION",
        caption,
        "",
        "OUTPUT BOUNDARY",
        "- Output the artwork itself, not a gallery wall, product mockup, catalog page, UI screen, or photographed display.",
        "- Do not add sample IDs, filenames, watermarks, UI labels, explanatory captions, or unrelated footer credits.",
        "- Fill the image with the artwork surface unless the official caption explicitly asks for scroll, album, page, or panel support.",
        "",
        "PRIMARY V5 ROUTE",
        f"- {route}: {_route_directive(route)}",
    ]
    if row.get("secondary_routes"):
        lines.append("- Secondary guard: " + ", ".join(str(item) for item in row["secondary_routes"]) + ".")
    if aspect:
        lines.extend(
            [
                "",
                "ASPECT AND SUPPORT",
                f"- Canvas: {aspect.get('width')}x{aspect.get('height')} ({aspect.get('label')}).",
                f"- {aspect.get('prompt_directive')}",
            ]
        )
    lines.extend(
        [
            "",
            "CAPTION-FAITHFUL CONTENT LOCK",
            "- Preserve the named subjects, relationships, supports, text/script requirements, color/light attributes, and style words from the official caption.",
            "- If the caption asks for a poster, scroll, album page, document, or panel layout, keep that support visible as part of the artwork, not as an external display scene.",
        ]
    )
    if hard_constraints:
        lines.append("- Hard constraints: " + "; ".join(hard_constraints) + ".")
    if composition_hints:
        lines.append("- Composition cues: " + "; ".join(composition_hints[:8]) + ".")
    if medium_options:
        lines.append("- Medium cues: " + "; ".join(medium_options[:8]) + ".")
    lines.extend(
        [
            "",
            "FID-FIRST STYLE DISTRIBUTION",
            "- Match the official reference family through medium, crop, density, brushwork, color, line, lighting, and natural artwork texture.",
            "- Avoid a repeated generic generated-poster template: no automatic thick border, no identical centered hero figure layout, no unnecessary white mat, no stock label bands.",
            "- Let the composition be as free as the official art references allow while keeping the caption content clear.",
        ]
    )
    if route == CAPTION_FAITHFUL_ROUTE:
        lines.append("- Because this sample has hard AAS risks, content and relation correctness outrank decorative style freedom.")
    elif route == ANTI_TEMPLATE_ROUTE:
        lines.append("- Prefer a less common crop, viewpoint, figure scale, edge treatment, or painterly handling when the caption permits it.")
    else:
        lines.append("- Prioritize looking like a natural member of the official art style family over looking like a prompt template.")
    if reference_notes:
        lines.extend(["", "REFERENCE FAMILY NOTES"])
        lines.extend(f"- {note}" for note in reference_notes[:4])
    text = "\n".join(lines).strip()
    text = text.replace(sample_id, "") if sample_id else text
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text + "\n"


def summarize_v5_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference_counts = count_reference_reuse(rows)
    return {
        "version": V5_VERSION,
        "total": len(rows),
        "route_counts": dict(Counter(str(row.get("v5_route") or "") for row in rows).most_common()),
        "style_family_counts": dict(Counter(str(row.get("style_family") or "") for row in rows).most_common()),
        "v7_family_counts": dict(Counter(str(row.get("v7_family_id") or "") for row in rows).most_common()),
        "aspect_counts": dict(Counter(str((row.get("aspect_plan") or {}).get("label") or "") for row in rows).most_common()),
        "risk_tag_counts": dict(Counter(tag for row in rows for tag in row.get("risk_tags", [])).most_common()),
        "candidate_budget": sum(int(row.get("candidate_budget") or 0) for row in rows),
        "top_reference_reuse": [
            {"file": file_name, "count": count}
            for file_name, count in Counter(reference_counts).most_common(25)
        ],
    }


def write_v5_reports(
    rows: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    current_image_dir: str | Path | None = None,
    html_limit: int = 1000,
) -> dict[str, Any]:
    out_dir = _safe_output_dir(out_dir)
    prompt_dir = out_dir / "provider_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        prompt_path = prompt_dir / f"{row['sample_id']}_{row['v5_route']}.txt"
        prompt_path.write_text(str(row["provider_prompt"]).rstrip() + "\n", encoding="utf-8")
        row["provider_prompt_path"] = str(prompt_path)

    summary = summarize_v5_plan(rows)
    payload_rows = [_without_prompt(row) for row in rows]
    (out_dir / "track1_v5_fid_breakthrough_plan.json").write_text(
        json.dumps({"summary": summary, "rows": payload_rows}, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "track1_v5_fid_breakthrough_plan.jsonl").open("w", encoding="utf-8") as fh:
        for row in payload_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_csv(payload_rows, out_dir / "track1_v5_fid_breakthrough_plan.csv")
    (out_dir / "track1_v5_fid_breakthrough_plan_zh.md").write_text(_render_md(rows, summary), encoding="utf-8")
    (out_dir / "track1_v5_fid_breakthrough_review_zh.html").write_text(
        _strip_trailing_line_space(
            render_v5_html(
                rows[:html_limit],
                summary,
                html_path=out_dir / "track1_v5_fid_breakthrough_review_zh.html",
                current_image_dir=current_image_dir,
            )
        ),
        encoding="utf-8",
    )
    return {
        "out_dir": str(out_dir),
        "summary": summary,
        "json": str(out_dir / "track1_v5_fid_breakthrough_plan.json"),
        "jsonl": str(out_dir / "track1_v5_fid_breakthrough_plan.jsonl"),
        "csv": str(out_dir / "track1_v5_fid_breakthrough_plan.csv"),
        "md": str(out_dir / "track1_v5_fid_breakthrough_plan_zh.md"),
        "html": str(out_dir / "track1_v5_fid_breakthrough_review_zh.html"),
    }


def render_v5_html(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    html_path: str | Path,
    current_image_dir: str | Path | None = None,
) -> str:
    html_path = Path(html_path)
    cards = "\n".join(_render_card(row, html_path=html_path, current_image_dir=current_image_dir) for row in rows)
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <title>Track1 v5 FID-first route review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f6f6f4; color: #1f2933; }}
    header {{ position: sticky; top: 0; z-index: 2; background: #ffffff; border-bottom: 1px solid #d8d8d2; padding: 16px 24px; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 28px 0 12px; }}
    .summary {{ display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 8px; }}
    .metric {{ background: #ffffff; border: 1px solid #d8d8d2; border-radius: 6px; padding: 10px; }}
    .metric strong {{ display: block; font-size: 18px; }}
    main {{ padding: 18px 24px 40px; }}
    .card {{ background: #ffffff; border: 1px solid #d8d8d2; border-radius: 6px; padding: 14px; margin: 0 0 16px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }}
    .tag {{ font-size: 12px; padding: 3px 7px; border-radius: 999px; background: #e8edf0; }}
    .route-caption_faithful_guard {{ background: #ffe7d6; }}
    .route-reference_family_primary {{ background: #e1f0e6; }}
    .route-anti_template_diversifier {{ background: #e7e3f6; }}
    .grid {{ display: grid; grid-template-columns: 180px 1fr; gap: 14px; }}
    .thumbs {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; align-content: start; }}
    img {{ max-width: 100%; height: 120px; object-fit: contain; background: #f0f0ee; border: 1px solid #ddd; }}
    .caption {{ font-size: 13px; line-height: 1.45; }}
    pre {{ white-space: pre-wrap; max-height: 260px; overflow: auto; background: #f7f7f5; border: 1px solid #ddd; padding: 10px; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; margin: 10px 0 18px; }}
    th, td {{ text-align: left; border-bottom: 1px solid #e2e2dc; padding: 7px; font-size: 13px; }}
  </style>
</head>
<body>
<header>
  <h1>Track1 v5 FID-first 离线路由审核</h1>
  <div class=\"summary\">
    <div class=\"metric\"><span>样本</span><strong>{summary['total']}</strong></div>
    <div class=\"metric\"><span>候选预算</span><strong>{summary['candidate_budget']}</strong></div>
    <div class=\"metric\"><span>路由</span><strong>{len(summary['route_counts'])}</strong></div>
    <div class=\"metric\"><span>风格族</span><strong>{len(summary['style_family_counts'])}</strong></div>
    <div class=\"metric\"><span>画幅</span><strong>{len(summary['aspect_counts'])}</strong></div>
  </div>
</header>
<main>
  <h2>全局统计</h2>
  {_render_summary_tables(summary)}
  <h2>样本审核</h2>
  {cards}
</main>
</body>
</html>
"""


def count_reference_reuse(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for asset in row.get("reference_assets") or []:
            counter[Path(str(asset)).name] += 1
    return dict(counter)


def _aspect_plan(contract: dict[str, Any], official_route: dict[str, Any]) -> dict[str, Any]:
    aspect = dict(contract.get("aspect_plan") or {})
    if not aspect:
        hints = official_route.get("aspect_hints") or []
        if hints and isinstance(hints[0], dict):
            aspect = dict(hints[0])
    return aspect


def _recommended_model(route: str, expert: dict[str, Any]) -> str:
    original = str(expert.get("recommended_model") or "")
    if not original:
        return DEFAULT_TRACK1_IMAGE_MODEL
    if original.startswith("imagen-"):
        return DEFAULT_TRACK1_IMAGE_MODEL
    if "image" in original:
        return original
    return DEFAULT_TRACK1_IMAGE_MODEL


def _route_directive(route: str) -> str:
    if route == CAPTION_FAITHFUL_ROUTE:
        return "protect exact caption content, relation logic, requested text, real-world anchors, and artwork support."
    if route == ANTI_TEMPLATE_ROUTE:
        return "break repeated layout templates while staying inside the official caption and reference style family."
    return "prioritize official reference-family distribution, natural crop, medium, texture, color, and brushwork."


def _reference_note_summary(notes: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for note in notes:
        text = str(note)
        style = re.search(r"style=([^;]+)", text)
        source = re.search(r"source=Images[\\/]+(.+)$", text)
        if style and source:
            result.append(f"{style.group(1)} reference: {source.group(1)}")
        elif style:
            result.append(f"{style.group(1)} reference")
    return _unique(result)


def _clean_reference_assets(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return _unique(str(value).strip() for value in values if str(value).strip())


def _humanize_list(values: Iterable[Any]) -> list[str]:
    return _unique(str(value).replace("_", " ").strip() for value in values if str(value).strip())


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _safe_output_dir(path: str | Path) -> Path:
    out_dir = Path(path).resolve()
    repo = Path(__file__).resolve().parents[1]
    protected = {
        (repo / "submissions" / "track1_submission.json").resolve(),
        (repo / "submissions" / "track1_submission.zip").resolve(),
    }
    protected_image_dir = (repo / "submissions" / "track1" / "images").resolve()
    if out_dir in protected or out_dir == protected_image_dir or protected_image_dir in out_dir.parents:
        raise ValueError(f"protected Track1 output path rejected: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _without_prompt(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "provider_prompt"}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            lineterminator="\n",
            fieldnames=[
                "sample_id",
                "v5_route",
                "secondary_routes",
                "generation_strategy",
                "candidate_budget",
                "style_family",
                "v7_family_id",
                "aspect_label",
                "risk_tags",
                "overused_reference_assets",
                "recommended_model",
                "provider_prompt_path",
                "caption",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "v5_route": row["v5_route"],
                    "secondary_routes": ";".join(row.get("secondary_routes", [])),
                    "generation_strategy": row["generation_strategy"],
                    "candidate_budget": row["candidate_budget"],
                    "style_family": row["style_family"],
                    "v7_family_id": row["v7_family_id"],
                    "aspect_label": (row.get("aspect_plan") or {}).get("label", ""),
                    "risk_tags": ";".join(row.get("risk_tags", [])),
                    "overused_reference_assets": ";".join(Path(asset).name for asset in row.get("overused_reference_assets", [])),
                    "recommended_model": row.get("recommended_model", ""),
                    "provider_prompt_path": row.get("provider_prompt_path", ""),
                    "caption": row["caption"],
                }
            )


def _render_md(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Track1 v5 FID Breakthrough Offline Plan",
        "",
        "这不是提交包，也不会调用 Gemini。它是最后两次提交前的全 1000 离线路由和 prompt 审核包。",
        "",
        "## Summary",
        "",
        f"- total: `{summary['total']}`",
        f"- candidate budget: `{summary['candidate_budget']}`",
        f"- route counts: `{summary['route_counts']}`",
        f"- aspect counts: `{summary['aspect_counts']}`",
        f"- risk tags: `{summary['risk_tag_counts']}`",
        "",
        "## First 40 Samples",
        "",
    ]
    for row in rows[:40]:
        lines.append(
            f"- `{row['sample_id']}` route=`{row['v5_route']}` family=`{row['style_family']}` "
            f"v7_family=`{row['v7_family_id']}` aspect=`{(row.get('aspect_plan') or {}).get('label', '')}` "
            f"risks={','.join(row.get('risk_tags', [])) or 'none'}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_summary_tables(summary: dict[str, Any]) -> str:
    sections = [
        ("v5 路由", summary.get("route_counts", {})),
        ("画幅", summary.get("aspect_counts", {})),
        ("风险标签", summary.get("risk_tag_counts", {})),
        ("v7 family", summary.get("v7_family_counts", {})),
    ]
    html_parts = []
    for title, counts in sections:
        rows = "".join(
            f"<tr><td>{html.escape(str(key))}</td><td>{value}</td></tr>"
            for key, value in list(counts.items())[:20]
        )
        html_parts.append(f"<h3>{html.escape(title)}</h3><table><tbody>{rows}</tbody></table>")
    ref_rows = "".join(
        f"<tr><td>{html.escape(str(item['file']))}</td><td>{item['count']}</td></tr>"
        for item in summary.get("top_reference_reuse", [])[:20]
    )
    html_parts.append(f"<h3>reference 复用 Top20</h3><table><tbody>{ref_rows}</tbody></table>")
    return "\n".join(html_parts)


def _render_card(row: dict[str, Any], *, html_path: Path, current_image_dir: str | Path | None) -> str:
    route = html.escape(str(row.get("v5_route") or ""))
    tags = "".join(
        f"<span class=\"tag\">{html.escape(str(tag))}</span>" for tag in row.get("risk_tags", [])
    )
    refs = "".join(
        f"<img src=\"{html.escape(_relative_url(asset, html_path))}\" alt=\"reference\">"
        for asset in row.get("reference_assets", [])[:4]
    )
    current = ""
    if current_image_dir:
        image_path = Path(current_image_dir) / f"{row['sample_id']}.jpg"
        if image_path.exists():
            current = f"<img src=\"{html.escape(_relative_url(image_path, html_path))}\" alt=\"current/v3\">"
    prompt = html.escape(str(row.get("provider_prompt") or "")[:1800])
    reasons = "".join(f"<li>{html.escape(str(reason))}</li>" for reason in row.get("route_reasons", []))
    return f"""
<section class=\"card\">
  <h3>{html.escape(str(row['sample_id']))} <span class=\"tag route-{route}\">{route}</span></h3>
  <div class=\"caption\">{html.escape(str(row.get('caption') or ''))}</div>
  <div class=\"meta\">
    <span class=\"tag\">style: {html.escape(str(row.get('style_family') or ''))}</span>
    <span class=\"tag\">v7: {html.escape(str(row.get('v7_family_id') or ''))}</span>
    <span class=\"tag\">aspect: {html.escape(str((row.get('aspect_plan') or {}).get('label') or ''))}</span>
    <span class=\"tag\">model: {html.escape(str(row.get('recommended_model') or ''))}</span>
    {tags}
  </div>
  <div class=\"grid\">
    <div>
      <p><strong>当前/v3</strong></p>
      {current}
      <p><strong>官方 reference</strong></p>
      <div class=\"thumbs\">{refs}</div>
    </div>
    <div>
      <p><strong>路由原因</strong></p>
      <ul>{reasons}</ul>
      <p><strong>Prompt 摘要</strong></p>
      <pre>{prompt}</pre>
    </div>
  </div>
</section>
"""


def _relative_url(path: str | Path, html_path: Path) -> str:
    target = Path(path)
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    repo = Path(__file__).resolve().parents[1]
    try:
        return "/" + target.relative_to(repo).as_posix()
    except ValueError:
        pass
    try:
        return target.relative_to(html_path.parent.resolve()).as_posix()
    except ValueError:
        return target.as_uri()


def _strip_trailing_line_space(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"
