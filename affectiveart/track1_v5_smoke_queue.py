from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SMOKE_ANCHORS = [
    "track1_0077",
    "track1_0091",
    "track1_0128",
    "track1_0476",
    "track1_0628",
    "track1_0747",
    "track1_0803",
    "track1_0063",
    "track1_0019",
    "track1_0534",
    "track1_0665",
    "track1_0717",
    "track1_0353",
    "track1_0491",
    "track1_0881",
    "track1_0816",
    "track1_0151",
    "track1_0160",
]

ROUTE_ORDER = [
    "caption_faithful_guard",
    "reference_family_primary",
    "anti_template_diversifier",
]

ROUTE_FILL_ORDER = [
    "caption_faithful_guard",
    "anti_template_diversifier",
    "reference_family_primary",
]

SUPPORT_ASPECTS = {"vertical_scroll", "album_spread", "horizontal_scroll", "panel_story"}

ASPECT_PRIORITY = [
    "portrait_poster",
    "vertical_scroll",
    "album_spread",
    "horizontal_scroll",
    "panel_story",
    "square_artwork",
]

STYLE_PRIORITY = [
    "socialist_realism_poster",
    "ukiyoe_woodblock",
    "ink_wash_painting",
    "gongbi_scroll",
    "ink_wash_scroll",
    "album_leaf_ink",
    "generic_painting",
    "generic_artwork",
    "watercolor_painting",
    "pencil_drawing",
    "pastel_drawing",
]

RISK_PRIORITY = [
    "text_or_symbol_guard",
    "real_world_reference_guard",
    "relation_logic_guard",
    "support_surface_guard",
    "template_distribution_guard",
    "overused_reference_guard",
]


def load_v5_plan(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list):
        raise ValueError(f"expected v5 rows in {path}")
    return [dict(row) for row in rows if isinstance(row, dict) and row.get("sample_id")]


def select_smoke_rows(
    rows: Iterable[dict[str, Any]],
    *,
    target_count: int = 48,
    anchors: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    all_rows = sorted((dict(row) for row in rows), key=lambda row: str(row.get("sample_id") or ""))
    by_id = {str(row.get("sample_id")): row for row in all_rows}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(row: dict[str, Any] | None) -> None:
        if not row or len(selected) >= target_count:
            return
        sample_id = str(row.get("sample_id") or "")
        if sample_id and sample_id not in selected_ids:
            selected.append(row)
            selected_ids.add(sample_id)

    for sample_id in anchors if anchors is not None else DEFAULT_SMOKE_ANCHORS:
        add(by_id.get(str(sample_id)))

    route_targets = _route_targets(target_count)
    for aspect in ASPECT_PRIORITY:
        if aspect in {_aspect_label(row) for row in selected}:
            continue
        add(
            _best_unselected(
                all_rows,
                selected_ids,
                lambda row, aspect=aspect: _aspect_label(row) == aspect
                and (
                    _can_add_aspect_with_route_budget(
                        selected,
                        candidate=row,
                        aspect=aspect,
                        route_targets=route_targets,
                        target_count=target_count,
                    )
                ),
            )
        )

    for route in ROUTE_FILL_ORDER:
        while _count_route(selected, route) < route_targets.get(route, 0) and len(selected) < target_count:
            candidate = _best_unselected(all_rows, selected_ids, lambda row, route=route: row.get("v5_route") == route)
            if not candidate:
                break
            add(candidate)

    for style in STYLE_PRIORITY:
        add(_best_unselected(all_rows, selected_ids, lambda row, style=style: row.get("style_family") == style))

    for risk in RISK_PRIORITY:
        add(_best_unselected(all_rows, selected_ids, lambda row, risk=risk: risk in set(row.get("risk_tags") or [])))

    for row in sorted(all_rows, key=_selection_score, reverse=True):
        add(row)

    return selected[:target_count]


def _can_add_aspect_with_route_budget(
    selected: list[dict[str, Any]],
    *,
    candidate: dict[str, Any],
    aspect: str,
    route_targets: dict[str, int],
    target_count: int,
) -> bool:
    route = str(candidate.get("v5_route") or "")
    if _count_route(selected, route) < route_targets.get(route, target_count):
        return True
    if aspect not in SUPPORT_ASPECTS:
        return False
    remaining_after_add = target_count - (len(selected) + 1)
    min_reference = max(1, round(target_count * 0.25))
    min_anti = max(1, round(target_count * 0.16))
    needed_after_add = max(0, min_reference - _count_route(selected, "reference_family_primary")) + max(
        0,
        min_anti - _count_route(selected, "anti_template_diversifier"),
    )
    return remaining_after_add >= needed_after_add


def build_smoke_packets(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        row = dict(row)
        route = str(row.get("v5_route") or "")
        sample_id = str(row.get("sample_id") or "")
        packet = {
            "rank": rank,
            "sample_id": sample_id,
            "caption": str(row.get("caption") or ""),
            "candidate_strategy": route,
            "provider_prompt": _load_prompt(row),
            "aspect_plan": dict(row.get("aspect_plan") or {}),
            "reference_assets": list(row.get("reference_assets") or []),
            "style_family": str(row.get("style_family") or ""),
            "v7_family_id": str(row.get("v7_family_id") or ""),
            "risk_tags": list(row.get("risk_tags") or []),
            "review_metadata": {
                "candidate_count": 1,
                "recommended_model": str(row.get("recommended_model") or "gemini-3.1-flash-image"),
                "primary_expert": route,
                "v5_route": route,
                "style_family": str(row.get("style_family") or ""),
                "v7_family_id": str(row.get("v7_family_id") or ""),
                "risk_tags": list(row.get("risk_tags") or []),
            },
        }
        packets.append(packet)
    return packets


def write_smoke_queue(rows: list[dict[str, Any]], *, out_dir: str | Path) -> dict[str, str]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    packets = build_smoke_packets(rows)
    summary = _summary(packets)
    json_path = out_dir / "track1_v5_smoke_queue.json"
    jsonl_path = out_dir / "track1_v5_smoke_queue.jsonl"
    csv_path = out_dir / "track1_v5_smoke_queue.csv"
    md_path = out_dir / "track1_v5_smoke_queue_zh.md"
    html_path = out_dir / "track1_v5_smoke_queue_zh.html"

    json_path.write_text(
        json.dumps({"summary": summary, "packets": packets}, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for packet in packets:
            fh.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
    _write_csv(packets, csv_path)
    md_path.write_text(_render_md(packets, summary), encoding="utf-8")
    html_path.write_text(_render_html(packets, summary, html_path=html_path), encoding="utf-8")
    return {
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "md": str(md_path),
        "html": str(html_path),
    }


def write_generated_review(
    *,
    queue_json: str | Path,
    manifest_json: str | Path,
    current_images_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, str]:
    queue_payload = json.loads(Path(queue_json).read_text(encoding="utf-8"))
    manifest_payload = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    packets = [dict(packet) for packet in queue_payload.get("packets", queue_payload.get("rows", []))]
    manifest_rows = [dict(row) for row in manifest_payload.get("rows", manifest_payload if isinstance(manifest_payload, list) else [])]
    packet_by_id = {str(packet.get("sample_id") or ""): packet for packet in packets}
    review_rows = [
        _review_row(row, packet_by_id=packet_by_id, current_images_dir=Path(current_images_dir))
        for row in manifest_rows
    ]
    review_rows.sort(key=lambda row: int(row.get("rank") or 999999))
    summary = _generated_review_summary(review_rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "track1_v5_smoke_generated_review.json"
    html_path = out_dir / "track1_v5_smoke_generated_review_zh.html"
    md_path = out_dir / "track1_v5_smoke_generated_review_zh.md"
    payload = {"summary": summary, "rows": review_rows}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(_render_generated_review_html(review_rows, summary, html_path=html_path), encoding="utf-8")
    md_path.write_text(_render_generated_review_md(review_rows, summary), encoding="utf-8")
    return {"json": str(json_path), "html": str(html_path), "md": str(md_path)}


def _best_unselected(
    rows: list[dict[str, Any]],
    selected_ids: set[str],
    predicate,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if str(row.get("sample_id") or "") not in selected_ids and predicate(row)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_selection_score, reverse=True)[0]


def _selection_score(row: dict[str, Any]) -> tuple[int, int, int, str]:
    risks = set(row.get("risk_tags") or [])
    route = str(row.get("v5_route") or "")
    style = str(row.get("style_family") or "")
    return (
        len(risks),
        len(row.get("reference_assets") or []),
        1 if route == "caption_faithful_guard" and style == "socialist_realism_poster" else 0,
        str(row.get("sample_id") or ""),
    )


def _route_targets(target_count: int) -> dict[str, int]:
    if target_count <= 0:
        return {}
    caption = max(1, round(target_count * 0.46))
    reference = max(1, round(target_count * 0.34))
    anti = max(1, target_count - caption - reference)
    return {
        "caption_faithful_guard": caption,
        "reference_family_primary": reference,
        "anti_template_diversifier": anti,
    }


def _count_route(rows: Iterable[dict[str, Any]], route: str) -> int:
    return sum(1 for row in rows if row.get("v5_route") == route)


def _aspect_label(row: dict[str, Any]) -> str:
    return str((row.get("aspect_plan") or {}).get("label") or "")


def _load_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("provider_prompt") or "")
    if prompt:
        return prompt if prompt.endswith("\n") else prompt + "\n"
    path_text = str(row.get("provider_prompt_path") or "")
    if path_text:
        path = Path(path_text)
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise ValueError(f"missing provider prompt for {row.get('sample_id')}")


def _summary(packets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(packets),
        "route_counts": dict(Counter(packet.get("candidate_strategy", "") for packet in packets).most_common()),
        "style_family_counts": dict(Counter(packet.get("style_family", "") for packet in packets).most_common()),
        "aspect_counts": dict(Counter(_aspect_label(packet) for packet in packets).most_common()),
        "risk_tag_counts": dict(Counter(tag for packet in packets for tag in packet.get("risk_tags", [])).most_common()),
        "model_counts": dict(
            Counter(str((packet.get("review_metadata") or {}).get("recommended_model") or "") for packet in packets).most_common()
        ),
    }


def _write_csv(packets: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            lineterminator="\n",
            fieldnames=[
                "sample_id",
                "candidate_strategy",
                "style_family",
                "v7_family_id",
                "aspect_label",
                "recommended_model",
                "risk_tags",
                "caption",
            ],
        )
        writer.writeheader()
        for packet in packets:
            writer.writerow(
                {
                    "sample_id": packet["sample_id"],
                    "candidate_strategy": packet["candidate_strategy"],
                    "style_family": packet["style_family"],
                    "v7_family_id": packet["v7_family_id"],
                    "aspect_label": _aspect_label(packet),
                    "recommended_model": (packet.get("review_metadata") or {}).get("recommended_model", ""),
                    "risk_tags": ";".join(packet.get("risk_tags") or []),
                    "caption": packet["caption"],
                }
            )


def _review_row(
    manifest_row: dict[str, Any],
    *,
    packet_by_id: dict[str, dict[str, Any]],
    current_images_dir: Path,
) -> dict[str, Any]:
    sample_id = str(manifest_row.get("sample_id") or "")
    packet = packet_by_id.get(sample_id, {})
    metadata = _load_json_file(manifest_row.get("metadata_path"))
    actual_model = str(metadata.get("image_model") or (metadata.get("provider_metadata") or {}).get("model") or manifest_row.get("actual_model") or manifest_row.get("model") or "")
    current_image = current_images_dir / f"{sample_id}.jpg"
    return {
        "rank": packet.get("rank", manifest_row.get("rank", "")),
        "sample_id": sample_id,
        "caption": packet.get("caption", manifest_row.get("caption", "")),
        "candidate_strategy": packet.get("candidate_strategy", manifest_row.get("candidate_strategy", "")),
        "style_family": packet.get("style_family", ""),
        "aspect_label": _aspect_label(packet),
        "risk_tags": packet.get("risk_tags", []),
        "status": manifest_row.get("status", ""),
        "requested_model": manifest_row.get("model", ""),
        "actual_model": actual_model,
        "current_image": str(current_image),
        "current_exists": current_image.exists(),
        "candidate_image": str(manifest_row.get("image_path") or ""),
        "candidate_exists": Path(str(manifest_row.get("image_path") or "")).exists(),
        "reference_board": str(manifest_row.get("reference_image_path") or ""),
        "reference_board_exists": Path(str(manifest_row.get("reference_image_path") or "")).exists(),
        "reference_assets": packet.get("reference_assets", [])[:4],
        "reference_fallback_without_reference": bool(
            metadata.get("reference_image_fallback_without_reference")
            or manifest_row.get("reference_image_fallback_without_reference")
        ),
        "reference_fallback_reason": str(
            metadata.get("reference_image_fallback_reason")
            or manifest_row.get("reference_image_fallback_reason")
            or ""
        ),
    }


def _load_json_file(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _generated_review_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "missing_current": sum(1 for row in rows if not row.get("current_exists")),
        "missing_candidate": sum(1 for row in rows if not row.get("candidate_exists")),
        "missing_reference_board": sum(1 for row in rows if not row.get("reference_board_exists")),
        "reference_fallback_without_reference": sum(1 for row in rows if row.get("reference_fallback_without_reference")),
        "status_counts": dict(Counter(str(row.get("status") or "") for row in rows).most_common()),
        "route_counts": dict(Counter(str(row.get("candidate_strategy") or "") for row in rows).most_common()),
        "actual_model_counts": dict(Counter(str(row.get("actual_model") or "") for row in rows).most_common()),
    }


def _render_generated_review_md(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Track1 v5 Smoke Generated Review",
        "",
        "这是 v5 smoke 真实生成后的人工审核入口。它不是提交包，也不自动接受替换。",
        "",
        f"- total: `{summary['total']}`",
        f"- missing current: `{summary['missing_current']}`",
        f"- missing candidate: `{summary['missing_candidate']}`",
        f"- reference fallback without reference: `{summary['reference_fallback_without_reference']}`",
        f"- actual models: `{summary['actual_model_counts']}`",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        fallback = "fallback" if row.get("reference_fallback_without_reference") else "reference_ok"
        lines.append(
            f"- `{row['sample_id']}` route=`{row['candidate_strategy']}` "
            f"model=`{row['actual_model']}` {fallback}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_md(packets: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Track1 v5 Smoke Queue",
        "",
        "这是 v5 full-1000 生成前的分层 smoke queue。它只用于小样本生成和人工/自动审核，不是提交包。",
        "",
        f"- total: `{summary['total']}`",
        f"- routes: `{summary['route_counts']}`",
        f"- aspects: `{summary['aspect_counts']}`",
        f"- styles: `{summary['style_family_counts']}`",
        "",
        "## Samples",
        "",
    ]
    for packet in packets:
        lines.append(
            f"- `{packet['sample_id']}` route=`{packet['candidate_strategy']}` "
            f"style=`{packet['style_family']}` aspect=`{_aspect_label(packet)}` "
            f"risks={','.join(packet.get('risk_tags') or []) or 'none'}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_html(packets: list[dict[str, Any]], summary: dict[str, Any], *, html_path: Path) -> str:
    cards = "\n".join(_render_card(packet, html_path=html_path) for packet in packets)
    text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Track1 v5 smoke queue</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f6f6f4; color: #1f2933; }}
    header {{ position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #ddd; padding: 14px 22px; }}
    main {{ padding: 18px 22px 40px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .metric, .card {{ background: #fff; border: 1px solid #d8d8d2; border-radius: 6px; }}
    .metric {{ padding: 8px 10px; }}
    .card {{ display: grid; grid-template-columns: 230px 1fr; gap: 14px; padding: 14px; margin-bottom: 14px; }}
    .tag {{ display: inline-block; margin: 2px 4px 2px 0; padding: 3px 7px; border-radius: 999px; background: #e8edf0; font-size: 12px; }}
    img {{ max-width: 100%; height: 116px; object-fit: contain; background: #f0f0ee; border: 1px solid #ddd; }}
    .refs {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; }}
    pre {{ white-space: pre-wrap; max-height: 220px; overflow: auto; background: #f7f7f5; border: 1px solid #ddd; padding: 10px; font-size: 12px; }}
  </style>
</head>
<body>
<header>
  <h1>Track1 v5 smoke queue 审核</h1>
  <div class="summary">
    <div class="metric">样本 <strong>{summary['total']}</strong></div>
    <div class="metric">路由 <strong>{len(summary['route_counts'])}</strong></div>
    <div class="metric">画幅 <strong>{len(summary['aspect_counts'])}</strong></div>
    <div class="metric">风格 <strong>{len(summary['style_family_counts'])}</strong></div>
  </div>
</header>
<main>
{cards}
</main>
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"


def _render_card(packet: dict[str, Any], *, html_path: Path) -> str:
    refs = "\n".join(
        f'<img src="{html.escape(_relative_url(path, html_path))}" alt="reference">'
        for path in packet.get("reference_assets", [])[:4]
    )
    tags = "\n".join(f'<span class="tag">{html.escape(str(tag))}</span>' for tag in packet.get("risk_tags", []))
    prompt = html.escape(str(packet.get("provider_prompt") or "")[:1600])
    return f"""
<section class="card">
  <div>
    <h3>{html.escape(str(packet.get("sample_id") or ""))}</h3>
    <div class="refs">{refs}</div>
  </div>
  <div>
    <div><span class="tag">{html.escape(str(packet.get("candidate_strategy") or ""))}</span><span class="tag">{html.escape(str(packet.get("style_family") or ""))}</span><span class="tag">{html.escape(_aspect_label(packet))}</span></div>
    <p>{html.escape(str(packet.get("caption") or ""))}</p>
    <div>{tags}</div>
    <pre>{prompt}</pre>
  </div>
</section>
"""


def _render_generated_review_html(rows: list[dict[str, Any]], summary: dict[str, Any], *, html_path: Path) -> str:
    cards = "\n".join(_render_generated_review_card(row, html_path=html_path) for row in rows)
    text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Track1 v5 smoke generated review</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f2; color: #1f2933; }}
    header {{ position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #d9d9d4; padding: 14px 22px; }}
    main {{ padding: 18px 22px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .metric, .sample {{ background: #fff; border: 1px solid #d8d8d2; border-radius: 6px; }}
    .metric {{ padding: 8px 10px; }}
    .sample {{ padding: 14px; margin-bottom: 16px; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 5px; margin: 7px 0; }}
    .tag {{ display: inline-block; padding: 3px 7px; border-radius: 999px; background: #e8edf0; font-size: 12px; }}
    .warn {{ background: #fff1d6; color: #7c3f00; }}
    .bad {{ background: #fde2e2; color: #8f1d1d; }}
    .caption {{ font-size: 14px; line-height: 1.45; margin: 8px 0 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; align-items: start; }}
    figure {{ margin: 0; background: #f8f8f6; border: 1px solid #ddd; border-radius: 6px; padding: 8px; }}
    img {{ width: 100%; height: 360px; object-fit: contain; background: #eee; }}
    figcaption {{ font-size: 12px; line-height: 1.35; color: #4b5563; margin-top: 6px; word-break: break-word; }}
    .refs {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; margin-top: 8px; }}
    .refs img {{ height: 110px; }}
    .path {{ display: block; color: #6b7280; }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} img {{ height: auto; max-height: 520px; }} }}
  </style>
</head>
<body>
<header>
  <h1>Track1 v5 smoke 生成结果审核</h1>
  <div class="summary">
    <div class="metric">样本 <strong>{summary['total']}</strong></div>
    <div class="metric">缺 current <strong>{summary['missing_current']}</strong></div>
    <div class="metric">缺 candidate <strong>{summary['missing_candidate']}</strong></div>
    <div class="metric">reference fallback <strong>{summary['reference_fallback_without_reference']}</strong></div>
    <div class="metric">真实模型 <strong>{html.escape(str(summary['actual_model_counts']))}</strong></div>
  </div>
</header>
<main>
{cards}
</main>
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"


def _render_generated_review_card(row: dict[str, Any], *, html_path: Path) -> str:
    fallback = bool(row.get("reference_fallback_without_reference"))
    warnings: list[str] = []
    if not row.get("current_exists"):
        warnings.append("current 缺失")
    if not row.get("candidate_exists"):
        warnings.append("candidate 缺失")
    if not row.get("reference_board_exists"):
        warnings.append("reference board 缺失")
    if fallback:
        warnings.append("reference fallback：是")
    warning_tags = "\n".join(f'<span class="tag warn">{html.escape(item)}</span>' for item in warnings)
    risk_tags = "\n".join(f'<span class="tag">{html.escape(str(tag))}</span>' for tag in row.get("risk_tags", []))
    refs = "\n".join(
        f'<img src="{html.escape(_relative_url(path, html_path))}" alt="official reference asset">'
        for path in row.get("reference_assets", [])[:4]
    )
    fallback_reason = ""
    if fallback:
        fallback_reason = f"<p><strong>fallback reason:</strong> {html.escape(str(row.get('reference_fallback_reason') or ''))}</p>"
    return f"""
<section class="sample">
  <h2>{html.escape(str(row.get("sample_id") or ""))}</h2>
  <div class="tags">
    <span class="tag">{html.escape(str(row.get("candidate_strategy") or ""))}</span>
    <span class="tag">{html.escape(str(row.get("style_family") or ""))}</span>
    <span class="tag">{html.escape(str(row.get("aspect_label") or ""))}</span>
    <span class="tag">actual model: {html.escape(str(row.get("actual_model") or ""))}</span>
    {warning_tags}
    {risk_tags}
  </div>
  <p class="caption">{html.escape(str(row.get("caption") or ""))}</p>
  <div class="grid">
    <figure>
      <img src="{html.escape(_relative_url(row.get("current_image"), html_path))}" alt="current champion image">
      <figcaption><strong>current：当前 champion 提交包</strong><span class="path">{html.escape(str(row.get("current_image") or ""))}</span></figcaption>
    </figure>
    <figure>
      <img src="{html.escape(_relative_url(row.get("candidate_image"), html_path))}" alt="v5 smoke candidate image">
      <figcaption><strong>candidate：v5 smoke 新候选</strong><span class="path">{html.escape(str(row.get("candidate_image") or ""))}</span></figcaption>
    </figure>
    <figure>
      <img src="{html.escape(_relative_url(row.get("reference_board"), html_path))}" alt="reference board">
      <figcaption><strong>reference board：官方 reference 组合图</strong><span class="path">{html.escape(str(row.get("reference_board") or ""))}</span></figcaption>
    </figure>
  </div>
  <div class="refs">{refs}</div>
  {fallback_reason}
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
