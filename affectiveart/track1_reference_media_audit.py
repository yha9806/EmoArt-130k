from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


POSTER_CAPTION_KEYWORDS = (
    "propaganda poster",
    "poster",
    "cyrillic typography",
    "bold cyrillic",
    "typography",
)

POSTER_ASSET_KEYWORDS = (
    "poster",
    "tass",
    "frontpage",
    "allforthefront",
    "allforvictory",
    "liberate",
    "tovictory",
)


def load_routes_json(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("routes"), list):
        rows = payload["routes"]
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise ValueError("routes JSON must be a list or an object with routes/rows")
    return [dict(row) for row in rows if isinstance(row, dict)]


def build_reference_media_audit(
    old_routes: Iterable[dict[str, Any]],
    new_routes: Iterable[dict[str, Any]],
    *,
    partial_candidate_image_dir: str | Path | None = None,
) -> dict[str, Any]:
    old_by_id = _index_routes(old_routes)
    new_by_id = _index_routes(new_routes)
    changed_rows = _changed_rows(old_by_id, new_by_id)
    sample_source_risks = _sample_source_risks(new_by_id)
    partial_candidates = _partial_candidates(partial_candidate_image_dir)
    tainted_ids = sorted(
        set(partial_candidates)
        & ({row["sample_id"] for row in changed_rows} | {row["sample_id"] for row in sample_source_risks})
    )
    tainted_rows = [
        {
            "sample_id": sample_id,
            "caption": str(new_by_id.get(sample_id, old_by_id.get(sample_id, {})).get("caption") or ""),
            "partial_candidate_images": partial_candidates.get(sample_id, []),
            "reason": _taint_reason(sample_id, changed_rows, sample_source_risks),
        }
        for sample_id in tainted_ids
    ]
    return {
        "summary": {
            "old_total": len(old_by_id),
            "new_total": len(new_by_id),
            "changed_samples": len(changed_rows),
            "sample_source_manual_review": len(sample_source_risks),
            "partial_candidate_samples": len(partial_candidates),
            "tainted_partial_samples": len(tainted_ids),
        },
        "changed_rows": changed_rows,
        "sample_source_risks": sample_source_risks,
        "tainted_partial_samples": tainted_ids,
        "tainted_partial_rows": tainted_rows,
    }


def write_reference_media_audit_artifacts(report: dict[str, Any], *, out_dir: str | Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(report, ensure_ascii=False))
    _write_boards(payload, out_dir)
    json_path = out_dir / "track1_reference_media_audit.json"
    md_path = out_dir / "track1_reference_media_audit_zh.md"
    html_path = out_dir / "track1_reference_media_audit_zh.html"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_reference_media_audit_md(payload), encoding="utf-8")
    html_path.write_text(render_reference_media_audit_html(payload, out_html=html_path), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path), "html": str(html_path)}


def render_reference_media_audit_md(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Track1 Reference Media Audit",
        "",
        "这份报告用于审查 reference board 是否满足 caption 的媒介要求，尤其是 poster/propaganda/Cyrillic typography 是否误用了油画、风景或普通 family reference。",
        "",
        "## 摘要",
        "",
        f"- Changed samples: {summary.get('changed_samples', 0)}",
        f"- Sample-source manual review: {summary.get('sample_source_manual_review', 0)}",
        f"- Partial candidate samples: {summary.get('partial_candidate_samples', 0)}",
        f"- Tainted partial samples: {summary.get('tainted_partial_samples', 0)}",
        "",
        "## Changed Samples",
        "",
    ]
    for row in report.get("changed_rows", []):
        lines.append(
            f"- `{row.get('sample_id')}`: {row.get('old_source')} -> {row.get('new_source')} "
            f"({row.get('new_style_key', '')})"
        )
    lines.extend(["", "## Sample-Source Manual Review", ""])
    for row in report.get("sample_source_risks", []):
        lines.append(
            f"- `{row.get('sample_id')}`: poster-like refs {row.get('poster_like_assets')}/"
            f"{row.get('reference_asset_count')} ({row.get('manual_review_reason')})"
        )
    lines.extend(["", "## Tainted Partial Samples", ""])
    for sample_id in report.get("tainted_partial_samples", []):
        lines.append(f"- `{sample_id}`")
    return "\n".join(lines).rstrip() + "\n"


def render_reference_media_audit_html(report: dict[str, Any], *, out_html: str | Path) -> str:
    out_html = Path(out_html)
    summary = report.get("summary", {})
    changed_cards = "\n".join(_render_changed_card(row, out_html) for row in report.get("changed_rows", []))
    sample_cards = "\n".join(_render_sample_risk_card(row, out_html) for row in report.get("sample_source_risks", []))
    tainted_cards = "\n".join(_render_tainted_card(row, out_html) for row in report.get("tainted_partial_rows", []))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track1 Reference Media Audit</title>
  <style>
    :root {{
      --bg: #f6f4ef;
      --panel: #fff;
      --line: #d8d2c7;
      --text: #202225;
      --muted: #5f6872;
      --bad: #b42318;
      --good: #0f766e;
      --warn: #a15c07;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 244, 239, 0.96);
    }}
    h1 {{ margin: 0 0 8px; font-size: 23px; letter-spacing: 0; }}
    .intro {{ margin: 0; max-width: 1180px; color: var(--muted); font-size: 14px; }}
    main {{ padding: 20px 24px 48px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin-bottom: 22px;
    }}
    .metric {{
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .metric strong {{ display: block; font-size: 23px; color: var(--good); }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    h2 {{ margin: 28px 0 12px; font-size: 19px; letter-spacing: 0; }}
    .card {{
      margin: 0 0 18px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .head {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 14px;
      margin-bottom: 12px;
    }}
    h3 {{ margin: 0 0 6px; font-size: 17px; letter-spacing: 0; }}
    .caption {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .badge {{
      display: inline-block;
      align-self: start;
      padding: 6px 8px;
      border-radius: 6px;
      border: 1px solid var(--line);
      color: var(--warn);
      background: #fff8eb;
      font-size: 12px;
      white-space: nowrap;
    }}
    .board-grid, .candidate-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #eee9df;
    }}
    img {{
      display: block;
      width: 100%;
      height: 330px;
      object-fit: contain;
      background: #e8e1d6;
    }}
    .candidate-grid img {{ height: 260px; }}
    figcaption {{
      min-height: 48px;
      padding: 8px 10px;
      border-top: 1px solid var(--line);
      background: #fff;
      font-size: 13px;
    }}
    .meta {{ display: block; color: var(--muted); overflow-wrap: anywhere; }}
    .old {{ color: var(--bad); }}
    .new {{ color: var(--good); }}
    details {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      padding: 10px;
      border-radius: 6px;
      background: #f7f7f5;
      color: #27323a;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Track1 Reference Media Audit</h1>
    <p class="intro">目标：检查 reference board 是否满足官方 caption 的媒介要求。重点看 poster/propaganda/Cyrillic typography 是否误用了油画、风景、普通 family reference；同时隔离旧 partial run 里被污染的候选图。</p>
  </header>
  <main>
    <section class="summary">
      {_metric('Changed samples', summary.get('changed_samples', 0))}
      {_metric('Sample-source risks', summary.get('sample_source_manual_review', 0))}
      {_metric('Partial candidate samples', summary.get('partial_candidate_samples', 0))}
      {_metric('Tainted partial samples', summary.get('tainted_partial_samples', 0))}
    </section>
    <h2>一、v1 -> v2 reference board 变化</h2>
    {changed_cards or '<p>没有变化样本。</p>'}
    <h2>二、sample-specific reference 残留风险</h2>
    {sample_cards or '<p>没有 sample-specific 残留风险。</p>'}
    <h2>三、旧 partial run 污染候选隔离</h2>
    {tainted_cards or '<p>没有检测到污染候选。</p>'}
  </main>
</body>
</html>
"""


def _changed_rows(old_by_id: dict[str, dict[str, Any]], new_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(set(old_by_id) & set(new_by_id)):
        old = old_by_id[sample_id]
        new = new_by_id[sample_id]
        if _asset_names(old) == _asset_names(new) and old.get("reference_asset_source") == new.get("reference_asset_source"):
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "caption": str(new.get("caption") or old.get("caption") or ""),
                "family_id": str(new.get("family_id") or old.get("family_id") or ""),
                "old_source": str(old.get("reference_asset_source") or ""),
                "new_source": str(new.get("reference_asset_source") or ""),
                "old_style_key": str(old.get("reference_style_key") or ""),
                "new_style_key": str(new.get("reference_style_key") or ""),
                "old_reference_assets": _as_list(old.get("reference_assets")),
                "new_reference_assets": _as_list(new.get("reference_assets")),
            }
        )
    return rows


def _sample_source_risks(new_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, route in sorted(new_by_id.items()):
        if route.get("reference_asset_source") != "sample" or not _requires_poster_reference(route.get("caption")):
            continue
        assets = _as_list(route.get("reference_assets"))
        poster_like_count = sum(1 for asset in assets if _is_poster_like_asset(asset))
        required_count = min(2, len(assets)) if assets else 1
        if poster_like_count >= required_count:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "caption": str(route.get("caption") or ""),
                "family_id": str(route.get("family_id") or ""),
                "source": str(route.get("reference_asset_source") or ""),
                "style_key": str(route.get("reference_style_key") or ""),
                "poster_like_assets": poster_like_count,
                "reference_asset_count": len(assets),
                "manual_review_reason": "sample-specific reference bypasses caption_media and has too few poster-like assets",
                "old_reference_assets": assets,
                "new_reference_assets": assets,
            }
        )
    return rows


def _partial_candidates(path_value: str | Path | None) -> dict[str, list[str]]:
    if not path_value:
        return {}
    root = Path(path_value)
    if not root.exists() or not root.is_dir():
        return {}
    rows: dict[str, list[str]] = {}
    for path in sorted(root.glob("*.png")):
        sample_id = _sample_id_from_candidate_name(path.name)
        if not sample_id:
            continue
        rows.setdefault(sample_id, []).append(str(path))
    return rows


def _sample_id_from_candidate_name(name: str) -> str:
    parts = Path(name).stem.split("_")
    if len(parts) < 2 or parts[0] != "track1":
        return ""
    return f"{parts[0]}_{parts[1]}"


def _safe_board_filename(sample_id: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in sample_id)
    safe = safe.strip("._-") or "sample"
    return f"{safe}.jpg"


def _taint_reason(sample_id: str, changed_rows: list[dict[str, Any]], sample_source_risks: list[dict[str, Any]]) -> str:
    if sample_id in {row["sample_id"] for row in changed_rows}:
        return "旧 partial candidate 使用了 v1 reference；v2 reference 已变化，不能进入候选池。"
    if sample_id in {row["sample_id"] for row in sample_source_risks}:
        return "sample-specific reference 仍需人工复核，旧 partial candidate 暂停使用。"
    return "需要复核。"


def _write_boards(report: dict[str, Any], out_dir: Path) -> None:
    for section in ("changed_rows", "sample_source_risks"):
        for row in report.get(section, []):
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                continue
            safe_name = _safe_board_filename(sample_id)
            old_path = out_dir / "reference_boards" / "old" / safe_name
            new_path = out_dir / "reference_boards" / "new" / safe_name
            row["old_reference_board_path"] = _maybe_write_board(row.get("old_reference_assets", []), old_path)
            row["new_reference_board_path"] = _maybe_write_board(row.get("new_reference_assets", []), new_path)


def _maybe_write_board(values: Any, output_path: Path) -> str:
    paths = [Path(value) for value in _as_list(values) if Path(value).exists()]
    if not paths:
        return ""
    _write_reference_board(paths, output_path)
    return str(output_path)


def _write_reference_board(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = paths[:4]
    cell_w, cell_h = 384, 288
    cols = 2 if len(selected) > 1 else 1
    rows = (len(selected) + cols - 1) // cols
    board = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for index, path in enumerate(selected):
        with Image.open(path) as image:
            tile = image.convert("RGB")
            tile.thumbnail((cell_w - 12, cell_h - 12), Image.Resampling.LANCZOS)
            x = (index % cols) * cell_w + (cell_w - tile.width) // 2
            y = (index // cols) * cell_h + (cell_h - tile.height) // 2
            board.paste(tile, (x, y))
    board.save(output_path, format="JPEG", quality=92)


def _render_changed_card(row: dict[str, Any], out_html: Path) -> str:
    return _render_board_card(
        row,
        out_html,
        badge=f"{row.get('old_source', '')} -> {row.get('new_source', '')}",
        reason="v2 已改 reference source；旧 reference board 不能继续用于生成。",
    )


def _render_sample_risk_card(row: dict[str, Any], out_html: Path) -> str:
    badge = f"manual review {row.get('poster_like_assets', 0)}/{row.get('reference_asset_count', 0)}"
    return _render_board_card(row, out_html, badge=badge, reason=str(row.get("manual_review_reason") or ""))


def _render_board_card(row: dict[str, Any], out_html: Path, *, badge: str, reason: str) -> str:
    old_board = _img_figure(
        row.get("old_reference_board_path"),
        "旧 reference board",
        f"{row.get('old_source', row.get('source', ''))} {row.get('old_style_key', '')}",
        out_html,
        "old",
    )
    new_board = _img_figure(
        row.get("new_reference_board_path"),
        "v2 reference board",
        f"{row.get('new_source', row.get('source', ''))} {row.get('new_style_key', row.get('style_key', ''))}",
        out_html,
        "new",
    )
    assets = "\n".join(_escape(Path(asset).name) for asset in row.get("new_reference_assets", []))
    return f"""
<section class="card">
  <div class="head">
    <div>
      <h3>{_escape(row.get('sample_id', ''))}</h3>
      <p class="caption">{_escape(row.get('caption', ''))}</p>
    </div>
    <span class="badge">{_escape(badge)}</span>
  </div>
  <p class="caption">{_escape(reason)}</p>
  <div class="board-grid">{old_board}{new_board}</div>
  <details><summary>v2 reference asset filenames</summary><pre>{assets}</pre></details>
</section>
"""


def _render_tainted_card(row: dict[str, Any], out_html: Path) -> str:
    figures = "".join(
        _img_figure(path, Path(path).name, "旧 partial candidate，暂停使用", out_html, "old")
        for path in row.get("partial_candidate_images", [])
    )
    return f"""
<section class="card">
  <div class="head">
    <div>
      <h3>{_escape(row.get('sample_id', ''))}</h3>
      <p class="caption">{_escape(row.get('caption', ''))}</p>
    </div>
    <span class="badge">tainted</span>
  </div>
  <p class="caption">{_escape(row.get('reason', ''))}</p>
  <div class="candidate-grid">{figures}</div>
</section>
"""


def _img_figure(path_value: Any, title: str, meta: str, out_html: Path, class_name: str) -> str:
    if not path_value:
        return f"<figure><figcaption><strong>{_escape(title)}</strong><span class=\"meta\">无图片</span></figcaption></figure>"
    src = _relative_src(Path(str(path_value)), out_html.parent)
    return f"""
<figure>
  <img src="{_escape(src)}" alt="{_escape(title)}" loading="lazy">
  <figcaption><strong class="{_escape(class_name)}">{_escape(title)}</strong><span class="meta">{_escape(meta)}</span></figcaption>
</figure>
"""


def _relative_src(path: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return str(path)


def _metric(label: str, value: Any) -> str:
    return f'<div class="metric"><strong>{_escape(value)}</strong><span>{_escape(label)}</span></div>'


def _index_routes(routes: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): dict(row) for row in routes if str(row.get("sample_id") or "")}


def _asset_names(row: dict[str, Any]) -> list[str]:
    return [Path(value).name for value in _as_list(row.get("reference_assets"))]


def _as_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


def _requires_poster_reference(caption_value: Any) -> bool:
    caption = str(caption_value or "").lower()
    return any(keyword in caption for keyword in POSTER_CAPTION_KEYWORDS)


def _is_poster_like_asset(path_value: str) -> bool:
    name = Path(path_value).name.lower()
    return any(keyword in name for keyword in POSTER_ASSET_KEYWORDS)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)
