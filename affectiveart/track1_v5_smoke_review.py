from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from typing import Any, Iterable


def build_review_rows(
    manifest_paths: Iterable[str | Path],
    *,
    current_image_dir: str | Path = "submissions/track1/images",
) -> list[dict[str, Any]]:
    attempts_by_sample: dict[str, list[dict[str, Any]]] = {}
    for manifest_path in manifest_paths:
        for row in _load_manifest_rows(manifest_path):
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id:
                attempts_by_sample.setdefault(sample_id, []).append(dict(row))

    current_dir = Path(current_image_dir)
    review_rows: list[dict[str, Any]] = []
    for sample_id in sorted(attempts_by_sample):
        attempts = attempts_by_sample[sample_id]
        chosen = _choose_candidate(attempts)
        first = attempts[0]
        review_rows.append(
            {
                "sample_id": sample_id,
                "caption": chosen.get("caption") or first.get("caption") or "",
                "current_image_path": str(current_dir / f"{sample_id}.jpg"),
                "candidate_image_path": str(chosen.get("image_path") or ""),
                "reference_image_path": _first_nonempty(attempts, "reference_image_path"),
                "candidate_status": str(chosen.get("status") or ""),
                "candidate_model": str(chosen.get("model") or ""),
                "candidate_strategy": str(chosen.get("candidate_strategy") or ""),
                "attempt_statuses": " -> ".join(str(row.get("status") or "") for row in attempts),
                "attempt_models": " -> ".join(str(row.get("model") or "") for row in attempts),
                "error": str(chosen.get("error") or _first_nonempty(attempts, "error")),
                "notes": _notes_for_attempts(attempts, chosen),
                "provider_prompt": str(chosen.get("provider_prompt") or first.get("provider_prompt") or ""),
            }
        )
    return review_rows


def write_review_reports(rows: list[dict[str, Any]], *, out_dir: str | Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "track1_v5_smoke_review_rows.json"
    csv_path = out_dir / "track1_v5_smoke_review_rows.csv"
    md_path = out_dir / "track1_v5_smoke_review_summary_zh.md"
    html_path = out_dir / "track1_v5_smoke_review_zh.html"

    summary = _summary(rows)
    json_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(rows, csv_path)
    md_path.write_text(_render_markdown(rows, summary), encoding="utf-8")
    html_path.write_text(_render_html(rows, summary, html_path), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path), "html": str(html_path)}


def _load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    return [dict(row) for row in payload.get("rows", [])]


def _choose_candidate(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [row for row in attempts if row.get("status") in {"generated", "cached"} and row.get("image_path")]
    if generated:
        return generated[-1]
    return attempts[-1]


def _first_nonempty(rows: list[dict[str, Any]], key: str) -> str:
    for row in rows:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _notes_for_attempts(attempts: list[dict[str, Any]], chosen: dict[str, Any]) -> str:
    errors = [str(row.get("error") or "") for row in attempts if row.get("status") == "error"]
    if chosen.get("status") in {"generated", "cached"} and errors:
        if any("imagen-4-ultra" in str(row.get("model") or "") for row in attempts):
            return "前序 Imagen/API 失败，已用 Gemini retry 救回。"
        return "前序生成失败，后续 retry 救回。"
    if chosen.get("status") == "error":
        return "仍需重跑或人工保留 current。"
    return ""


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "generated_or_cached": sum(1 for row in rows if row.get("candidate_status") in {"generated", "cached"}),
        "error": sum(1 for row in rows if row.get("candidate_status") == "error"),
        "models": _counts(str(row.get("candidate_model") or "") for row in rows),
        "strategies": _counts(str(row.get("candidate_strategy") or "") for row in rows),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "sample_id",
        "candidate_status",
        "candidate_strategy",
        "candidate_model",
        "attempt_statuses",
        "attempt_models",
        "notes",
        "caption",
        "current_image_path",
        "candidate_image_path",
        "reference_image_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Track1 v5 smoke review summary",
        "",
        f"- total: {summary['total']}",
        f"- generated_or_cached: {summary['generated_or_cached']}",
        f"- error: {summary['error']}",
        f"- models: {summary['models']}",
        f"- strategies: {summary['strategies']}",
        "",
        "## 需要人工看的重点",
    ]
    for row in rows:
        note = row.get("notes") or ""
        if note or row.get("candidate_status") == "error":
            lines.append(f"- {row['sample_id']}: {row.get('candidate_status')} / {note}")
    return "\n".join(lines) + "\n"


def _render_html(rows: list[dict[str, Any]], summary: dict[str, Any], html_path: Path) -> str:
    cards = "\n".join(_render_card(row, html_path) for row in rows)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track1 v5 smoke 中文审核</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #181818; background: #f6f7f8; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 14px 18px; background: rgba(255,255,255,.96); border-bottom: 1px solid #d8dde3; }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; }}
    .pill {{ padding: 4px 8px; border-radius: 6px; background: #eef2f5; border: 1px solid #d5dde5; }}
    main {{ padding: 16px; }}
    .card {{ margin: 0 0 18px; padding: 14px; border: 1px solid #d9dee4; border-radius: 8px; background: #fff; }}
    .meta {{ display: grid; grid-template-columns: minmax(120px, 180px) 1fr; gap: 6px 12px; margin-bottom: 12px; font-size: 13px; }}
    .label {{ color: #5d6670; }}
    .caption {{ font-size: 14px; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; align-items: start; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; max-height: 620px; object-fit: contain; background: #f2f2f2; border: 1px solid #dde1e6; border-radius: 4px; }}
    figcaption {{ margin-top: 6px; font-size: 12px; color: #4d5660; word-break: break-all; }}
    details {{ margin-top: 10px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f7f8fa; border: 1px solid #dfe4ea; border-radius: 6px; padding: 10px; font-size: 12px; line-height: 1.4; }}
    .warn {{ color: #9a3d00; font-weight: 600; }}
    @media (max-width: 980px) {{ .grid {{ grid-template-columns: 1fr; }} .meta {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Track1 v5 smoke 中文审核</h1>
    <div class="summary">
      <span class="pill">总样本 {summary['total']}</span>
      <span class="pill">可看候选 {summary['generated_or_cached']}</span>
      <span class="pill">错误 {summary['error']}</span>
      <span class="pill">current = 当前冠军包</span>
      <span class="pill">v5候选 = 本轮新生成或 retry 结果</span>
      <span class="pill">reference = 官方 reference board</span>
    </div>
  </header>
  <main>
    {cards}
  </main>
</body>
</html>
"""


def _render_card(row: dict[str, Any], html_path: Path) -> str:
    current_src = _relative_src(row.get("current_image_path"), html_path)
    candidate_src = _relative_src(row.get("candidate_image_path"), html_path)
    reference_src = _relative_src(row.get("reference_image_path"), html_path)
    note = html.escape(str(row.get("notes") or ""))
    note_html = f'<div><span class="label">备注</span><span class="warn">{note}</span></div>' if note else ""
    return f"""
    <section class="card" id="{html.escape(str(row.get('sample_id') or ''))}">
      <div class="meta">
        <div class="label">样本</div><div>{html.escape(str(row.get('sample_id') or ''))}</div>
        <div class="label">官方 caption</div><div class="caption">{html.escape(str(row.get('caption') or ''))}</div>
        <div class="label">路由 / 模型</div><div>{html.escape(str(row.get('candidate_strategy') or ''))} / {html.escape(str(row.get('candidate_model') or ''))}</div>
        <div class="label">尝试状态</div><div>{html.escape(str(row.get('attempt_statuses') or ''))}</div>
        <div class="label">尝试模型</div><div>{html.escape(str(row.get('attempt_models') or ''))}</div>
        {note_html}
      </div>
      <div class="grid">
        {_figure(current_src, 'current：当前冠军包', row.get('current_image_path'))}
        {_figure(candidate_src, 'v5候选：本轮生成/重试', row.get('candidate_image_path'))}
        {_figure(reference_src, '官方 reference board：风格/媒介参考', row.get('reference_image_path'))}
      </div>
      <details>
        <summary>展开 provider prompt</summary>
        <pre>{html.escape(str(row.get('provider_prompt') or ''))}</pre>
      </details>
    </section>
"""


def _figure(src: str, title: str, path: Any) -> str:
    if not src:
        return f"<figure><figcaption>{html.escape(title)}：无图片</figcaption></figure>"
    return (
        f'<figure><img src="{html.escape(src)}" alt="{html.escape(title)}">'
        f"<figcaption>{html.escape(title)}<br>{html.escape(str(path or ''))}</figcaption></figure>"
    )


def _relative_src(path_value: Any, html_path: Path) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return os.path.relpath(path, html_path.parent)
