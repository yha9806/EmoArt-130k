#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_reference_family_bank import _safe_output_paths  # noqa: E402


STRATEGY_LABELS = {
    "aas_safe": "AAS 安全",
    "reference_style": "参考风格",
    "fid_diverse": "FID 多样化",
    "legacy": "旧策略",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Track1 strategy review HTML.")
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--out-html", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        (out_html,) = _safe_output_paths([args.out_html], repo_root=ROOT)
        rows = load_review_rows(json.loads(args.manifest_json.read_text(encoding="utf-8")))
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(
            render_html(rows, out_html=out_html, asset_base_dir=args.manifest_json.parent),
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(out_html)
    return 0


def load_review_rows(payload: Any) -> list[dict[str, Any]]:
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
    if not rows:
        raise ValueError("no review rows found")
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"review row {index} must be an object")
        if not str(row.get("sample_id") or "").strip():
            raise ValueError(f"review row {index} missing required sample_id")
        validated.append(dict(row))
    return validated


def render_html(
    rows: list[dict[str, Any]],
    *,
    out_html: Path | None = None,
    asset_base_dir: Path | None = None,
) -> str:
    grouped = _group_rows(rows)
    cards = "\n".join(
        _render_sample_card(sample, out_html=out_html, asset_base_dir=asset_base_dir)
        for sample in grouped
    )
    if not cards:
        cards = '<p class="empty">没有可展示的候选。</p>'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track1 三策略候选评审</title>
  <style>
    :root {{
      --bg: #f5f3ef;
      --panel: #ffffff;
      --line: #d8d2c7;
      --text: #202225;
      --muted: #606872;
      --accent: #0f766e;
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
      z-index: 3;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(245, 243, 239, 0.96);
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .legend {{
      margin: 8px 0 0;
      max-width: 1120px;
      color: var(--muted);
      font-size: 14px;
    }}
    main {{ padding: 20px 24px 40px; }}
    .sample-card {{
      margin: 0 0 24px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .sample-head {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: start;
      margin-bottom: 14px;
    }}
    h2 {{
      margin: 0 0 6px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .caption {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .family {{
      min-width: 180px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8faf9;
      font-size: 13px;
    }}
    .family strong {{
      display: block;
      margin-bottom: 3px;
      color: var(--accent);
      font-size: 12px;
    }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #f0eee8;
    }}
    img {{
      display: block;
      width: 100%;
      height: 360px;
      object-fit: contain;
      background: #e9e5dc;
    }}
    figcaption {{
      min-height: 52px;
      padding: 8px 10px;
      border-top: 1px solid var(--line);
      background: #fff;
      font-size: 13px;
    }}
    .figure-title {{
      display: block;
      font-weight: 700;
    }}
    .figure-meta {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    details {{
      margin-top: 10px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 8px 0 0;
      padding: 10px;
      border-radius: 6px;
      background: #f7f7f5;
      color: #27323a;
      font-size: 12px;
    }}
    .empty {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>Track1 三策略候选评审</h1>
    <p class="legend">current = 当前提交包；旧候选 = 已有候选或上一轮候选；AAS 安全、参考风格、FID 多样化是这次 anti-template 路由产生的三个候选策略。参考图只用于判断风格分布和题材边界，不等同于必须复制。</p>
  </header>
  <main>
    {cards}
  </main>
</body>
</html>
"""


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        if sample_id not in grouped:
            order.append(sample_id)
            grouped[sample_id] = {
                "sample_id": sample_id,
                "caption": str(row.get("caption") or ""),
                "family_id": _family_id(row),
                "current_image_path": _first_text(row, ("current_image_path", "baseline_image_path", "current_path")),
                "old_candidate_image_path": _first_text(
                    row,
                    ("old_candidate_image_path", "old_image_path", "previous_candidate_image_path"),
                ),
                "reference_paths": [],
                "candidates": [],
                "prompts": [],
            }
        sample = grouped[sample_id]
        if not sample["caption"] and row.get("caption"):
            sample["caption"] = str(row["caption"])
        if not sample["family_id"]:
            sample["family_id"] = _family_id(row)
        if not sample["current_image_path"]:
            sample["current_image_path"] = _first_text(row, ("current_image_path", "baseline_image_path", "current_path"))
        if not sample["old_candidate_image_path"]:
            sample["old_candidate_image_path"] = _first_text(
                row,
                ("old_candidate_image_path", "old_image_path", "previous_candidate_image_path"),
            )
        _extend_unique(sample["reference_paths"], _reference_paths(row))
        candidate_path = _first_text(row, ("image_path", "candidate_image_path", "path"))
        if candidate_path:
            strategy = str(row.get("candidate_strategy") or row.get("strategy") or "candidate")
            sample["candidates"].append(
                {
                    "label": _strategy_label(strategy),
                    "strategy": strategy,
                    "path": candidate_path,
                }
            )
        prompt = _first_text(row, ("provider_prompt", "prompt"))
        if prompt:
            sample["prompts"].append(
                {
                    "strategy": str(row.get("candidate_strategy") or row.get("strategy") or "candidate"),
                    "text": prompt,
                }
            )
    return [grouped[sample_id] for sample_id in order]


def _render_sample_card(
    sample: dict[str, Any],
    *,
    out_html: Path | None,
    asset_base_dir: Path | None,
) -> str:
    sample_id = html.escape(str(sample["sample_id"]))
    family_id = html.escape(str(sample.get("family_id") or "未标注"))
    figures = []
    if sample.get("current_image_path"):
        figures.append(
            _figure(
                "当前提交包",
                str(sample["current_image_path"]),
                out_html=out_html,
                asset_base_dir=asset_base_dir,
            )
        )
    if sample.get("old_candidate_image_path"):
        figures.append(
            _figure(
                "旧候选",
                str(sample["old_candidate_image_path"]),
                out_html=out_html,
                asset_base_dir=asset_base_dir,
            )
        )
    for candidate in sample.get("candidates", []):
        title = f"{candidate['label']} ({candidate['strategy']})"
        figures.append(_figure(title, candidate["path"], out_html=out_html, asset_base_dir=asset_base_dir))
    for index, ref_path in enumerate(sample.get("reference_paths", []), start=1):
        figures.append(
            _figure(
                f"参考图 {index}",
                str(ref_path),
                out_html=out_html,
                asset_base_dir=asset_base_dir,
            )
        )
    prompt_details = _render_prompt_details(sample.get("prompts", []))
    return f"""
<section class="sample-card" id="{sample_id}">
  <div class="sample-head">
    <div>
      <h2>{sample_id}</h2>
      <p class="caption">{html.escape(str(sample.get("caption") or ""))}</p>
    </div>
    <div class="family"><strong>参考族</strong>{family_id}</div>
  </div>
  <div class="image-grid">{''.join(figures)}</div>
  {prompt_details}
</section>
"""


def _figure(label: str, path: str, *, out_html: Path | None, asset_base_dir: Path | None) -> str:
    source = _image_src(path, out_html=out_html, asset_base_dir=asset_base_dir)
    escaped_label = html.escape(label)
    return (
        f"<figure><img src=\"{html.escape(source)}\" alt=\"{escaped_label}\">"
        f"<figcaption><span class=\"figure-title\">{escaped_label}</span>"
        f"<span class=\"figure-meta\">{html.escape(path)}</span></figcaption></figure>"
    )


def _render_prompt_details(prompts: list[dict[str, str]]) -> str:
    if not prompts:
        return ""
    blocks = []
    for prompt in prompts:
        strategy = html.escape(str(prompt.get("strategy") or "candidate"))
        text = html.escape(str(prompt.get("text") or ""))
        blocks.append(f"<h3>{strategy}</h3><pre>{text}</pre>")
    return f"<details><summary>展开 provider prompts</summary>{''.join(blocks)}</details>"


def _image_src(path: str, *, out_html: Path | None, asset_base_dir: Path | None) -> str:
    if path.startswith(("http://", "https://", "data:")):
        return path
    image_path = _resolve_local_image_path(path, asset_base_dir=asset_base_dir)
    if out_html is not None:
        return os.path.relpath(
            image_path.resolve(strict=False),
            out_html.parent.resolve(strict=False),
        )
    return image_path.as_posix()


def _resolve_local_image_path(path: str, *, asset_base_dir: Path | None) -> Path:
    image_path = Path(path)
    if image_path.is_absolute():
        return image_path
    candidates: list[Path] = []
    if asset_base_dir is not None:
        candidates.append(asset_base_dir / image_path)
    candidates.append(ROOT / image_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _family_id(row: dict[str, Any]) -> str:
    if row.get("family_id"):
        return str(row["family_id"])
    metadata = row.get("review_metadata")
    if isinstance(metadata, dict) and metadata.get("family_id"):
        return str(metadata["family_id"])
    return ""


def _reference_paths(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    if row.get("reference_path"):
        values.append(str(row["reference_path"]))
    references = row.get("reference_paths") or row.get("references") or []
    if isinstance(references, list):
        for item in references:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                path = item.get("path") or item.get("reference_path") or item.get("image_path")
                if path:
                    values.append(str(path))
    return values


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extend_unique(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values:
        if value not in seen:
            target.append(value)
            seen.add(value)


def _strategy_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy or "候选")


if __name__ == "__main__":
    raise SystemExit(main())
