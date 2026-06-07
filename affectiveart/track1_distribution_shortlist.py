from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DELTA_THRESHOLDS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)


def parse_report_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"report spec must be LABEL=PATH: {spec}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label or not path.strip():
        raise ValueError(f"invalid report spec: {spec}")
    return label, Path(path)


def load_metric_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"metric report must contain rows: {path}")
    return payload


def build_distribution_shortlist(
    reports_by_source: dict[str, dict[str, Any]],
    *,
    captions: dict[str, str] | None = None,
    top_n: int = 80,
    min_delta: float = 0.03,
    high_delta: float = 0.05,
) -> dict[str, Any]:
    captions = captions or {}
    candidates = []
    source_counts: dict[str, dict[str, Any]] = {}
    for source, report in reports_by_source.items():
        source_candidates = [_candidate_from_row(source, row, captions=captions) for row in report.get("rows", [])]
        source_candidates = [item for item in source_candidates if item is not None]
        candidates.extend(source_candidates)
        source_counts[source] = _source_count(source_candidates)

    candidates.sort(key=_candidate_sort_key)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_sample[str(item["sample_id"])].append(item)

    primary_rows = []
    for sample_id in sorted(by_sample):
        options = sorted(by_sample[sample_id], key=_candidate_sort_key)
        primary = dict(options[0])
        primary["alternatives"] = [
            _compact_alternative(option) for option in options[1:]
        ]
        primary["duplicate_candidate_count"] = len(options)
        primary_rows.append(primary)
    primary_rows.sort(key=_candidate_sort_key)

    review_rows = [
        _with_review_tier(row, min_delta=min_delta, high_delta=high_delta)
        for row in primary_rows
        if float(row["delta"]) >= min_delta
    ][: max(0, top_n)]
    watch_rows = [
        _with_review_tier(row, min_delta=min_delta, high_delta=high_delta)
        for row in primary_rows
        if 0.0 < float(row["delta"]) < min_delta
    ]

    return {
        "method": {
            "name": "track1_distribution_shortlist_v1",
            "warning": (
                "Review queue only. Positive local proxy delta does not imply official FID or AAS improvement. "
                "No row is accepted until visual, VLM, human, and package-level FID-like gates pass."
            ),
            "top_n": top_n,
            "min_delta": min_delta,
            "high_delta": high_delta,
            "thresholds": list(DELTA_THRESHOLDS),
        },
        "summary": {
            "source_count": len(reports_by_source),
            "candidate_pairs_positive": len(candidates),
            "unique_positive_samples": len(primary_rows),
            "review_queue_samples": len(review_rows),
            "watchlist_samples": len(watch_rows),
            "high_delta_review_samples": sum(1 for row in review_rows if float(row["delta"]) >= high_delta),
            "source_counts": source_counts,
        },
        "review_queue": review_rows,
        "positive_watchlist": watch_rows,
    }


def write_shortlist_reports(
    report: dict[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
    md_path: str | Path,
    html_path: str | Path,
) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    md_path = Path(md_path)
    html_path = Path(html_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(report, csv_path)
    md_path.write_text(_strip_trailing_whitespace(render_shortlist_markdown(report)), encoding="utf-8")
    html_path.write_text(_strip_trailing_whitespace(render_shortlist_html(report, out_html=html_path)), encoding="utf-8")


def render_shortlist_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    method = report["method"]
    lines = [
        "# Track1 V4 Distribution Shortlist",
        "",
        "这个报告只用于找高价值审图队列，不是 accepted replacement manifest。",
        "",
        "## 结论",
        "",
        f"- 来源报告数：`{summary['source_count']}`",
        f"- 正向 candidate pair：`{summary['candidate_pairs_positive']}`",
        f"- 去重后正向 sample：`{summary['unique_positive_samples']}`",
        f"- 进入 review queue：`{summary['review_queue_samples']}`",
        f"- 高 delta 队列：`{summary['high_delta_review_samples']}`，阈值 `delta >= {method['high_delta']}`",
        f"- watchlist：`{summary['watchlist_samples']}`，低于 review 阈值但 delta 为正",
        "",
        "## 风险规则",
        "",
        "- 这里没有候选 AAS/VLM 审计，不能自动替换。",
        "- FID 是包级分布指标，单样本 delta 只能作为候选优先级。",
        "- 下一步必须做：官方 caption 约束检查、视觉审图、Gemini redteam、hybrid 包级 FID-like sanity。",
        "",
        "## 来源统计",
        "",
    ]
    for source, counts in sorted(summary["source_counts"].items()):
        threshold_bits = ", ".join(
            f">{threshold}: {counts['threshold_counts'][str(threshold)]}"
            for threshold in method["thresholds"]
        )
        lines.append(
            f"- `{source}` positive=`{counts['positive']}` changed=`{counts['changed']}` "
            f"mean_positive_delta=`{counts['mean_positive_delta']}` ({threshold_bits})"
        )
    lines.extend(["", "## Review Queue", ""])
    for index, row in enumerate(report["review_queue"], start=1):
        lines.append(
            f"{index}. `{row['sample_id']}` source=`{row['source']}` tier=`{row['review_tier']}` "
            f"delta=`{row['delta']}` distribution_delta=`{row['distribution_delta']}` "
            f"perceptual_delta=`{row['perceptual_delta']}` duplicate_options=`{row['duplicate_candidate_count']}`"
        )
        caption = str(row.get("caption") or "").strip()
        if caption:
            lines.append(f"   caption: {caption}")
    return "\n".join(lines).rstrip() + "\n"


def render_shortlist_html(report: dict[str, Any], *, out_html: str | Path) -> str:
    out_html = Path(out_html)
    cards = "\n".join(_render_html_card(row, out_html=out_html) for row in report.get("review_queue", []))
    if not cards:
        cards = '<p class="empty">没有候选进入 review queue。</p>'
    summary = report["summary"]
    source_stats = "\n".join(
        f"<li><strong>{_esc(source)}</strong>: positive {counts['positive']}, changed {counts['changed']}, "
        f"mean positive delta {counts['mean_positive_delta']}</li>"
        for source, counts in sorted(summary["source_counts"].items())
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track1 V4 FID 候选审查队列</title>
  <style>
    :root {{
      --bg: #f6f4ef;
      --panel: #ffffff;
      --line: #d9d3c8;
      --text: #202226;
      --muted: #5f6670;
      --accent: #0f766e;
      --risk: #b45309;
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
      z-index: 2;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 244, 239, 0.96);
    }}
    h1 {{ margin: 0 0 8px; font-size: 22px; letter-spacing: 0; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }}
    .pill {{
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      font-size: 13px;
    }}
    .note {{ margin: 8px 0 0; max-width: 1180px; color: var(--risk); font-size: 14px; }}
    main {{ padding: 22px 24px 48px; }}
    .stats {{
      margin: 0 0 18px;
      padding: 14px 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .sample {{
      margin: 0 0 24px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .sample h2 {{ margin: 0 0 6px; font-size: 18px; letter-spacing: 0; }}
    .caption {{ margin: 0 0 12px; color: var(--muted); font-size: 14px; }}
    .metric-line {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
    .metric {{
      padding: 5px 8px;
      border-radius: 6px;
      background: #eef5f3;
      color: #164e45;
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(260px, 1fr));
      gap: 12px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #ede8dd;
    }}
    img {{
      display: block;
      width: 100%;
      height: min(60vh, 560px);
      object-fit: contain;
      background: #e8e3d8;
    }}
    figcaption {{
      min-height: 52px;
      padding: 8px 10px;
      border-top: 1px solid var(--line);
      background: #fff;
      font-size: 13px;
    }}
    .title {{ display: block; font-weight: 700; }}
    .path {{ display: block; margin-top: 2px; color: var(--muted); overflow-wrap: anywhere; }}
    details {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 760px) {{
      .grid {{ grid-template-columns: 1fr; }}
      img {{ height: 420px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Track1 V4 FID 候选审查队列</h1>
    <div class="meta">
      <span class="pill">review queue: {summary['review_queue_samples']}</span>
      <span class="pill">unique positive: {summary['unique_positive_samples']}</span>
      <span class="pill">high delta: {summary['high_delta_review_samples']}</span>
      <span class="pill">watchlist: {summary['watchlist_samples']}</span>
    </div>
    <p class="note">这些不是已接受替换。current = 当前冠军提交包；candidate = 从 full1000/partial757 中按本地分布 proxy 选出的候选。每张都还需要 caption、文本、空间逻辑和包级 FID-like 审核。</p>
  </header>
  <main>
    <section class="stats">
      <strong>来源统计</strong>
      <ul>{source_stats}</ul>
    </section>
    {cards}
  </main>
</body>
</html>
"""


def _candidate_from_row(source: str, row: dict[str, Any], *, captions: dict[str, str]) -> dict[str, Any] | None:
    if not row.get("changed"):
        return None
    delta = float(row.get("delta") or 0.0)
    if delta <= 0.0:
        return None
    sample_id = str(row.get("sample_id") or "")
    if not sample_id:
        return None
    current = row.get("current") or {}
    candidate = row.get("candidate") or {}
    return {
        "sample_id": sample_id,
        "caption": captions.get(sample_id, ""),
        "source": source,
        "delta": round(delta, 6),
        "distribution_delta": round(float(candidate.get("distribution_proxy") or 0.0) - float(current.get("distribution_proxy") or 0.0), 6),
        "perceptual_delta": round(float(candidate.get("perceptual_proxy") or 0.0) - float(current.get("perceptual_proxy") or 0.0), 6),
        "surface_delta": round(float(candidate.get("surface_gate") or 0.0) - float(current.get("surface_gate") or 0.0), 6),
        "candidate_aas_proxy": float(candidate.get("aas_proxy") or 0.0),
        "current_aas_proxy": float(current.get("aas_proxy") or 0.0),
        "candidate_review_missing": float(candidate.get("aas_proxy") or 0.0) == 0.0 and float(current.get("aas_proxy") or 0.0) == 0.0,
        "current_score": float(current.get("official_like_score") or 0.0),
        "candidate_score": float(candidate.get("official_like_score") or 0.0),
        "current_image": str(row.get("current_image") or ""),
        "candidate_image": str(row.get("candidate_image") or ""),
        "candidate_hard_reject_reasons": candidate.get("hard_reject_reasons") or [],
        "recommendation_from_metric_proxy": str(row.get("recommendation") or ""),
    }


def _source_count(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    threshold_counts = {
        str(threshold): sum(1 for item in candidates if float(item["delta"]) > threshold)
        for threshold in DELTA_THRESHOLDS
    }
    positive = [float(item["delta"]) for item in candidates]
    changed = len(candidates)
    return {
        "changed": changed,
        "positive": len(positive),
        "mean_positive_delta": round(sum(positive) / len(positive), 6) if positive else 0.0,
        "threshold_counts": threshold_counts,
    }


def _candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, str, str]:
    return (
        -float(item.get("delta") or 0.0),
        -float(item.get("distribution_delta") or 0.0),
        str(item.get("sample_id") or ""),
        str(item.get("source") or ""),
    )


def _compact_alternative(option: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": option["source"],
        "delta": option["delta"],
        "candidate_image": option["candidate_image"],
        "distribution_delta": option["distribution_delta"],
        "perceptual_delta": option["perceptual_delta"],
    }


def _with_review_tier(row: dict[str, Any], *, min_delta: float, high_delta: float) -> dict[str, Any]:
    item = dict(row)
    delta = float(item["delta"])
    if delta >= high_delta:
        tier = "high_delta_review"
    elif delta >= min_delta:
        tier = "medium_delta_review"
    else:
        tier = "positive_watchlist"
    item["review_tier"] = tier
    item["acceptance_status"] = "review_needed_not_accepted"
    item["blocking_gates"] = [
        "candidate_aas_vlm_review_missing",
        "human_visual_review_missing",
        "hybrid_package_fid_like_missing",
    ]
    if item.get("candidate_hard_reject_reasons"):
        item["blocking_gates"].append("candidate_has_proxy_hard_reject_reason")
    return item


def _write_csv(report: dict[str, Any], csv_path: Path) -> None:
    rows = report.get("review_queue", [])
    fieldnames = [
        "rank",
        "sample_id",
        "source",
        "review_tier",
        "delta",
        "distribution_delta",
        "perceptual_delta",
        "candidate_review_missing",
        "duplicate_candidate_count",
        "current_image",
        "candidate_image",
        "caption",
        "acceptance_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow({key: row.get(key, "") for key in fieldnames if key != "rank"} | {"rank": index})


def _render_html_card(row: dict[str, Any], *, out_html: Path) -> str:
    current_src = _relative_src(row.get("current_image"), out_html)
    candidate_src = _relative_src(row.get("candidate_image"), out_html)
    alternatives = row.get("alternatives") or []
    alt_html = ""
    if alternatives:
        items = "".join(
            f"<li>{_esc(item['source'])}: delta {_esc(item['delta'])}, path {_esc(item['candidate_image'])}</li>"
            for item in alternatives
        )
        alt_html = f"<details><summary>其它候选来源</summary><ul>{items}</ul></details>"
    return f"""<article class="sample">
  <h2>{_esc(row.get('sample_id'))}</h2>
  <p class="caption">{_esc(row.get('caption') or 'caption 未载入')}</p>
  <div class="metric-line">
    <span class="metric">来源：{_esc(row.get('source'))}</span>
    <span class="metric">tier：{_esc(row.get('review_tier'))}</span>
    <span class="metric">delta：{_esc(row.get('delta'))}</span>
    <span class="metric">distribution：{_esc(row.get('distribution_delta'))}</span>
    <span class="metric">perceptual：{_esc(row.get('perceptual_delta'))}</span>
    <span class="metric">候选 AAS 审计：{'缺失' if row.get('candidate_review_missing') else '已有'}</span>
  </div>
  <div class="grid">
    <figure>
      <img src="{_esc(current_src)}" alt="{_esc(row.get('sample_id'))} current">
      <figcaption><span class="title">current：当前冠军包</span><span class="path">{_esc(row.get('current_image'))}</span></figcaption>
    </figure>
    <figure>
      <img src="{_esc(candidate_src)}" alt="{_esc(row.get('sample_id'))} candidate">
      <figcaption><span class="title">candidate：本地 FID/proxy 候选</span><span class="path">{_esc(row.get('candidate_image'))}</span></figcaption>
    </figure>
  </div>
  {alt_html}
</article>"""


def _relative_src(path_value: Any, out_html: Path) -> str:
    path = Path(str(path_value or ""))
    if not path.is_absolute():
        path = Path.cwd() / path
    import os

    return os.path.relpath(path.resolve(), out_html.parent.resolve())


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


__all__ = [
    "build_distribution_shortlist",
    "load_metric_report",
    "parse_report_spec",
    "render_shortlist_html",
    "render_shortlist_markdown",
    "write_shortlist_reports",
]
