"""Build a local HTML review console for Track2 Gemini 3.5 guarded rows.

The output is a draft/manual-review artifact. It does not alter any submission
JSON and does not mark rows as human-confirmed evidence.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from PIL import Image


ROOT = Path("/Users/yhryzy/dev/emoart-130k")
DEFAULT_AUDIT_REPORT = (
    ROOT
    / "experiments/track2_scaled_human_gate_20260602/gemini35_hardcase_audit/gemini35_hardcase_audit_report.json"
)
DEFAULT_ROLLBACK_REPORT = (
    ROOT
    / "experiments/track2_scaled_human_gate_20260602/gemini35_guarded_v1/rollback_report.json"
)
DEFAULT_BASE_JSON = ROOT / "submissions/final_track2_20260512_description_v2_vulca_audited.json"
DEFAULT_DESC_JSON = ROOT / "submissions/final_track2_20260602_scaled_human_gate_desc_v1.json"
DEFAULT_GUARDED_JSON = ROOT / "submissions/final_track2_20260602_scaled_human_gate_gemini35_guarded_v1.json"
DEFAULT_TRACK2_ZIP = ROOT / "data/raw/Track2_testset.zip"
DEFAULT_OUT_DIR = (
    ROOT
    / "experiments/track2_scaled_human_gate_20260602/gemini35_guarded_v1/html_review"
)

LABEL_ZH = {
    "alarmed": "惊恐/警觉",
    "amazed": "惊奇",
    "amused": "愉悦/觉得有趣",
    "annoyed": "恼怒",
    "aroused": "激发/兴奋",
    "bored": "厌倦",
    "calm": "平静",
    "content": "满足/安适",
    "disgusted": "厌恶",
    "excited": "兴奋",
    "frustrated": "挫败/受阻",
    "glad": "高兴",
    "happy": "快乐",
    "sad": "悲伤",
    "tired": "疲惫",
}


@dataclass(frozen=True)
class ReviewRow:
    sample_id: str
    bucket: str
    risk_focus: list[str]
    allowlist_transition: str
    allowlist_reason: str
    gemini: dict[str, Any]
    base: dict[str, Any]
    desc: dict[str, Any]
    guarded: dict[str, Any]
    image_rel: str
    default_decision: str


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def index_submission(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"submission JSON must be a list: {path}")
    return {str(row["sample_id"]): row for row in rows}


def label_tuple(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["emotion"]),
        str(row["emotional_valence"]),
        str(row["emotional_arousal_level"]),
    )


def label_html(row: dict[str, Any], css_class: str = "") -> str:
    emotion, valence, arousal = label_tuple(row)
    zh = LABEL_ZH.get(emotion, "")
    return (
        f'<div class="label {css_class}">'
        f"<strong>{html.escape(emotion)}</strong>"
        f'<span class="muted">{html.escape(zh)}</span>'
        f"<span>{html.escape(valence)} / {html.escape(arousal)}</span>"
        "</div>"
    )


def normalize_rows(audit_report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return risk rows with high-confidence blocks first and no duplicates."""
    high_ids = [row["sample_id"] for row in audit_report.get("high_conf_blocks", [])]
    by_id: dict[str, dict[str, Any]] = {}
    for row in audit_report.get("risk_rows", []):
        by_id[row["sample_id"]] = row
    for row in audit_report.get("high_conf_blocks", []):
        by_id[row["sample_id"]] = row
    ordered_ids = list(dict.fromkeys(high_ids + sorted(by_id)))
    return [by_id[sample_id] for sample_id in ordered_ids]


def extract_image(track2_zip: Path, sample_id: str, assets_dir: Path, max_side: int) -> str:
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path = assets_dir / f"{sample_id}.jpg"
    member = f"track2_testset/images/{sample_id}.jpg"
    with ZipFile(track2_zip) as zf:
        with zf.open(member) as fh:
            image = Image.open(BytesIO(fh.read())).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    image.save(out_path, format="JPEG", quality=92, optimize=True)
    return f"assets/{out_path.name}"


def build_review_rows(
    audit_report: dict[str, Any],
    base_index: dict[str, dict[str, Any]],
    desc_index: dict[str, dict[str, Any]],
    guarded_index: dict[str, dict[str, Any]],
    track2_zip: Path,
    out_dir: Path,
    max_side: int,
) -> list[ReviewRow]:
    high_ids = {row["sample_id"] for row in audit_report.get("high_conf_blocks", [])}
    rows: list[ReviewRow] = []
    for item in normalize_rows(audit_report):
        sample_id = item["sample_id"]
        if sample_id not in base_index or sample_id not in desc_index or sample_id not in guarded_index:
            raise KeyError(f"missing sample in submission JSON: {sample_id}")
        gemini = dict(item.get("gemini") or {})
        bucket = "high_conf_rollback" if sample_id in high_ids else "risk_context"
        default_decision = "keep_guarded_rollback" if bucket == "high_conf_rollback" else "keep_guarded"
        image_rel = extract_image(track2_zip, sample_id, out_dir / "assets", max_side=max_side)
        rows.append(
            ReviewRow(
                sample_id=sample_id,
                bucket=bucket,
                risk_focus=list(item.get("risk_focus") or []),
                allowlist_transition=str(item.get("allowlist_transition") or ""),
                allowlist_reason=str(item.get("allowlist_reason") or ""),
                gemini=gemini,
                base=base_index[sample_id],
                desc=desc_index[sample_id],
                guarded=guarded_index[sample_id],
                image_rel=image_rel,
                default_decision=default_decision,
            )
        )
    return rows


def row_summary(row: ReviewRow) -> dict[str, str]:
    gemini = row.gemini
    return {
        "sample_id": row.sample_id,
        "bucket": row.bucket,
        "base_label": " / ".join(label_tuple(row.base)),
        "desc_v1_label": " / ".join(label_tuple(row.desc)),
        "guarded_v1_label": " / ".join(label_tuple(row.guarded)),
        "gemini_decision": str(gemini.get("decision", "")),
        "gemini_suggested": " / ".join(
            [
                str(gemini.get("suggested_emotion", "")),
                str(gemini.get("suggested_valence", "")),
                str(gemini.get("suggested_arousal", "")),
            ]
        ),
        "gemini_confidence": str(gemini.get("confidence_1_5", "")),
        "text_consistency": str(gemini.get("text_consistency", "")),
        "allowlist_transition": row.allowlist_transition,
        "default_decision": row.default_decision,
        "human_decision": "",
        "human_reviewer": "",
        "human_notes": "",
    }


def write_review_csv(rows: list[ReviewRow], out_path: Path) -> None:
    fields = list(row_summary(rows[0]).keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row_summary(row))


def render_chips(items: list[str]) -> str:
    return "".join(f'<span class="chip">{html.escape(item)}</span>' for item in items)


def render_review_card(row: ReviewRow, index: int) -> str:
    gemini = row.gemini
    bucket_label = (
        "高置信回滚 / High-confidence rollback"
        if row.bucket == "high_conf_rollback"
        else "风险旁证 / Risk context"
    )
    desc_diff = "changed" if label_tuple(row.desc) != label_tuple(row.guarded) else "same"
    confidence = html.escape(str(gemini.get("confidence_1_5", "")))
    text_consistency = html.escape(str(gemini.get("text_consistency", "")))
    rationale = html.escape(str(gemini.get("rationale", "")))
    caption = html.escape(str(row.guarded.get("overall_caption", "")))
    fields = []
    for key, title in [
        ("brushstroke", "Brushstroke / 笔触"),
        ("composition", "Composition / 构图"),
        ("color", "Color / 色彩"),
        ("line", "Line / 线条"),
        ("light", "Light / 光线"),
    ]:
        fields.append(
            f"<dt>{title}</dt><dd>{html.escape(str(row.guarded.get(key, '')))}</dd>"
        )
    fields_html = "".join(fields)
    options = [
        ("keep_guarded_rollback", "保留 guarded 回滚 / Keep guarded rollback"),
        ("restore_desc_v1", "恢复 scaled desc_v1 / Restore desc_v1"),
        ("hold_manual", "暂缓人工再议 / Hold for manual decision"),
    ]
    if row.bucket != "high_conf_rollback":
        options = [
            ("keep_guarded", "保留 guarded 当前值 / Keep guarded current"),
            ("change_after_review", "人工认为需改 / Human thinks change needed"),
            ("hold_manual", "暂缓人工再议 / Hold for manual decision"),
        ]
    option_html = "".join(
        (
            f'<label class="choice"><input type="radio" name="decision_{html.escape(row.sample_id)}" '
            f'value="{value}" data-sample="{html.escape(row.sample_id)}" '
            f'{"checked" if value == row.default_decision else ""}> {label}</label>'
        )
        for value, label in options
    )
    return f"""
<article class="review-card" id="{html.escape(row.sample_id)}" data-bucket="{row.bucket}" data-diff="{desc_diff}">
  <div class="card-head">
    <div>
      <div class="index">#{index:02d}</div>
      <h2>{html.escape(row.sample_id)}</h2>
      <p class="bucket">{bucket_label}</p>
    </div>
    <a class="anchor" href="#{html.escape(row.sample_id)}">#{html.escape(row.sample_id)}</a>
  </div>
  <div class="card-grid">
    <figure>
      <img src="{html.escape(row.image_rel)}" alt="{html.escape(row.sample_id)} artwork">
      <figcaption>{html.escape(row.sample_id)} test image</figcaption>
    </figure>
    <section class="decision-panel">
      <div class="label-grid">
        <div><h3>Base 20260512</h3>{label_html(row.base, "base")}</div>
        <div><h3>Scaled desc_v1</h3>{label_html(row.desc, "desc")}</div>
        <div><h3>Guarded v1</h3>{label_html(row.guarded, "guarded")}</div>
      </div>
      <div class="meta-row">
        <span><strong>Transition:</strong> {html.escape(row.allowlist_transition)}</span>
        <span><strong>Confidence:</strong> {confidence}/5</span>
        <span><strong>Text:</strong> {text_consistency}</span>
      </div>
      <div class="chips">{render_chips(row.risk_focus)}</div>
      <p class="rationale"><strong>Gemini 3.5:</strong> {rationale}</p>
      <div class="human-box">
        <h3>人工复审 / Manual decision</h3>
        <div class="choices">{option_html}</div>
        <textarea data-note="{html.escape(row.sample_id)}" placeholder="Reviewer notes / 人工备注"></textarea>
      </div>
    </section>
  </div>
  <details>
    <summary>查看 caption 和 attribute 文本 / Show caption and attributes</summary>
    <p><strong>Overall caption:</strong> {caption}</p>
    <dl>{fields_html}</dl>
  </details>
</article>
"""


def render_html(rows: list[ReviewRow], audit_report: dict[str, Any], rollback_report: dict[str, Any]) -> str:
    high_count = sum(1 for row in rows if row.bucket == "high_conf_rollback")
    risk_count = len(rows)
    transition_counts = Counter(row.allowlist_transition for row in rows)
    transition_html = "".join(
        f"<li><span>{html.escape(k)}</span><strong>{v}</strong></li>"
        for k, v in transition_counts.most_common()
    )
    nav_links = "".join(
        f'<a href="#{html.escape(row.sample_id)}" data-bucket="{row.bucket}">{html.escape(row.sample_id)}</a>'
        for row in rows
    )
    cards = "\n".join(render_review_card(row, i + 1) for i, row in enumerate(rows))
    summary = {
        "audit_model": audit_report.get("model"),
        "queue_count": audit_report.get("queue_count"),
        "decision_counts": audit_report.get("decision_counts"),
        "high_conf_block_count": audit_report.get("high_conf_block_count"),
        "guarded_rollback_count": rollback_report.get("rollback_count"),
        "formal_submission_overwritten": rollback_report.get("formal_submission_overwritten"),
    }
    summary_json = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AffectiveArt Track2 Gemini 3.5 Guarded Review</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #202124;
      --muted: #5f6368;
      --line: #d7dbe0;
      --panel: #f7f8fa;
      --white: #ffffff;
      --blue: #1f5eff;
      --green: #0b7f58;
      --orange: #b75e00;
      --red: #b3261e;
      --shadow: 0 8px 22px rgba(32, 33, 36, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background: #eef1f5;
      letter-spacing: 0;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }}
    .topbar {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 14px 20px;
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto;
      gap: 16px;
      align-items: center;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.25;
    }}
    .subtitle {{
      margin: 5px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    button {{
      border: 1px solid var(--line);
      background: var(--white);
      color: var(--ink);
      padding: 8px 11px;
      border-radius: 7px;
      font-size: 13px;
      cursor: pointer;
    }}
    button.active {{
      color: white;
      border-color: var(--blue);
      background: var(--blue);
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px 20px 48px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric, .nav-panel, .review-card {{
      background: var(--white);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .metric {{
      padding: 14px;
      min-height: 88px;
    }}
    .metric strong {{
      display: block;
      font-size: 25px;
      line-height: 1.2;
      margin-top: 4px;
    }}
    .metric span, .muted, figcaption, .bucket {{
      color: var(--muted);
      font-size: 12px;
    }}
    .nav-panel {{
      padding: 12px;
      margin-bottom: 18px;
    }}
    .nav-panel h2 {{
      margin: 0 0 8px;
      font-size: 14px;
    }}
    .nav-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .nav-links a {{
      color: var(--blue);
      text-decoration: none;
      border: 1px solid #cbd6ff;
      background: #f4f6ff;
      padding: 5px 8px;
      border-radius: 6px;
      font-size: 12px;
    }}
    .nav-links a[data-bucket="high_conf_rollback"] {{
      color: var(--red);
      border-color: #ffd0cc;
      background: #fff5f4;
    }}
    .review-card {{
      margin: 0 0 18px;
      padding: 16px;
      scroll-margin-top: 104px;
    }}
    .card-head, .meta-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .index {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    h2 {{
      margin: 2px 0;
      font-size: 24px;
      line-height: 1.15;
    }}
    h3 {{
      margin: 0 0 7px;
      font-size: 13px;
    }}
    .anchor {{
      color: var(--blue);
      text-decoration: none;
      font-size: 13px;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: minmax(320px, 45%) minmax(420px, 1fr);
      gap: 18px;
      margin-top: 12px;
    }}
    figure {{
      margin: 0;
    }}
    img {{
      display: block;
      width: 100%;
      max-height: 680px;
      object-fit: contain;
      background: #f0f0f0;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .label-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(150px, 1fr));
      gap: 10px;
    }}
    .label {{
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
      min-height: 76px;
      display: grid;
      gap: 5px;
      background: var(--panel);
    }}
    .label strong {{
      font-size: 18px;
    }}
    .label.base {{ border-left: 5px solid var(--green); }}
    .label.desc {{ border-left: 5px solid var(--orange); }}
    .label.guarded {{ border-left: 5px solid var(--blue); }}
    .meta-row {{
      margin: 12px 0;
      justify-content: flex-start;
      color: #363a3f;
      font-size: 13px;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 10px;
    }}
    .chip {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      background: #fbfbfc;
      color: #41464c;
      font-size: 12px;
    }}
    .rationale {{
      line-height: 1.55;
      margin: 10px 0 12px;
      padding: 11px;
      border-left: 4px solid var(--red);
      background: #fff7f5;
      border-radius: 5px;
    }}
    .human-box {{
      border: 1px solid var(--line);
      background: #fbfcfd;
      border-radius: 7px;
      padding: 12px;
    }}
    .choices {{
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .choice {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
    }}
    textarea {{
      width: 100%;
      min-height: 72px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      font: inherit;
      background: white;
    }}
    details {{
      margin-top: 14px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }}
    summary {{
      cursor: pointer;
      color: var(--blue);
      font-weight: 600;
      font-size: 13px;
    }}
    dl {{
      display: grid;
      grid-template-columns: 160px 1fr;
      gap: 8px 14px;
      margin: 12px 0 0;
    }}
    dt {{
      color: var(--muted);
      font-weight: 700;
    }}
    dd {{
      margin: 0;
      line-height: 1.5;
    }}
    .transitions {{
      columns: 2;
      margin: 8px 0 0;
      padding-left: 18px;
      color: #40444a;
      font-size: 13px;
    }}
    .transitions li {{
      break-inside: avoid;
      margin-bottom: 5px;
    }}
    .transitions strong {{
      margin-left: 6px;
    }}
    pre {{
      overflow: auto;
      max-height: 180px;
      padding: 10px;
      background: #202124;
      color: #f8fafd;
      border-radius: 7px;
      font-size: 12px;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 980px) {{
      .topbar, .card-grid, .label-grid, .summary-grid {{
        grid-template-columns: 1fr;
      }}
      .toolbar {{
        justify-content: flex-start;
      }}
      .review-card {{
        scroll-margin-top: 156px;
      }}
      dl {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
<header>
  <div class="topbar">
    <div>
      <h1>Track2 Gemini 3.5 Guarded 人工复审 / Manual Review</h1>
      <p class="subtitle">Draft review console. It does not modify submission files or mark rows as human-confirmed.</p>
    </div>
    <div class="toolbar">
      <button class="active" data-filter="all">全部 All</button>
      <button data-filter="high_conf_rollback">只看 19 个回滚</button>
      <button data-filter="risk_context">旁证 Risk rows</button>
      <button id="export-json">导出决策 JSON</button>
    </div>
  </div>
</header>
<main>
  <section class="summary-grid">
    <div class="metric"><span>Risk rows / 风险样本</span><strong>{risk_count}</strong></div>
    <div class="metric"><span>High-conf rollback / 高置信回滚</span><strong>{high_count}</strong></div>
    <div class="metric"><span>Guarded rollback count</span><strong>{html.escape(str(rollback_report.get("rollback_count", "")))}</strong></div>
    <div class="metric"><span>Formal submission overwritten?</span><strong>{html.escape(str(rollback_report.get("formal_submission_overwritten", "")))}</strong></div>
  </section>
  <section class="nav-panel">
    <h2>快速跳转 / Quick jump</h2>
    <div class="nav-links">{nav_links}</div>
    <details>
      <summary>Transition summary / 标签迁移概览</summary>
      <ul class="transitions">{transition_html}</ul>
      <pre>{summary_json}</pre>
    </details>
  </section>
  {cards}
</main>
<script>
  const stateKey = "track2_gemini35_guarded_manual_review_v1";

  function loadState() {{
    try {{ return JSON.parse(localStorage.getItem(stateKey) || "{{}}"); }}
    catch {{ return {{}}; }}
  }}

  function saveState(state) {{
    localStorage.setItem(stateKey, JSON.stringify(state));
  }}

  function collectState() {{
    const out = loadState();
    document.querySelectorAll(".review-card").forEach(card => {{
      const id = card.id;
      const selected = card.querySelector("input[type=radio]:checked");
      const note = card.querySelector("textarea").value;
      out[id] = {{
        sample_id: id,
        decision: selected ? selected.value : "",
        note,
        bucket: card.dataset.bucket
      }};
    }});
    return out;
  }}

  function applyState() {{
    const state = loadState();
    for (const [id, row] of Object.entries(state)) {{
      const input = document.querySelector(`input[data-sample="${{id}}"][value="${{row.decision}}"]`);
      if (input) input.checked = true;
      const note = document.querySelector(`textarea[data-note="${{id}}"]`);
      if (note && row.note) note.value = row.note;
    }}
  }}

  document.querySelectorAll("input[type=radio], textarea").forEach(el => {{
    el.addEventListener("change", () => saveState(collectState()));
    el.addEventListener("input", () => saveState(collectState()));
  }});

  document.querySelectorAll("button[data-filter]").forEach(button => {{
    button.addEventListener("click", () => {{
      document.querySelectorAll("button[data-filter]").forEach(b => b.classList.remove("active"));
      button.classList.add("active");
      const filter = button.dataset.filter;
      document.querySelectorAll(".review-card").forEach(card => {{
        card.classList.toggle("hidden", filter !== "all" && card.dataset.bucket !== filter);
      }});
    }});
  }});

  document.getElementById("export-json").addEventListener("click", () => {{
    const payload = JSON.stringify(Object.values(collectState()), null, 2);
    const blob = new Blob([payload], {{ type: "application/json" }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "track2_gemini35_manual_review_decisions.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }});

  applyState();
</script>
</body>
</html>
"""


def write_summary(rows: list[ReviewRow], out_path: Path, html_path: Path, csv_path: Path) -> None:
    high_count = sum(1 for row in rows if row.bucket == "high_conf_rollback")
    summary = {
        "method": "track2_gemini35_guarded_manual_review_html_v1",
        "html": str(html_path),
        "csv_template": str(csv_path),
        "row_count": len(rows),
        "high_conf_rollback_count": high_count,
        "risk_context_count": len(rows) - high_count,
        "formal_submission_overwritten": False,
        "human_final_confirmed": False,
        "default_action": "review high_conf_rollback first; do not write final manifest until manual decisions are exported and validated",
        "sample_ids": [row.sample_id for row in rows],
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    parser.add_argument("--rollback-report", type=Path, default=DEFAULT_ROLLBACK_REPORT)
    parser.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    parser.add_argument("--desc-json", type=Path, default=DEFAULT_DESC_JSON)
    parser.add_argument("--guarded-json", type=Path, default=DEFAULT_GUARDED_JSON)
    parser.add_argument("--track2-zip", type=Path, default=DEFAULT_TRACK2_ZIP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-side", type=int, default=1200)
    args = parser.parse_args()

    audit_report = read_json(args.audit_report)
    rollback_report = read_json(args.rollback_report)
    rows = build_review_rows(
        audit_report=audit_report,
        base_index=index_submission(args.base_json),
        desc_index=index_submission(args.desc_json),
        guarded_index=index_submission(args.guarded_json),
        track2_zip=args.track2_zip,
        out_dir=args.out_dir,
        max_side=args.max_side,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.out_dir / "track2_gemini35_guarded_manual_review.html"
    csv_path = args.out_dir / "track2_gemini35_guarded_manual_review_template.csv"
    summary_path = args.out_dir / "track2_gemini35_guarded_manual_review_summary.json"
    html_path.write_text(strip_trailing_whitespace(render_html(rows, audit_report, rollback_report)), encoding="utf-8")
    write_review_csv(rows, csv_path)
    write_summary(rows, summary_path, html_path, csv_path)
    print(json.dumps({"html": str(html_path), "csv": str(csv_path), "summary": str(summary_path), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
