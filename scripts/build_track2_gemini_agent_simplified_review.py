"""Build a simplified Track2 manual review page from Gemini-agent triage.

This page is a draft review aid. It keeps model recommendations separate from
human-confirmed decisions and does not edit submission JSON files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path("/Users/yhryzy/dev/emoart-130k")
DEFAULT_REVIEW_DIR = (
    ROOT
    / "experiments/track2_scaled_human_gate_20260602/gemini35_guarded_v1/html_review"
)
DEFAULT_AUDIT_REPORT = (
    ROOT
    / "experiments/track2_scaled_human_gate_20260602/gemini35_hardcase_audit/gemini35_hardcase_audit_report.json"
)
DEFAULT_BASE_JSON = ROOT / "submissions/final_track2_20260512_description_v2_vulca_audited.json"
DEFAULT_DESC_JSON = ROOT / "submissions/final_track2_20260602_scaled_human_gate_desc_v1.json"
DEFAULT_GUARDED_JSON = ROOT / "submissions/final_track2_20260602_scaled_human_gate_gemini35_guarded_v1.json"

LABEL_ZH = {
    "alarmed": "惊恐/警觉",
    "annoyed": "恼怒",
    "aroused": "激发/兴奋",
    "bored": "厌倦",
    "calm": "平静",
    "content": "满足/安适",
    "excited": "兴奋",
    "frustrated": "挫败/受阻",
    "glad": "高兴",
    "happy": "快乐",
    "sad": "悲伤",
    "tired": "疲惫",
}

EMOTION_GUIDE = {
    "alarmed": "负向高唤醒：危险、惊惧、警戒、突发威胁。",
    "annoyed": "负向高唤醒：恼怒、不耐烦、冲突姿态。",
    "aroused": "高唤醒但不一定快乐：激情、震撼、紧张、强刺激。",
    "bored": "负向低唤醒：无聊、冷漠、缺乏兴趣。",
    "calm": "正向低唤醒：安静、松弛、平和，没有明显欲望或庆祝。",
    "content": "正向低唤醒：满足、安适、生活圆满，比 calm 更有享受或拥有感。",
    "excited": "正向高唤醒：活跃、热闹、期待、节庆、运动感。",
    "frustrated": "负向高唤醒：努力受阻、压抑、挣扎、失败感。",
    "glad": "正向低或中唤醒：明显喜悦、庆贺、成功感，但不一定很热闹。",
    "happy": "正向高唤醒：快乐、欢快、友好或节庆感。",
    "sad": "负向低唤醒：失落、哀悼、孤独、沉重。",
    "tired": "负向低唤醒：耗尽、困倦、无力；闭眼不等于 tired。",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def index_submission(path: Path) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in read_json(path)}


def label(row: dict[str, Any]) -> str:
    emotion = row["emotion"]
    zh = LABEL_ZH.get(emotion, "")
    return f"{emotion} / {zh} / {row['emotional_valence']} / {row['emotional_arousal_level']}"


def merge_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    high_ids = [row["sample_id"] for row in audit.get("high_conf_blocks", [])]
    by_id = {row["sample_id"]: dict(row) for row in audit.get("risk_rows", [])}
    for row in audit.get("high_conf_blocks", []):
        by_id[row["sample_id"]] = dict(row)
    ordered = list(dict.fromkeys(high_ids + sorted(by_id)))
    return [by_id[sample_id] for sample_id in ordered]


def triage(row: dict[str, Any], high_ids: set[str]) -> dict[str, str]:
    sample_id = row["sample_id"]
    gemini = row.get("gemini", {})
    if sample_id in high_ids:
        return {
            "tier": "A",
            "action": "默认保留 Guarded v1",
            "decision": "keep_guarded_v1",
            "human_task": "快速看图确认是否明显错；没有明显错就通过。",
            "agent_note": "Gemini-agent contact sheet second opinion: 19/19 high-conf rollback rows favor Guarded/Base over Scaled desc_v1.",
        }
    if gemini.get("decision") == "keep_final":
        return {
            "tier": "B",
            "action": "保留 Guarded v1",
            "decision": "keep_guarded_v1",
            "human_task": "低优先级抽查；如果图像明显不符再改。",
            "agent_note": "Gemini-agent risk sheet agrees with the current Guarded v1 direction.",
        }
    return {
        "tier": "C",
        "action": "重点人工复核",
        "decision": "hold_manual_consider_base",
        "human_task": "必须人工二选一：若图像更平静/快乐，改回 Base；否则保留 Guarded v1。",
        "agent_note": "Gemini-agent risk sheet points back toward Base, but confidence is only 3, so do not auto-change.",
    }


def render_label_box(title: str, row: dict[str, Any], cls: str) -> str:
    return f"""
      <div class="label {cls}">
        <span>{html.escape(title)}</span>
        <strong>{html.escape(row["emotion"])}</strong>
        <em>{html.escape(LABEL_ZH.get(row["emotion"], ""))}</em>
        <b>{html.escape(row["emotional_valence"])} / {html.escape(row["emotional_arousal_level"])}</b>
      </div>
    """


def render_candidate_guide(*rows: dict[str, Any]) -> str:
    emotions: list[str] = []
    seen = set()
    for row in rows:
        emotion = row["emotion"]
        if emotion not in seen:
            seen.add(emotion)
            emotions.append(emotion)
    items = "".join(
        f"<li><strong>{html.escape(emotion)}</strong> / {html.escape(LABEL_ZH.get(emotion, ''))}: {html.escape(EMOTION_GUIDE.get(emotion, ''))}</li>"
        for emotion in emotions
    )
    return f"<ul>{items}</ul>"


def render_card(
    row: dict[str, Any],
    base: dict[str, Any],
    desc: dict[str, Any],
    guarded: dict[str, Any],
    triage_info: dict[str, str],
    review_dir: Path,
    idx: int,
) -> str:
    sample_id = row["sample_id"]
    gemini = row.get("gemini", {})
    rationale = html.escape(str(gemini.get("rationale", "")))
    agent_note = html.escape(triage_info["agent_note"])
    human_task = html.escape(triage_info["human_task"])
    caption = html.escape(str(guarded.get("overall_caption", "")))
    candidate_guide = render_candidate_guide(base, desc, guarded)
    return f"""
<article class="card tier-{triage_info["tier"]}" id="{html.escape(sample_id)}" data-tier="{triage_info["tier"]}">
  <div class="head">
    <div>
      <span class="num">#{idx:02d}</span>
      <h2>{html.escape(sample_id)}</h2>
    </div>
    <div class="status">
      <strong>{html.escape(triage_info["action"])}</strong>
      <span>Tier {html.escape(triage_info["tier"])}</span>
    </div>
  </div>
  <div class="grid">
    <img src="assets/{html.escape(sample_id)}.jpg" alt="{html.escape(sample_id)}">
    <section>
      <div class="labels">
        {render_label_box("Base", base, "base")}
        {render_label_box("Scaled desc_v1", desc, "desc")}
        {render_label_box("Guarded v1", guarded, "guarded")}
      </div>
      <div class="instruction">
        <h3>你要做什么 / Human task</h3>
        <p>{human_task}</p>
      </div>
      <div class="wizard">
        <h3>三步判断 / Evidence workflow</h3>
        <ol>
          <li><strong>正负：</strong>画面主导感受是安宁、快乐、满足，还是痛苦、威胁、疲惫、冲突？</li>
          <li><strong>唤醒：</strong>画面是静止低强度，还是有危险、打斗、骚动、强烈动作？</li>
          <li><strong>二选一：</strong>先排除正负/唤醒不匹配的候选；如果同象限，再按下面候选定义选择。</li>
        </ol>
        <div class="candidate-guide">
          <h4>本图候选标签边界</h4>
          {candidate_guide}
        </div>
      </div>
      <div class="agent">
        <h3>Gemini-agent second opinion</h3>
        <p>{agent_note}</p>
        <p><strong>Prior Gemini 3.5:</strong> {html.escape(str(gemini.get("decision", "")))}; conf={html.escape(str(gemini.get("confidence_1_5", "")))}; suggested={html.escape(str(gemini.get("suggested_emotion", "")))} / {html.escape(str(gemini.get("suggested_valence", "")))} / {html.escape(str(gemini.get("suggested_arousal", "")))}</p>
        <p>{rationale}</p>
      </div>
      <details>
        <summary>Caption / 文本</summary>
        <p>{caption}</p>
      </details>
      <div class="choices">
        <label><input type="radio" name="{html.escape(sample_id)}" value="accept_recommendation" checked> 接受推荐 / accept recommendation</label>
        <label><input type="radio" name="{html.escape(sample_id)}" value="use_base"> 用 Base</label>
        <label><input type="radio" name="{html.escape(sample_id)}" value="use_scaled_desc_v1"> 用 Scaled desc_v1</label>
        <label><input type="radio" name="{html.escape(sample_id)}" value="hold"> Hold</label>
        <textarea data-note="{html.escape(sample_id)}" placeholder="人工备注 / reviewer notes"></textarea>
      </div>
    </section>
  </div>
</article>
"""


def render_html(rows: list[dict[str, Any]], base_index: dict[str, dict[str, Any]], desc_index: dict[str, dict[str, Any]], guarded_index: dict[str, dict[str, Any]], high_ids: set[str], review_dir: Path) -> str:
    triaged = [(row, triage(row, high_ids)) for row in rows]
    counts = {tier: sum(1 for _, info in triaged if info["tier"] == tier) for tier in ["A", "B", "C"]}
    cards = "\n".join(
        render_card(
            row=row,
            base=base_index[row["sample_id"]],
            desc=desc_index[row["sample_id"]],
            guarded=guarded_index[row["sample_id"]],
            triage_info=info,
            review_dir=review_dir,
            idx=i + 1,
        )
        for i, (row, info) in enumerate(triaged)
    )
    quick = "".join(
        f'<a href="#{html.escape(row["sample_id"])}" data-tier="{info["tier"]}">{html.escape(row["sample_id"])}</a>'
        for row, info in triaged
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track2 Simplified Manual Review</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif; background: #f0f2f5; color: #202124; letter-spacing: 0; }}
    header {{ position: sticky; top: 0; z-index: 5; background: rgba(255,255,255,.96); border-bottom: 1px solid #d7dbe0; }}
    .top {{ max-width: 1380px; margin: 0 auto; padding: 14px 20px; display: grid; grid-template-columns: 1fr auto; gap: 16px; align-items: center; }}
    h1 {{ margin: 0; font-size: 21px; }}
    .sub {{ margin: 4px 0 0; color: #5f6368; font-size: 13px; }}
    .filters {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    button {{ border: 1px solid #d7dbe0; background: white; padding: 8px 11px; border-radius: 7px; cursor: pointer; }}
    button.active {{ background: #1f5eff; border-color: #1f5eff; color: white; }}
    main {{ max-width: 1380px; margin: 0 auto; padding: 18px 20px 56px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; margin-bottom: 14px; }}
    .metric, .quick, .card, .protocol {{ background: white; border: 1px solid #d7dbe0; border-radius: 8px; box-shadow: 0 8px 22px rgba(32,33,36,.08); }}
    .metric {{ padding: 14px; }}
    .metric span {{ color: #5f6368; font-size: 13px; }}
    .metric strong {{ display: block; font-size: 30px; margin-top: 4px; }}
    .protocol {{ padding: 14px; margin-bottom: 16px; }}
    .protocol h2 {{ margin: 0 0 10px; font-size: 17px; }}
    .protocol-grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 10px; }}
    .protocol-grid div {{ border: 1px solid #d7dbe0; border-radius: 7px; padding: 10px; background: #fbfcfd; }}
    .protocol-grid strong {{ display: block; margin-bottom: 5px; }}
    .protocol-grid span {{ color: #5f6368; font-size: 13px; line-height: 1.45; }}
    .quick {{ padding: 12px; margin-bottom: 16px; display: flex; flex-wrap: wrap; gap: 7px; }}
    .quick a {{ text-decoration: none; padding: 5px 8px; border-radius: 6px; border: 1px solid #d7dbe0; color: #202124; font-size: 12px; }}
    .quick a[data-tier="A"] {{ border-color: #b8dfd1; background: #f0fbf7; color: #0b7f58; }}
    .quick a[data-tier="B"] {{ border-color: #cbd6ff; background: #f4f6ff; color: #1f5eff; }}
    .quick a[data-tier="C"] {{ border-color: #ffd0cc; background: #fff5f4; color: #b3261e; }}
    .card {{ margin-bottom: 18px; padding: 16px; scroll-margin-top: 94px; }}
    .head {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 12px; }}
    .num {{ color: #5f6368; font-size: 12px; font-weight: 700; }}
    h2 {{ margin: 2px 0; font-size: 25px; }}
    .status {{ text-align: right; }}
    .status strong {{ display: block; font-size: 18px; }}
    .status span {{ color: #5f6368; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: minmax(320px, 42%) 1fr; gap: 18px; }}
    img {{ width: 100%; max-height: 660px; object-fit: contain; border: 1px solid #d7dbe0; border-radius: 6px; background: #f3f4f6; }}
    .labels {{ display: grid; grid-template-columns: repeat(3, minmax(150px, 1fr)); gap: 10px; }}
    .label {{ min-height: 92px; border: 1px solid #d7dbe0; border-left-width: 5px; border-radius: 7px; padding: 10px; display: grid; gap: 4px; background: #fbfcfd; }}
    .label span, .label em {{ color: #5f6368; font-size: 12px; font-style: normal; }}
    .label strong {{ font-size: 20px; }}
    .base {{ border-left-color: #0b7f58; }}
    .desc {{ border-left-color: #b75e00; }}
    .guarded {{ border-left-color: #1f5eff; }}
    h3 {{ margin: 0 0 7px; font-size: 14px; }}
    .instruction, .agent, .choices, .wizard {{ margin-top: 12px; padding: 12px; border-radius: 7px; border: 1px solid #d7dbe0; background: #fbfcfd; }}
    .tier-C .instruction {{ border-color: #ffd0cc; background: #fff7f5; }}
    .wizard ol {{ margin: 0 0 10px 20px; padding: 0; }}
    .wizard li {{ margin-bottom: 6px; line-height: 1.45; }}
    .candidate-guide {{ border-top: 1px solid #d7dbe0; padding-top: 10px; }}
    .candidate-guide h4 {{ margin: 0 0 6px; font-size: 13px; }}
    .candidate-guide ul {{ margin: 0 0 0 18px; padding: 0; }}
    .candidate-guide li {{ margin-bottom: 5px; line-height: 1.45; }}
    p {{ line-height: 1.55; margin: 6px 0; }}
    details {{ margin-top: 12px; }}
    summary {{ cursor: pointer; color: #1f5eff; font-weight: 600; }}
    .choices {{ display: grid; gap: 8px; }}
    label {{ display: flex; gap: 8px; align-items: center; }}
    textarea {{ width: 100%; min-height: 64px; border: 1px solid #d7dbe0; border-radius: 6px; padding: 8px; font: inherit; resize: vertical; }}
    .hidden {{ display: none; }}
    @media (max-width: 980px) {{ .top, .grid, .labels, .metrics, .protocol-grid {{ grid-template-columns: 1fr; }} .filters {{ justify-content: flex-start; }} .status {{ text-align: left; }} }}
  </style>
</head>
<body>
<header>
  <div class="top">
    <div>
      <h1>Track2 低抽象人工复审 / Simplified Review</h1>
      <p class="sub">不要从 12 类自由标注。只按“正负 → 唤醒 → 候选二选一”仲裁冲突样本。AI recommendations are draft second opinions, not human-confirmed evidence.</p>
    </div>
    <div class="filters">
      <button class="active" data-filter="all">全部</button>
      <button data-filter="A">Tier A 默认通过</button>
      <button data-filter="B">Tier B 抽查</button>
      <button data-filter="C">Tier C 必看</button>
      <button id="export">导出 JSON</button>
    </div>
  </div>
</header>
<main>
  <section class="metrics">
    <div class="metric"><span>Tier A: high-conf rollback confirmed</span><strong>{counts["A"]}</strong></div>
    <div class="metric"><span>Tier B: keep guarded, low priority</span><strong>{counts["B"]}</strong></div>
    <div class="metric"><span>Tier C: must review manually</span><strong>{counts["C"]}</strong></div>
  </section>
  <section class="protocol">
    <h2>人评员只做这件事 / What the rater actually does</h2>
    <div class="protocol-grid">
      <div><strong>1. 不自由打标签</strong><span>只在 Base、Scaled desc_v1、Guarded v1 给出的候选中选。</span></div>
      <div><strong>2. 先判正负和唤醒</strong><span>正负/高低唤醒错了，比 calm/content 边界更严重。</span></div>
      <div><strong>3. 没有强证据就保守</strong><span>Tier A/B 默认保留 guarded；Tier C 才认真二选一。</span></div>
      <div><strong>4. 备注只写画面证据</strong><span>例如“静态山水、无冲突、低唤醒，选 calm”。</span></div>
    </div>
    <details>
      <summary>查看完整人评协议 / full protocol</summary>
      <p>完整文档：<code>docs/track2_human_review_protocol_zh.md</code></p>
      <p>不要按模型权威投票，不要按 caption 文采投票，不要猜艺术家原意。只看画面证据是否支持候选标签。</p>
    </details>
  </section>
  <nav class="quick">{quick}</nav>
  {cards}
</main>
<script>
const key = "track2_simplified_manual_review_v1";
function collect() {{
  const rows = [];
  document.querySelectorAll(".card").forEach(card => {{
    const id = card.id;
    rows.push({{
      sample_id: id,
      tier: card.dataset.tier,
      decision: card.querySelector("input[type=radio]:checked")?.value || "",
      note: card.querySelector("textarea")?.value || ""
    }});
  }});
  return rows;
}}
function save() {{ localStorage.setItem(key, JSON.stringify(collect(), null, 2)); }}
function restore() {{
  try {{
    const rows = JSON.parse(localStorage.getItem(key) || "[]");
    rows.forEach(row => {{
      const radio = document.querySelector(`input[name="${{row.sample_id}}"][value="${{row.decision}}"]`);
      if (radio) radio.checked = true;
      const note = document.querySelector(`textarea[data-note="${{row.sample_id}}"]`);
      if (note) note.value = row.note || "";
    }});
  }} catch {{}}
}}
document.querySelectorAll("input, textarea").forEach(el => {{
  el.addEventListener("change", save);
  el.addEventListener("input", save);
}});
document.querySelectorAll("button[data-filter]").forEach(button => {{
  button.addEventListener("click", () => {{
    document.querySelectorAll("button[data-filter]").forEach(b => b.classList.remove("active"));
    button.classList.add("active");
    const filter = button.dataset.filter;
    document.querySelectorAll(".card").forEach(card => {{
      card.classList.toggle("hidden", filter !== "all" && card.dataset.tier !== filter);
    }});
  }});
}});
document.getElementById("export").addEventListener("click", () => {{
  const blob = new Blob([JSON.stringify(collect(), null, 2)], {{type: "application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "track2_simplified_manual_review_decisions.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}});
restore();
</script>
</body>
</html>
"""


def strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def write_csv(rows: list[dict[str, Any]], base_index: dict[str, dict[str, Any]], desc_index: dict[str, dict[str, Any]], guarded_index: dict[str, dict[str, Any]], high_ids: set[str], out_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "tier",
        "recommended_decision",
        "human_task",
        "base_label",
        "scaled_desc_v1_label",
        "guarded_v1_label",
        "prior_gemini_decision",
        "prior_gemini_confidence",
        "prior_gemini_rationale",
        "human_decision",
        "human_notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            sample_id = row["sample_id"]
            info = triage(row, high_ids)
            gemini = row.get("gemini", {})
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "tier": info["tier"],
                    "recommended_decision": info["decision"],
                    "human_task": info["human_task"],
                    "base_label": label(base_index[sample_id]),
                    "scaled_desc_v1_label": label(desc_index[sample_id]),
                    "guarded_v1_label": label(guarded_index[sample_id]),
                    "prior_gemini_decision": gemini.get("decision", ""),
                    "prior_gemini_confidence": gemini.get("confidence_1_5", ""),
                    "prior_gemini_rationale": gemini.get("rationale", ""),
                    "human_decision": "",
                    "human_notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    parser.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    parser.add_argument("--desc-json", type=Path, default=DEFAULT_DESC_JSON)
    parser.add_argument("--guarded-json", type=Path, default=DEFAULT_GUARDED_JSON)
    args = parser.parse_args()

    audit = read_json(args.audit_report)
    rows = merge_rows(audit)
    high_ids = {row["sample_id"] for row in audit.get("high_conf_blocks", [])}
    base_index = index_submission(args.base_json)
    desc_index = index_submission(args.desc_json)
    guarded_index = index_submission(args.guarded_json)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.review_dir / "track2_gemini_agent_simplified_review.html"
    csv_path = args.review_dir / "track2_gemini_agent_simplified_review.csv"
    summary_path = args.review_dir / "track2_gemini_agent_simplified_review_summary.json"
    html_path.write_text(
        strip_trailing_whitespace(render_html(rows, base_index, desc_index, guarded_index, high_ids, args.review_dir)),
        encoding="utf-8",
    )
    write_csv(rows, base_index, desc_index, guarded_index, high_ids, csv_path)
    tier_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for row in rows:
        tier_counts[triage(row, high_ids)["tier"]] += 1
    summary = {
        "method": "track2_gemini_agent_simplified_review_v1",
        "html": str(html_path),
        "csv": str(csv_path),
        "tier_counts": tier_counts,
        "human_final_confirmed": False,
        "formal_submission_overwritten": False,
        "gemini_agent_result_summary": {
            "high_conf_contact_sheet": "Gemini-agent judged the 19 high-confidence rollback rows as favoring guarded/base labels.",
            "risk_context_contact_sheet": "Gemini-agent agreed with guarded for 0139 and 0635, and suggested base for 0547, 0869, 0964 at confidence 3.",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
