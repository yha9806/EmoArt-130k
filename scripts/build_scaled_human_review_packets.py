#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "experiments" / "scaled_human_review_20260528"

TRACK1_BASE = ROOT / "submissions" / "track1_candidate_human_v21_plus0816_20260514"
TRACK1_FEEDBACK = ROOT / "experiments" / "human_review_feedback_20260528" / "track1_merged_human_review.json"
TRACK1_ZIP = ROOT / "data" / "raw" / "Track1_testset.zip"

TRACK2_BASE_JSON = ROOT / "submissions" / "final_track2_20260512_description_v2_vulca_audited.json"
TRACK2_FEEDBACK = ROOT / "experiments" / "human_review_feedback_20260528" / "track2_merged_human_review.json"
TRACK2_ZIP = ROOT / "data" / "raw" / "Track2_testset.zip"


def main() -> int:
    validate_inputs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    track1 = build_track1_packet()
    track2 = build_track2_packet()
    combined_zip = OUT_ROOT / "scaled_human_review_packets_20260528.zip"
    write_combined_zip(combined_zip, [Path(track1["packet_dir"]), Path(track2["packet_dir"])])
    summary = {
        "created_at": "2026-05-28",
        "track1": track1,
        "track2": track2,
        "combined_zip": str(combined_zip.relative_to(ROOT)),
        "combined_zip_sha256": file_sha256(combined_zip),
    }
    write_json(OUT_ROOT / "scaled_human_review_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def validate_inputs() -> None:
    required_files = [
        TRACK1_BASE / "submission.json",
        TRACK1_ZIP,
        TRACK2_BASE_JSON,
        TRACK2_ZIP,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    if not path_is_relative_to(OUT_ROOT.resolve(), (ROOT / "experiments").resolve()):
        raise ValueError(f"Unsafe output root outside experiments/: {OUT_ROOT}")
    if OUT_ROOT.resolve() in {TRACK1_BASE.resolve(), TRACK2_BASE_JSON.parent.resolve(), ROOT.resolve()}:
        raise ValueError(f"Unsafe output root overlaps a source path: {OUT_ROOT}")
    if not track1_candidate_sources():
        raise ValueError("No Track1 candidate sources found")
    if not track2_candidate_sources():
        raise ValueError("No Track2 candidate sources found")


def build_track1_packet() -> dict[str, Any]:
    packet_dir = OUT_ROOT / "track1_scaled_human_review_packet_20260528"
    recreate_dir(packet_dir)
    assets_dir = packet_dir / "assets"
    left_dir = assets_dir / "left"
    right_dir = assets_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    captions = load_track1_captions(TRACK1_ZIP)
    base_rows = index_submission(TRACK1_BASE / "submission.json")
    base_hashes = {
        sample_id: file_sha256(TRACK1_BASE / row["path"])
        for sample_id, row in base_rows.items()
    }
    feedback = load_feedback(TRACK1_FEEDBACK)

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    missing_assets: list[str] = []
    source_counts = Counter()
    source_dirs = track1_candidate_sources()
    for source_dir in source_dirs:
        source_rows = index_submission(source_dir / "submission.json")
        if set(source_rows) != set(base_rows):
            continue
        for sample_id, row in source_rows.items():
            candidate_path = source_dir / row["path"]
            if not candidate_path.exists():
                missing_assets.append(str(candidate_path.relative_to(ROOT)))
                continue
            candidate_hash = file_sha256(candidate_path)
            if candidate_hash == base_hashes.get(sample_id):
                continue
            key = (sample_id, candidate_hash)
            source_counts[source_dir.name] += 1
            if key not in rows_by_key:
                base_path = TRACK1_BASE / base_rows[sample_id]["path"]
                pair_id = f"{sample_id}__{short_hash(candidate_hash)}"
                left_asset = left_dir / f"{pair_id}_A{base_path.suffix.lower()}"
                right_asset = right_dir / f"{pair_id}_B{candidate_path.suffix.lower()}"
                shutil.copy2(base_path, left_asset)
                shutil.copy2(candidate_path, right_asset)
                prior = feedback.get(sample_id, {})
                rows_by_key[key] = {
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "caption": captions.get(sample_id, ""),
                    "left_label": "A_current",
                    "right_label": "B_candidate",
                    "left_source": TRACK1_BASE.name,
                    "candidate_sources": [],
                    "left_image": rel_for_packet(left_asset, packet_dir),
                    "right_image": rel_for_packet(right_asset, packet_dir),
                    "prior_merged_decision": prior.get("merged_decision", ""),
                    "prior_recommended_action": prior.get("recommended_action", ""),
                    "prior_chw_decision": prior.get("chw_decision", ""),
                    "prior_nanhe_decision": prior.get("nanhe_decision", ""),
                    "prior_notes": compact_notes(prior.get("chw_notes", ""), prior.get("nanhe_notes", "")),
                    "decision": "",
                    "confidence_1_3": "",
                    "issue_tags": "",
                    "reviewer_name": "",
                    "comment": "",
                }
            rows_by_key[key]["candidate_sources"].append(source_dir.name)

    rows = sorted(rows_by_key.values(), key=lambda item: (item["sample_id"], item["pair_id"]))
    for row in rows:
        row["candidate_sources"] = ";".join(sorted(set(row["candidate_sources"])))

    csv_path = packet_dir / "track1_scaled_review_template.csv"
    write_csv(csv_path, rows, TRACK1_COLUMNS)
    write_json(packet_dir / "manifest.json", {
        "name": "track1_scaled_human_review_packet_20260528",
        "policy": "A/B review. A is the preserved current baseline; B is one candidate image from historical Track1 candidate packages. Do not accept B unless it is clearly stronger for official AAS/expert review.",
        "baseline": str(TRACK1_BASE.relative_to(ROOT)),
        "row_count": len(rows),
        "unique_sample_count": len({row["sample_id"] for row in rows}),
        "source_count": dict(source_counts.most_common()),
        "missing_assets": missing_assets,
        "csv": csv_path.name,
        "html": "index.html",
    })
    (packet_dir / "README_human_review_zh.md").write_text(render_track1_readme(len(rows)), encoding="utf-8")
    (packet_dir / "index.html").write_text(render_track1_html(rows), encoding="utf-8")
    zip_path = OUT_ROOT / "track1_scaled_human_review_packet_20260528.zip"
    write_zip(packet_dir, zip_path)
    validation = validate_packet(packet_dir, csv_path, ["left_image", "right_image"], zip_path)
    if missing_assets:
        raise RuntimeError(f"Track1 skipped missing candidate assets: {missing_assets[:10]}")
    return {
        "packet_dir": str(packet_dir.relative_to(ROOT)),
        "zip": str(zip_path.relative_to(ROOT)),
        "zip_sha256": file_sha256(zip_path),
        "row_count": len(rows),
        "unique_sample_count": len({row["sample_id"] for row in rows}),
        "missing_asset_count": len(missing_assets),
        "validation": validation,
    }


def build_track2_packet() -> dict[str, Any]:
    packet_dir = OUT_ROOT / "track2_scaled_human_review_packet_20260528"
    recreate_dir(packet_dir)
    image_dir = packet_dir / "assets" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    base_rows = index_track2_rows(TRACK2_BASE_JSON)
    feedback = load_feedback(TRACK2_FEEDBACK)
    source_files = track2_candidate_sources()
    rows_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    source_counts = Counter()
    missing_assets: list[str] = []

    for source_json in source_files:
        candidate_rows = index_track2_rows(source_json)
        if set(candidate_rows) != set(base_rows):
            continue
        for sample_id, candidate in candidate_rows.items():
            base = base_rows[sample_id]
            base_label = label_triplet(base)
            proposed_label = label_triplet(candidate)
            if proposed_label == base_label:
                continue
            key = (sample_id, *proposed_label)
            source_counts[source_json.name] += 1
            if key not in rows_by_key:
                asset_path = image_dir / f"{sample_id}.jpg"
                if not asset_path.exists() and not extract_track2_image(TRACK2_ZIP, sample_id, asset_path):
                    missing_assets.append(sample_id)
                    continue
                prior = feedback.get(sample_id, {})
                rows_by_key[key] = {
                    "proposal_id": f"{sample_id}__{'_'.join(proposed_label)}",
                    "sample_id": sample_id,
                    "image": rel_for_packet(asset_path, packet_dir),
                    "current_emotion": base_label[0],
                    "current_valence": base_label[1],
                    "current_arousal": base_label[2],
                    "proposed_emotion": proposed_label[0],
                    "proposed_valence": proposed_label[1],
                    "proposed_arousal": proposed_label[2],
                    "proposal_sources": [],
                    "prior_merged_decision": prior.get("merged_decision", ""),
                    "prior_recommended_action": prior.get("recommended_action", ""),
                    "prior_chw_decision": prior.get("chw_decision", ""),
                    "prior_nanhe_decision": prior.get("nanhe_decision", ""),
                    "prior_research_recommendation": prior.get("research_recommendation", ""),
                    "prior_research_reasons": prior.get("research_reasons", ""),
                    "overall_caption": base.get("overall_caption", ""),
                    "brushstroke": base.get("brushstroke", ""),
                    "composition": base.get("composition", ""),
                    "color": base.get("color", ""),
                    "line": base.get("line", ""),
                    "light": base.get("light", ""),
                    "human_decision": "",
                    "reviewer_confidence_1_5": "",
                    "reviewer_name": "",
                    "comment": "",
                }
            rows_by_key[key]["proposal_sources"].append(source_json.name)

    rows = sorted(rows_by_key.values(), key=lambda item: (item["sample_id"], item["proposed_emotion"]))
    for row in rows:
        row["proposal_sources"] = ";".join(sorted(set(row["proposal_sources"])))

    csv_path = packet_dir / "track2_scaled_review_template.csv"
    write_csv(csv_path, rows, TRACK2_COLUMNS)
    write_json(packet_dir / "manifest.json", {
        "name": "track2_scaled_human_review_packet_20260528",
        "policy": "Label-change review. Accept proposed only when it is clearly better than the current frozen label. Unsure defaults to keep_current.",
        "baseline_json": str(TRACK2_BASE_JSON.relative_to(ROOT)),
        "row_count": len(rows),
        "unique_sample_count": len({row["sample_id"] for row in rows}),
        "source_count": dict(source_counts.most_common()),
        "missing_assets": missing_assets,
        "csv": csv_path.name,
        "html": "index.html",
    })
    (packet_dir / "README_human_review_zh.md").write_text(render_track2_readme(len(rows)), encoding="utf-8")
    (packet_dir / "index.html").write_text(render_track2_html(rows), encoding="utf-8")
    zip_path = OUT_ROOT / "track2_scaled_human_review_packet_20260528.zip"
    write_zip(packet_dir, zip_path)
    validation = validate_packet(packet_dir, csv_path, ["image"], zip_path)
    if missing_assets:
        raise RuntimeError(f"Track2 skipped missing source images: {missing_assets[:10]}")
    return {
        "packet_dir": str(packet_dir.relative_to(ROOT)),
        "zip": str(zip_path.relative_to(ROOT)),
        "zip_sha256": file_sha256(zip_path),
        "row_count": len(rows),
        "unique_sample_count": len({row["sample_id"] for row in rows}),
        "missing_asset_count": len(missing_assets),
        "validation": validation,
    }


TRACK1_COLUMNS = [
    "pair_id",
    "sample_id",
    "caption",
    "left_label",
    "right_label",
    "left_source",
    "candidate_sources",
    "left_image",
    "right_image",
    "prior_merged_decision",
    "prior_recommended_action",
    "prior_chw_decision",
    "prior_nanhe_decision",
    "prior_notes",
    "decision",
    "confidence_1_3",
    "issue_tags",
    "reviewer_name",
    "comment",
]

TRACK2_COLUMNS = [
    "proposal_id",
    "sample_id",
    "image",
    "current_emotion",
    "current_valence",
    "current_arousal",
    "proposed_emotion",
    "proposed_valence",
    "proposed_arousal",
    "proposal_sources",
    "prior_merged_decision",
    "prior_recommended_action",
    "prior_chw_decision",
    "prior_nanhe_decision",
    "prior_research_recommendation",
    "prior_research_reasons",
    "overall_caption",
    "brushstroke",
    "composition",
    "color",
    "line",
    "light",
    "human_decision",
    "reviewer_confidence_1_5",
    "reviewer_name",
    "comment",
]


def track1_candidate_sources() -> list[Path]:
    excluded = {
        "track1",
        TRACK1_BASE.name,
        "track1_smoke",
        "track1_smoke_v2",
        "track1_smoke_v3",
        "track1_smoke_v4",
    }
    sources = []
    for submission_json in sorted((ROOT / "submissions").glob("track1*/submission.json")):
        source_dir = submission_json.parent
        if source_dir.name in excluded:
            continue
        if "template" in source_dir.name:
            continue
        if not (source_dir / "images").exists():
            continue
        sources.append(source_dir)
    return sources


def track2_candidate_sources() -> list[Path]:
    excluded_names = {
        TRACK2_BASE_JSON.name,
        "track2_submission_template.json",
    }
    sources = []
    for path in sorted((ROOT / "submissions").glob("*.json")):
        name = path.name
        if name in excluded_names or "template" in name:
            continue
        if not (name.startswith("track2_") or name.startswith("final_track2_")):
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "emotion" in rows[0]:
            sources.append(path)
    return sources


def load_track1_captions(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as zf:
        rows = json.loads(zf.read("track1_testset/track1_captions.json").decode("utf-8"))
    return {str(row["sample_id"]): str(row.get("caption", "")) for row in rows}


def index_submission(path: Path) -> dict[str, dict[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): dict(row) for row in rows}


def index_track2_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): dict(row) for row in rows}


def load_feedback(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): row for row in payload.get("rows", []) if row.get("sample_id")}


def label_triplet(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("emotion", "")),
        str(row.get("emotional_valence", "")),
        str(row.get("emotional_arousal_level", "")),
    )


def extract_track2_image(zip_path: Path, sample_id: str, out_path: Path) -> bool:
    member = f"track2_testset/images/{sample_id}.jpg"
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(zf.read(member))
    return True


def render_track1_html(rows: list[dict[str, Any]]) -> str:
    data = json.dumps(rows, ensure_ascii=False)
    cards = []
    for row in rows:
        cards.append(
            f"""
<article class="card" data-pair="{h(row['pair_id'])}" data-prior="{h(row['prior_merged_decision'])}">
  <header>
    <h2>{h(row['sample_id'])}</h2>
    <div class="meta">pair: {h(row['pair_id'])} · prior: {h(row['prior_merged_decision'] or 'none')}</div>
  </header>
  <p class="caption">{h(row['caption'])}</p>
  <div class="compare">
    <figure><img src="{h(row['left_image'])}" alt="A current"><figcaption>A current · {h(row['left_source'])}</figcaption></figure>
    <figure><img src="{h(row['right_image'])}" alt="B candidate"><figcaption>B candidate · {h(row['candidate_sources'])}</figcaption></figure>
  </div>
  <p class="notes">{h(row['prior_notes'])}</p>
  <div class="controls" data-row="{h(row['pair_id'])}">
    <select class="decision"><option value=""></option><option>left</option><option>right</option><option>tie_keep_current</option><option>both_bad_rerun</option><option>unsure</option></select>
    <select class="confidence"><option value=""></option><option>1</option><option>2</option><option>3</option></select>
    <input class="issue" placeholder="issue tags">
    <input class="reviewer" placeholder="reviewer">
    <input class="comment" placeholder="comment">
  </div>
</article>"""
        )
    return render_html_page(
        title="Track1 Scaled Human Review",
        intro=f"{len(rows)} A/B comparisons. Decision values: left, right, tie_keep_current, both_bad_rerun, unsure.",
        rows_json=data,
        body="\n".join(cards),
        export_name="track1_scaled_review_filled.csv",
        columns=TRACK1_COLUMNS,
        mode="track1",
    )


def render_track2_html(rows: list[dict[str, Any]]) -> str:
    data = json.dumps(rows, ensure_ascii=False)
    cards = []
    for row in rows:
        cards.append(
            f"""
<article class="card" data-proposal="{h(row['proposal_id'])}" data-prior="{h(row['prior_merged_decision'])}">
  <header>
    <h2>{h(row['sample_id'])}</h2>
    <div class="meta">proposal: {h(row['proposal_id'])} · prior: {h(row['prior_merged_decision'] or 'none')}</div>
  </header>
  <div class="track2-layout">
    <figure><img src="{h(row['image'])}" alt="{h(row['sample_id'])}"><figcaption>{h(row['sample_id'])}</figcaption></figure>
    <section>
      <div class="labels"><span>current</span> {h(row['current_emotion'])} / {h(row['current_valence'])} / {h(row['current_arousal'])}</div>
      <div class="labels proposed"><span>proposed</span> {h(row['proposed_emotion'])} / {h(row['proposed_valence'])} / {h(row['proposed_arousal'])}</div>
      <p>{h(row['overall_caption'])}</p>
      <p class="notes">sources: {h(row['proposal_sources'])}</p>
      <p class="notes">prior: {h(row['prior_research_recommendation'])} {h(row['prior_research_reasons'])}</p>
    </section>
  </div>
  <div class="controls" data-row="{h(row['proposal_id'])}">
    <select class="decision"><option value=""></option><option>accept_proposed</option><option>keep_current</option><option>unsure</option></select>
    <select class="confidence"><option value=""></option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select>
    <input class="reviewer" placeholder="reviewer">
    <input class="comment" placeholder="comment">
  </div>
</article>"""
        )
    return render_html_page(
        title="Track2 Scaled Human Review",
        intro=f"{len(rows)} unique label-change proposals. Decision values: accept_proposed, keep_current, unsure.",
        rows_json=data,
        body="\n".join(cards),
        export_name="track2_scaled_review_filled.csv",
        columns=TRACK2_COLUMNS,
        mode="track2",
    )


def render_html_page(*, title: str, intro: str, rows_json: str, body: str, export_name: str, columns: list[str], mode: str) -> str:
    columns_json = json.dumps(columns)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{h(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f5f5f2; color: #222; }}
    .top {{ position: sticky; top: 0; z-index: 2; background: #ffffff; border-bottom: 1px solid #ccc; padding: 12px 18px; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    .warning {{ color: #8a2500; font-weight: 650; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 10px; }}
    button, input, select {{ font-size: 14px; padding: 7px 8px; border: 1px solid #aaa; border-radius: 6px; background: #fff; }}
    button {{ cursor: pointer; background: #16202a; color: #fff; }}
    main {{ max-width: 1280px; margin: 18px auto; padding: 0 14px 40px; }}
    .card {{ background: #fff; border: 1px solid #d0d0cc; border-radius: 8px; margin: 14px 0; padding: 14px; }}
    h2 {{ margin: 0; font-size: 18px; }}
    .meta, .notes {{ color: #666; font-size: 13px; }}
    .caption {{ font-size: 15px; line-height: 1.45; }}
    .compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .track2-layout {{ display: grid; grid-template-columns: minmax(260px, 420px) 1fr; gap: 14px; align-items: start; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; max-height: 640px; object-fit: contain; background: #eee; border: 1px solid #bbb; }}
    figcaption {{ font-size: 13px; color: #555; margin-top: 4px; }}
    .labels {{ padding: 8px; background: #f1f4f7; border-radius: 6px; margin-bottom: 8px; }}
    .labels span {{ display: inline-block; width: 80px; color: #555; }}
    .proposed {{ background: #fff5df; }}
    .controls {{ display: grid; grid-template-columns: 180px 120px 1fr 160px 1.5fr; gap: 8px; margin-top: 12px; }}
    .controls[data-row] input {{ min-width: 0; }}
    @media (max-width: 760px) {{
      .compare, .track2-layout {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <section class="top">
    <h1>{h(title)}</h1>
    <div>{h(intro)}</div>
    <div class="warning">Private challenge material. Do not upload images, ZIPs, screenshots, or exported review files to public services.</div>
    <div class="toolbar">
      <button onclick="downloadCsv()">Export filled CSV</button>
      <input id="filter" placeholder="filter sample/source/prior" oninput="filterCards()">
    </div>
  </section>
  <main>{body}</main>
  <script>
    const rows = {rows_json};
    const columns = {columns_json};
    const mode = {json.dumps(mode)};
    function csvEscape(value) {{
      const s = String(value ?? "");
      return /[",\\n]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s;
    }}
    function downloadCsv() {{
      const byId = new Map(rows.map(r => [mode === "track1" ? r.pair_id : r.proposal_id, {{...r}}]));
      document.querySelectorAll(".controls").forEach(el => {{
        const id = el.dataset.row;
        const row = byId.get(id);
        if (!row) return;
        if (mode === "track1") {{
          row.decision = el.querySelector(".decision").value;
          row.confidence_1_3 = el.querySelector(".confidence").value;
          row.issue_tags = el.querySelector(".issue")?.value || "";
          row.reviewer_name = el.querySelector(".reviewer").value;
          row.comment = el.querySelector(".comment").value;
        }} else {{
          row.human_decision = el.querySelector(".decision").value;
          row.reviewer_confidence_1_5 = el.querySelector(".confidence").value;
          row.reviewer_name = el.querySelector(".reviewer").value;
          row.comment = el.querySelector(".comment").value;
        }}
      }});
      const outRows = Array.from(byId.values());
      const csv = [columns.join(","), ...outRows.map(row => columns.map(c => csvEscape(row[c])).join(","))].join("\\n");
      const blob = new Blob([csv + "\\n"], {{ type: "text/csv;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = {json.dumps(export_name)};
      a.click();
      URL.revokeObjectURL(url);
    }}
    function filterCards() {{
      const q = document.getElementById("filter").value.toLowerCase();
      document.querySelectorAll(".card").forEach(card => {{
        card.style.display = card.innerText.toLowerCase().includes(q) ? "" : "none";
      }});
    }}
  </script>
</body>
</html>
"""


def render_track1_readme(row_count: int) -> str:
    return f"""# Track1 扩大人评包 2026-05-28

本包用于扩大 Track1 A/B 人工评审，共 `{row_count}` 条候选对比。

## 使用方式

1. 解压 ZIP。
2. 打开 `index.html`。
3. 每条样本比较 A/B 两张图：A 是当前保守底盘，B 是历史候选包中的候选图。
4. 在页面中填写 decision、confidence、issue tags、reviewer、comment。
5. 点击 `Export filled CSV`，把导出的 CSV 私下发回。

## Decision

- `left`: A/current 更好。
- `right`: B/candidate 明显更好。
- `tie_keep_current`: 差不多，保守保留 current。
- `both_bad_rerun`: 两张都不行，需要重跑。
- `unsure`: 不确定，需要仲裁。

## 评审重点

优先看官方 AAS 和专家评审相关项：caption 内容、关系语义、风格、构图、文字/符号、support surface、artifact。不要把局部小瑕疵看得比主体语义更重要，但 gallery/mockup/footer microtext/sample id/关系语义错误要重罚。

## 隐私

本包包含 challenge 私有图像和生成候选。不要上传到 GitHub issue、公开网盘、公共聊天工具或公共视觉服务。
"""


def render_track2_readme(row_count: int) -> str:
    return f"""# Track2 扩大人评包 2026-05-28

本包用于扩大 Track2 标签变更人工评审，共 `{row_count}` 条去重 label-change proposals。

## 使用方式

1. 解压 ZIP。
2. 打开 `index.html`。
3. 对照图像、current label、proposed label、caption 和历史来源。
4. 填写 human_decision、reviewer_confidence_1_5、reviewer_name、comment。
5. 点击 `Export filled CSV`，把导出的 CSV 私下发回。

## Decision

- `accept_proposed`: proposed 明显更符合图像。
- `keep_current`: current 更好，或 proposed 只是也说得通。
- `unsure`: 无法稳定判断，需要仲裁。

## 评审重点

Track2 默认保守。跨象限变更尤其要谨慎，例如 `sad -> glad`、`alarmed -> glad`、`frustrated -> aroused`、`bored -> aroused`。`unsure` 默认不改。

## 隐私

本包包含 challenge 私有测试图。不要上传到 GitHub issue、公开网盘、公共聊天工具或公共视觉服务。
"""


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_zip(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir))


def write_combined_zip(zip_path: Path, source_dirs: list[Path]) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source_dir in source_dirs:
            for path in sorted(source_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, Path(source_dir.name) / path.relative_to(source_dir))


def recreate_dir(path: Path) -> None:
    resolved = path.resolve()
    output_root = OUT_ROOT.resolve()
    if resolved == output_root or not path_is_relative_to(resolved, output_root):
        raise ValueError(f"Refusing to recreate unsafe output path: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def validate_packet(packet_dir: Path, csv_path: Path, asset_columns: list[str], zip_path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    missing_refs = []
    for row in rows:
        row_id = row.get("sample_id") or row.get("pair_id") or row.get("proposal_id") or "unknown"
        for column in asset_columns:
            rel_path = row.get(column, "")
            if not rel_path:
                missing_refs.append(f"{row_id}:{column}:blank")
                continue
            if not (packet_dir / rel_path).exists():
                missing_refs.append(f"{row_id}:{column}:{rel_path}")
    if missing_refs:
        raise RuntimeError(f"Packet has missing asset references: {missing_refs[:10]}")

    required_names = {csv_path.name, "index.html", "README_human_review_zh.md", "manifest.json"}
    required_names.update(row[column] for row in rows for column in asset_columns)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    missing_zip_members = sorted(required_names - names)
    if missing_zip_members:
        raise RuntimeError(f"ZIP missing required members: {missing_zip_members[:10]}")
    return {
        "csv_rows_checked": len(rows),
        "asset_references_checked": len(rows) * len(asset_columns),
        "zip_members": len(names),
    }


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(value: str) -> str:
    return value[:12]


def rel_for_packet(path: Path, packet_dir: Path) -> str:
    return str(path.relative_to(packet_dir)).replace("\\", "/")


def compact_notes(*values: str) -> str:
    return " / ".join(value.strip() for value in values if value and value.strip())


def h(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
