from __future__ import annotations

import csv
import html
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from affectiveart.track2_visual_audit import find_track2_image_member


DECISION_COLUMNS = [
    "sample_id",
    "category",
    "current_emotion",
    "current_valence",
    "current_arousal",
    "clean_emotion",
    "clean_confidence",
    "clean_margin",
    "clean_knn_emotion",
    "clean_knn_confidence",
    "inclusive_emotion",
    "inclusive_confidence",
    "inclusive_margin",
    "inclusive_knn_emotion",
    "inclusive_knn_confidence",
    "high_similarity_public_reference",
    "recommended_decision",
    "manual_decision",
    "reviewer_rationale",
]

TEXT_FIELDS = [
    "overall_caption",
    "brushstroke",
    "composition",
    "color",
    "line",
    "light",
]

CATEGORY_ORDER = {
    "inclusive_only_change": 0,
    "clean_only_change": 1,
    "clean_inclusive_conflict": 2,
}

OUTPUT_FILENAMES = {
    "html": "html_review/track2_clean_inclusive_disagreement_review.html",
    "csv": "track2_clean_inclusive_disagreement_decisions.csv",
    "json": "track2_clean_inclusive_disagreement_report.json",
    "markdown": "track2_clean_inclusive_disagreement_report.md",
}


def build_disagreement_review_rows(
    current_rows: list[dict[str, Any]],
    clean_payload: dict[str, Any] | list[Any],
    inclusive_payload: dict[str, Any] | list[Any],
    *,
    high_similarity_sample_ids: set[str] | None = None,
    image_members: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    high_similarity_sample_ids = high_similarity_sample_ids or set()
    image_members = image_members or {}
    current_by_id = _index_current_rows(current_rows)
    clean_by_id = _index_prediction_entries(clean_payload)
    inclusive_by_id = _index_prediction_entries(inclusive_payload)

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(current_by_id):
        clean = clean_by_id.get(sample_id)
        inclusive = inclusive_by_id.get(sample_id)
        if not clean or not inclusive:
            continue

        clean_summary = _prediction_summary(clean)
        inclusive_summary = _prediction_summary(inclusive)
        if clean_summary["emotion"] == inclusive_summary["emotion"]:
            continue

        current = current_by_id[sample_id]
        current_emotion = str(current.get("emotion", ""))
        category = _category(current_emotion, clean_summary["emotion"], inclusive_summary["emotion"])
        row = {
            "sample_id": sample_id,
            "category": category,
            "current_emotion": current_emotion,
            "current_valence": str(current.get("emotional_valence", "")),
            "current_arousal": str(current.get("emotional_arousal_level", "")),
            "clean": clean_summary,
            "inclusive": inclusive_summary,
            "clean_emotion": clean_summary["emotion"],
            "clean_confidence": clean_summary["confidence"],
            "clean_margin": clean_summary["margin"],
            "clean_knn_emotion": clean_summary["knn_emotion"],
            "clean_knn_confidence": clean_summary["knn_confidence"],
            "inclusive_emotion": inclusive_summary["emotion"],
            "inclusive_confidence": inclusive_summary["confidence"],
            "inclusive_margin": inclusive_summary["margin"],
            "inclusive_knn_emotion": inclusive_summary["knn_emotion"],
            "inclusive_knn_confidence": inclusive_summary["knn_confidence"],
            "high_similarity_public_reference": sample_id in high_similarity_sample_ids,
            "image_member": image_members.get(sample_id, ""),
            "image_asset": f"assets/{_safe_asset_filename(sample_id)}",
            "recommended_decision": "hold",
            "manual_decision": "",
            "reviewer_rationale": "",
        }
        for field in TEXT_FIELDS:
            row[field] = str(current.get(field, ""))
        rows.append(row)

    return sorted(rows, key=lambda row: (_category_rank(row["category"]), row["sample_id"]))


def write_disagreement_review_outputs(
    *,
    current_json: str | Path,
    clean_predictions_json: str | Path,
    inclusive_predictions_json: str | Path,
    image_zip: str | Path,
    out_dir: str | Path,
    high_similarity_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    current_rows = _read_current_rows(current_json)
    clean_payload = _read_json(clean_predictions_json)
    inclusive_payload = _read_json(inclusive_predictions_json)

    out_dir = Path(out_dir)
    html_dir = out_dir / "html_review"
    assets_dir = html_dir / "assets"
    html_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [str(row["sample_id"]) for row in current_rows if row.get("sample_id")]
    image_members = _image_members_by_sample_id(image_zip, sample_ids)
    rows = build_disagreement_review_rows(
        current_rows,
        clean_payload,
        inclusive_payload,
        high_similarity_sample_ids=high_similarity_sample_ids,
        image_members=image_members,
    )
    _extract_assets(image_zip, rows, assets_dir)

    outputs = {key: str(out_dir / name) for key, name in OUTPUT_FILENAMES.items()}
    report = {
        "method": "track2_clean_inclusive_disagreement_review_v1",
        "row_count": len(rows),
        "category_counts": _category_counts(rows),
        "high_similarity_count": sum(1 for row in rows if row["high_similarity_public_reference"]),
        "outputs": outputs,
        "inputs": {
            "current_json": str(current_json),
            "clean_predictions_json": str(clean_predictions_json),
            "inclusive_predictions_json": str(inclusive_predictions_json),
            "image_zip": str(image_zip),
        },
        "rows": rows,
    }

    _write_decision_csv(Path(outputs["csv"]), rows)
    Path(outputs["html"]).write_text(render_html(rows, report), encoding="utf-8")
    Path(outputs["json"]).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(outputs["markdown"]).write_text(render_markdown(report), encoding="utf-8")
    return report


def render_html(rows: list[dict[str, Any]], report: dict[str, Any]) -> str:
    cards = "\n".join(_render_row_card(row) for row in rows)
    counts = report.get("category_counts", {})
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track2 Clean/Inclusive Disagreement Review</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #5f6c7b;
      --line: #d8dee7;
      --panel: #ffffff;
      --band: #f4f7fb;
      --accent: #0f766e;
      --warning: #9f580a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--band);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 16px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .summary, .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .pill, button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 7px 10px;
      font-size: 13px;
    }}
    button {{
      cursor: pointer;
    }}
    button.active {{
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 650;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 20px 24px 48px;
    }}
    .sop {{
      margin: 0 0 18px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      line-height: 1.45;
    }}
    .sop h2 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .review-card {{
      display: grid;
      grid-template-columns: minmax(220px, 340px) 1fr;
      gap: 18px;
      margin: 0 0 18px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .review-card[hidden] {{
      display: none;
    }}
    .artwork {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #eef2f7;
    }}
    .row-title {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .row-title h2 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .badge {{
      border-radius: 6px;
      padding: 4px 7px;
      background: #eaf7f5;
      color: #0f615b;
      font-size: 12px;
      font-weight: 650;
    }}
    .badge.warning {{
      background: #fff3dc;
      color: var(--warning);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
      min-width: 0;
    }}
    .panel h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    .label {{
      font-size: 18px;
      font-weight: 700;
    }}
    .meta, .caption, .small {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    .caption {{
      margin: 8px 0 12px;
    }}
    .manual {{
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 10px;
      align-items: start;
    }}
    select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      font: inherit;
      background: #fff;
    }}
    textarea {{
      min-height: 76px;
      resize: vertical;
    }}
    @media (max-width: 820px) {{
      .review-card, .grid, .manual {{
        grid-template-columns: 1fr;
      }}
      header, main {{
        padding-left: 14px;
        padding-right: 14px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Track2 Clean/Inclusive Disagreement Review</h1>
    <div class="summary">
      <span class="pill">Rows: {int(report.get("row_count", len(rows)))}</span>
      <span class="pill">Inclusive-only: {int(counts.get("inclusive_only_change", 0))}</span>
      <span class="pill">Clean-only: {int(counts.get("clean_only_change", 0))}</span>
      <span class="pill">Conflict: {int(counts.get("clean_inclusive_conflict", 0))}</span>
      <span class="pill">High-similarity: {int(report.get("high_similarity_count", 0))}</span>
    </div>
    <div class="filters" aria-label="Review filters">
      <button type="button" class="active" data-filter="all">All</button>
      <button type="button" data-filter="inclusive_only_change">Inclusive-only</button>
      <button type="button" data-filter="clean_only_change">Clean-only</button>
      <button type="button" data-filter="clean_inclusive_conflict">Conflict</button>
      <button type="button" data-filter="high_similarity">High-similarity</button>
    </div>
  </header>
  <main>
    <section class="sop" aria-label="Review SOP">
      <h2>Review SOP</h2>
      <div>For calm/content disagreements, prefer holding unless the visible expression clearly supports a boundary change.</div>
      <div>For frustrated/aroused disagreements, check negative affect and activation separately before accepting any change.</div>
    </section>
    {cards or '<p class="small">No clean/inclusive emotion disagreements found.</p>'}
  </main>
  <script>
    const buttons = Array.from(document.querySelectorAll("[data-filter]"));
    const cards = Array.from(document.querySelectorAll(".review-card"));
    buttons.forEach((button) => {{
      button.addEventListener("click", () => {{
        buttons.forEach((item) => item.classList.toggle("active", item === button));
        const filter = button.dataset.filter;
        cards.forEach((card) => {{
          const matches = filter === "all"
            || card.dataset.category === filter
            || (filter === "high_similarity" && card.dataset.highSimilarity === "true");
          card.hidden = !matches;
        }});
      }});
    }});
  </script>
</body>
</html>
"""
    return html_text


def render_markdown(report: dict[str, Any]) -> str:
    counts = report.get("category_counts", {})
    outputs = report.get("outputs", {})
    inputs = report.get("inputs", {})
    lines = [
        "# Track2 Clean/Inclusive Disagreement Review",
        "",
        f"- Rows: {report.get('row_count', 0)}",
        f"- Inclusive-only changes: {counts.get('inclusive_only_change', 0)}",
        f"- Clean-only changes: {counts.get('clean_only_change', 0)}",
        f"- Clean/inclusive conflicts: {counts.get('clean_inclusive_conflict', 0)}",
        f"- High-similarity public references: {report.get('high_similarity_count', 0)}",
        "",
        "## Outputs",
        "",
        f"- HTML: `{outputs.get('html', '')}`",
        f"- CSV: `{outputs.get('csv', '')}`",
        f"- JSON: `{outputs.get('json', '')}`",
        f"- Markdown: `{outputs.get('markdown', '')}`",
        "",
        "## Inputs",
        "",
        f"- Current JSON: `{inputs.get('current_json', '')}`",
        f"- Clean predictions JSON: `{inputs.get('clean_predictions_json', '')}`",
        f"- Inclusive predictions JSON: `{inputs.get('inclusive_predictions_json', '')}`",
        f"- Image ZIP: `{inputs.get('image_zip', '')}`",
        "",
    ]
    return "\n".join(lines)


def _render_row_card(row: dict[str, Any]) -> str:
    sample_id = _e(row["sample_id"])
    category = _e(row["category"])
    high_similarity = bool(row.get("high_similarity_public_reference"))
    high_similarity_badge = (
        '<span class="badge warning">High-similarity public reference</span>' if high_similarity else ""
    )
    text_bits = [
        f"<strong>{_e(field.replace('_', ' ').title())}:</strong> {_e(row.get(field, ''))}"
        for field in TEXT_FIELDS
        if row.get(field)
    ]
    text_html = "<br>".join(text_bits)
    return f"""
    <article class="review-card" data-category="{category}" data-high-similarity="{str(high_similarity).lower()}">
      <div>
        <img class="artwork" src="{_e(row.get('image_asset', ''))}" alt="{sample_id}">
        <div class="small">Image member: {_e(row.get('image_member', ''))}</div>
      </div>
      <div>
        <div class="row-title">
          <h2>{sample_id}</h2>
          <span class="badge">{_category_label(str(row.get('category', '')))}</span>
          {high_similarity_badge}
        </div>
        <div class="grid">
          <section class="panel">
            <h3>Current label</h3>
            <div class="label">{_e(row.get('current_emotion', ''))}</div>
            <div class="meta">Valence: {_e(row.get('current_valence', ''))}</div>
            <div class="meta">Arousal: {_e(row.get('current_arousal', ''))}</div>
          </section>
          <section class="panel">
            <h3>Clean prediction</h3>
            <div class="label">{_e(row.get('clean_emotion', ''))}</div>
            <div class="meta">Confidence: {_format_number(row.get('clean_confidence', ''))}</div>
            <div class="meta">Margin: {_format_number(row.get('clean_margin', ''))}</div>
            <div class="meta">KNN: {_e(row.get('clean_knn_emotion', ''))} ({_format_number(row.get('clean_knn_confidence', ''))})</div>
          </section>
          <section class="panel">
            <h3>Inclusive prediction</h3>
            <div class="label">{_e(row.get('inclusive_emotion', ''))}</div>
            <div class="meta">Confidence: {_format_number(row.get('inclusive_confidence', ''))}</div>
            <div class="meta">Margin: {_format_number(row.get('inclusive_margin', ''))}</div>
            <div class="meta">KNN: {_e(row.get('inclusive_knn_emotion', ''))} ({_format_number(row.get('inclusive_knn_confidence', ''))})</div>
          </section>
        </div>
        <div class="caption">{text_html}</div>
        <div class="manual">
          <label>
            <span class="small">Manual decision</span>
            <select name="manual_decision_{sample_id}">
              <option value=""></option>
              <option value="hold">hold</option>
              <option value="accept_clean">accept_clean</option>
              <option value="accept_inclusive">accept_inclusive</option>
            </select>
          </label>
          <label>
            <span class="small">Reviewer rationale</span>
            <textarea name="reviewer_rationale_{sample_id}"></textarea>
          </label>
        </div>
      </div>
    </article>
"""


def _read_current_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected Track2 JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _index_current_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(row.get("sample_id", "")): row for row in rows if row.get("sample_id")}
    if len(indexed) != len(rows):
        raise ValueError("Track2 rows must have unique non-empty sample_id values")
    return indexed


def _index_prediction_entries(payload: dict[str, Any] | list[Any]) -> dict[str, dict[str, Any]]:
    entries = _prediction_entries(payload)
    indexed = {str(row.get("sample_id", "")): row for row in entries if row.get("sample_id")}
    if len(indexed) != len(entries):
        raise ValueError("prediction entries must have unique non-empty sample_id values")
    return indexed


def _prediction_entries(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        entries = payload.get("entries")
    else:
        entries = payload
    if not isinstance(entries, list):
        raise ValueError("prediction payload must be a list or an object with an entries list")
    return [dict(row) for row in entries if isinstance(row, dict)]


def _category(current: str, clean: str, inclusive: str) -> str:
    clean_changes = clean != current
    inclusive_changes = inclusive != current
    if inclusive_changes and not clean_changes:
        return "inclusive_only_change"
    if clean_changes and not inclusive_changes:
        return "clean_only_change"
    return "clean_inclusive_conflict"


def _prediction_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "emotion": str(row.get("emotion", "")),
        "confidence": _float(row.get("confidence", 0.0)),
        "margin": _float(row.get("margin", 0.0)),
        "knn_emotion": str(row.get("knn_emotion", "")),
        "knn_confidence": _float(row.get("knn_confidence", 0.0)),
        "top3": _optional_prediction_list(row.get("top3")),
        "knn_top3": _optional_prediction_list(row.get("knn_top3")),
    }


def _category_rank(category: str) -> int:
    return CATEGORY_ORDER.get(category, 99)


def _category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("category", "")) for row in rows)
    return {category: int(counts.get(category, 0)) for category in CATEGORY_ORDER}


def _image_members_by_sample_id(image_zip: str | Path, sample_ids: Iterable[str]) -> dict[str, str]:
    with zipfile.ZipFile(image_zip) as zf:
        names = zf.namelist()
    return {sample_id: find_track2_image_member(names, sample_id) for sample_id in sample_ids}


def _extract_assets(image_zip: str | Path, rows: list[dict[str, Any]], assets_dir: Path) -> None:
    with zipfile.ZipFile(image_zip) as zf:
        for row in rows:
            member = str(row.get("image_member", ""))
            if not member:
                continue
            target = _asset_target(assets_dir, str(row["sample_id"]))
            with zf.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            row["image_asset"] = f"assets/{target.name}"


def _safe_asset_filename(sample_id: str) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", str(sample_id))
    if not safe_stem:
        safe_stem = "sample"
    return f"{safe_stem}.jpg"


def _asset_target(assets_dir: Path, sample_id: str) -> Path:
    assets_root = assets_dir.resolve()
    target = assets_dir / _safe_asset_filename(sample_id)
    resolved_target = target.resolve()
    if not resolved_target.is_relative_to(assets_root):
        raise ValueError(f"asset path escaped assets directory: {sample_id}")
    return target


def _optional_prediction_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    return []


def _write_decision_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(DECISION_COLUMNS)
        for row in rows:
            writer.writerow([row.get(column, "") for column in DECISION_COLUMNS])


def _category_label(category: str) -> str:
    return {
        "inclusive_only_change": "Inclusive-only",
        "clean_only_change": "Clean-only",
        "clean_inclusive_conflict": "Conflict",
    }.get(category, category)


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return _e(value)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)
