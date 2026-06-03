"""Build compact image contact sheets for Track2 Gemini-agent review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/yhryzy/dev/emoart-130k")
DEFAULT_HTML_REVIEW_DIR = (
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
    "alarmed": "jingkong/jingjue",
    "amazed": "jingqi",
    "amused": "yuyue/youqu",
    "annoyed": "naonu",
    "aroused": "jifa/xingfen",
    "bored": "yanjuan",
    "calm": "pingjing",
    "content": "manzu/anshi",
    "disgusted": "yanwu",
    "excited": "xingfen",
    "frustrated": "cuobai/shouzu",
    "glad": "gaoxing",
    "happy": "kuaile",
    "sad": "beishang",
    "tired": "pibei",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def index_submission(path: Path) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in read_json(path)}


def label(row: dict[str, Any]) -> str:
    emotion = row["emotion"]
    return f"{emotion} ({LABEL_ZH.get(emotion, '')}) / {row['emotional_valence']} / {row['emotional_arousal_level']}"


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def high_conf_rows(audit_report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(audit_report.get("high_conf_blocks") or [])


def risk_context_rows(audit_report: dict[str, Any]) -> list[dict[str, Any]]:
    high_ids = {row["sample_id"] for row in audit_report.get("high_conf_blocks", [])}
    return [row for row in audit_report.get("risk_rows", []) if row["sample_id"] not in high_ids]


def draw_sheet(
    rows: list[dict[str, Any]],
    out_path: Path,
    review_dir: Path,
    base_index: dict[str, dict[str, Any]],
    desc_index: dict[str, dict[str, Any]],
    guarded_index: dict[str, dict[str, Any]],
    title: str,
) -> None:
    cols = 2
    card_w = 920
    image_w = 300
    image_h = 240
    card_h = 340
    margin = 28
    gap = 20
    header_h = 94
    rows_n = (len(rows) + cols - 1) // cols
    width = margin * 2 + cols * card_w + (cols - 1) * gap
    height = header_h + margin + rows_n * card_h + max(0, rows_n - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), "#eef1f5")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34, bold=True)
    h_font = load_font(22, bold=True)
    text_font = load_font(18)
    small_font = load_font(15)
    draw.text((margin, 24), title, fill="#202124", font=title_font)
    draw.text(
        (margin, 64),
        "Use this sheet for Gemini-agent second-opinion labeling. Default question: keep Guarded v1 or restore Scaled desc_v1?",
        fill="#5f6368",
        font=small_font,
    )
    for idx, row in enumerate(rows):
        sample_id = row["sample_id"]
        col = idx % cols
        line = idx // cols
        x = margin + col * (card_w + gap)
        y = header_h + margin + line * (card_h + gap)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=12, fill="#ffffff", outline="#d7dbe0")
        image_path = review_dir / "assets" / f"{sample_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((image_w, image_h), Image.Resampling.LANCZOS)
        image_bg = Image.new("RGB", (image_w, image_h), "#f3f4f6")
        image_bg.paste(image, ((image_w - image.width) // 2, (image_h - image.height) // 2))
        canvas.paste(image_bg, (x + 18, y + 72))
        draw.text((x + 18, y + 18), f"{idx + 1:02d}. {sample_id}", fill="#202124", font=h_font)
        draw.text((x + 18, y + 46), "High-conf rollback" if row in high_conf_rows(read_json(DEFAULT_AUDIT_REPORT)) else "Risk context", fill="#b3261e", font=small_font)
        tx = x + 338
        ty = y + 72
        base = base_index[sample_id]
        desc = desc_index[sample_id]
        guarded = guarded_index[sample_id]
        lines = [
            ("Base:", label(base), "#0b7f58"),
            ("Scaled desc_v1:", label(desc), "#b75e00"),
            ("Guarded v1:", label(guarded), "#1f5eff"),
            ("Gemini prior:", f"{row.get('gemini', {}).get('decision', '')}; conf={row.get('gemini', {}).get('confidence_1_5', '')}", "#5f6368"),
        ]
        for prefix, text, color in lines:
            draw.text((tx, ty), prefix, fill=color, font=small_font)
            ty += 20
            for wrapped in wrap(draw, text, text_font, card_w - image_w - 80)[:2]:
                draw.text((tx, ty), wrapped, fill="#202124", font=text_font)
                ty += 24
            ty += 4
        rationale = str(row.get("gemini", {}).get("rationale", ""))
        draw.text((tx, ty), "One-line reason:", fill="#5f6368", font=small_font)
        ty += 20
        for wrapped in wrap(draw, rationale, small_font, card_w - image_w - 80)[:4]:
            draw.text((tx, ty), wrapped, fill="#3c4043", font=small_font)
            ty += 19
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_HTML_REVIEW_DIR)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    parser.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    parser.add_argument("--desc-json", type=Path, default=DEFAULT_DESC_JSON)
    parser.add_argument("--guarded-json", type=Path, default=DEFAULT_GUARDED_JSON)
    args = parser.parse_args()

    audit_report = read_json(args.audit_report)
    base_index = index_submission(args.base_json)
    desc_index = index_submission(args.desc_json)
    guarded_index = index_submission(args.guarded_json)
    high_path = args.review_dir / "track2_gemini35_high_conf_rollback_contact_sheet.jpg"
    risk_path = args.review_dir / "track2_gemini35_risk_context_contact_sheet.jpg"
    draw_sheet(
        high_conf_rows(audit_report),
        high_path,
        args.review_dir,
        base_index,
        desc_index,
        guarded_index,
        "Track2 High-Confidence Rollback Review",
    )
    draw_sheet(
        risk_context_rows(audit_report),
        risk_path,
        args.review_dir,
        base_index,
        desc_index,
        guarded_index,
        "Track2 Risk-Context Review",
    )
    print(json.dumps({"high_conf_sheet": str(high_path), "risk_context_sheet": str(risk_path)}, indent=2))


if __name__ == "__main__":
    main()
