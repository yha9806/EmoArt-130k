from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


BANK_VERSION = "track1_reference_family_bank_v1"

FAMILY_RULES: list[dict[str, Any]] = [
    {
        "family_id": "kremlin_red_square",
        "terms": ["kremlin", "red square", "searchlight", "spasskaya"],
        "composition_hints": ["red brick tower", "searchlight", "night scene"],
    },
    {
        "family_id": "battle_tank_cavalry",
        "terms": ["battle", "tank", "cavalry", "mounted", "soldiers", "infantry"],
        "composition_hints": ["battlefield depth", "armored vehicle", "mounted movement", "infantry group"],
    },
    {
        "family_id": "naval_aviation_vehicle",
        "terms": ["naval", "sailor", "ship", "airplane", "pilot", "airmen", "train"],
        "composition_hints": ["uniformed sailor or pilot", "ship or vehicle silhouette", "transport machinery"],
    },
    {
        "family_id": "surrender_document_tableau",
        "terms": ["surrender", "document", "officer", "treaty", "declaration"],
        "composition_hints": ["central document", "officer figures", "formal table arrangement"],
    },
    {
        "family_id": "agriculture_industry_worker",
        "terms": ["wheat", "peasants", "factory", "worker", "industrial", "forge", "harvest"],
        "composition_hints": ["field or factory setting", "working hands and tools", "productive labor"],
    },
    {
        "family_id": "commemorative_medal_institution",
        "terms": ["medal", "anniversary", "academy", "commemorative", "ribbon"],
        "composition_hints": ["medal ribbon", "institutional emblem", "anniversary composition"],
    },
    {
        "family_id": "scroll_album_paper_support",
        "terms": ["scroll", "album", "graph paper", "folded paper", "ruled page"],
        "composition_hints": ["visible paper support", "folds or ruled grid", "album or scroll framing"],
    },
]

GENERIC_FAMILY: dict[str, Any] = {
    "family_id": "generic_artwork",
    "terms": [],
    "composition_hints": ["caption-led subject", "medium-appropriate composition", "avoid batch template"],
}


def classify_aspect(width: int, height: int) -> dict[str, Any]:
    width = int(width)
    height = int(height)
    ratio = width / height if height else None
    if ratio is None or width <= 0 or height <= 0:
        label = "unknown"
    elif ratio >= 1.12:
        label = "landscape"
    elif ratio <= 1 / 1.12:
        label = "portrait"
    else:
        label = "square_ish"
    return {
        "width": width,
        "height": height,
        "ratio": ratio,
        "label": label,
    }


def infer_reference_family(
    sample_id: str,
    caption: str,
    reference_path: str | Path = "",
) -> dict[str, Any]:
    searchable = _normalise_search_text(sample_id, caption, reference_path)
    best_rule = GENERIC_FAMILY
    best_terms: list[str] = []
    for rule in FAMILY_RULES:
        matched_terms = [term for term in rule["terms"] if _term_matches(searchable, term)]
        if len(matched_terms) > len(best_terms):
            best_rule = rule
            best_terms = matched_terms
    return {
        "sample_id": str(sample_id),
        "family_id": best_rule["family_id"],
        "matched_terms": best_terms,
        "composition_hints": list(best_rule["composition_hints"]),
    }


def build_reference_family_bank(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        caption = str(row.get("caption", ""))
        reference_path = _reference_path_from_row(row)
        image_status = _read_image_status(reference_path)
        family = infer_reference_family(sample_id, caption, reference_path)
        output_rows.append(
            {
                "sample_id": sample_id,
                "caption": caption,
                "reference_path": reference_path,
                "image_exists": image_status["image_exists"],
                "image_error": image_status.get("image_error", ""),
                "aspect": image_status["aspect"],
                "family_id": family["family_id"],
                "matched_terms": family["matched_terms"],
                "composition_hints": family["composition_hints"],
            }
        )

    aspect_counts = Counter(row["aspect"]["label"] for row in output_rows)
    family_counts = Counter(row["family_id"] for row in output_rows)
    return {
        "version": BANK_VERSION,
        "summary": {
            "version": BANK_VERSION,
            "total": len(output_rows),
            "missing_images": sum(1 for row in output_rows if not row["image_exists"]),
            "aspect_labels": dict(sorted(aspect_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
        },
        "rows": output_rows,
        "families": _summarize_families(output_rows),
    }


def write_reference_family_reports(
    bank: dict[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
    md_path: str | Path,
) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _write_rows_csv(bank.get("rows", []), csv_path)
    md_path.write_text(_render_markdown_report(bank), encoding="utf-8")


def load_reference_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise ValueError("reference JSON must be a list or an object with a 'rows' list")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _normalise_search_text(*parts: object) -> str:
    text = " ".join(str(part) for part in parts if part is not None)
    return re.sub(r"[\W_]+", " ", text.lower())


def _term_matches(searchable: str, term: str) -> bool:
    raw_words = term.lower().split()
    if not raw_words:
        return False
    words = [re.escape(word) for word in raw_words]
    raw_last = raw_words[-1]
    if raw_last.endswith("y"):
        words[-1] = f"{re.escape(raw_last[:-1])}(?:y|ies)"
    elif not raw_last.endswith("s"):
        words[-1] = f"{re.escape(raw_last)}s?"
    pattern = r"\b" + r"\s+".join(words) + r"\b"
    return re.search(pattern, searchable) is not None


def _reference_path_from_row(row: dict[str, Any]) -> str:
    value = row.get("reference_path") or row.get("path") or ""
    return str(value) if value else ""


def _read_image_status(reference_path: str) -> dict[str, Any]:
    if not reference_path:
        return {
            "image_exists": False,
            "aspect": {"width": None, "height": None, "ratio": None, "label": "missing"},
        }
    path = Path(reference_path)
    if not path.exists():
        return {
            "image_exists": False,
            "aspect": {"width": None, "height": None, "ratio": None, "label": "missing"},
        }

    from PIL import Image

    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:  # pragma: no cover - defensive for corrupt local assets
        return {
            "image_exists": False,
            "image_error": str(exc),
            "aspect": {"width": None, "height": None, "ratio": None, "label": "unreadable"},
        }
    return {
        "image_exists": True,
        "aspect": classify_aspect(width, height),
    }


def _summarize_families(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family_id in sorted({row["family_id"] for row in rows}):
        family_rows = [row for row in rows if row["family_id"] == family_id]
        aspect_counts = Counter(row["aspect"]["label"] for row in family_rows)
        families[family_id] = {
            "family_id": family_id,
            "count": len(family_rows),
            "sample_ids": [row["sample_id"] for row in family_rows],
            "aspect_labels": dict(sorted(aspect_counts.items())),
            "matched_terms": _unique(term for row in family_rows for term in row["matched_terms"]),
            "composition_hints": _unique(hint for row in family_rows for hint in row["composition_hints"]),
        }
    return families


def _write_rows_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "family_id",
        "aspect_label",
        "width",
        "height",
        "ratio",
        "image_exists",
        "reference_path",
        "matched_terms",
        "composition_hints",
        "caption",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            aspect = row.get("aspect", {})
            ratio = aspect.get("ratio")
            writer.writerow(
                {
                    "sample_id": row.get("sample_id", ""),
                    "family_id": row.get("family_id", ""),
                    "aspect_label": aspect.get("label", ""),
                    "width": aspect.get("width") or "",
                    "height": aspect.get("height") or "",
                    "ratio": f"{ratio:.6f}" if isinstance(ratio, float) else "",
                    "image_exists": row.get("image_exists", False),
                    "reference_path": row.get("reference_path", ""),
                    "matched_terms": " | ".join(row.get("matched_terms", [])),
                    "composition_hints": " | ".join(row.get("composition_hints", [])),
                    "caption": row.get("caption", ""),
                }
            )


def _render_markdown_report(bank: dict[str, Any]) -> str:
    summary = bank.get("summary", {})
    lines = [
        "# Track1 Reference Family Bank",
        "",
        f"- Version: {bank.get('version', BANK_VERSION)}",
        f"- Total references: {summary.get('total', 0)}",
        f"- Missing images: {summary.get('missing_images', 0)}",
        "",
        "## Aspect Labels",
        "",
    ]
    aspect_labels = summary.get("aspect_labels", {})
    if aspect_labels:
        for label, count in aspect_labels.items():
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none: 0")

    lines.extend(["", "## Families", "", "| Family | Count | Aspect labels | Hints |", "| --- | ---: | --- | --- |"])
    families = bank.get("families", {})
    for family_id, family in families.items():
        aspect_text = ", ".join(f"{label}:{count}" for label, count in family.get("aspect_labels", {}).items())
        hint_text = "; ".join(family.get("composition_hints", []))
        lines.append(f"| {family_id} | {family.get('count', 0)} | {aspect_text} | {hint_text} |")

    lines.extend(["", "## Rows", "", "| Sample | Family | Aspect | Reference |", "| --- | --- | --- | --- |"])
    for row in bank.get("rows", []):
        aspect = row.get("aspect", {})
        lines.append(
            f"| {row.get('sample_id', '')} | {row.get('family_id', '')} | "
            f"{aspect.get('label', '')} | {row.get('reference_path', '')} |"
        )
    return "\n".join(lines).strip() + "\n"


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
