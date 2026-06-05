from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


BANK_VERSION = "track1_reference_family_bank_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]

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

MEDIUM_RULES: list[dict[str, Any]] = [
    {
        "medium_id": "propaganda_poster_print",
        "terms": [
            "poster",
            "propaganda",
            "cyrillic typography",
            "cyrillic",
            "typography",
            "sailor",
            "socialist realism",
        ],
    },
    {
        "medium_id": "scroll_or_album_paper_support",
        "terms": ["scroll", "album", "graph paper", "folded paper", "ruled page"],
    },
    {
        "medium_id": "painting_or_brushwork_surface",
        "terms": ["oil painting", "watercolor", "ink", "brushwork", "canvas"],
    },
    {
        "medium_id": "document_or_tableau_surface",
        "terms": ["document", "surrender", "treaty", "tableau"],
    },
]

GENERIC_MEDIUM_HINT = "generic_artwork_surface"


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


def infer_medium_hints(
    sample_id: str,
    caption: str,
    reference_path: str | Path = "",
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    searchable = _normalise_search_text(sample_id, caption, reference_path, metadata or {})
    hints: list[str] = []
    for rule in MEDIUM_RULES:
        if any(_term_matches(searchable, term) for term in rule["terms"]):
            hints.append(rule["medium_id"])
    return hints or [GENERIC_MEDIUM_HINT]


def build_reference_family_bank(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        caption = str(row.get("caption", ""))
        reference_path = _reference_path_from_row(row)
        image_status = _read_image_status(reference_path)
        family = infer_reference_family(sample_id, caption, reference_path)
        medium_hints = infer_medium_hints(sample_id, caption, reference_path, row)
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
                "medium_hints": medium_hints,
            }
        )

    aspect_counts = Counter(row["aspect"]["label"] for row in output_rows)
    family_counts = Counter(row["family_id"] for row in output_rows)
    medium_counts = Counter(hint for row in output_rows for hint in row["medium_hints"])
    return {
        "version": BANK_VERSION,
        "summary": {
            "version": BANK_VERSION,
            "total": len(output_rows),
            "missing_images": sum(1 for row in output_rows if not row["image_exists"]),
            "aspect_labels": dict(sorted(aspect_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "medium_counts": dict(sorted(medium_counts.items())),
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
    repo_root: str | Path | None = None,
) -> None:
    json_path, csv_path, md_path = _safe_output_paths(
        [json_path, csv_path, md_path],
        repo_root=repo_root,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _write_rows_csv(bank.get("rows", []), csv_path)
    md_path.write_text(_render_markdown_report(bank), encoding="utf-8")


def load_reference_rows(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise ValueError("reference JSON must be a list or an object with a 'rows' list")
    return [_validated_reference_row(row, index, source_path.parent) for index, row in enumerate(rows)]


def _safe_output_paths(
    paths: Iterable[str | Path],
    *,
    repo_root: str | Path | None = None,
) -> tuple[Path, ...]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    root = root.resolve(strict=False)
    resolved_paths = tuple(_resolve_under_root(path, root) for path in paths)
    protected_files = {
        (root / "submissions" / "track1_submission.json").resolve(strict=False),
        (root / "submissions" / "track1_submission.zip").resolve(strict=False),
    }
    protected_images = (root / "submissions" / "track1" / "images").resolve(strict=False)
    for output_path in resolved_paths:
        if output_path in protected_files or _is_relative_to(output_path, protected_images):
            raise ValueError(f"protected output path is not allowed: {output_path}")
    return resolved_paths


def _resolve_under_root(path: str | Path, root: Path) -> Path:
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = root / output_path
    return output_path.resolve(strict=False)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validated_reference_row(row: Any, index: int, base_dir: Path) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"reference row {index} must be an object")
    if not row.get("sample_id"):
        raise ValueError(f"reference row {index} missing required sample_id")
    if not row.get("caption"):
        raise ValueError(f"reference row {index} missing required caption")
    if not (row.get("reference_path") or row.get("path")):
        raise ValueError(f"reference row {index} missing required reference_path or path")

    validated = dict(row)
    for key in ("reference_path", "path"):
        if validated.get(key):
            reference_path = Path(str(validated[key]))
            if not reference_path.is_absolute():
                reference_path = base_dir / reference_path
            validated[key] = str(reference_path)
    return validated


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
            "reference_assets": _unique(
                row["reference_path"]
                for row in family_rows
                if row.get("reference_path") and row.get("image_exists")
            ),
            "matched_terms": _unique(term for row in family_rows for term in row["matched_terms"]),
            "composition_hints": _unique(hint for row in family_rows for hint in row["composition_hints"]),
            "medium_hints": _unique(hint for row in family_rows for hint in row["medium_hints"]),
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
        "medium_hints",
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
                    "medium_hints": " | ".join(row.get("medium_hints", [])),
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

    lines.extend(
        [
            "",
            "## Families",
            "",
            "| Family | Count | Aspect labels | Composition hints | Medium hints |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    families = bank.get("families", {})
    for family_id, family in families.items():
        aspect_text = ", ".join(f"{label}:{count}" for label, count in family.get("aspect_labels", {}).items())
        composition_text = "; ".join(family.get("composition_hints", []))
        medium_text = "; ".join(family.get("medium_hints", []))
        lines.append(
            f"| {family_id} | {family.get('count', 0)} | {aspect_text} | "
            f"{composition_text} | {medium_text} |"
        )

    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Sample | Family | Aspect | Medium hints | Reference |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in bank.get("rows", []):
        aspect = row.get("aspect", {})
        medium_text = "; ".join(row.get("medium_hints", []))
        lines.append(
            f"| {row.get('sample_id', '')} | {row.get('family_id', '')} | "
            f"{aspect.get('label', '')} | {medium_text} | {row.get('reference_path', '')} |"
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
