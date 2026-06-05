from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from affectiveart.track1_reference_family_bank import (
    _safe_output_paths,
    infer_medium_hints,
    infer_reference_family,
)


ROUTER_VERSION = "track1_distribution_router_v1"
DEFAULT_STRATEGIES = ("aas_safe", "reference_style", "fid_diverse")

SUPPORT_TERMS = (
    "scroll",
    "album",
    "graph paper",
    "folded paper",
    "ruled page",
    "paper support",
    "silk mounting",
    "roller rods",
)
LANDMARK_TERMS = ("kremlin", "red square", "spasskaya")


def build_distribution_route(
    contract: dict[str, Any],
    reference_family_bank: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_id = str(contract.get("sample_id", ""))
    caption = str(contract.get("caption", ""))
    bank_row = _bank_row_for_sample(reference_family_bank, sample_id)
    inferred_family = infer_reference_family(sample_id, caption)
    family_id = _route_family_id(contract, inferred_family["family_id"], bank_row)
    family_summary = _bank_family(reference_family_bank, family_id)

    hard_constraints = _hard_constraints(contract, caption, family_id)
    style_freedom = _style_freedom(caption, family_id)
    composition_hints = _composition_hints(bank_row, family_summary, inferred_family)
    medium_options = _medium_options(sample_id, caption, bank_row, family_summary)
    aspect_hints = _aspect_hints(contract, bank_row, family_summary)
    reference_assets = _reference_assets(bank_row, family_summary)

    return {
        "version": ROUTER_VERSION,
        "sample_id": sample_id,
        "caption": caption,
        "family_id": family_id,
        "hard_constraints": hard_constraints,
        "style_freedom": style_freedom,
        "composition_hints": composition_hints,
        "medium_options": medium_options,
        "aspect_hints": aspect_hints,
        "reference_assets": reference_assets,
        "candidate_strategies": [{"strategy": strategy} for strategy in DEFAULT_STRATEGIES],
    }


def build_distribution_routes(
    contracts: Iterable[dict[str, Any]],
    reference_family_bank: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        build_distribution_route(contract, reference_family_bank=reference_family_bank)
        for contract in contracts
    ]


def write_distribution_route_reports(
    routes: Iterable[dict[str, Any]],
    json_path: str | Path,
    csv_path: str | Path,
    md_path: str | Path,
    repo_root: str | Path | None = None,
) -> None:
    route_list = list(routes)
    json_path, csv_path, md_path = _safe_output_paths(
        [json_path, csv_path, md_path],
        repo_root=repo_root,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"summary": _summary(route_list), "routes": route_list}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _write_routes_csv(route_list, csv_path)
    md_path.write_text(_render_markdown_report(payload), encoding="utf-8")


def load_contract_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    elif isinstance(payload, dict) and isinstance(payload.get("contracts"), list):
        rows = payload["contracts"]
    else:
        raise ValueError("contracts JSON must be a list or an object with 'rows' or 'contracts'")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"contract row {index} must be an object")
        if not row.get("sample_id"):
            raise ValueError(f"contract row {index} missing required sample_id")
        if not row.get("caption"):
            raise ValueError(f"contract row {index} missing required caption")
    return [dict(row) for row in rows]


def load_reference_family_bank(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _route_family_id(contract: dict[str, Any], inferred_family_id: str, bank_row: dict[str, Any] | None) -> str:
    if bank_row and bank_row.get("family_id"):
        return str(bank_row["family_id"])
    caption = str(contract.get("caption", ""))
    if _has_landmark_reference(contract, caption):
        return "kremlin_red_square"
    if _has_any_term(caption, SUPPORT_TERMS):
        return "scroll_album_paper_support"
    return inferred_family_id


def _hard_constraints(contract: dict[str, Any], caption: str, family_id: str) -> list[str]:
    constraints = ["caption_content"]
    if family_id == "kremlin_red_square" or _has_landmark_reference(contract, caption):
        constraints.append("landmark_reference")
    if family_id == "scroll_album_paper_support" or _has_any_term(caption, SUPPORT_TERMS):
        constraints.append("support_surface_required")
    if _contract_required(contract, "text_contract"):
        constraints.append("requested_text_policy")
    if _contract_required(contract, "relation_contract"):
        constraints.append("relation_logic")
    return _unique(constraints)


def _style_freedom(caption: str, family_id: str) -> list[str]:
    if family_id == "scroll_album_paper_support" or _has_any_term(caption, SUPPORT_TERMS):
        return ["preserve_support_aspect", "medium_texture_variation_allowed", "viewpoint_variation_allowed"]
    if family_id == "kremlin_red_square" or _has_any_term(caption, LANDMARK_TERMS):
        return ["aspect_variation_allowed", "style_variation_allowed", "crop_variation_allowed"]
    return ["free_crop_allowed", "aspect_variation_allowed", "style_variation_allowed"]


def _composition_hints(
    bank_row: dict[str, Any] | None,
    family_summary: dict[str, Any] | None,
    inferred_family: dict[str, Any],
) -> list[str]:
    if bank_row and isinstance(bank_row.get("composition_hints"), list):
        return [str(hint) for hint in bank_row["composition_hints"]]
    if family_summary and isinstance(family_summary.get("composition_hints"), list):
        return [str(hint) for hint in family_summary["composition_hints"]]
    return [str(hint) for hint in inferred_family.get("composition_hints", [])]


def _medium_options(
    sample_id: str,
    caption: str,
    bank_row: dict[str, Any] | None,
    family_summary: dict[str, Any] | None,
) -> list[str]:
    if bank_row and isinstance(bank_row.get("medium_hints"), list):
        return [str(hint) for hint in bank_row["medium_hints"]]
    if family_summary and isinstance(family_summary.get("medium_hints"), list):
        return [str(hint) for hint in family_summary["medium_hints"]]
    return infer_medium_hints(sample_id, caption)


def _aspect_hints(
    contract: dict[str, Any],
    bank_row: dict[str, Any] | None,
    family_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if bank_row and isinstance(bank_row.get("aspect"), dict):
        return [dict(bank_row["aspect"])]
    if family_summary and isinstance(family_summary.get("aspect_labels"), dict):
        return [
            {"label": label, "count": count}
            for label, count in sorted(family_summary["aspect_labels"].items())
        ]
    aspect_plan = contract.get("aspect_plan")
    if isinstance(aspect_plan, dict):
        return [dict(aspect_plan)]
    return []


def _reference_assets(
    bank_row: dict[str, Any] | None,
    family_summary: dict[str, Any] | None,
) -> list[str]:
    assets: list[str] = []
    if bank_row:
        direct = bank_row.get("reference_asset") or bank_row.get("reference_path") or bank_row.get("path")
        if direct:
            assets.append(str(direct))
        if isinstance(bank_row.get("reference_assets"), list):
            assets.extend(str(item) for item in bank_row["reference_assets"] if str(item))
    if family_summary and isinstance(family_summary.get("reference_assets"), list):
        assets.extend(str(item) for item in family_summary["reference_assets"] if str(item))
    return _unique(assets)


def _bank_row_for_sample(bank: dict[str, Any] | None, sample_id: str) -> dict[str, Any] | None:
    if not bank:
        return None
    rows = bank.get("rows", [])
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and str(row.get("sample_id", "")) == sample_id:
            return row
    return None


def _bank_family(bank: dict[str, Any] | None, family_id: str) -> dict[str, Any] | None:
    if not bank or not isinstance(bank.get("families"), dict):
        return None
    family = bank["families"].get(family_id)
    return family if isinstance(family, dict) else None


def _summary(routes: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(route.get("family_id", "") for route in routes)
    strategy_counts = Counter(
        strategy.get("strategy", "")
        for route in routes
        for strategy in route.get("candidate_strategies", [])
        if isinstance(strategy, dict)
    )
    return {
        "version": ROUTER_VERSION,
        "total": len(routes),
        "families": dict(sorted(family_counts.items())),
        "strategies": dict(sorted(strategy_counts.items())),
    }


def _write_routes_csv(routes: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "family_id",
        "hard_constraints",
        "style_freedom",
        "composition_hints",
        "medium_options",
        "aspect_hints",
        "reference_assets",
        "candidate_strategies",
        "caption",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for route in routes:
            writer.writerow(
                {
                    "sample_id": route.get("sample_id", ""),
                    "family_id": route.get("family_id", ""),
                    "hard_constraints": " | ".join(route.get("hard_constraints", [])),
                    "style_freedom": " | ".join(route.get("style_freedom", [])),
                    "composition_hints": " | ".join(route.get("composition_hints", [])),
                    "medium_options": " | ".join(route.get("medium_options", [])),
                    "aspect_hints": json.dumps(route.get("aspect_hints", []), sort_keys=True),
                    "reference_assets": " | ".join(route.get("reference_assets", [])),
                    "candidate_strategies": " | ".join(
                        strategy.get("strategy", "")
                        for strategy in route.get("candidate_strategies", [])
                        if isinstance(strategy, dict)
                    ),
                    "caption": route.get("caption", ""),
                }
            )


def _render_markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Track1 Distribution Router Report",
        "",
        f"- Version: {summary['version']}",
        f"- Total routes: {summary['total']}",
        "",
        "## Families",
        "",
    ]
    for family_id, count in summary["families"].items():
        lines.append(f"- {family_id}: {count}")
    lines.extend(["", "## Routes", ""])
    for route in payload["routes"]:
        lines.extend(
            [
                f"### {route.get('sample_id', '')}",
                "",
                f"- Family: {route.get('family_id', '')}",
                f"- Hard constraints: {', '.join(route.get('hard_constraints', []))}",
                f"- Style freedom: {', '.join(route.get('style_freedom', []))}",
                f"- Medium options: {', '.join(route.get('medium_options', []))}",
                f"- Candidate strategies: {', '.join(strategy['strategy'] for strategy in route.get('candidate_strategies', []))}",
                "",
            ]
        )
    return "\n".join(lines)


def _has_landmark_reference(contract: dict[str, Any], caption: str) -> bool:
    reference_contract = contract.get("reference_contract")
    categories: list[str] = []
    if isinstance(reference_contract, dict):
        raw_categories = reference_contract.get("categories", [])
        if isinstance(raw_categories, list):
            categories = [str(category).lower() for category in raw_categories]
    return "landmark" in categories or _has_any_term(caption, LANDMARK_TERMS)


def _contract_required(contract: dict[str, Any], key: str) -> bool:
    value = contract.get(key)
    return isinstance(value, dict) and bool(value.get("required"))


def _has_any_term(text: str, terms: Iterable[str]) -> bool:
    text_lower = text.lower()
    return any(term in text_lower for term in terms)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output
