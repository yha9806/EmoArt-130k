from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from affectiveart.track1_reference_family_bank import _safe_output_paths


BINDING_VERSION = "track1_reference_asset_bindings_v4"

VALID_SELECTION_MODES = {"stable", "diversity_balanced"}
PROTECTED_FAMILY_POSTER_NOTE = "preserved family poster/print reference"

POSTER_CAPTION_KEYWORDS = (
    "propaganda poster",
    "poster",
    "cyrillic typography",
    "bold cyrillic",
    "typography",
)

POSTER_ASSET_NAME_KEYWORDS = (
    "poster",
    "tasswindow",
    "tass_window",
    "frontpage",
    "allforthefront",
    "allforvictory",
    "liberate",
    "tovictory",
)

FAMILY_POSTER_ASSET_NAME_KEYWORDS = POSTER_ASSET_NAME_KEYWORDS + (
    "kukryniksy",
)

DEFAULT_FAMILY_REFERENCE_FILES: dict[str, list[str]] = {
    "kremlin_red_square": [
        "ref_0109226_KonstantinYuon-MilitaryParadeinRedSquare_7thNovember1941.jpg",
        "ref_0123502_KonstantinYuon-TheRedArmyonparadeinRedSquare_Moscow_November1940.jpg",
        "ref_0117410_DmitryNalbandyan-ReceptionintheKremlin_May24_1945.jpg",
        "ref_0161653_ElLissitzky-Allforthefront_AllforVictory_.jpg",
    ],
    "battle_tank_cavalry": [
        "ref_0041614_Kukryniksy-Splendidlyanddesperatelydowefight_theissueofSuvorovandChapayevdoitallright_.jpg",
        "ref_0013135_Kukryniksy-Liberate.jpg",
        "ref_0100747_Kukryniksy-Tovictory_.jpg",
        "ref_0108635_Kukryniksy-OnfrontlinesbytheDnieper_TheTASSWindow_906_.jpg",
    ],
    "naval_aviation_vehicle": [
        "ref_0038204_VeniaminKremer-DneprflotillaboatsontheoutskirtsofthetownofPinsk.jpg",
        "ref_0051830_VeniaminKremer-TheMarinesontheFightattheLeningradFrontinJanuary1941.jpg",
        "ref_0094218_SergiyGrigoriev-OntheDniperRiver.jpg",
        "ref_0077202_VictorPuzyrkov-OutskirtsoftheVillage.jpg",
    ],
    "generic_artwork": [
        "ref_0057522_VictorZaretsky-TheVictoryBanner.ASketchForMosaicInKrasnodon_MadeInCooperationWithVictorSmirnovAndAllaHorska_.jpg",
        "ref_0161653_ElLissitzky-Allforthefront_AllforVictory_.jpg",
        "ref_0011707_BorisKustodiev-Subscribeto1927thedailynewspaperIzvestiaUSSRCentralExecutiveCommittee.jpg",
        "ref_0038493_PavelFilonov-Udarnitzi_RecordBreakingWorkers_attheFactoryKrasnayaZaria.jpg",
    ],
}


def load_reference_asset_index(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reference asset index JSON must be an object")
    references = payload.get("references", {})
    if references is not None and not isinstance(references, dict):
        raise ValueError("reference asset index 'references' must be an object")
    route_references = payload.get("route_references", {})
    if route_references is not None and not isinstance(route_references, dict):
        raise ValueError("reference asset index 'route_references' must be an object")
    family_references = payload.get("family_references", {})
    if family_references is not None and not isinstance(family_references, dict):
        raise ValueError("reference asset index 'family_references' must be an object")
    caption_style_references = payload.get("caption_style_references", {})
    if caption_style_references is not None and not isinstance(caption_style_references, dict):
        raise ValueError("reference asset index 'caption_style_references' must be an object")
    return dict(payload)


def load_route_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("routes"), list):
        rows = payload["routes"]
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise ValueError("routes JSON must be a list or an object with 'routes' or 'rows'")
    return [_validated_route(row, index) for index, row in enumerate(rows)]


def attach_reference_assets_to_routes(
    routes: Iterable[dict[str, Any]],
    reference_asset_index: dict[str, Any],
    *,
    asset_root: str | Path,
    max_assets_per_route: int = 4,
    selection_mode: str = "stable",
) -> list[dict[str, Any]]:
    if max_assets_per_route <= 0:
        raise ValueError("max_assets_per_route must be positive")
    if selection_mode not in VALID_SELECTION_MODES:
        valid = ", ".join(sorted(VALID_SELECTION_MODES))
        raise ValueError(f"selection_mode must be one of: {valid}")
    asset_root_path = Path(asset_root).expanduser()
    usage_counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for route in routes:
        normalized = _validated_route(route, len(output))
        candidates = _candidate_items_for_route(normalized, reference_asset_index)
        selected = _select_existing_assets(
            candidates,
            asset_root_path,
            max_assets_per_route=max_assets_per_route,
            selection_mode=selection_mode,
            sample_id=normalized["sample_id"],
            usage_counts=usage_counts,
        )
        route_out = dict(normalized)
        route_out["reference_assets"] = [str(item["path"]) for item in selected["items"]]
        route_out["reference_asset_notes"] = [str(item["note"]) for item in selected["items"] if str(item["note"])]
        route_out["reference_asset_source"] = selected["source"]
        route_out["reference_style_key"] = selected["style_key"]
        route_out["reference_asset_missing"] = selected["missing"]
        route_out["reference_selection_mode"] = selection_mode
        route_out["reference_candidate_pool_size"] = selected["candidate_pool_size"]
        route_out["reference_selected_identities"] = selected["selected_identities"]
        route_out["reference_diversity_limited"] = selected["diversity_limited"]
        output.append(route_out)
    return output


def write_reference_asset_route_reports(
    routes: list[dict[str, Any]],
    *,
    json_path: str | Path,
    csv_path: str | Path,
    md_path: str | Path,
    repo_root: str | Path | None = None,
) -> None:
    json_path, csv_path, md_path = _safe_output_paths([json_path, csv_path, md_path], repo_root=repo_root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": _summary(routes), "routes": routes}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(routes, csv_path)
    md_path.write_text(_render_md(payload), encoding="utf-8")


def _validated_route(row: Any, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"route row {index} must be an object")
    if not row.get("sample_id"):
        raise ValueError(f"route row {index} missing required sample_id")
    normalized = dict(row)
    normalized["sample_id"] = str(row["sample_id"])
    if row.get("family_id") is not None:
        normalized["family_id"] = str(row.get("family_id") or "")
    return normalized


def _candidate_items_for_route(route: dict[str, Any], index: dict[str, Any]) -> list[dict[str, Any]]:
    sample_id = str(route.get("sample_id") or "")
    family_id = str(route.get("family_id") or "")
    references = index.get("references", {})
    route_references = index.get("route_references", {})
    family_references = index.get("family_references", {})
    caption_style_references = index.get("caption_style_references", {})
    official_route_items = _normalise_index_items(route_references.get(sample_id, []), source="official_retrieval")
    if official_route_items:
        return official_route_items
    direct_items = _normalise_index_items(references.get(sample_id, []), source="sample")
    if direct_items:
        return direct_items
    family_items = _normalise_index_items(family_references.get(family_id, []), source="family")
    media_key, media_items = _caption_media_items(route, caption_style_references)
    if media_items:
        return _reranked_caption_media_items(family_items, media_items, media_key)
    if family_items:
        return family_items
    style_key, style_values = _caption_style_items(route, caption_style_references)
    if style_values:
        items = _normalise_index_items(style_values, source="caption_style")
        for item in items:
            item["style_key"] = style_key
        return items
    return _normalise_index_items(DEFAULT_FAMILY_REFERENCE_FILES.get(family_id, []), source="family_default")


def _normalise_index_items(values: Any, *, source: str) -> list[dict[str, str]]:
    if not isinstance(values, list):
        return []
    output: list[dict[str, str]] = []
    for value in values:
        if isinstance(value, dict):
            file_value = str(value.get("file") or value.get("path") or "").strip()
            note = str(value.get("note") or value.get("source_name") or "").strip()
        else:
            file_value = str(value).strip()
            note = ""
        if file_value:
            output.append({"file": file_value, "note": note, "source": source})
    return output


def _select_existing_assets(
    candidates: list[dict[str, str]],
    asset_root: Path,
    *,
    max_assets_per_route: int,
    selection_mode: str,
    sample_id: str,
    usage_counts: Counter[str],
) -> dict[str, Any]:
    resolved: list[dict[str, str]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(candidates):
        path = _asset_path(asset_root, item["file"])
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.exists() or not path.is_file():
            missing.append(str(path))
            continue
        resolved.append(
            {
                "path": str(path),
                "note": item.get("note", ""),
                "source": item["source"],
                "style_key": item.get("style_key", ""),
                "identity": _reference_identity(path.name),
                "original_index": str(index),
            }
        )
    selected = _select_resolved_assets(
        resolved,
        max_assets_per_route=max_assets_per_route,
        selection_mode=selection_mode,
        sample_id=sample_id,
        usage_counts=usage_counts,
    )
    for item in selected:
        usage_counts.update([item["identity"]])
    source = str(selected[0]["source"]) if selected else "missing"
    style_key = str(selected[0].get("style_key") or "") if selected else ""
    diversity_limited = bool(selected and source != "sample" and len(resolved) <= max_assets_per_route)
    return {
        "items": selected,
        "source": source,
        "style_key": style_key,
        "missing": missing,
        "candidate_pool_size": len(resolved),
        "selected_identities": [str(item["identity"]) for item in selected],
        "diversity_limited": diversity_limited,
    }


def _select_resolved_assets(
    resolved: list[dict[str, str]],
    *,
    max_assets_per_route: int,
    selection_mode: str,
    sample_id: str,
    usage_counts: Counter[str],
) -> list[dict[str, str]]:
    if selection_mode == "stable" or len(resolved) <= max_assets_per_route:
        return resolved[:max_assets_per_route]
    if resolved and resolved[0].get("source") == "sample":
        return resolved[:max_assets_per_route]
    protected = [
        item
        for item in resolved
        if PROTECTED_FAMILY_POSTER_NOTE in str(item.get("note", ""))
    ][: min(2, max_assets_per_route)]
    protected_keys = {str(item["path"]) for item in protected}
    remaining = [item for item in resolved if str(item["path"]) not in protected_keys]
    remaining.sort(
        key=lambda item: (
            usage_counts[str(item["identity"])],
            _stable_diversity_tiebreaker(sample_id, str(item["identity"])),
            int(item["original_index"]),
        )
    )
    return (protected + remaining)[:max_assets_per_route]


def _stable_diversity_tiebreaker(sample_id: str, identity: str) -> int:
    digest = hashlib.sha1(f"{sample_id}:{identity}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _caption_style_items(
    route: dict[str, Any],
    caption_style_references: Any,
) -> tuple[str, list[Any]]:
    if not isinstance(caption_style_references, dict):
        return "", []
    caption = str(route.get("caption") or "").lower()
    for style_key in sorted(caption_style_references):
        key = str(style_key).strip().lower()
        if key and key in caption:
            values = caption_style_references.get(style_key, [])
            return str(style_key), values if isinstance(values, list) else []
    return "", []


def _caption_media_items(
    route: dict[str, Any],
    caption_style_references: Any,
) -> tuple[str, list[Any]]:
    if not _requires_poster_reference(route):
        return "", []
    style_key, style_values = _caption_style_items(route, caption_style_references)
    poster_items = _poster_like_items(style_values)
    if poster_items:
        return f"{style_key}:poster_print", poster_items
    return "", []


def _reranked_caption_media_items(
    family_items: list[dict[str, str]],
    media_values: list[Any],
    media_key: str,
) -> list[dict[str, str]]:
    family_poster_items = _poster_like_items(family_items, keywords=FAMILY_POSTER_ASSET_NAME_KEYWORDS)
    if not family_poster_items:
        return _media_items_with_note(media_values, source="caption_media", style_key=media_key)
    media_items = _media_items_with_note(media_values, source="caption_media_reranked", style_key=media_key)
    reranked = _normalise_index_items(family_poster_items, source="caption_media_reranked")
    for item in reranked:
        item["style_key"] = media_key
        note = item.get("note", "")
        media_note = "preserved family poster/print reference for caption media fit"
        item["note"] = f"{media_note}; {note}" if note else media_note
    return _dedupe_reference_items(reranked + media_items)


def _media_items_with_note(values: list[Any], *, source: str, style_key: str) -> list[dict[str, str]]:
    items = _normalise_index_items(values, source=source)
    for item in items:
        item["style_key"] = style_key
        note = item.get("note", "")
        media_note = "caption requires poster/propaganda print medium"
        item["note"] = f"{media_note}; {note}" if note else media_note
    return items


def _requires_poster_reference(route: dict[str, Any]) -> bool:
    caption = str(route.get("caption") or "").lower()
    return any(keyword in caption for keyword in POSTER_CAPTION_KEYWORDS)


def _poster_like_items(values: list[Any], *, keywords: tuple[str, ...] = POSTER_ASSET_NAME_KEYWORDS) -> list[Any]:
    output: list[Any] = []
    for value in values:
        file_value = ""
        if isinstance(value, dict):
            file_value = str(value.get("file") or value.get("path") or "")
        else:
            file_value = str(value)
        normalized = _compact_reference_name(file_value)
        if any(keyword in normalized for keyword in keywords):
            output.append(value)
    return output


def _compact_reference_name(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum() or char == "_")


def _dedupe_reference_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        key = _reference_identity(item.get("file", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _reference_identity(file_value: str) -> str:
    stem = Path(file_value).stem.lower()
    for part in stem.replace("-", "_").split("_"):
        if part.isdigit() and len(part) >= 6:
            return part
    return _compact_reference_name(stem)


def _asset_path(asset_root: Path, file_value: str) -> Path:
    path = Path(file_value).expanduser()
    if path.is_absolute():
        return path
    return asset_root / path


def _summary(routes: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(str(route.get("family_id") or "") for route in routes)
    source_counts = Counter(str(route.get("reference_asset_source") or "") for route in routes)
    asset_counts = Counter(
        Path(str(asset)).name
        for route in routes
        for asset in route.get("reference_assets", [])
    )
    return {
        "version": BINDING_VERSION,
        "total": len(routes),
        "routes_with_reference_assets": sum(1 for route in routes if route.get("reference_assets")),
        "reference_asset_slots": sum(len(route.get("reference_assets", [])) for route in routes),
        "unique_reference_assets": len(asset_counts),
        "diversity_limited_routes": sum(1 for route in routes if route.get("reference_diversity_limited")),
        "top_reference_reuse": [
            {"file": file, "count": count}
            for file, count in asset_counts.most_common(20)
        ],
        "reference_asset_sources": dict(sorted(source_counts.items())),
        "families": dict(sorted(family_counts.items())),
    }


def _write_csv(routes: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "family_id",
        "reference_asset_source",
        "reference_style_key",
        "reference_assets",
        "reference_asset_notes",
        "reference_asset_missing",
        "reference_selection_mode",
        "reference_candidate_pool_size",
        "reference_diversity_limited",
        "reference_selected_identities",
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
                    "reference_asset_source": route.get("reference_asset_source", ""),
                    "reference_style_key": route.get("reference_style_key", ""),
                    "reference_assets": " | ".join(route.get("reference_assets", [])),
                    "reference_asset_notes": " | ".join(route.get("reference_asset_notes", [])),
                    "reference_asset_missing": " | ".join(route.get("reference_asset_missing", [])),
                    "reference_selection_mode": route.get("reference_selection_mode", ""),
                    "reference_candidate_pool_size": route.get("reference_candidate_pool_size", ""),
                    "reference_diversity_limited": route.get("reference_diversity_limited", ""),
                    "reference_selected_identities": " | ".join(route.get("reference_selected_identities", [])),
                    "caption": route.get("caption", ""),
                }
            )


def _render_md(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Track1 Reference Asset Route Bindings",
        "",
        "这份报告只记录 reference 图片资产如何绑定到 Track1 distribution routes。它不修改当前 champion submission。",
        "",
        "## 摘要",
        "",
        f"- Version: {summary['version']}",
        f"- Routes: {summary['total']}",
        f"- Routes with reference assets: {summary['routes_with_reference_assets']}",
        f"- Reference asset slots: {summary['reference_asset_slots']}",
        f"- Unique reference assets: {summary['unique_reference_assets']}",
        f"- Diversity-limited routes: {summary['diversity_limited_routes']}",
        "",
        "## Sources",
        "",
    ]
    for source, count in summary["reference_asset_sources"].items():
        lines.append(f"- {source}: {count}")
    lines.extend(["", "## Top Reference Reuse", ""])
    for row in summary["top_reference_reuse"]:
        lines.append(f"- {row['count']}: {row['file']}")
    lines.extend(["", "## Routes", ""])
    for route in payload["routes"]:
        assets = route.get("reference_assets", [])
        lines.extend(
            [
                f"### {route.get('sample_id', '')}",
                "",
                f"- Family: {route.get('family_id', '')}",
                f"- Source: {route.get('reference_asset_source', '')}",
                f"- Style key: {route.get('reference_style_key', '')}",
                f"- Selection mode: {route.get('reference_selection_mode', '')}",
                f"- Candidate pool size: {route.get('reference_candidate_pool_size', '')}",
                f"- Diversity limited: {route.get('reference_diversity_limited', '')}",
                f"- Reference assets: {len(assets)}",
            ]
        )
        for asset in assets:
            lines.append(f"  - {asset}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
