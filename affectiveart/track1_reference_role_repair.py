from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from affectiveart.track1_reference_role_gate import (
    REFERENCE_ROLES,
    build_reference_role_gate,
    _normalise_text,
)


def repair_reference_routes(
    routes: Iterable[dict[str, Any]],
    *,
    repair_roles: list[str] | tuple[str, ...] = ("poster_print",),
    max_assets_per_route: int = 4,
) -> dict[str, Any]:
    if max_assets_per_route <= 0:
        raise ValueError("max_assets_per_route must be positive")
    route_rows = [dict(route) for route in routes if isinstance(route, dict)]
    role_names = _valid_role_names(repair_roles)
    donor_pool = _donor_pool(route_rows, role_names)

    repaired_routes: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    for route in route_rows:
        repaired, repairs, unresolved = _repair_route(
            route,
            donor_pool=donor_pool,
            repair_roles=role_names,
            max_assets_per_route=max_assets_per_route,
        )
        repaired_routes.append(repaired)
        repair_rows.extend(repairs)
        unresolved_rows.extend(unresolved)

    return {
        "summary": {
            "total": len(repaired_routes),
            "repair_roles": role_names,
            "repaired_routes": len({row["sample_id"] for row in repair_rows}),
            "added_assets": len(repair_rows),
            "unresolved_routes": len({row["sample_id"] for row in unresolved_rows}),
            "unresolved_missing_roles": dict(Counter(row["missing_role"] for row in unresolved_rows)),
        },
        "routes": repaired_routes,
        "repair_rows": repair_rows,
        "unresolved_rows": unresolved_rows,
    }


def write_reference_role_repair_artifacts(
    result: dict[str, Any],
    *,
    out_dir: str | Path,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "track1_reference_role_repair.json"
    routes_path = out_dir / "track1_distribution_routes_role_repaired.json"
    md_path = out_dir / "track1_reference_role_repair_zh.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    routes_path.write_text(
        json.dumps({"summary": result.get("summary", {}), "routes": result.get("routes", [])}, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_reference_role_repair_md(result), encoding="utf-8")
    return {"json": str(json_path), "routes_json": str(routes_path), "md": str(md_path)}


def render_reference_role_repair_md(result: dict[str, Any]) -> str:
    summary = result.get("summary", {})
    lines = [
        "# Track1 Reference Role Repair",
        "",
        "这份报告记录 reference role gate 之后的自动修复。当前默认只修复 `poster_print`：caption 要求海报/宣传印刷时，reference board 必须包含真正的海报格式 reference。",
        "",
        "## 摘要",
        "",
        f"- Total: {summary.get('total', 0)}",
        f"- Repair roles: {', '.join(str(item) for item in summary.get('repair_roles', []))}",
        f"- Repaired routes: {summary.get('repaired_routes', 0)}",
        f"- Added assets: {summary.get('added_assets', 0)}",
        f"- Unresolved routes: {summary.get('unresolved_routes', 0)}",
        "",
        "## Repaired Rows",
        "",
    ]
    for row in result.get("repair_rows", []):
        lines.append(
            f"- `{row.get('sample_id')}` + `{row.get('role')}` from `{Path(str(row.get('donor_asset'))).name}` "
            f"(donor `{row.get('donor_sample_id')}`)"
        )
    lines.extend(["", "## Unresolved Rows", ""])
    for row in result.get("unresolved_rows", []):
        lines.append(f"- `{row.get('sample_id')}` missing `{row.get('missing_role')}`")
    return "\n".join(lines).rstrip() + "\n"


def _repair_route(
    route: dict[str, Any],
    *,
    donor_pool: dict[str, list[dict[str, str]]],
    repair_roles: list[str],
    max_assets_per_route: int,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    sample_id = str(route.get("sample_id") or "")
    gate_row = build_reference_role_gate([route])["rows"][0]
    missing_roles = [role for role in gate_row.get("missing_required_roles", []) if role in repair_roles]
    if not missing_roles:
        return dict(route), [], []

    repaired = dict(route)
    assets = [str(item) for item in _as_list(route.get("reference_assets"))]
    notes = [str(item) for item in _as_list(route.get("reference_asset_notes"))]
    while len(notes) < len(assets):
        notes.append("")

    repairs: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    for role in missing_roles:
        donor = _select_donor(role, donor_pool.get(role, []), route, assets)
        if donor is None:
            unresolved.append({"sample_id": sample_id, "missing_role": role})
            continue
        assets = [donor["asset"], *assets]
        notes = [
            (
                f"role_repair: added {role} reference; "
                f"donor_sample_id={donor['sample_id']}; donor_note={donor['note']}"
            ),
            *notes,
        ]
        repairs.append(
            {
                "sample_id": sample_id,
                "role": role,
                "donor_sample_id": donor["sample_id"],
                "donor_asset": donor["asset"],
            }
        )

    assets, notes = _dedupe_assets_and_notes(assets, notes)
    assets, notes, trimmed = _trim_preserving_required_role_assets(repaired, assets, notes, max_assets_per_route)
    repaired["reference_assets"] = assets
    repaired["reference_asset_notes"] = notes
    if trimmed:
        repaired["reference_role_trimmed_assets"] = trimmed
    if repairs:
        repaired["reference_role_repaired"] = True
        repaired["reference_role_added"] = [row["role"] for row in repairs]
        repaired["reference_asset_source"] = _repaired_source(route)
    return repaired, repairs, unresolved


def _donor_pool(routes: list[dict[str, Any]], role_names: list[str]) -> dict[str, list[dict[str, str]]]:
    roles = {role.name: role for role in REFERENCE_ROLES if role.name in role_names}
    donors: dict[str, list[dict[str, str]]] = {role: [] for role in role_names}
    seen: set[tuple[str, str]] = set()
    for route in routes:
        sample_id = str(route.get("sample_id") or "")
        family_id = str(route.get("family_id") or "")
        assets = [str(item) for item in _as_list(route.get("reference_assets"))]
        notes = [str(item) for item in _as_list(route.get("reference_asset_notes"))]
        for index, asset in enumerate(assets):
            note = notes[index] if index < len(notes) else ""
            text = _normalise_text(f"{asset} {note}")
            for role_name, role in roles.items():
                if not any(term in text for term in role.evidence_terms):
                    continue
                key = (role_name, asset)
                if key in seen:
                    continue
                seen.add(key)
                donors[role_name].append(
                    {
                        "asset": asset,
                        "note": note,
                        "sample_id": sample_id,
                        "family_id": family_id,
                    }
                )
    return donors


def _select_donor(
    role: str,
    donors: list[dict[str, str]],
    route: dict[str, Any],
    current_assets: list[str],
) -> dict[str, str] | None:
    current = set(current_assets)
    sample_id = str(route.get("sample_id") or "")
    family_id = str(route.get("family_id") or "")
    usable = [donor for donor in donors if donor["asset"] not in current and donor["sample_id"] != sample_id]
    if not usable:
        return None
    usable.sort(key=lambda donor: (donor.get("family_id") != family_id, donor["sample_id"], donor["asset"]))
    return usable[0]


def _valid_role_names(values: list[str] | tuple[str, ...]) -> list[str]:
    valid = {role.name for role in REFERENCE_ROLES}
    output = []
    for value in values:
        role = str(value)
        if role not in valid:
            raise ValueError(f"unknown reference role: {role}")
        if role not in output:
            output.append(role)
    return output or ["poster_print"]


def _repaired_source(route: dict[str, Any]) -> str:
    source = str(route.get("reference_asset_source") or "")
    if not source:
        return "role_repair"
    if "role_repair" in source:
        return source
    return f"{source}+role_repair"


def _dedupe_assets_and_notes(assets: list[str], notes: list[str]) -> tuple[list[str], list[str]]:
    output_assets: list[str] = []
    output_notes: list[str] = []
    seen: set[str] = set()
    for index, asset in enumerate(assets):
        if asset in seen:
            continue
        seen.add(asset)
        output_assets.append(asset)
        output_notes.append(notes[index] if index < len(notes) else "")
    return output_assets, output_notes


def _trim_preserving_required_role_assets(
    route: dict[str, Any],
    assets: list[str],
    notes: list[str],
    max_assets_per_route: int,
) -> tuple[list[str], list[str], list[str]]:
    if len(assets) <= max_assets_per_route:
        return assets, notes, []
    protected = _required_role_protected_indices(route, assets, notes)
    kept = list(range(min(max_assets_per_route, len(assets))))
    for index in sorted(protected):
        if index in kept:
            continue
        replacement_position = _last_unprotected_position(kept, protected)
        if replacement_position is None:
            continue
        kept[replacement_position] = index
    kept = sorted(set(kept))
    trimmed = [asset for index, asset in enumerate(assets) if index not in kept]
    return [assets[index] for index in kept], [notes[index] for index in kept], trimmed


def _required_role_protected_indices(route: dict[str, Any], assets: list[str], notes: list[str]) -> set[int]:
    caption = str(route.get("caption") or "")
    protected: set[int] = set()
    for role in REFERENCE_ROLES:
        if not role.required_if(caption, route):
            continue
        matches = [
            index
            for index, asset in enumerate(assets)
            if _asset_matches_role(asset, notes[index] if index < len(notes) else "", role)
        ]
        if len(matches) == 1:
            protected.add(matches[0])
    return protected


def _asset_matches_role(asset: str, note: str, role: Any) -> bool:
    text = _normalise_text(f"{asset} {note}")
    return any(term in text for term in role.evidence_terms)


def _last_unprotected_position(indices: list[int], protected: set[int]) -> int | None:
    for position in range(len(indices) - 1, -1, -1):
        if indices[position] not in protected:
            return position
    return None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
