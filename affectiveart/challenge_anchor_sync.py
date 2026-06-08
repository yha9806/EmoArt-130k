from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


ANCHOR_REGISTRY_VERSION = "challenge_anchor_registry_v1"


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_json_payload(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_unified_anchor_registry(
    *,
    track1_api_rows: list[dict[str, Any]] | None = None,
    track1_anchor_rows: list[dict[str, Any]] | None = None,
    track2_exact_rows: list[dict[str, Any]] | None = None,
    track2_scoreboard_rows: list[dict[str, Any]] | None = None,
    source_index_rows: list[dict[str, Any]] | None = None,
    public_resource_scan: dict[str, Any] | None = None,
    method_cards: dict[str, Any] | None = None,
) -> dict[str, Any]:
    anchors_by_key: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []

    _merge_track1_anchors(
        anchors_by_key,
        issues,
        api_rows=track1_api_rows or [],
        anchor_rows=track1_anchor_rows or [],
    )
    _merge_track2_anchors(
        anchors_by_key,
        issues,
        exact_rows=track2_exact_rows or [],
        scoreboard_rows=track2_scoreboard_rows or [],
    )
    resource_index = _build_resource_index(source_index_rows or [], method_cards or {})
    author_index = _build_author_index(source_index_rows or [], public_resource_scan or {}, method_cards or {})
    anchors = sorted(anchors_by_key.values(), key=lambda row: (row["track"], row["anchor_type"], row["anchor_key"]))
    exact_track2 = [row for row in anchors if row["track"] == "track2" and row.get("calibration_kind") == "exact_official"]
    estimated_track2 = [row for row in anchors if row["track"] == "track2" and row.get("calibration_kind") == "estimated"]
    return {
        "version": ANCHOR_REGISTRY_VERSION,
        "summary": {
            "anchor_count": len(anchors),
            "track1_anchor_count": sum(1 for row in anchors if row["track"] == "track1"),
            "track2_anchor_count": sum(1 for row in anchors if row["track"] == "track2"),
            "track2_exact_anchor_count": len(exact_track2),
            "track2_estimated_anchor_count": len(estimated_track2),
            "author_count": len(author_index),
            "resource_count": len(resource_index),
        },
        "anchors": anchors,
        "anchors_by_key": {row["anchor_key"]: row for row in anchors},
        "consistency": {
            "issue_count": len(issues),
            "blocking_issue_count": sum(1 for issue in issues if issue.get("blocking")),
            "issues": issues,
        },
        "resource_summary": {
            "source_index_count": len(source_index_rows or []),
            "method_card_count": len((method_cards or {}).get("cards") or []),
            "public_scan_work_count": _public_scan_work_count(public_resource_scan or {}),
        },
        "resource_index": resource_index,
        "author_index": author_index,
    }


def write_anchor_registry_outputs(
    registry: dict[str, Any],
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
    json_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    _write_anchor_csv(registry["anchors"], csv_path)
    md_path.write_text(_render_registry_md(registry), encoding="utf-8")


def _merge_track1_anchors(
    anchors_by_key: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
    *,
    api_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
) -> None:
    for row in api_rows:
        submission_id = _text(row.get("submission_id") or row.get("id"))
        if not submission_id:
            continue
        key = f"track1:{submission_id}"
        _merge_anchor(
            anchors_by_key,
            issues,
            key,
            _track1_anchor_from_row(row, source="track1_api"),
            prefer_existing=False,
        )
    for row in anchor_rows:
        submission_id = _text(row.get("submission_id") or row.get("id"))
        if not submission_id:
            continue
        key = f"track1:{submission_id}"
        _merge_anchor(
            anchors_by_key,
            issues,
            key,
            _track1_anchor_from_row(row, source="track1_anchor_csv"),
            prefer_existing=True,
        )


def _merge_track2_anchors(
    anchors_by_key: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
    *,
    exact_rows: list[dict[str, Any]],
    scoreboard_rows: list[dict[str, Any]],
) -> None:
    for row in exact_rows:
        submission_id = _text(row.get("submission_id") or row.get("id"))
        if not submission_id:
            continue
        key = f"track2:{submission_id}"
        _merge_anchor(
            anchors_by_key,
            issues,
            key,
            _track2_exact_anchor_from_row(row, source="track2_exact_scores"),
            prefer_existing=False,
        )
    for row in scoreboard_rows:
        calibration_kind = _text(row.get("calibration_kind")) or "estimated"
        submission_id = _text(row.get("anchor_submission_id") or row.get("submission_id"))
        candidate_name = _text(row.get("candidate_name"))
        if calibration_kind in {"exact_official", "visible_official"} and submission_id:
            key = f"track2:{submission_id}"
        elif candidate_name:
            key = f"track2_estimated:{candidate_name}"
        else:
            continue
        _merge_anchor(
            anchors_by_key,
            issues,
            key,
            _track2_scoreboard_anchor_from_row(row, key=key),
            prefer_existing=key in anchors_by_key,
        )


def _merge_anchor(
    anchors_by_key: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
    key: str,
    incoming: dict[str, Any],
    *,
    prefer_existing: bool,
) -> None:
    incoming["anchor_key"] = key
    existing = anchors_by_key.get(key)
    if not existing:
        anchors_by_key[key] = incoming
        return
    for metric in (
        "official_overall",
        "official_fid",
        "official_fid_score",
        "official_aas",
        "official_classification",
        "official_description",
    ):
        _check_metric_consistency(issues, key, metric, existing.get(metric), incoming.get(metric))
    merged = dict(existing if prefer_existing else incoming)
    secondary = incoming if prefer_existing else existing
    for field, value in secondary.items():
        if value not in (None, "", []):
            if field not in merged or merged[field] in (None, "", []):
                merged[field] = value
    if (
        existing.get("calibration_kind") == "own_official"
        or incoming.get("calibration_kind") == "own_official"
        or merged.get("local_package")
    ) and merged.get("track") == "track1":
        merged["calibration_kind"] = "own_official"
    merged["source_refs"] = sorted(set(_as_list(existing.get("source_refs")) + _as_list(incoming.get("source_refs"))))
    anchors_by_key[key] = merged


def _track1_anchor_from_row(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    fid_score = _float_or_none(row.get("official_fid_score") or row.get("fid_score"))
    aas = _float_or_none(row.get("official_aas") or row.get("aas"))
    official_overall = _float_or_none(row.get("official_overall") or row.get("overall"))
    if official_overall is None and fid_score is not None and aas is not None:
        official_overall = round(0.5 * (fid_score + aas), 10)
    return {
        "track": "track1",
        "anchor_type": "official_score",
        "calibration_kind": _track1_calibration_kind(row),
        "participant": _text(row.get("participant") or row.get("owner")),
        "submission_id": _text(row.get("submission_id") or row.get("id")),
        "file_name": _text(row.get("file_name") or row.get("filename")),
        "local_package": _text(row.get("local_package") or row.get("package")),
        "local_fid_like": _float_or_none(row.get("local_fid_like")),
        "official_overall": official_overall,
        "official_fid": _float_or_none(row.get("official_fid") or row.get("fid")),
        "official_fid_score": fid_score,
        "official_aas": aas,
        "content_alignment": _float_or_none(row.get("content_alignment")),
        "style_alignment": _float_or_none(row.get("style_alignment")),
        "attribute_alignment": _float_or_none(row.get("attribute_alignment")),
        "notes": _text(row.get("notes")),
        "source_refs": [source],
    }


def _track2_exact_anchor_from_row(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    classification = _float_or_none(row.get("official_classification") or row.get("classification_expected"))
    description = _float_or_none(row.get("official_description") or row.get("description_expected"))
    official_overall = _float_or_none(row.get("official_overall") or row.get("overall_expected"))
    if official_overall is None and classification is not None and description is not None:
        official_overall = round(0.5 * (classification + description), 10)
    return {
        "track": "track2",
        "anchor_type": "official_score",
        "calibration_kind": "exact_official",
        "participant": _text(row.get("participant") or row.get("owner")),
        "submission_id": _text(row.get("submission_id") or row.get("id")),
        "file_name": _text(row.get("file_name") or row.get("filename")),
        "official_overall": official_overall,
        "official_classification": classification,
        "official_description": description,
        "emotion_accuracy": _float_or_none(row.get("emotion_accuracy")),
        "emotion_macro_f1": _float_or_none(row.get("emotion_macro_f1")),
        "valence_accuracy": _float_or_none(row.get("valence_accuracy")),
        "valence_macro_f1": _float_or_none(row.get("valence_macro_f1")),
        "arousal_accuracy": _float_or_none(row.get("arousal_accuracy")),
        "arousal_macro_f1": _float_or_none(row.get("arousal_macro_f1")),
        "notes": _text(row.get("notes")),
        "source_refs": [source],
    }


def _track2_scoreboard_anchor_from_row(row: dict[str, Any], *, key: str) -> dict[str, Any]:
    calibration_kind = _text(row.get("calibration_kind")) or "estimated"
    return {
        "track": "track2",
        "anchor_type": "estimated_candidate" if calibration_kind == "estimated" else "official_score",
        "calibration_kind": calibration_kind,
        "candidate_name": _text(row.get("candidate_name")),
        "submission_id": _text(row.get("anchor_submission_id") or row.get("submission_id")) if key.startswith("track2:") else "",
        "anchor_submission_id": _text(row.get("anchor_submission_id")),
        "json_path": _text(row.get("json_path")),
        "official_overall": _float_or_none(row.get("official_overall") or row.get("overall_expected")),
        "official_classification": _float_or_none(row.get("official_classification") or row.get("classification_expected")),
        "official_description": _float_or_none(row.get("official_description") or row.get("description_expected")),
        "official_visible": _float_or_none(row.get("overall_visible")),
        "label_changes_vs_anchor": _int_or_none(row.get("label_changes_vs_anchor")),
        "text_changed_rows_vs_anchor": _int_or_none(row.get("text_changed_rows_vs_anchor")),
        "transition_counts": _text(row.get("transition_counts")),
        "warnings": _text(row.get("warnings")),
        "notes": _text(row.get("notes")),
        "source_refs": ["track2_official_anchor_scoreboard"],
    }


def _build_resource_index(source_rows: list[dict[str, Any]], method_cards: dict[str, Any]) -> list[dict[str, Any]]:
    resources = []
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        resources.append(
            {
                "source_type": _text(row.get("source_type")),
                "name": _text(row.get("name")),
                "url": _text(row.get("url")),
                "verified_public_fact": _text(row.get("verified_public_fact")),
                "track_use": _text(row.get("track2_use") or row.get("track_use")),
                "status": _text(row.get("status")),
            }
        )
    for card in method_cards.get("cards") or []:
        resources.append(
            {
                "source_type": "method_card",
                "name": _text(card.get("id")),
                "url": ";".join(_as_list(card.get("public_sources"))),
                "verified_public_fact": _text(card.get("core_method")),
                "track_use": _text(card.get("engineering_module")),
                "status": _text(card.get("priority")),
            }
        )
    return [row for row in resources if row.get("name") or row.get("url")]


def _build_author_index(
    source_rows: list[dict[str, Any]],
    public_scan: dict[str, Any],
    method_cards: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    authors: dict[str, dict[str, Any]] = {}
    for work in _iter_public_scan_works(public_scan):
        title = _text(work.get("title"))
        year = _int_or_none(work.get("publication_year"))
        for authorship in work.get("authorships") or []:
            name = _canonical_author_name(authorship.get("author"))
            if not name:
                continue
            author = _author_record(authors, name)
            _append_unique(author["institutions"], _text(authorship.get("institution")))
            _append_unique(author["works"], f"{title} ({year})" if year else title)
    for row in source_rows:
        name = _text(row.get("name"))
        if not name:
            continue
        source_type = _text(row.get("source_type")).lower()
        if source_type not in {"profile", "lab"}:
            continue
        for author_name in _author_names_from_source_name(name):
            author = _author_record(authors, author_name)
            _append_unique(author["source_names"], name)
            _append_unique(author["urls"], _text(row.get("url")))
    for card in method_cards.get("cards") or []:
        card_id = _text(card.get("id"))
        for author_name in _author_names_from_source_group(card.get("source_group")):
            author = _author_record(authors, author_name)
            _append_unique(author["method_card_ids"], card_id)
            for url in _as_list(card.get("public_sources")):
                _append_unique(author["urls"], _text(url))
    return {name: _sort_author_record(record) for name, record in sorted(authors.items())}


def _write_anchor_csv(anchors: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "anchor_key",
        "track",
        "anchor_type",
        "calibration_kind",
        "participant",
        "candidate_name",
        "submission_id",
        "anchor_submission_id",
        "file_name",
        "local_package",
        "local_fid_like",
        "official_overall",
        "official_fid",
        "official_fid_score",
        "official_aas",
        "official_classification",
        "official_description",
        "json_path",
        "warnings",
        "source_refs",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in anchors:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _render_registry_md(registry: dict[str, Any]) -> str:
    summary = registry["summary"]
    consistency = registry["consistency"]
    lines = [
        "# Unified Challenge Anchor Registry",
        "",
        "这是 Track1 / Track2 官方锚点、候选锚点和作者/方法资源的统一同步表。",
        "",
        "## Summary",
        "",
        f"- Anchors: `{summary['anchor_count']}`",
        f"- Track1 anchors: `{summary['track1_anchor_count']}`",
        f"- Track2 anchors: `{summary['track2_anchor_count']}`",
        f"- Track2 exact anchors: `{summary['track2_exact_anchor_count']}`",
        f"- Track2 estimated anchors: `{summary['track2_estimated_anchor_count']}`",
        f"- Authors/resources: `{summary['author_count']}` / `{summary['resource_count']}`",
        f"- Consistency issues: `{consistency['issue_count']}`; blocking `{consistency['blocking_issue_count']}`",
        "",
        "## Anchors",
        "",
        "| key | track | kind | overall | components | local/package |",
        "|---|---|---|---:|---|---|",
    ]
    for row in registry["anchors"][:40]:
        components = _component_summary(row)
        local = row.get("local_package") or row.get("candidate_name") or row.get("json_path") or ""
        lines.append(
            f"| `{row['anchor_key']}` | `{row['track']}` | `{row.get('calibration_kind', '')}` | "
            f"{_csv_value(row.get('official_overall'))} | {components} | `{local}` |"
        )
    if len(registry["anchors"]) > 40:
        lines.append(f"| ... | ... | ... | ... | omitted {len(registry['anchors']) - 40} rows | ... |")
    lines += [
        "",
        "## Author / Method Index",
        "",
        "| author | institutions | method cards | works/resources |",
        "|---|---|---|---|",
    ]
    for name, author in _author_display_rows(registry["author_index"])[:30]:
        lines.append(
            f"| `{name}` | {_join(author.get('institutions'))} | "
            f"{_join(author.get('method_card_ids'))} | {_join((author.get('works') or [])[:2] + (author.get('source_names') or [])[:2])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Track1 和 Track2 scorer 后续应优先读取这份 registry，避免重复维护不一致锚点。",
        "- `exact_official` 才能用于数值校准；`estimated` 只能用于候选排序和风险分析。",
        "- 作者/方法索引用来把 EmoArt/FAB-G/EmoVIT/PromptFix/AICA 等公开线索同步到两个 track 的策略层。",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _track1_calibration_kind(row: dict[str, Any]) -> str:
    if _text(row.get("source_type")) == "own_submission":
        return "own_official"
    if _text(row.get("source_type")).startswith("public"):
        return "public_leaderboard"
    if _text(row.get("local_package")):
        return "own_official"
    return "public_or_api_official"


def _author_display_rows(author_index: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    priority_names = {
        "Cheng Zhang": 0,
        "Hongxia Xie": 1,
        "Wen-Huang Cheng": 2,
        "Hong-Han Shuai": 3,
        "Jianlong Fu": 4,
        "Sicheng Zhao": 5,
        "Bin Wen": 6,
        "Songhan Zuo": 7,
        "Ruoxuan Zhang": 8,
    }
    return sorted(
        author_index.items(),
        key=lambda item: (
            priority_names.get(item[0], 100),
            -len(item[1].get("method_card_ids") or []),
            -len(item[1].get("source_names") or []),
            -len(item[1].get("works") or []),
            item[0],
        ),
    )


def _check_metric_consistency(
    issues: list[dict[str, Any]],
    anchor_key: str,
    metric: str,
    existing: Any,
    incoming: Any,
) -> None:
    left = _float_or_none(existing)
    right = _float_or_none(incoming)
    if left is None or right is None:
        return
    if abs(left - right) <= 0.005:
        return
    issues.append(
        {
            "anchor_key": anchor_key,
            "metric": metric,
            "existing": left,
            "incoming": right,
            "blocking": True,
            "reason": "metric mismatch between anchor sources",
        }
    )


def _author_record(authors: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    if name not in authors:
        authors[name] = {
            "name": name,
            "institutions": [],
            "works": [],
            "source_names": [],
            "method_card_ids": [],
            "urls": [],
        }
    return authors[name]


def _sort_author_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: sorted(value) if isinstance(value, list) else value
        for key, value in record.items()
    }


def _iter_public_scan_works(public_scan: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for rows in (public_scan.get("openalex_works") or {}).values():
        for work in rows or []:
            if isinstance(work, dict):
                yield work


def _public_scan_work_count(public_scan: dict[str, Any]) -> int:
    return sum(1 for _ in _iter_public_scan_works(public_scan))


def _author_names_from_source_name(name: str) -> list[str]:
    cleaned = re.sub(r"\s*/\s*(AVC Lab|BASIC Lab|MSRA publications|research page|homepage)$", "", name, flags=re.I)
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0]
    if "Lab" in cleaned or "Challenge" in cleaned or "official" in cleaned.lower():
        return []
    return [_canonical_author_name(cleaned)] if _canonical_author_name(cleaned) else []


def _author_names_from_source_group(source_group: Any) -> list[str]:
    text = _text(source_group)
    if not text:
        return []
    return [
        name
        for name in (_canonical_author_name(part) for part in re.split(r"\s*/\s*", text))
        if name
    ]


def _canonical_author_name(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    aliases = {
        "xie, Hongxia": "Hongxia Xie",
        "Cheng, Wen-huang": "Wen-Huang Cheng",
        "Wen-Huang Cheng": "Wen-Huang Cheng",
        "Wen Huang Cheng": "Wen-Huang Cheng",
    }
    return aliases.get(text, text)


def _component_summary(row: dict[str, Any]) -> str:
    if row.get("track") == "track1":
        return f"FID={_csv_value(row.get('official_fid'))}; FIDScore={_csv_value(row.get('official_fid_score'))}; AAS={_csv_value(row.get('official_aas'))}"
    return f"Cls={_csv_value(row.get('official_classification'))}; Desc={_csv_value(row.get('official_description'))}"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    if isinstance(value, str) and ";" in value:
        return [item.strip() for item in value.split(";") if item.strip()]
    text = _text(value)
    return [text] if text else []


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _join(values: Any) -> str:
    items = _as_list(values)
    return ", ".join(f"`{item}`" for item in items) if items else ""


def _csv_value(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(_text(item) for item in value if _text(item))
    if isinstance(value, float):
        return f"{value:.10g}"
    if value is None:
        return ""
    return str(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    return int(number) if number is not None else None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
