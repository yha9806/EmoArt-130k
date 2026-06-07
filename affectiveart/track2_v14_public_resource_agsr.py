from __future__ import annotations

import csv
import html
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_audit import compute_track2_distribution, strict_track2_label_issues
from affectiveart.track2_fused_shadow_evaluator import write_fused_shadow_evaluator_outputs
from affectiveart.track2_local_shadow_evaluator import write_shadow_evaluator_outputs
from affectiveart.track2_repair import _repair_row


FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}
TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
ATTRIBUTE_FIELDS = ("brushstroke", "composition", "color", "line", "light")
DEFAULT_SALIENT_ATTRIBUTES = ("composition", "color")
DEFAULT_OUT_DIR = Path("experiments/track2_v14_public_resource_agsr_20260607")
DEFAULT_HISTORICAL_CANDIDATES = (
    ("official_779605_moe_v2_anchor", "submissions/track2_submission_moe_v2_accept5_candidate.json"),
    ("official_781601_v3_mid", "submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json"),
    ("official_782683_v12_stable_probe", "submissions/track2_submission_v12_stable_probe_candidate.json"),
)
OFFICIAL_SCORE_ORDER = {
    "official_779605_moe_v2_anchor": 0.836408,
    "official_781601_v3_mid": 0.834027,
    "official_782683_v12_stable_probe": 0.84,
}


def classify_public_evidence(row: dict[str, Any]) -> str:
    clip = _safe_float(row.get("clip_cosine") or row.get("top1_clip"))
    dhash = _safe_int(row.get("dhash_distance"), default=99)
    bucket = str(row.get("duplicate_bucket") or row.get("group") or "")
    majority = _safe_int(row.get("topk_majority_count") or row.get("majority_count"), default=0)
    if clip >= 0.99 or (clip >= 0.985 and dhash <= 2) or bucket == "visual_duplicate_likely":
        return "exact_same_work"
    if clip >= 0.96 and (bucket in {"same_work_or_series_review", "near_duplicate"} or majority >= 7):
        return "near_same_work"
    if clip >= 0.93 and (
        bucket in {"same_work_or_series_review", "same_series_high_clip", "style_neighbor_review"} or majority >= 5
    ):
        return "same_series_style"
    return "style_prior_only"


def build_public_evidence_matrix(
    current_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_by_id = {str(row.get("sample_id", "")): row for row in current_rows if row.get("sample_id")}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for public in public_rows:
        sample_id = str(public.get("sample_id", ""))
        if sample_id in current_by_id:
            grouped[sample_id].append(public)

    matrix: list[dict[str, Any]] = []
    for sample_id in sorted(current_by_id):
        current = current_by_id[sample_id]
        candidates = grouped.get(sample_id, [])
        if not candidates:
            continue
        enriched = [_enrich_public_row(candidate, candidates=candidates) for candidate in candidates]
        ranked = sorted(enriched, key=_public_evidence_sort_key, reverse=True)
        best = ranked[0]
        target = _canonical_emotion(best.get("public_emotion") or best.get("nearest_emotion") or best.get("emotion"))
        if not target:
            continue
        current_emotion = _canonical_emotion(current.get("emotion"))
        salient = _normalize_salient_attributes(best)
        level = classify_public_evidence(best)
        matrix.append(
            {
                "sample_id": sample_id,
                "current_emotion": current_emotion,
                "target_emotion": target,
                "current_valence": str(current.get("emotional_valence", "")),
                "target_valence": str(best.get("public_valence") or best.get("valence") or _expected_valence(target)),
                "current_arousal": str(current.get("emotional_arousal_level", "")),
                "target_arousal": str(best.get("public_arousal") or best.get("arousal") or _expected_arousal(target)),
                "evidence_level": level,
                "clip_cosine": round(_safe_float(best.get("clip_cosine") or best.get("top1_clip")), 6),
                "dhash_distance": _safe_int(best.get("dhash_distance"), default=99),
                "topk_majority_count": _safe_int(best.get("topk_majority_count") or best.get("majority_count"), default=0),
                "duplicate_bucket": str(best.get("duplicate_bucket") or best.get("group") or ""),
                "public_member": str(best.get("public_member") or best.get("nearest_member") or best.get("member") or ""),
                "public_request_id": str(best.get("public_request_id") or best.get("request_id") or ""),
                "public_description": str(
                    best.get("public_description")
                    or best.get("public_caption")
                    or best.get("description")
                    or best.get("caption")
                    or ""
                ),
                "salient_attributes": ",".join(salient),
                "sources": _source_name(best),
                "same_quadrant": _same_quadrant(current_emotion, target),
                "transition": f"{current_emotion}->{target}",
                "neighbor_count": len(candidates),
            }
        )
    return matrix


def select_v14_label_changes(evidence_rows: list[dict[str, Any]], *, profile: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in sorted(evidence_rows, key=lambda item: str(item.get("sample_id", ""))):
        if row.get("current_emotion") == row.get("target_emotion"):
            continue
        level = str(row.get("evidence_level", ""))
        same_quadrant = bool(row.get("same_quadrant"))
        if profile == "description_max_safe":
            if level == "exact_same_work" and same_quadrant:
                selected.append(row)
            continue
        if profile != "public_resource_agsr_max":
            raise ValueError(f"unknown v14 profile: {profile}")
        if level == "exact_same_work" and same_quadrant:
            selected.append(row)
        elif level == "near_same_work" and same_quadrant:
            selected.append(row)
    return selected


def apply_v14_candidate_rows(
    current_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    *,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = select_v14_label_changes(evidence_rows, profile=profile)
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    candidate_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    text_changes = 0
    for current in current_rows:
        sample_id = str(current.get("sample_id", ""))
        row = dict(current)
        before = _label_triplet(row)
        evidence = selected_by_id.get(sample_id)
        if evidence:
            row["emotion"] = str(evidence["target_emotion"])
            if evidence.get("target_valence"):
                row["emotional_valence"] = str(evidence["target_valence"])
            if evidence.get("target_arousal"):
                row["emotional_arousal_level"] = str(evidence["target_arousal"])
            _repair_row(row)
            fields = _apply_agsr_text(row, evidence)
            text_changes += len(fields)
            changes.append(
                {
                    "sample_id": sample_id,
                    "from": before,
                    "to": _label_triplet(row),
                    "text_fields": fields,
                    "evidence_level": evidence.get("evidence_level", ""),
                    "transition": f"{before['emotion']}->{row.get('emotion', '')}",
                }
            )
        elif profile in {"public_resource_agsr_max", "description_max_safe"}:
            fields = _apply_safe_text_cleanup(row)
            text_changes += len(fields)
            if fields:
                changes.append(
                    {
                        "sample_id": sample_id,
                        "from": before,
                        "to": _label_triplet(row),
                        "text_fields": fields,
                        "evidence_level": "text_cleanup",
                        "transition": f"{before['emotion']}->{row.get('emotion', '')}",
                    }
                )
        candidate_rows.append({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})
    report = {
        "profile": profile,
        "row_count": len(candidate_rows),
        "classification_label_changes": sum(1 for item in changes if item["from"] != item["to"]),
        "changed_rows": len(changes),
        "changed_fields": text_changes,
        "changes": changes,
        "distribution": compute_track2_distribution(candidate_rows),
        "label_consistency_issue_count": sum(len(strict_track2_label_issues(row)) for row in candidate_rows),
        "label_transition_counts": dict(
            Counter(item["transition"] for item in changes if item["from"] != item["to"])
        ),
    }
    return candidate_rows, report


def build_v14_outputs(
    *,
    current_json: str | Path,
    public_audit_jsons: list[str | Path],
    out_dir: str | Path,
    submission_dir: str | Path,
    image_dir: str | Path | None = None,
    run_shadow: bool = True,
) -> dict[str, Any]:
    current_json = Path(current_json)
    out_dir = Path(out_dir)
    submission_dir = Path(submission_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    current_rows = _load_json_rows(current_json)
    public_rows = []
    for path in public_audit_jsons:
        public_rows.extend(_load_public_rows(Path(path)))
    matrix = build_public_evidence_matrix(current_rows, public_rows)
    _write_json(out_dir / "track2_public_evidence_matrix.json", matrix)
    _write_csv(out_dir / "track2_public_evidence_matrix.csv", matrix)
    _write_jsonl(out_dir / "public_resource_registry.jsonl", public_rows)

    candidates: dict[str, Any] = {}
    for profile in ("public_resource_agsr_max", "description_max_safe"):
        rows, candidate_report = apply_v14_candidate_rows(current_rows, matrix, profile=profile)
        out_json = submission_dir / f"track2_submission_v14_{profile}_candidate.json"
        out_zip = submission_dir / f"track2_submission_v14_{profile}_candidate.zip"
        _assert_side_path(out_json)
        _assert_side_path(out_zip)
        _write_json(out_json, rows)
        _write_zip(out_zip, out_json)
        candidates[profile] = {**candidate_report, "json": str(out_json), "zip": str(out_zip)}

    report = {
        "method": "track2_v14_public_resource_agsr_v1",
        "current_json": str(current_json),
        "public_audit_jsons": [str(path) for path in public_audit_jsons],
        "image_dir": str(image_dir or ""),
        "row_count": len(current_rows),
        "public_row_count": len(public_rows),
        "evidence_level_counts": dict(Counter(row["evidence_level"] for row in matrix)),
        "transition_counts": dict(Counter(row["transition"] for row in matrix if row["current_emotion"] != row["target_emotion"])),
        "candidates": candidates,
        "formal_submission_overwritten": False,
    }
    _write_json(out_dir / "v14_run_report.json", report)
    (out_dir / "v14_agsr_rewrite_report.md").write_text(_render_rewrite_report(report), encoding="utf-8")
    (out_dir / "v14_candidate_score_proxy.md").write_text(_render_score_proxy(report), encoding="utf-8")
    html_dir = out_dir / "html_review"
    html_dir.mkdir(parents=True, exist_ok=True)
    (html_dir / "track2_v14_public_resource_agsr_review.html").write_text(
        _render_html_review(matrix, candidates),
        encoding="utf-8",
    )
    if run_shadow:
        _run_shadow_reports(current_json=current_json, out_dir=out_dir, candidates=candidates)
    return report


def build_emotion_boundary_report(*, evidence_matrix_json: str | Path | None = None, out_dir: str | Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    out_dir = Path(out_dir)
    matrix_path = Path(evidence_matrix_json) if evidence_matrix_json else out_dir / "track2_public_evidence_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.exists() else []
    accepted = [
        row for row in matrix
        if row.get("current_emotion") != row.get("target_emotion")
        and row.get("evidence_level") in {"exact_same_work", "near_same_work"}
        and row.get("same_quadrant")
    ]
    blocked = [
        row for row in matrix
        if row.get("current_emotion") != row.get("target_emotion")
        and row.get("evidence_level") not in {"exact_same_work", "near_same_work"}
    ]
    report = {
        "method": "track2_v14_emotion_boundary_report_v1",
        "matrix_path": str(matrix_path),
        "accepted_transition_counts": dict(Counter(str(row.get("transition", "")) for row in accepted)),
        "blocked_transition_counts": dict(Counter(str(row.get("transition", "")) for row in blocked)),
        "evidence_level_counts": dict(Counter(str(row.get("evidence_level", "")) for row in matrix)),
        "accepted_count": len(accepted),
        "blocked_count": len(blocked),
        "calm_content_glad_accepts": [
            row for row in accepted
            if str(row.get("current_emotion")) in {"calm", "content", "glad"}
            or str(row.get("target_emotion")) in {"calm", "content", "glad"}
        ],
    }
    _write_json(out_dir / "v14_emotion_boundary_report.json", report)
    (out_dir / "v14_emotion_boundary_report.md").write_text(_render_boundary_report(report), encoding="utf-8")
    return report


def build_threshold_ablation_report(*, evidence_matrix_json: str | Path | None = None, out_dir: str | Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    out_dir = Path(out_dir)
    matrix_path = Path(evidence_matrix_json) if evidence_matrix_json else out_dir / "track2_public_evidence_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.exists() else []
    rows = []
    for threshold in (0.93, 0.95, 0.96, 0.98, 0.99):
        candidates = [row for row in matrix if _safe_float(row.get("clip_cosine")) >= threshold]
        changed = [row for row in candidates if row.get("current_emotion") != row.get("target_emotion")]
        rows.append(
            {
                "threshold": threshold,
                "candidate_count": len(candidates),
                "changed_count": len(changed),
                "exact_or_near_count": sum(1 for row in candidates if row.get("evidence_level") in {"exact_same_work", "near_same_work"}),
                "style_only_count": sum(1 for row in candidates if row.get("evidence_level") not in {"exact_same_work", "near_same_work"}),
            }
        )
    report = {"method": "track2_v14_threshold_ablation_v1", "matrix_path": str(matrix_path), "rows": rows}
    _write_json(out_dir / "v14_threshold_ablation_report.json", report)
    (out_dir / "v14_threshold_ablation_report.md").write_text(_render_threshold_report(report), encoding="utf-8")
    return report


def build_historical_correlation_report(*, out_dir: str | Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    out_dir = Path(out_dir)
    fused_path = out_dir / "fused_shadow_compare" / "fused_shadow_score_report.json"
    if not fused_path.exists():
        fused_path = out_dir / "fused_shadow_eval" / "fused_shadow_score_report.json"
    ranking = json.loads(fused_path.read_text(encoding="utf-8")).get("ranking", []) if fused_path.exists() else []
    by_name = {str(item.get("candidate_name", "")): item for item in ranking}
    anchor = by_name.get("official_779605_moe_v2_anchor")
    failed = by_name.get("official_781601_v3_mid")
    alignment = "unknown"
    if anchor and failed:
        alignment = "pass" if float(anchor.get("overall_lower", 0.0)) >= float(failed.get("overall_lower", 0.0)) else "fail"
    report = {
        "method": "track2_v14_historical_correlation_v1",
        "fused_report": str(fused_path),
        "official_best": "779605",
        "shadow_must_not_prefer": "781601",
        "historical_alignment": alignment,
        "spearman_proxy": _historical_spearman_proxy(ranking),
        "ranking": [
            {
                "candidate_name": item.get("candidate_name", ""),
                "overall_lower": item.get("overall_lower", 0.0),
                "overall_expected": item.get("overall_expected", 0.0),
            }
            for item in ranking
            if str(item.get("candidate_name", "")).startswith("official_")
        ],
    }
    _write_json(out_dir / "v14_historical_correlation_report.json", report)
    (out_dir / "v14_historical_correlation_report.md").write_text(_render_historical_report(report), encoding="utf-8")
    return report


def _run_shadow_reports(*, current_json: Path, out_dir: Path, candidates: dict[str, Any]) -> None:
    shadow_candidates = [
        {"name": "v14_public_resource_agsr_max", "json": candidates["public_resource_agsr_max"]["json"]},
        {"name": "v14_description_max_safe", "json": candidates["description_max_safe"]["json"]},
        *[
            {"name": name, "json": path}
            for name, path in DEFAULT_HISTORICAL_CANDIDATES
        ],
    ]
    existing = [item for item in shadow_candidates if Path(item["json"]).exists()]
    write_shadow_evaluator_outputs(baseline_json=current_json, candidates=existing, out_dir=out_dir / "shadow_eval")
    write_fused_shadow_evaluator_outputs(baseline_json=current_json, candidates=existing, out_dir=out_dir / "fused_shadow_eval")


def _enrich_public_row(row: dict[str, Any], *, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(row)
    target = _canonical_emotion(enriched.get("public_emotion") or enriched.get("nearest_emotion") or enriched.get("emotion"))
    if target:
        majority = sum(
            1 for item in candidates
            if _canonical_emotion(item.get("public_emotion") or item.get("nearest_emotion") or item.get("emotion")) == target
        )
        enriched.setdefault("topk_majority_count", majority)
    if "member" in enriched and "public_member" not in enriched:
        enriched["public_member"] = enriched["member"]
    if "request_id" in enriched and "public_request_id" not in enriched:
        enriched["public_request_id"] = enriched["request_id"]
    if "emotion" in enriched and "public_emotion" not in enriched:
        enriched["public_emotion"] = enriched["emotion"]
    if "caption" in enriched and "public_description" not in enriched:
        enriched["public_description"] = enriched["caption"]
    return enriched


def _public_evidence_sort_key(row: dict[str, Any]) -> tuple[int, float, int, int]:
    level_weight = {
        "exact_same_work": 4,
        "near_same_work": 3,
        "same_series_style": 2,
        "style_prior_only": 1,
    }[classify_public_evidence(row)]
    return (
        level_weight,
        _safe_float(row.get("clip_cosine") or row.get("top1_clip")),
        _safe_int(row.get("topk_majority_count") or row.get("majority_count"), default=0),
        -_safe_int(row.get("dhash_distance"), default=99),
    )


def _apply_agsr_text(row: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    salient = [item for item in str(evidence.get("salient_attributes", "")).split(",") if item]
    if not salient:
        salient = list(DEFAULT_SALIENT_ATTRIBUTES)
    changed: list[str] = []
    public_description = str(evidence.get("public_description", "")).strip()
    emotion = str(row.get("emotion", "")).lower()
    caption = public_description or str(row.get("overall_caption", "")).strip()
    clause = _salience_clause(salient, emotion)
    if clause.lower() not in caption.lower():
        row["overall_caption"] = _join_sentence(caption, clause)
        changed.append("overall_caption")
    elif public_description and public_description != row.get("overall_caption"):
        row["overall_caption"] = public_description
        changed.append("overall_caption")
    for field in ATTRIBUTE_FIELDS:
        original = str(row.get(field, ""))
        if field in salient:
            updated = _attribute_sentence(field, emotion)
        else:
            updated = _remove_unsupported_emotion_terms(original, emotion)
        if updated != original:
            row[field] = updated
            changed.append(field)
    return changed


def _apply_safe_text_cleanup(row: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    emotion = str(row.get("emotion", "")).lower()
    for field in TEXT_FIELDS:
        original = str(row.get(field, ""))
        updated = _remove_unsupported_emotion_terms(original, emotion)
        if updated != original:
            row[field] = updated
            changed.append(field)
    return changed


def _source_name(row: dict[str, Any]) -> str:
    source = str(row.get("resource") or row.get("source") or "")
    if source:
        return source
    if row.get("public_member") or row.get("nearest_member") or row.get("member"):
        return "public_duplicate"
    return "public_resource"


def _canonical_emotion(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_salient_attributes(row: dict[str, Any]) -> list[str]:
    value = row.get("salient_attributes") or row.get("tags") or row.get("salient_attribute_tags")
    if isinstance(value, list):
        items = [str(item).strip().lower() for item in value]
    else:
        text = str(value or "")
        items = [part.strip().lower() for part in text.replace(";", ",").split(",")]
    filtered = [item for item in ATTRIBUTE_FIELDS if item in set(items)]
    return filtered or list(DEFAULT_SALIENT_ATTRIBUTES)


def _same_quadrant(current: str, target: str) -> bool:
    return _expected_valence(current) == _expected_valence(target) and _expected_arousal(current) == _expected_arousal(target)


def _expected_valence(emotion: str) -> str:
    return "Negative" if emotion in {"alarmed", "annoyed", "bored", "frustrated", "sad", "tired"} else "Positive"


def _expected_arousal(emotion: str) -> str:
    return "High" if emotion in {"alarmed", "annoyed", "aroused", "excited", "frustrated", "happy"} else "Low"


def _label_triplet(row: dict[str, Any]) -> dict[str, str]:
    return {
        "emotion": str(row.get("emotion", "")),
        "emotional_valence": str(row.get("emotional_valence", "")),
        "emotional_arousal_level": str(row.get("emotional_arousal_level", "")),
    }


def _salience_clause(salient: list[str], emotion: str) -> str:
    if "composition" in salient and "color" in salient:
        return f"Balanced composition and restrained color support a {emotion} emotional atmosphere."
    if "composition" in salient:
        return f"The composition organizes the image into a {emotion} emotional atmosphere."
    if "color" in salient:
        return f"The color relationships support a {emotion} emotional atmosphere."
    if "line" in salient:
        return f"The line rhythm supports a {emotion} emotional atmosphere."
    if "light" in salient:
        return f"The light shapes a {emotion} emotional atmosphere."
    return f"The selected visual attributes support a {emotion} emotional atmosphere."


def _attribute_sentence(field: str, emotion: str) -> str:
    mapping = {
        "brushstroke": f"The brushwork is controlled enough to sustain the {emotion} mood without distracting texture.",
        "composition": f"Balanced composition guides attention and stabilizes the {emotion} mood.",
        "color": f"Restrained color relationships reinforce the {emotion} atmosphere.",
        "line": f"Measured line rhythm supports the {emotion} expression.",
        "light": f"Soft light clarifies forms and preserves the {emotion} tone.",
    }
    return mapping[field]


def _join_sentence(base: str, clause: str) -> str:
    base = base.strip()
    if not base:
        return clause
    if base.endswith("."):
        return f"{base} {clause}"
    return f"{base}. {clause}"


def _remove_unsupported_emotion_terms(text: str, emotion: str) -> str:
    replacements = {
        "weary": "quiet",
        "fatigued": "quiet",
        "tired": "quiet",
        "irritated": "composed",
        "annoyed": "composed",
        "frustrated": "focused",
        "alarmed": "alert",
    }
    updated = str(text)
    if emotion in {"calm", "content", "glad"}:
        for old, new in replacements.items():
            updated = updated.replace(old, new).replace(old.capitalize(), new.capitalize())
    return updated


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _load_public_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("entries", "rows", "selected", "neighbors", "audits"):
            value = payload.get(key)
            if isinstance(value, list):
                return _flatten_public_rows(value)
    return []


def _flatten_public_rows(rows: list[Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("neighbors"), list):
            sample_id = str(item.get("sample_id", ""))
            for neighbor in item["neighbors"]:
                if isinstance(neighbor, dict):
                    merged = dict(neighbor)
                    merged["sample_id"] = sample_id
                    if "emotion" in merged:
                        merged["public_emotion"] = merged["emotion"]
                    if "member" in merged:
                        merged["public_member"] = merged["member"]
                    if "request_id" in merged:
                        merged["public_request_id"] = merged["request_id"]
                    flattened.append(merged)
        else:
            flattened.append(dict(item))
    return flattened


def _assert_side_path(path: Path) -> None:
    if path.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError(f"refusing to write formal submission path: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_zip(out_zip: Path, out_json: Path) -> None:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "submission.json")


def _render_rewrite_report(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v14 AGSR Rewrite Report",
        "",
        f"- Method: `{report['method']}`",
        f"- Rows: {report['row_count']}",
        f"- Public evidence rows: {report['public_row_count']}",
        f"- Evidence levels: {report['evidence_level_counts']}",
        "",
        "## Candidates",
        "",
    ]
    for name, candidate in report["candidates"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- JSON: `{candidate['json']}`",
                f"- ZIP: `{candidate['zip']}`",
                f"- Changed rows: {candidate['changed_rows']}",
                f"- Classification label changes: {candidate['classification_label_changes']}",
                f"- Label transition counts: {candidate['label_transition_counts']}",
                f"- Label consistency issues: {candidate['label_consistency_issue_count']}",
                "",
            ]
        )
    return "\n".join(lines)


def _render_score_proxy(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v14 Candidate Score Proxy",
        "",
        "This report is a local proxy. It is not the official Codabench score.",
        "",
        "| candidate | label changes | changed rows | top emotion | missing emotions | consistency issues |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
    for name, candidate in report["candidates"].items():
        dist = candidate["distribution"]
        lines.append(
            f"| {name} | {candidate['classification_label_changes']} | {candidate['changed_rows']} | "
            f"{dist.get('top_emotion', '')} {float(dist.get('top_emotion_share', 0.0)):.1%} | "
            f"{', '.join(dist.get('missing_emotions', [])) or 'none'} | {candidate['label_consistency_issue_count']} |"
        )
    return "\n".join(lines) + "\n"


def _render_html_review(matrix: list[dict[str, Any]], candidates: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('sample_id', '')))}</td>"
        f"<td>{html.escape(str(item.get('evidence_level', '')))}</td>"
        f"<td>{html.escape(str(item.get('transition', '')))}</td>"
        f"<td>{html.escape(str(item.get('clip_cosine', '')))}</td>"
        f"<td>{html.escape(str(item.get('salient_attributes', '')))}</td>"
        f"<td>{html.escape(str(item.get('public_member', '')))}</td>"
        "</tr>"
        for item in matrix[:300]
    )
    cards = "".join(
        f"<section><h2>{html.escape(name)}</h2><p>label changes: {candidate['classification_label_changes']}; "
        f"changed rows: {candidate['changed_rows']}; consistency issues: {candidate['label_consistency_issue_count']}</p></section>"
        for name, candidate in candidates.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Track2 v14 Public Resource AGSR Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f6; }}
    section {{ border: 1px solid #d7dde5; padding: 12px; margin: 10px 0; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Track2 v14 Public Resource AGSR Review</h1>
  {cards}
  <table>
    <thead><tr><th>sample</th><th>evidence</th><th>transition</th><th>clip</th><th>salient attrs</th><th>public member</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def _render_boundary_report(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v14 Emotion Boundary Report",
        "",
        f"- Matrix: `{report['matrix_path']}`",
        f"- Accepted changes: {report['accepted_count']}",
        f"- Blocked changes: {report['blocked_count']}",
        "",
        "## Accepted Transitions",
        "",
    ]
    for key, count in sorted(report["accepted_transition_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Blocked Transitions", ""])
    for key, count in sorted(report["blocked_transition_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Calm/Content/Glad Accepted Rows", ""])
    for row in report["calm_content_glad_accepts"][:120]:
        lines.append(f"- {row['sample_id']}: {row['transition']} ({row['evidence_level']}, clip={row['clip_cosine']})")
    return "\n".join(lines) + "\n"


def _render_threshold_report(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v14 Threshold Ablation",
        "",
        f"- Matrix: `{report['matrix_path']}`",
        "",
        "| threshold | candidates | changed | exact/near | style-only |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['candidate_count']} | {row['changed_count']} | "
            f"{row['exact_or_near_count']} | {row['style_only_count']} |"
        )
    return "\n".join(lines) + "\n"


def _render_historical_report(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v14 Historical Correlation",
        "",
        f"- official_best={report['official_best']}",
        f"- shadow_must_not_prefer={report['shadow_must_not_prefer']}",
        f"- historical_alignment={report['historical_alignment']}",
        f"- spearman_proxy={report['spearman_proxy']}",
        "",
        "| candidate | overall lower | overall expected |",
        "| --- | ---: | ---: |",
    ]
    for row in report["ranking"]:
        lines.append(f"| {row['candidate_name']} | {float(row['overall_lower']):.6f} | {float(row['overall_expected']):.6f} |")
    return "\n".join(lines) + "\n"


def _historical_spearman_proxy(ranking: list[dict[str, Any]]) -> float | None:
    official = []
    local = []
    for item in ranking:
        name = str(item.get("candidate_name", ""))
        if name in OFFICIAL_SCORE_ORDER:
            official.append(OFFICIAL_SCORE_ORDER[name])
            local.append(float(item.get("overall_lower", 0.0)))
    if len(official) < 2:
        return None
    return _spearman(official, local)


def _spearman(a: list[float], b: list[float]) -> float:
    ra = _ranks(a)
    rb = _ranks(b)
    mean_a = sum(ra) / len(ra)
    mean_b = sum(rb) / len(rb)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb, strict=True))
    denom_a = sum((x - mean_a) ** 2 for x in ra) ** 0.5
    denom_b = sum((y - mean_b) ** 2 for y in rb) ** 0.5
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    for rank, (_value, index) in enumerate(ordered, start=1):
        ranks[index] = float(rank)
    return ranks
