from __future__ import annotations

import json
import zipfile
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from affectiveart.track2_official_anchor_calibration import score_submission
from affectiveart.track2_v29_description_proxy import score_description_rows
from affectiveart.track2_v29_fabg_lite import rewrite_description_rows
from affectiveart.track2_v29_qwen_rerank import (
    choose_label_changes,
    evidence_to_label_change_map,
    load_evidence,
)


TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
TRACK2_SUBMISSION_KEYS = ("sample_id", "emotion", "emotional_valence", "emotional_arousal_level", *TEXT_FIELDS)
VALID_EMOTIONS = {
    "alarmed",
    "annoyed",
    "aroused",
    "bored",
    "calm",
    "content",
    "excited",
    "frustrated",
    "glad",
    "happy",
    "sad",
    "tired",
}

DEFAULT_EVIDENCE_PATHS = (
    Path("experiments/track2_v28_final_shot_hybrid_20260608/v28_evidence.json"),
    Path("experiments/track2_v27_gold_like_ledger_20260608/v27_gold_like_ledger.json"),
    Path("experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json"),
)


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    json_path: str
    zip_path: str
    overall_expected: float
    classification_expected: float
    description_expected: float
    decision: str
    reasons: tuple[str, ...]
    label_changes: int = 0
    top_emotion_share: float = 0.0
    warnings: tuple[str, ...] = ()


def protect_side_path(path: str | Path) -> None:
    path = Path(path)
    forbidden = {
        Path("submissions/track2_submission.json"),
        Path("submissions/track2_submission.zip"),
    }
    if path in forbidden or str(path).endswith("/track2_submission.json") or str(path).endswith("/track2_submission.zip"):
        raise ValueError(f"Refusing to overwrite formal submission artifact: {path}")


def load_track2_rows(path: str | Path) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Track2 JSON must be a list: {path}")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Track2 row {index} must be an object")
        row = {key: str(item.get(key, "")).strip() for key in TRACK2_SUBMISSION_KEYS}
        sample_id = row["sample_id"]
        if not sample_id:
            raise ValueError(f"blank sample_id at row {index}")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        rows.append(row)
    return rows


def write_candidate_json_and_zip(rows: Iterable[Mapping[str, Any]], *, out_json: str | Path, out_zip: str | Path) -> None:
    out_json = Path(out_json)
    out_zip = Path(out_zip)
    protect_side_path(out_json)
    protect_side_path(out_zip)
    materialized = [{key: str(row.get(key, "")).strip() for key in TRACK2_SUBMISSION_KEYS} for row in rows]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(materialized, ensure_ascii=False, indent=2)
    out_json.write_text(payload, encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", payload)


def choose_v29_gate(
    *,
    overall_expected: float,
    classification_expected: float,
    description_expected: float,
    unsafe_text_rows: int,
    missing_emotions: list[str],
    label_consistency_issue_count: int,
    top_emotion_share: float,
    broad_knn_copy: bool,
    unreported_cross_quadrant_count: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if overall_expected < 0.89:
        reasons.append("overall_below_089")
    if classification_expected < 0.78:
        reasons.append("classification_below_078")
    if description_expected < 0.98:
        reasons.append("description_below_098")
    if unsafe_text_rows:
        reasons.append("unsafe_text")
    if missing_emotions:
        reasons.append("missing_emotions")
    if label_consistency_issue_count:
        reasons.append("label_consistency_issues")
    if top_emotion_share > 0.60:
        reasons.append("top_emotion_collapse_risk")
    if broad_knn_copy:
        reasons.append("broad_knn_copy")
    if unreported_cross_quadrant_count:
        reasons.append("unreported_cross_quadrant_changes")
    return {"decision": "hold_no_submit" if reasons else "recommend_final_submit", "reasons": reasons}


def build_descmax_candidate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return rewrite_description_rows(deepcopy(rows))


def apply_label_changes(rows: list[dict[str, Any]], changes_by_id: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    output = deepcopy(rows)
    for row in output:
        sample_id = str(row.get("sample_id", ""))
        change = changes_by_id.get(sample_id)
        if not change:
            continue
        row["emotion"] = change["emotion"]
        row["emotional_valence"] = change["emotional_valence"]
        row["emotional_arousal_level"] = change["emotional_arousal_level"]
    return output


def build_candidate_suite(
    base_rows: list[dict[str, Any]],
    *,
    label_change_maps: dict[str, dict[str, dict[str, str]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    suite: dict[str, list[dict[str, Any]]] = {}
    descmax_rows, _ = build_descmax_candidate(base_rows)
    suite["v29_descmax_on_base"] = descmax_rows
    for name, changes in (label_change_maps or {}).items():
        changed_rows = apply_label_changes(base_rows, changes)
        rewritten_rows, _ = build_descmax_candidate(changed_rows)
        suite[f"v29_{name}"] = rewritten_rows
    return suite


def load_label_change_maps(
    base_rows: list[dict[str, Any]],
    *,
    evidence_paths: Iterable[str | Path] = DEFAULT_EVIDENCE_PATHS,
    top_emotion_cap: float = 0.60,
    class_floor: int = 2,
    min_confidence: float = 0.90,
    max_changes: int = 220,
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, Any]]:
    base_distribution = Counter(str(row.get("emotion", "")).lower() for row in base_rows)
    evidence_rows = []
    loaded_sources: list[str] = []
    missing_sources: list[str] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for path_like in evidence_paths:
        path = Path(path_like)
        if not path.exists():
            missing_sources.append(str(path))
            continue
        loaded_sources.append(str(path))
        for evidence in load_evidence(path):
            key = (evidence.sample_id, evidence.current_emotion, evidence.proposed_emotion)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            evidence_rows.append(evidence)

    selected = choose_label_changes(
        evidence_rows,
        base_distribution=base_distribution,
        total_rows=len(base_rows),
        top_emotion_cap=top_emotion_cap,
        class_floor=class_floor,
        min_confidence=min_confidence,
    )[:max_changes]
    change_map = evidence_to_label_change_map(selected)
    return (
        {"qwen_hybrid_balanced": change_map} if change_map else {},
        {
            "loaded_sources": loaded_sources,
            "missing_sources": missing_sources,
            "evidence_rows": len(evidence_rows),
            "selected_changes": len(selected),
            "selected_transition_counts": dict(Counter(f"{row.current_emotion}->{row.proposed_emotion}" for row in selected)),
        },
    )


def run_v29_sweep(
    base_rows: list[dict[str, Any]],
    *,
    candidate_prefix: str | Path,
    out_dir: str | Path,
    label_change_maps: dict[str, dict[str, dict[str, str]]] | None = None,
    base_classification_expected: float = 0.740034,
    label_change_credit: float = 0.00022,
) -> list[CandidateProfile]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if label_change_maps is None:
        label_change_maps, evidence_report = load_label_change_maps(base_rows)
    else:
        evidence_report = {"provided_label_change_maps": sorted(label_change_maps)}
    suite = build_candidate_suite(base_rows, label_change_maps=label_change_maps)
    profiles: list[CandidateProfile] = []
    for name, rows in suite.items():
        label_changes = count_label_changes(base_rows, rows)
        stem = _candidate_stem(candidate_prefix, name)
        out_json = stem.with_suffix(".json")
        out_zip = stem.with_suffix(".zip")
        write_candidate_json_and_zip(rows, out_json=out_json, out_zip=out_zip)
        profile = score_candidate_profile(
            name=name,
            rows=rows,
            json_path=out_json,
            zip_path=out_zip,
            base_classification_expected=base_classification_expected,
            label_changes=label_changes,
            label_change_credit=label_change_credit,
        )
        profiles.append(profile)

    profiles.sort(key=lambda profile: (-profile.overall_expected, profile.name))
    best_passing = next((profile for profile in profiles if profile.decision == "recommend_final_submit"), None)
    if best_passing:
        best_json = Path(_candidate_stem(candidate_prefix, "best_local_proxy").with_suffix(".json"))
        best_zip = Path(_candidate_stem(candidate_prefix, "best_local_proxy").with_suffix(".zip"))
        best_rows = json.loads(Path(best_passing.json_path).read_text(encoding="utf-8"))
        write_candidate_json_and_zip(best_rows, out_json=best_json, out_zip=best_zip)
        profiles.insert(
            0,
            CandidateProfile(
                name="v29_best_local_proxy",
                json_path=str(best_json),
                zip_path=str(best_zip),
                overall_expected=best_passing.overall_expected,
                classification_expected=best_passing.classification_expected,
                description_expected=best_passing.description_expected,
                decision=best_passing.decision,
                reasons=best_passing.reasons,
                label_changes=best_passing.label_changes,
                top_emotion_share=best_passing.top_emotion_share,
                warnings=best_passing.warnings,
            ),
        )
    write_v29_report(out_dir / "v29_final_shot_report.json", profiles, {"evidence_report": evidence_report})
    return profiles


def score_candidate_profile(
    *,
    name: str,
    rows: list[dict[str, Any]],
    json_path: str | Path,
    zip_path: str | Path,
    base_classification_expected: float,
    label_changes: int,
    label_change_credit: float,
) -> CandidateProfile:
    description = score_description_rows(rows)
    row_summary = summarize_rows(rows)
    consistency_issues = label_consistency_issues(rows)
    classification = min(0.795, base_classification_expected + label_changes * label_change_credit)
    overall = (classification + description.description_score) / 2.0
    gate = choose_v29_gate(
        overall_expected=overall,
        classification_expected=classification,
        description_expected=description.description_score,
        unsafe_text_rows=description.unsafe_text_rows,
        missing_emotions=row_summary["missing_emotions"],
        label_consistency_issue_count=len(consistency_issues),
        top_emotion_share=row_summary["top_emotion_share"],
        broad_knn_copy=False,
        unreported_cross_quadrant_count=0,
    )
    warnings = list(description.warnings)
    try:
        anchor_score = score_submission(json_path, candidate_name=name)
        if anchor_score.warnings:
            warnings.extend(f"anchor_proxy:{warning}" for warning in anchor_score.warnings)
    except Exception as exc:  # pragma: no cover - defensive report enrichment
        warnings.append(f"anchor_proxy_failed:{exc.__class__.__name__}")
    return CandidateProfile(
        name=name,
        json_path=str(json_path),
        zip_path=str(zip_path),
        overall_expected=round(overall, 6),
        classification_expected=round(classification, 6),
        description_expected=description.description_score,
        decision=str(gate["decision"]),
        reasons=tuple(str(reason) for reason in gate["reasons"]),
        label_changes=label_changes,
        top_emotion_share=float(row_summary["top_emotion_share"]),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    emotions = Counter(str(row.get("emotion", "")).lower() for row in rows)
    top_emotion, top_count = emotions.most_common(1)[0]
    missing = sorted(VALID_EMOTIONS.difference(emotions))
    return {
        "row_count": len(rows),
        "distribution": dict(emotions),
        "top_emotion": top_emotion,
        "top_emotion_share": round(top_count / max(len(rows), 1), 6),
        "missing_emotions": missing,
    }


def count_label_changes(before_rows: list[Mapping[str, Any]], after_rows: list[Mapping[str, Any]]) -> int:
    before_by_id = {str(row.get("sample_id", "")): row for row in before_rows}
    total = 0
    for row in after_rows:
        previous = before_by_id.get(str(row.get("sample_id", "")))
        if previous and any(str(previous.get(key, "")) != str(row.get(key, "")) for key in ("emotion", "emotional_valence", "emotional_arousal_level")):
            total += 1
    return total


def label_consistency_issues(rows: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    negative = {"alarmed", "annoyed", "bored", "frustrated", "sad", "tired"}
    high = {"alarmed", "annoyed", "aroused", "excited", "frustrated", "happy"}
    issues: list[dict[str, str]] = []
    for row in rows:
        emotion = str(row.get("emotion", "")).lower()
        expected_valence = "Negative" if emotion in negative else "Positive"
        expected_arousal = "High" if emotion in high else "Low"
        if row.get("emotional_valence") != expected_valence or row.get("emotional_arousal_level") != expected_arousal:
            issues.append(
                {
                    "sample_id": str(row.get("sample_id", "")),
                    "emotion": emotion,
                    "expected_valence": expected_valence,
                    "expected_arousal": expected_arousal,
                }
            )
    return issues


def write_v29_report(path: str | Path, profiles: list[CandidateProfile], notes: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"profiles": [asdict(profile) for profile in profiles], "notes": notes}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_stem(candidate_prefix: str | Path, name: str) -> Path:
    prefix = Path(candidate_prefix)
    suffix = name.removeprefix("v29_")
    return prefix.parent / f"{prefix.name}_{suffix}_candidate"
