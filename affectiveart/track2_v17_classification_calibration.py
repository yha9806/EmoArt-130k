from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_EMOTIONS, TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_audit import HIGH_AROUSAL_EMOTIONS, NEGATIVE_EMOTIONS, strict_track2_label_issues


FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}
DEFAULT_EXPERIMENT_DIR = Path("experiments/track2_v17_classification_calibration_20260607")
DEFAULT_BASE_JSON = Path("submissions/track2_submission_v15_desc_expand300_candidate.json")
DEFAULT_OFFICIAL_SCORES = Path(
    "experiments/track2_official_results_20260606/track2_known_official_exact_scores_from_ledger_20260606.csv"
)
DEFAULT_PAIRWISE_DIFFS = Path(
    "experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv"
)
DEFAULT_PREDICTION_SOURCES = {
    "siglip2": Path("experiments/track2_emoart130k_siglip2/predictions.json"),
    "clip": Path("experiments/track2_emoart130k_clip/predictions.json"),
    "dinov2": Path("experiments/track2_emoart130k_dinov2/predictions.json"),
    "gemini35": Path(
        "experiments/track2_moe_specialist_ensemble_20260603/dry_run_v2/"
        "gemini35_vlm_specialist_predictions.json"
    ),
    "public_clean": Path(
        "experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/clean_predictions.json"
    ),
    "public_inclusive": Path(
        "experiments/track2_public_style_distillation_20260603/siglip2_cached_logreg_v1/inclusive_predictions.json"
    ),
}
DEFAULT_DUPLICATE_SOURCES = [
    Path("experiments/track2_emoart130k_clip/deep_duplicate_audit_20260511/track2_deep_duplicate_top10_audit.json"),
    Path("experiments/track2_emoart130k_clip/overlap_reference/ge095_all/ge095_public_neighbor_audit.json"),
]
PUBLIC_STYLE_SOURCES = {"public_clean", "public_inclusive"}
PROFILE_CONFIG = {
    "safe": {
        "min_model_votes": 2,
        "min_support_score": 1.45,
        "min_confidence": 0.78,
        "near_duplicate_min_support_score": 0.95,
        "near_duplicate_min_confidence": 0.95,
        "failed_transition_block_count": 10,
        "total_cap": 48,
        "transition_family_caps": {"calm<->content": 2},
    },
    "balanced": {
        "min_model_votes": 2,
        "min_support_score": 1.25,
        "min_confidence": 0.72,
        "near_duplicate_min_support_score": 0.95,
        "near_duplicate_min_confidence": 0.95,
        "failed_transition_block_count": 10,
        "total_cap": 96,
        "transition_family_caps": {"calm<->content": 4},
    },
    "aggressive_probe": {
        "min_model_votes": 1,
        "min_support_score": 0.95,
        "min_confidence": 0.60,
        "near_duplicate_min_support_score": 0.90,
        "near_duplicate_min_confidence": 0.90,
        "failed_transition_block_count": 10,
        "total_cap": 160,
        "transition_family_caps": {"calm<->content": 8},
    },
}
CANDIDATE_OUTPUT_PATHS = {
    "safe": (
        Path("submissions/track2_submission_v17_safe_candidate.json"),
        Path("submissions/track2_submission_v17_safe_candidate.zip"),
    ),
    "balanced": (
        Path("submissions/track2_submission_v17_balanced_candidate.json"),
        Path("submissions/track2_submission_v17_balanced_candidate.zip"),
    ),
    "aggressive_probe": (
        Path("submissions/track2_submission_v17_aggressive_probe_candidate.json"),
        Path("submissions/track2_submission_v17_aggressive_probe_candidate.zip"),
    ),
}

EVIDENCE_MATRIX_FIELDS = [
    "sample_id",
    "current_emotion",
    "proposed_emotion",
    "transition",
    "current_valence",
    "current_arousal",
    "proposed_valence",
    "proposed_arousal",
    "same_valence",
    "same_arousal",
    "model_vote_count",
    "model_sources",
    "all_sources",
    "support_score",
    "model_support_score",
    "public_style_support_score",
    "public_duplicate_support_score",
    "max_confidence",
    "exact_duplicate",
    "near_duplicate",
    "failed_transition_count",
    "current_label_issue_count",
    "rationale",
]


def load_prediction_sources(paths: dict[str, str | Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _payload_rows(payload):
            sample_id = str(row.get("sample_id") or row.get("request_id") or row.get("id") or "").strip()
            emotion = _canonical_emotion(_first_present(row, ("target_emotion", "predicted_emotion", "emotion", "label")))
            if not sample_id or not emotion:
                continue
            grouped[sample_id].append(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "emotion": emotion,
                    "confidence": _safe_float(_first_present(row, ("confidence", "probability", "score"))),
                    "margin": _safe_float(row.get("margin")),
                    "decision": str(row.get("decision", "")).strip().lower(),
                    "rationale": str(row.get("rationale") or row.get("reason") or "").strip(),
                }
            )
    return dict(grouped)


def parse_official_failed_transition_counts(
    *,
    pairwise_rows: list[dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in pairwise_rows:
        candidate_a_id = _extract_submission_id(row.get("candidate_a"))
        candidate_b_id = _extract_submission_id(row.get("candidate_b"))
        if candidate_a_id not in score_rows or candidate_b_id not in score_rows:
            continue
        candidate_a_class = _safe_float(score_rows[candidate_a_id].get("classification"))
        candidate_b_class = _safe_float(score_rows[candidate_b_id].get("classification"))
        if candidate_a_class == candidate_b_class:
            continue
        if candidate_a_class < candidate_b_class:
            lower = "candidate_a"
            higher = "candidate_b"
        else:
            lower = "candidate_b"
            higher = "candidate_a"
        direction_from, direction_to = _parse_pairwise_direction(row.get("direction"))
        invert = (direction_from, direction_to) == (lower, higher)
        for transition, count in _parse_transition_counts(row.get("top_emotion_transitions", "")).items():
            if invert:
                transition = _invert_transition(transition)
            counts[transition] += count
    return dict(sorted(counts.items()))


def build_evidence_rows(
    *,
    base_rows: list[dict[str, Any]],
    predictions_by_sample: dict[str, list[dict[str, Any]]],
    duplicate_rows: list[dict[str, Any]],
    failed_transition_counts: dict[str, int],
) -> list[dict[str, Any]]:
    duplicates_by_id = _index_rows_by_sample_id(duplicate_rows)
    evidence_rows: list[dict[str, Any]] = []
    for base in base_rows:
        sample_id = str(base.get("sample_id", "")).strip()
        current = _canonical_emotion(base.get("emotion"))
        if not sample_id or not current:
            continue
        by_emotion: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions_by_sample.get(sample_id, []):
            emotion = _canonical_emotion(prediction.get("emotion"))
            if emotion and emotion != current:
                by_emotion[emotion].append(prediction)
        duplicate_payloads = _duplicate_payloads_for_sample(duplicates_by_id.get(sample_id, []), current=current)
        for emotion, payloads in duplicate_payloads.items():
            by_emotion[emotion].extend(payloads)
        for proposed, votes in sorted(by_emotion.items()):
            transition = f"{current}->{proposed}"
            sources = sorted({str(vote.get("source", "")).strip() for vote in votes if str(vote.get("source", "")).strip()})
            model_sources = sorted(
                source
                for source in sources
                if not source.startswith("public_") and not _is_public_style_source(source)
            )
            exact_duplicate = any(vote.get("evidence_type") == "exact_public_duplicate" for vote in votes)
            near_duplicate = any(vote.get("evidence_type") == "near_public_duplicate" for vote in votes)
            confidences = [_safe_float(vote.get("confidence")) for vote in votes]
            support_scores = _support_scores_by_family(votes)
            support_score = sum(support_scores.values())
            evidence_rows.append(
                {
                    "sample_id": sample_id,
                    "current_emotion": current,
                    "proposed_emotion": proposed,
                    "transition": transition,
                    "current_valence": str(base.get("emotional_valence", "")).strip(),
                    "current_arousal": str(base.get("emotional_arousal_level", "")).strip(),
                    "proposed_valence": _valence(proposed),
                    "proposed_arousal": _arousal(proposed),
                    "same_valence": _valence(current) == _valence(proposed),
                    "same_arousal": _arousal(current) == _arousal(proposed),
                    "model_vote_count": len(model_sources),
                    "model_sources": ",".join(model_sources),
                    "all_sources": ",".join(sources),
                    "support_score": round(float(support_score), 6),
                    "model_support_score": round(float(support_scores["model"]), 6),
                    "public_style_support_score": round(float(support_scores["public_style"]), 6),
                    "public_duplicate_support_score": round(float(support_scores["public_duplicate"]), 6),
                    "max_confidence": round(max(confidences or [0.0]), 6),
                    "exact_duplicate": exact_duplicate,
                    "near_duplicate": near_duplicate,
                    "failed_transition_count": int(failed_transition_counts.get(transition, 0)),
                    "current_label_issue_count": len(strict_track2_label_issues(base)),
                    "rationale": " | ".join(
                        str(vote.get("rationale", "")).strip()
                        for vote in votes
                        if str(vote.get("rationale", "")).strip()
                    )[:800],
                }
            )
    return sorted(
        evidence_rows,
        key=lambda row: (
            -float(row["support_score"]),
            int(row["failed_transition_count"]),
            str(row["sample_id"]),
            str(row["proposed_emotion"]),
        ),
    )


def v17_gate_for_evidence(
    row: dict[str, Any],
    *,
    profile: str,
    distribution: dict[str, Any] | Counter[str],
) -> dict[str, Any]:
    config = _profile_config(profile)
    reasons: list[str] = []
    current = _canonical_emotion(row.get("current_emotion"))
    proposed = _canonical_emotion(row.get("proposed_emotion"))
    if not current or not proposed:
        reasons.append("invalid_emotion")
    elif current == proposed:
        reasons.append("no_label_change")

    raw_exact_duplicate = _safe_bool(row.get("exact_duplicate"))
    exact_duplicate = raw_exact_duplicate and _has_exact_duplicate_override_support(row)
    near_duplicate = _safe_bool(row.get("near_duplicate"))
    failed_transition = _safe_int(row.get("failed_transition_count")) >= _safe_int(config["failed_transition_block_count"])
    model_votes = _safe_int(row.get("model_vote_count"))
    if current and _would_remove_rare_class(current, distribution):
        reasons.append("rare_current_class_floor")
    if failed_transition and not exact_duplicate:
        reasons.append("failed_transition_family")

    if not exact_duplicate and model_votes < _safe_int(config["min_model_votes"]):
        reasons.append("insufficient_model_families")

    min_support = float(config["min_support_score"])
    min_confidence = float(config["min_confidence"])
    if near_duplicate and not exact_duplicate:
        min_support = float(config["near_duplicate_min_support_score"])
        min_confidence = float(config["near_duplicate_min_confidence"])

    if not exact_duplicate and _safe_float(row.get("support_score")) < min_support:
        reasons.append("low_support")
    if not exact_duplicate and _safe_float(row.get("max_confidence")) < min_confidence:
        reasons.append("low_confidence")
    if raw_exact_duplicate and not exact_duplicate and reasons:
        reasons.append("insufficient_exact_duplicate_support")

    decision = "block" if reasons else "accept"
    if decision == "accept":
        reasons.append("exact_duplicate_override" if exact_duplicate else "meets_profile_thresholds")
    return {"decision": decision, "reasons": reasons}


def select_v17_changes(
    evidence_rows: list[dict[str, Any]],
    *,
    profile: str,
    current_distribution: dict[str, Any] | Counter[str],
) -> list[dict[str, Any]]:
    config = _profile_config(profile)
    total_cap = _safe_int(config["total_cap"])
    transition_family_caps = dict(config.get("transition_family_caps") or {})
    selected: list[dict[str, Any]] = []
    selected_sample_ids: set[str] = set()
    transition_family_counts: Counter[str] = Counter()
    projected_distribution: Counter[str] = Counter(
        {str(key): _safe_int(value) for key, value in dict(current_distribution).items()}
    )

    for row in sorted(evidence_rows, key=_selection_sort_key):
        if total_cap and len(selected) >= total_cap:
            break
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in selected_sample_ids:
            continue
        gate = v17_gate_for_evidence(row, profile=profile, distribution=projected_distribution)
        if gate["decision"] != "accept":
            continue
        transition = _row_transition(row)
        family = _transition_family_key(transition)
        family_cap = _safe_int(transition_family_caps.get(family))
        if family_cap and transition_family_counts[family] >= family_cap:
            continue
        selected_row = dict(row)
        selected_row["gate_decision"] = gate["decision"]
        selected_row["gate_reasons"] = ",".join(gate["reasons"])
        selected.append(selected_row)
        selected_sample_ids.add(sample_id)
        transition_family_counts[family] += 1
        current = _canonical_emotion(row.get("current_emotion"))
        proposed = _canonical_emotion(row.get("proposed_emotion"))
        if current and proposed:
            projected_distribution[current] -= 1
            projected_distribution[proposed] += 1
    return selected


def apply_v17_changes(
    base_rows: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _profile_config(profile)
    changes_by_id = {str(row.get("sample_id", "")).strip(): dict(row) for row in changes if row.get("sample_id")}
    candidate_rows: list[dict[str, Any]] = []
    accepted_changes: list[dict[str, Any]] = []

    for base in base_rows:
        row = dict(base)
        sample_id = str(row.get("sample_id", "")).strip()
        change = changes_by_id.get(sample_id)
        if change:
            current = _canonical_emotion(row.get("emotion"))
            proposed = _canonical_emotion(
                _first_present(change, ("proposed_emotion", "target_emotion", "emotion"))
            )
            if proposed and proposed != current:
                before = _label_triplet(row)
                row["emotion"] = proposed
                row["emotional_valence"] = _valence(proposed)
                row["emotional_arousal_level"] = _arousal(proposed)
                after = _label_triplet(row)
                transition = _row_transition(change) or f"{before[0]}->{after[0]}"
                accepted_changes.append(
                    {
                        "sample_id": sample_id,
                        "transition": transition,
                        "before": {
                            "emotion": before[0],
                            "emotional_valence": before[1],
                            "emotional_arousal_level": before[2],
                        },
                        "after": {
                            "emotion": after[0],
                            "emotional_valence": after[1],
                            "emotional_arousal_level": after[2],
                        },
                        "support_score": _safe_float(change.get("support_score")),
                        "max_confidence": _safe_float(change.get("max_confidence")),
                        "model_vote_count": _safe_int(change.get("model_vote_count")),
                        "model_sources": str(change.get("model_sources", "")),
                        "all_sources": str(change.get("all_sources", "")),
                        "gate_decision": str(change.get("gate_decision", "")),
                        "gate_reasons": str(change.get("gate_reasons", "")),
                        "rationale": str(change.get("rationale", "")),
                    }
                )
        candidate_rows.append({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})

    distribution = Counter(str(row.get("emotion", "")) for row in candidate_rows)
    label_issues = [issue for row in candidate_rows for issue in strict_track2_label_issues(row)]
    report = {
        "method": "track2_v17_classification_calibration_candidate_v1",
        "profile": profile,
        "row_count": len(candidate_rows),
        "input_change_count": len(changes),
        "accepted_label_changes": len(accepted_changes),
        "transition_counts": dict(Counter(item["transition"] for item in accepted_changes)),
        "distribution": dict(sorted(distribution.items())),
        "missing_emotions": sorted(TRACK2_JSON_EMOTIONS - set(distribution)),
        "top_emotion": distribution.most_common(1)[0][0] if distribution else "",
        "top_emotion_share": (distribution.most_common(1)[0][1] / len(candidate_rows)) if candidate_rows else 0.0,
        "label_consistency_issue_count": len(label_issues),
        "label_consistency_issues": label_issues[:80],
        "accepted_changes": accepted_changes,
        "formal_submission_overwritten": False,
    }
    return candidate_rows, report


def write_v17_candidate_outputs(
    *,
    base_json: str | Path,
    changes: list[dict[str, Any]],
    profile: str,
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
) -> dict[str, Any]:
    out_json_path = Path(out_json)
    out_zip_path = Path(out_zip)
    report_json_path = Path(report_json)
    report_md_path = Path(report_md)
    _assert_safe_candidate_path(out_json_path)
    _assert_safe_candidate_path(out_zip_path)
    _assert_safe_candidate_report_path(report_json_path, expected_suffix=".json")
    _assert_safe_candidate_report_path(report_md_path, expected_suffix=".md")
    base_rows = load_track2_rows(base_json)
    candidate_rows, report = apply_v17_changes(base_rows, changes, profile)
    report["paths"] = {
        "base_json": str(base_json),
        "out_json": str(out_json_path),
        "out_zip": str(out_zip_path),
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
    }
    _write_json(out_json_path, candidate_rows)
    _write_zip(out_zip_path, out_json_path)
    _write_json(report_json_path, report)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(render_candidate_report_markdown(report), encoding="utf-8")
    return report


def render_candidate_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v17 Classification Calibration Candidate",
        "",
        f"- Method: `{report['method']}`",
        f"- Profile: `{report['profile']}`",
        f"- Base JSON: `{report.get('paths', {}).get('base_json', '')}`",
        f"- Candidate JSON: `{report.get('paths', {}).get('out_json', '')}`",
        f"- Candidate ZIP: `{report.get('paths', {}).get('out_zip', '')}`",
        f"- Accepted label changes: {report['accepted_label_changes']}",
        f"- Label consistency issues: {report['label_consistency_issue_count']}",
        f"- Missing emotions: {', '.join(report['missing_emotions']) or 'none'}",
        f"- Top emotion: {report['top_emotion']} ({float(report['top_emotion_share']):.1%})",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        "",
        "## Transition Counts",
        "",
    ]
    if report.get("transition_counts"):
        for transition, count in sorted(dict(report["transition_counts"]).items()):
            lines.append(f"- {transition}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Distribution", ""])
    for emotion, count in sorted(dict(report["distribution"]).items()):
        lines.append(f"- {emotion}: {count}")
    lines.extend(["", "## Accepted Changes", ""])
    if report.get("accepted_changes"):
        for item in report["accepted_changes"][:120]:
            lines.append(
                "- "
                f"{item['sample_id']}: {item['transition']}; "
                f"support={float(item.get('support_score', 0.0)):.2f}; "
                f"confidence={float(item.get('max_confidence', 0.0)):.2f}; "
                f"sources={item.get('all_sources', '')}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_v17_profile_candidates(
    base_json: str | Path,
    evidence_json: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    base_rows = load_track2_rows(base_json)
    evidence_payload = json.loads(Path(evidence_json).read_text(encoding="utf-8"))
    evidence_rows = _payload_rows(evidence_payload)
    if not evidence_rows:
        raise ValueError(f"expected non-empty evidence rows: {evidence_json}")
    current_distribution = Counter(str(row.get("emotion", "")) for row in base_rows)
    out_dir_path = Path(out_dir)
    candidates: dict[str, dict[str, Any]] = {}

    for profile, (out_json, out_zip) in CANDIDATE_OUTPUT_PATHS.items():
        changes = select_v17_changes(evidence_rows, profile=profile, current_distribution=current_distribution)
        report_json = out_dir_path / f"track2_submission_v17_{profile}_candidate_report.json"
        report_md = out_dir_path / f"track2_submission_v17_{profile}_candidate_report.md"
        report = write_v17_candidate_outputs(
            base_json=base_json,
            changes=changes,
            profile=profile,
            out_json=out_json,
            out_zip=out_zip,
            report_json=report_json,
            report_md=report_md,
        )
        candidates[profile] = {
            key: value
            for key, value in report.items()
            if key not in {"accepted_changes", "label_consistency_issues"}
        }

    summary = {
        "method": "track2_v17_classification_calibration_candidates_v1",
        "base_json": str(base_json),
        "evidence_json": str(evidence_json),
        "out_dir": str(out_dir_path),
        "evidence_rows": len(evidence_rows),
        "profiles": list(CANDIDATE_OUTPUT_PATHS),
        "candidates": candidates,
        "formal_submission_overwritten": False,
    }
    _write_json(out_dir_path / "track2_v17_candidate_summary.json", summary)
    return summary


def load_track2_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".json")]
            preferred = [name for name in names if Path(name).name in {"submission.json", "track2_submission.json"}]
            if not preferred and not names:
                raise ValueError(f"no JSON payload in Track2 zip: {path}")
            with archive.open((preferred or names)[0]) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected Track2 JSON list: {path}")
    return [{key: dict(row).get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS} for row in payload if isinstance(row, dict)]


def load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_official_score_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in load_csv_rows(path):
        submission_id = str(row.get("submission_id", "")).strip()
        if not submission_id:
            continue
        rows[submission_id] = {
            "overall": _safe_float(_first_present(row, ("official_overall", "overall"))),
            "classification": _safe_float(_first_present(row, ("official_classification", "classification"))),
            "description": _safe_float(_first_present(row, ("official_description", "description"))),
        }
    return rows


def load_duplicate_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(_flatten_duplicate_rows(_payload_rows(payload)))
    return rows


def _duplicate_payloads_for_sample(rows: list[dict[str, Any]], current: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        emotion = _canonical_emotion(row.get("public_emotion") or row.get("nearest_emotion") or row.get("emotion"))
        if not emotion or emotion == current:
            continue
        cosine = _safe_float(_first_present(row, ("clip_cosine", "top1_clip_cosine", "top1_clip")))
        suspect_duplicate = row.get("suspect_duplicate") is True
        if suspect_duplicate or cosine >= 0.985:
            evidence_type = "exact_public_duplicate"
            confidence = max(cosine, 0.985)
        elif cosine >= 0.95:
            evidence_type = "near_public_duplicate"
            confidence = cosine
        else:
            continue
        grouped[emotion].append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "source": f"public_{evidence_type}",
                "emotion": emotion,
                "confidence": confidence,
                "margin": 0.0,
                "evidence_type": evidence_type,
                "rationale": f"{evidence_type} supports {emotion}",
            }
        )
    return dict(grouped)


def _index_rows_by_sample_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            grouped[sample_id].append(row)
    return dict(grouped)


def _canonical_emotion(value: Any) -> str:
    emotion = str(value or "").strip().lower()
    if emotion == "contentment":
        emotion = "content"
    return emotion if emotion in TRACK2_JSON_EMOTIONS else ""


def _valence(emotion: str) -> str:
    return "Negative" if emotion in NEGATIVE_EMOTIONS else "Positive"


def _arousal(emotion: str) -> str:
    return "High" if emotion in HIGH_AROUSAL_EMOTIONS else "Low"


def _first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row and row[name] is not None and row[name] != "":
            return row[name]
    return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    return int(_safe_float(value))


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _has_exact_duplicate_override_support(row: dict[str, Any]) -> bool:
    return _safe_float(row.get("public_duplicate_support_score")) >= 0.985


def _profile_config(profile: str) -> dict[str, Any]:
    if profile not in PROFILE_CONFIG:
        raise ValueError(f"unknown v17 profile: {profile}")
    return dict(PROFILE_CONFIG[profile])


def _would_remove_rare_class(current: str, distribution: dict[str, Any] | Counter[str]) -> bool:
    emotion = _canonical_emotion(current)
    if not emotion:
        return False
    return _safe_int(distribution.get(emotion, 0)) <= 2


def _selection_sort_key(row: dict[str, Any]) -> tuple[float, float, str, str]:
    return (
        -_safe_float(row.get("support_score")),
        -_safe_float(row.get("max_confidence")),
        str(row.get("sample_id", "")),
        _row_transition(row),
    )


def _row_transition(row: dict[str, Any]) -> str:
    transition = str(row.get("transition", "")).strip()
    if transition:
        return transition
    current = _canonical_emotion(row.get("current_emotion"))
    proposed = _canonical_emotion(row.get("proposed_emotion"))
    return f"{current}->{proposed}" if current and proposed else ""


def _transition_family_key(transition: str) -> str:
    left_right = [piece.strip() for piece in str(transition).split("->", 1)]
    if len(left_right) != 2:
        return str(transition)
    return "<->".join(sorted(left_right))


def _extract_submission_id(value: Any) -> str:
    match = re.match(r"^(\d{6,})", str(value or "").strip())
    return match.group(1) if match else ""


def _parse_pairwise_direction(value: Any) -> tuple[str, str]:
    match = re.search(r"(candidate_[ab])\s*->\s*(candidate_[ab])", str(value or "").strip().lower())
    if match:
        return match.group(1), match.group(2)
    return "candidate_b", "candidate_a"


def _invert_transition(transition: str) -> str:
    left_right = [piece.strip() for piece in str(transition).split("->", 1)]
    if len(left_right) != 2:
        return transition
    return f"{left_right[1]}->{left_right[0]}"


def _parse_transition_counts(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in str(value or "").split(";"):
        item = part.strip()
        if not item or ":" not in item:
            continue
        transition, count_text = item.rsplit(":", 1)
        transition = transition.strip()
        left_right = [piece.strip().lower() for piece in transition.split("->", 1)]
        if len(left_right) != 2:
            continue
        left = _canonical_emotion(left_right[0])
        right = _canonical_emotion(left_right[1])
        if left and right:
            counts[f"{left}->{right}"] = int(_safe_float(count_text))
    return counts


def _support_scores_by_family(votes: list[dict[str, Any]]) -> dict[str, float]:
    model_scores_by_source: dict[str, float] = defaultdict(float)
    public_style_scores: list[float] = []
    public_duplicate_scores: list[float] = []
    for vote in votes:
        source = str(vote.get("source", "")).strip()
        confidence = _safe_float(vote.get("confidence"))
        if _is_public_duplicate_vote(vote):
            public_duplicate_scores.append(confidence)
        elif _is_public_style_source(source):
            public_style_scores.append(confidence)
        elif source:
            contribution = confidence + 0.25 * max(0.0, _safe_float(vote.get("margin")))
            model_scores_by_source[source] = max(model_scores_by_source[source], contribution)
    return {
        "model": sum(model_scores_by_source.values()),
        "public_style": max(public_style_scores or [0.0]),
        "public_duplicate": max(public_duplicate_scores or [0.0]),
    }


def _is_public_style_source(source: str) -> bool:
    return source in PUBLIC_STYLE_SOURCES


def _is_public_duplicate_vote(vote: dict[str, Any]) -> bool:
    evidence_type = str(vote.get("evidence_type", "")).strip()
    source = str(vote.get("source", "")).strip()
    return evidence_type in {"exact_public_duplicate", "near_public_duplicate"} or source in {
        "public_exact_public_duplicate",
        "public_near_public_duplicate",
    }


def write_evidence_matrix_outputs(
    *,
    base_json: str | Path,
    official_scores: str | Path,
    pairwise_diffs: str | Path,
    out_json: str | Path,
    out_csv: str | Path,
    prediction_sources: dict[str, str | Path] | None = None,
    duplicate_sources: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    base_rows = load_track2_rows(base_json)
    predictions = load_prediction_sources(prediction_sources or DEFAULT_PREDICTION_SOURCES)
    duplicates = load_duplicate_rows(duplicate_sources or DEFAULT_DUPLICATE_SOURCES)
    failed_counts = parse_official_failed_transition_counts(
        pairwise_rows=load_csv_rows(pairwise_diffs),
        score_rows=load_official_score_rows(official_scores),
    )
    evidence = build_evidence_rows(
        base_rows=base_rows,
        predictions_by_sample=predictions,
        duplicate_rows=duplicates,
        failed_transition_counts=failed_counts,
    )
    _write_json(out_json, evidence)
    _write_csv(out_csv, evidence)
    return evidence


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    _assert_not_formal_submission(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    _assert_not_formal_submission(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = EVIDENCE_MATRIX_FIELDS + sorted({key for row in rows for key in row} - set(EVIDENCE_MATRIX_FIELDS))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Track2 v17 classification calibration tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-evidence", help="Build v17 evidence matrix")
    build.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    build.add_argument("--official-scores", type=Path, default=DEFAULT_OFFICIAL_SCORES)
    build.add_argument("--pairwise-diffs", type=Path, default=DEFAULT_PAIRWISE_DIFFS)
    build.add_argument("--out-json", type=Path, default=DEFAULT_EXPERIMENT_DIR / "evidence_matrix.json")
    build.add_argument("--out-csv", type=Path, default=DEFAULT_EXPERIMENT_DIR / "evidence_matrix.csv")
    candidates = subparsers.add_parser("build-candidates", help="Build v17 side-path candidates")
    candidates.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    candidates.add_argument("--evidence-json", type=Path, default=DEFAULT_EXPERIMENT_DIR / "evidence_matrix.json")
    candidates.add_argument("--out-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR / "candidate_reports")
    args = parser.parse_args(argv)
    if args.command == "build-evidence":
        rows = write_evidence_matrix_outputs(
            base_json=args.base_json,
            official_scores=args.official_scores,
            pairwise_diffs=args.pairwise_diffs,
            out_json=args.out_json,
            out_csv=args.out_csv,
        )
        print(
            json.dumps(
                {"evidence_rows": len(rows), "out_json": str(args.out_json), "out_csv": str(args.out_csv)},
                indent=2,
            )
        )
    elif args.command == "build-candidates":
        summary = write_v17_profile_candidates(
            base_json=args.base_json,
            evidence_json=args.evidence_json,
            out_dir=args.out_dir,
        )
        print(json.dumps(summary, indent=2))


def _payload_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("entries", "predictions", "rows", "items", "selected", "neighbors", "audits"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, dict)]
    if all(isinstance(value, dict) for value in payload.values()):
        rows = []
        for key, value in payload.items():
            row = dict(value)
            row.setdefault("sample_id", key)
            rows.append(row)
        return rows
    return []


def _flatten_duplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row.get("neighbors"), list):
            sample_id = str(row.get("sample_id", ""))
            for neighbor in row["neighbors"]:
                if not isinstance(neighbor, dict):
                    continue
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
            flattened.append(dict(row))
    return flattened


def _assert_not_formal_submission(path: Path) -> None:
    if path.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError(f"refusing to write formal Track2 submission path: {path}")


def _assert_safe_candidate_path(path: str | Path) -> None:
    path = Path(path)
    _assert_not_formal_submission(path)
    if path.suffix not in {".json", ".zip"}:
        raise ValueError(f"candidate output must be JSON or ZIP: {path}")
    if not path.stem.startswith("track2_submission_v17_") or not path.stem.endswith("_candidate"):
        raise ValueError(f"candidate output must be a v17 side-path candidate: {path}")


def _assert_safe_candidate_report_path(path: str | Path, *, expected_suffix: str) -> None:
    path = Path(path)
    _assert_not_formal_submission(path)
    if path.suffix != expected_suffix:
        raise ValueError(f"candidate report must use {expected_suffix}: {path}")
    if not path.name.startswith("track2_submission_v17_"):
        raise ValueError(f"candidate report must be a v17 report side path: {path}")
    expected_tail = f"_candidate_report{expected_suffix}"
    if not path.name.endswith(expected_tail):
        raise ValueError(f"candidate report filename must end with {expected_tail}: {path}")
    if not any("report" in part.lower() or part.lower() == "experiments" for part in path.parts):
        raise ValueError(f"candidate report must live under an experiment/report path: {path}")


def _write_zip(zip_path: str | Path, json_path: str | Path) -> None:
    zip_path = Path(zip_path)
    json_path = Path(json_path)
    _assert_safe_candidate_path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("submission.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(info, json_path.read_bytes())


def _label_triplet(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("emotion", "")).strip(),
        str(row.get("emotional_valence", "")).strip(),
        str(row.get("emotional_arousal_level", "")).strip(),
    )


if __name__ == "__main__":
    main()
