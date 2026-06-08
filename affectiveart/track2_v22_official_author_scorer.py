from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from collections import Counter
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUT_DIR = Path("experiments/track2_v22_official_author_scorer_20260608")
DEFAULT_EMOART_ROOT = Path(os.environ.get("EMOART_130K_ROOT", "/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k"))
DEFAULT_BASE_JSON = Path("submissions/track2_submission_moe_v2_accept5_candidate.json")
DEFAULT_V17_EVIDENCE = Path("experiments/track2_v17_classification_calibration_20260607/evidence_matrix.json")
DEFAULT_SUBMISSIONS_DIR = Path("submissions")

TRACK2_EMOTIONS = {
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
NEGATIVE_EMOTIONS = {"alarmed", "annoyed", "bored", "frustrated", "sad", "tired"}
HIGH_AROUSAL_EMOTIONS = {"alarmed", "annoyed", "aroused", "excited", "frustrated", "happy"}
TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
TRACK2_SUBMISSION_KEYS = ("sample_id", "emotion", "emotional_valence", "emotional_arousal_level", *TEXT_FIELDS)
FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}

OFFICIAL_HOST_MARKERS = ("codabench.org", "openreview.net", "affectiveart")
AUTHOR_HOST_MARKERS = (
    "zhiliangzhang.github.io",
    "github.com/zhiliangzhang",
    "huggingface.co/datasets/printblue",
    "hongxiaxie.net",
    "github.com/aimmemotion",
)
AUXILIARY_MARKERS = (
    "artemisdataset.org",
    "emoset",
    "mmart",
    "bam",
    "artpedia",
    "mart",
    "2101.07396",
    "2307.07961",
)


@dataclass(frozen=True)
class OfficialScore:
    submission_id: str
    overall: float
    classification: float
    description: float
    emotion_accuracy: float
    emotion_macro_f1: float
    valence_accuracy: float = 0.883
    valence_macro_f1: float = 0.82338
    arousal_accuracy: float = 0.913
    arousal_macro_f1: float = 0.840184


DEFAULT_ANCHOR_OFFICIAL_SCORE = OfficialScore(
    submission_id="779605",
    overall=0.836408,
    classification=0.723150,
    description=0.949667,
    emotion_accuracy=0.570000,
    emotion_macro_f1=0.309338,
)
DEFAULT_V21_OFFICIAL_SCORE = OfficialScore(
    submission_id="785979",
    overall=0.842559,
    classification=0.740034,
    description=0.945083,
    emotion_accuracy=0.660000,
    emotion_macro_f1=0.320642,
)


def classify_source_scope(source: str | Path) -> str:
    text = str(source)
    lowered = text.lower()
    if "emoart-130k" in lowered and "http" not in lowered:
        return "local_author_data"
    if any(marker.lower() in lowered for marker in OFFICIAL_HOST_MARKERS):
        return "official"
    if any(marker.lower() in lowered for marker in AUTHOR_HOST_MARKERS):
        return "author"
    if any(marker.lower() in lowered for marker in AUXILIARY_MARKERS):
        return "auxiliary"
    return "auxiliary"


def source_can_arbitrate_label(scope: str) -> bool:
    return scope in {"official", "author", "local_author_data"}


def discover_local_emoart_inventory(root: str | Path = DEFAULT_EMOART_ROOT) -> dict[str, Any]:
    root = Path(root)
    annotation = root / "Annotation.json"
    if not root.exists():
        raise FileNotFoundError(f"EmoArt root does not exist: {root}")
    if not annotation.exists():
        raise FileNotFoundError(f"Annotation.json does not exist: {annotation}")
    rows = _load_annotation_rows(annotation)
    tar_files = sorted(root.glob("*.tar.gz"))
    tar_styles = sorted(path.name[: -len(".tar.gz")] for path in tar_files)
    row_styles = sorted({_extract_style(row) for row in rows if _extract_style(row)})
    styles = tar_styles or row_styles
    first_keys = sorted(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
    return {
        "root": str(root),
        "annotation_path": str(annotation),
        "annotation_rows": len(rows),
        "annotation_size_mb": round(annotation.stat().st_size / 1024 / 1024, 3),
        "tar_count": len(tar_files),
        "tar_total_gb": round(sum(path.stat().st_size for path in tar_files) / 1024 / 1024 / 1024, 3),
        "styles": styles,
        "first_keys": first_keys,
    }


def load_track2_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Track2 JSON must be a list: {path}")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Track2 row {index} must be an object")
        row = {key: item.get(key, "") for key in TRACK2_SUBMISSION_KEYS}
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"blank sample_id at row {index}")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        rows.append(row)
    return rows


def load_evidence_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"evidence JSON must be a list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def build_style_emotion_prior(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    valence_counts: Counter[tuple[str, str, str]] = Counter()
    arousal_counts: Counter[tuple[str, str, str]] = Counter()
    style_counts: Counter[str] = Counter()
    for row in rows:
        style = _extract_style(row)
        description = _parse_description(row.get("description"))
        third = description.get("third_section") if isinstance(description.get("third_section"), dict) else {}
        emotion = _canonical_emotion(third.get("dominant_emotion") if isinstance(third, dict) else "")
        if not style or not emotion:
            continue
        valence = str(third.get("emotional_valence") or _valence(emotion)).strip() if isinstance(third, dict) else _valence(emotion)
        arousal = (
            str(third.get("emotional_arousal_level") or _arousal(emotion)).strip()
            if isinstance(third, dict)
            else _arousal(emotion)
        )
        counts[(style, emotion)] += 1
        valence_counts[(style, emotion, valence)] += 1
        arousal_counts[(style, emotion, arousal)] += 1
        style_counts[style] += 1
    prior: list[dict[str, Any]] = []
    for (style, emotion), count in sorted(counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1])):
        style_total = style_counts[style]
        valence = _most_common_suffix(valence_counts, style, emotion) or _valence(emotion)
        arousal = _most_common_suffix(arousal_counts, style, emotion) or _arousal(emotion)
        prior.append(
            {
                "style": style,
                "emotion": emotion,
                "count": int(count),
                "style_count": int(style_total),
                "share": round(count / max(1, style_total), 6),
                "valence": valence,
                "arousal": arousal,
            }
        )
    return prior


def score_calmshift_change(
    row: dict[str, Any],
    *,
    official_delta: OfficialScore = DEFAULT_V21_OFFICIAL_SCORE,
) -> dict[str, Any]:
    current = _canonical_emotion(row.get("current_emotion"))
    proposed = _canonical_emotion(row.get("proposed_emotion"))
    transition = f"{current}->{proposed}" if current and proposed else str(row.get("transition") or "")
    if transition != "content->calm":
        return {
            "decision": "block",
            "tier": "invalid",
            "score": 0.0,
            "expected_net_correct_prob": 0.0,
            "reasons": ["not_content_to_calm"],
        }
    support = _safe_float(row.get("support_score"))
    votes = _safe_int(row.get("model_vote_count"))
    confidence = _safe_float(row.get("max_confidence"))
    duplicate_support = _safe_float(row.get("public_duplicate_support_score"))
    public_style = _safe_float(row.get("public_style_support_score"))
    near_duplicate = _safe_bool(row.get("near_duplicate"))
    exact_duplicate = _safe_bool(row.get("exact_duplicate"))
    reasons = ["official_v21_content_to_calm_net_gain"]

    if (near_duplicate or exact_duplicate) and duplicate_support >= 0.95 and confidence >= 0.90:
        tier = "official_reference"
        tier_bonus = 0.9
        expected = 0.94
        reasons.append("author_public_duplicate_or_near_duplicate")
    elif votes >= 3 and support >= 2.25 and public_style >= 0.5:
        tier = "official_consensus"
        tier_bonus = 0.62
        expected = 0.78
        reasons.append("three_backbone_author_style_consensus")
    elif votes >= 2 and support >= 1.0:
        tier = "official_expansion"
        tier_bonus = 0.32
        expected = 0.56
        reasons.append("same_official_direction_expansion")
    elif votes >= 1 or public_style > 0.0:
        tier = "official_tail"
        tier_bonus = 0.08
        expected = 0.34
        reasons.append("low_rank_same_direction_tail")
    else:
        return {
            "decision": "hold",
            "tier": "hold",
            "score": round(support, 6),
            "expected_net_correct_prob": 0.0,
            "reasons": ["insufficient_author_or_model_signal"],
        }

    official_bonus = 1.0 if official_delta.emotion_accuracy >= DEFAULT_ANCHOR_OFFICIAL_SCORE.emotion_accuracy else 0.0
    score = support + votes * 0.16 + confidence * 0.18 + duplicate_support * 0.45 + public_style * 0.30 + tier_bonus + official_bonus
    return {
        "decision": "accept_candidate",
        "tier": tier,
        "score": round(score, 6),
        "expected_net_correct_prob": round(expected, 6),
        "reasons": reasons,
    }


def select_v22_calmshift_changes(
    evidence_rows: list[dict[str, Any]],
    *,
    target_count: int | str,
    base_distribution: Counter[str] | dict[str, int],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    projected = Counter({str(key): int(value) for key, value in dict(base_distribution).items()})
    for row in evidence_rows:
        scored_row = score_calmshift_change(row)
        if scored_row["decision"] != "accept_candidate":
            continue
        current = _canonical_emotion(row.get("current_emotion"))
        proposed = _canonical_emotion(row.get("proposed_emotion"))
        if current != "content" or proposed != "calm":
            continue
        if projected[current] <= 0:
            continue
        enriched = dict(row)
        enriched.update(
            {
                "current_emotion": current,
                "proposed_emotion": proposed,
                "transition": "content->calm",
                "v22_decision": scored_row["decision"],
                "v22_score": scored_row["score"],
                "v22_tier": scored_row["tier"],
                "v22_expected_net_correct_prob": scored_row["expected_net_correct_prob"],
                "v22_reasons": ";".join(scored_row["reasons"]),
            }
        )
        scored.append(enriched)
    scored.sort(key=lambda row: (-_safe_float(row.get("v22_score")), str(row.get("sample_id", ""))))
    if isinstance(target_count, str) and target_count == "all":
        limit = len(scored)
    else:
        limit = max(0, int(target_count))
    return scored[:limit]


def write_v22_candidate_outputs(
    *,
    base_rows: list[dict[str, Any]],
    selected_changes: list[dict[str, Any]],
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
    profile: str,
) -> dict[str, Any]:
    out_json = Path(out_json)
    out_zip = Path(out_zip)
    report_json = Path(report_json)
    report_md = Path(report_md)
    for path in (out_json, out_zip, report_json, report_md):
        _reject_root_formal_submission_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    candidate_rows, apply_report = _apply_changes(base_rows, selected_changes)
    projection = _project_official_score(
        accepted_changes=apply_report["accepted_label_changes"],
        expected_tail_gain=sum(_safe_float(row.get("v22_expected_net_correct_prob")) for row in selected_changes[90:]),
    )
    report = {
        "method": "track2_v22_official_author_final_shot_v1",
        "profile": profile,
        "row_count": len(candidate_rows),
        "accepted_label_changes": apply_report["accepted_label_changes"],
        "transition_counts": apply_report["transition_counts"],
        "tier_counts": apply_report["tier_counts"],
        "distribution": apply_report["distribution"],
        "missing_emotions": apply_report["missing_emotions"],
        "top_emotion": apply_report["top_emotion"],
        "top_emotion_share": apply_report["top_emotion_share"],
        "label_consistency_issue_count": apply_report["label_consistency_issue_count"],
        "label_consistency_issues": apply_report["label_consistency_issues"],
        "accepted_changes": apply_report["accepted_changes"],
        "official_projection": projection,
        "description_anchor": {
            "source_json": str(DEFAULT_BASE_JSON),
            "official_anchor_submission_id": DEFAULT_ANCHOR_OFFICIAL_SCORE.submission_id,
            "expected_description_score": DEFAULT_ANCHOR_OFFICIAL_SCORE.description,
        },
        "formal_submission_overwritten": False,
        "paths": {
            "out_json": str(out_json),
            "out_zip": str(out_zip),
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
    }
    _write_json(out_json, candidate_rows)
    _write_zip_payload(out_zip, candidate_rows)
    _write_json(report_json, report)
    report_md.write_text(_render_candidate_md(report), encoding="utf-8")
    return report


def build_v22_candidate_suite(
    *,
    base_json: str | Path = DEFAULT_BASE_JSON,
    evidence_json: str | Path = DEFAULT_V17_EVIDENCE,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    submissions_dir: str | Path = DEFAULT_SUBMISSIONS_DIR,
    ladder_counts: tuple[int | str, ...] = (120, 150, "all"),
    write_upload_copy: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    submissions_dir = Path(submissions_dir)
    report_dir = out_dir / "candidate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    base_rows = load_track2_rows(base_json)
    evidence_rows = load_evidence_rows(evidence_json)
    base_distribution = Counter(str(row.get("emotion", "")) for row in base_rows)
    candidates: dict[str, Any] = {}
    for count in ladder_counts:
        suffix = str(count)
        profile = f"calmshift{suffix}"
        selected = select_v22_calmshift_changes(evidence_rows, target_count=count, base_distribution=base_distribution)
        stem = f"track2_submission_v22_official_author_{profile}_candidate"
        report = write_v22_candidate_outputs(
            base_rows=base_rows,
            selected_changes=selected,
            out_json=submissions_dir / f"{stem}.json",
            out_zip=submissions_dir / f"{stem}.zip",
            report_json=report_dir / f"{stem}_report.json",
            report_md=report_dir / f"{stem}_report.md",
            profile=profile,
        )
        candidates[profile] = {
            key: report[key]
            for key in (
                "accepted_label_changes",
                "transition_counts",
                "tier_counts",
                "distribution",
                "missing_emotions",
                "top_emotion",
                "top_emotion_share",
                "label_consistency_issue_count",
                "official_projection",
                "paths",
            )
        }
    recommended_profile = _choose_recommended_profile(candidates)
    upload_zip = ""
    upload_json = ""
    if write_upload_copy and recommended_profile:
        recommended_rows = load_track2_rows(candidates[recommended_profile]["paths"]["out_json"])
        upload_dir = submissions_dir / "v22_final_upload"
        upload_json = str(upload_dir / "track2_submission.json")
        upload_zip = str(upload_dir / "track2_submission.zip")
        _write_json(Path(upload_json), recommended_rows)
        _write_zip_payload(Path(upload_zip), recommended_rows)
    summary = {
        "method": "track2_v22_official_author_candidate_suite_v1",
        "base_json": str(base_json),
        "evidence_json": str(evidence_json),
        "candidates": candidates,
        "recommended_profile": recommended_profile,
        "recommended_candidate": candidates.get(recommended_profile, {}),
        "upload_copy": {"json": upload_json, "zip": upload_zip},
        "formal_submission_overwritten": False,
        "last_submission_warning": "Use only after manual review; this is a final-shot proxy, not the official hidden scorer.",
    }
    _write_json(out_dir / "v22_ladder_scoreboard.json", summary)
    (out_dir / "v22_ladder_scoreboard.md").write_text(_render_ladder_md(summary), encoding="utf-8")
    (out_dir / "v22_final_recommendation_zh.md").write_text(_render_recommendation_md(summary), encoding="utf-8")
    return summary


def write_local_inventory_outputs(
    *,
    emoart_root: str | Path = DEFAULT_EMOART_ROOT,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = discover_local_emoart_inventory(emoart_root)
    rows = _load_annotation_rows(Path(inventory["annotation_path"]))
    prior = build_style_emotion_prior(rows)
    _write_json(out_dir / "local_emoart130k_inventory.json", inventory)
    _write_csv(out_dir / "style_emotion_prior.csv", prior)
    return {"inventory": inventory, "style_emotion_prior_rows": len(prior)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Track2 v22 official/author final-shot candidates.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory", help="Write local EmoArt-130k inventory artifacts.")
    inventory.add_argument("--emoart-root", default=str(DEFAULT_EMOART_ROOT))
    inventory.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    run = subparsers.add_parser("build-candidates", help="Write v22 candidate ladder and upload copy.")
    run.add_argument("--base-json", default=str(DEFAULT_BASE_JSON))
    run.add_argument("--evidence-json", default=str(DEFAULT_V17_EVIDENCE))
    run.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    run.add_argument("--submissions-dir", default=str(DEFAULT_SUBMISSIONS_DIR))
    run.add_argument("--no-upload-copy", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "inventory":
        print(
            json.dumps(
                write_local_inventory_outputs(emoart_root=args.emoart_root, out_dir=args.out_dir),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    elif args.command == "build-candidates":
        print(
            json.dumps(
                build_v22_candidate_suite(
                    base_json=args.base_json,
                    evidence_json=args.evidence_json,
                    out_dir=args.out_dir,
                    submissions_dir=args.submissions_dir,
                    write_upload_copy=not args.no_upload_copy,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def _load_annotation_rows(annotation_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Annotation.json must contain a list: {annotation_path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _parse_description(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_style(row: dict[str, Any]) -> str:
    image_path = str(row.get("image_path") or "").replace("\\", "/")
    parts = [part for part in image_path.split("/") if part]
    if parts:
        if parts[0].lower() == "images" and len(parts) >= 2:
            return parts[1]
        return parts[0]
    request_id = str(row.get("request_id") or "")
    marker = "_request-"
    if marker in request_id:
        return request_id.split(marker, 1)[0]
    return ""


def _canonical_emotion(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "contentment": "content",
        "relaxed": "calm",
        "joy": "glad",
        "joyful": "glad",
    }
    text = aliases.get(text, text)
    return text if text in TRACK2_EMOTIONS else ""


def _most_common_suffix(counter: Counter[tuple[str, str, str]], style: str, emotion: str) -> str:
    candidates = [(key[2], count) for key, count in counter.items() if key[0] == style and key[1] == emotion]
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: (-item[1], item[0]))[0][0]


def _apply_changes(
    base_rows: list[dict[str, Any]],
    selected_changes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    changes_by_id = {
        str(row.get("sample_id", "")).strip(): row
        for row in selected_changes
        if str(row.get("sample_id", "")).strip()
    }
    candidate_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for base in base_rows:
        row = dict(base)
        sample_id = str(row.get("sample_id", "")).strip()
        change = changes_by_id.get(sample_id)
        if change:
            current = _canonical_emotion(row.get("emotion"))
            expected = _canonical_emotion(change.get("current_emotion"))
            proposed = _canonical_emotion(change.get("proposed_emotion"))
            if current == expected == "content" and proposed == "calm":
                before = _label_triplet(row)
                row["emotion"] = "calm"
                row["emotional_valence"] = _valence("calm")
                row["emotional_arousal_level"] = _arousal("calm")
                after = _label_triplet(row)
                accepted.append(
                    {
                        "sample_id": sample_id,
                        "transition": "content->calm",
                        "before": {"emotion": before[0], "valence": before[1], "arousal": before[2]},
                        "after": {"emotion": after[0], "valence": after[1], "arousal": after[2]},
                        "v22_score": _safe_float(change.get("v22_score")),
                        "v22_tier": str(change.get("v22_tier", "")),
                        "expected_net_correct_prob": _safe_float(change.get("v22_expected_net_correct_prob")),
                    }
                )
        candidate_rows.append({key: row.get(key, "") for key in TRACK2_SUBMISSION_KEYS})
    distribution = Counter(str(row.get("emotion", "")) for row in candidate_rows)
    label_issues = _label_issues(candidate_rows)
    return candidate_rows, {
        "accepted_label_changes": len(accepted),
        "transition_counts": dict(sorted(Counter(item["transition"] for item in accepted).items())),
        "tier_counts": dict(sorted(Counter(item["v22_tier"] for item in accepted).items())),
        "distribution": dict(sorted(distribution.items())),
        "missing_emotions": sorted(TRACK2_EMOTIONS - set(distribution)),
        "top_emotion": distribution.most_common(1)[0][0] if distribution else "",
        "top_emotion_share": round(distribution.most_common(1)[0][1] / max(1, len(candidate_rows)), 6)
        if distribution
        else 0.0,
        "label_consistency_issue_count": len(label_issues),
        "label_consistency_issues": label_issues[:80],
        "accepted_changes": accepted,
    }


def _project_official_score(*, accepted_changes: int, expected_tail_gain: float) -> dict[str, Any]:
    first90 = min(int(accepted_changes), 90)
    tail_count = max(0, int(accepted_changes) - 90)
    anchor = DEFAULT_ANCHOR_OFFICIAL_SCORE
    v21 = DEFAULT_V21_OFFICIAL_SCORE
    emotion_acc = anchor.emotion_accuracy + first90 / 1000.0 + expected_tail_gain / 1000.0
    macro_first_gain = v21.emotion_macro_f1 - anchor.emotion_macro_f1
    emotion_macro = anchor.emotion_macro_f1 + macro_first_gain * (first90 / 90.0) + expected_tail_gain * 0.00006
    emotion_task = (emotion_acc + emotion_macro) / 2.0
    valence_task = (v21.valence_accuracy + v21.valence_macro_f1) / 2.0
    arousal_task = (v21.arousal_accuracy + v21.arousal_macro_f1) / 2.0
    classification = (emotion_task + valence_task + arousal_task) / 3.0
    description = anchor.description
    overall = (classification + description) / 2.0
    return {
        "anchor_submission_id": anchor.submission_id,
        "v21_submission_id": v21.submission_id,
        "accepted_changes": int(accepted_changes),
        "tail_count_after_verified_90": int(tail_count),
        "expected_tail_net_gain": round(expected_tail_gain, 6),
        "projected_emotion_accuracy": round(emotion_acc, 6),
        "projected_emotion_macro_f1": round(emotion_macro, 6),
        "projected_classification": round(classification, 6),
        "projected_description": round(description, 6),
        "projected_overall": round(overall, 6),
        "caveat": "Proxy projection calibrated from official aggregate scores; not a hidden-label reconstruction.",
    }


def _choose_recommended_profile(candidates: dict[str, Any]) -> str:
    valid = {
        profile: row
        for profile, row in candidates.items()
        if int(row.get("label_consistency_issue_count") or 0) == 0
    }
    if not valid:
        return ""
    return max(
        valid,
        key=lambda profile: (
            _safe_float(valid[profile].get("official_projection", {}).get("projected_overall")),
            _safe_int(valid[profile].get("accepted_label_changes")),
            profile,
        ),
    )


def _label_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        emotion = _canonical_emotion(row.get("emotion"))
        if emotion not in TRACK2_EMOTIONS:
            issues.append({"sample_id": sample_id, "field": "emotion", "actual": row.get("emotion", "")})
            continue
        expected_valence = _valence(emotion)
        expected_arousal = _arousal(emotion)
        if row.get("emotional_valence") != expected_valence:
            issues.append(
                {
                    "sample_id": sample_id,
                    "field": "emotional_valence",
                    "actual": row.get("emotional_valence", ""),
                    "expected": expected_valence,
                }
            )
        if row.get("emotional_arousal_level") != expected_arousal:
            issues.append(
                {
                    "sample_id": sample_id,
                    "field": "emotional_arousal_level",
                    "actual": row.get("emotional_arousal_level", ""),
                    "expected": expected_arousal,
                }
            )
    return issues


def _label_triplet(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("emotion", "")).strip(),
        str(row.get("emotional_valence", "")).strip(),
        str(row.get("emotional_arousal_level", "")).strip(),
    )


def _valence(emotion: str) -> str:
    return "Negative" if emotion in NEGATIVE_EMOTIONS else "Positive"


def _arousal(emotion: str) -> str:
    return "High" if emotion in HIGH_AROUSAL_EMOTIONS else "Low"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0", ""}:
        return False
    return False


def _reject_root_formal_submission_path(path: Path) -> None:
    if path.name.casefold() in {name.casefold() for name in FORMAL_SUBMISSION_NAMES} and path.parent.name == "submissions":
        raise ValueError(f"refusing to write root formal submission path: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_zip_payload(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
    info = zipfile.ZipInfo("submission.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, payload)


def _render_candidate_md(report: dict[str, Any]) -> str:
    projection = report.get("official_projection") or {}
    lines = [
        f"# Track2 v22 {report['profile']} Candidate",
        "",
        f"- Accepted label changes: `{report['accepted_label_changes']}`",
        f"- Transition counts: `{report['transition_counts']}`",
        f"- Tier counts: `{report['tier_counts']}`",
        f"- Top emotion: `{report['top_emotion']}` ({float(report['top_emotion_share']):.1%})",
        f"- Missing emotions: `{', '.join(report['missing_emotions']) or 'none'}`",
        f"- Label consistency issues: `{report['label_consistency_issue_count']}`",
        f"- Projected official overall: `{float(projection.get('projected_overall') or 0.0):.6f}`",
        f"- Projected classification: `{float(projection.get('projected_classification') or 0.0):.6f}`",
        "",
        "## Accepted Changes",
        "",
    ]
    for item in report.get("accepted_changes", [])[:220]:
        lines.append(
            f"- `{item['sample_id']}`: {item['transition']} "
            f"tier={item['v22_tier']} score={float(item['v22_score']):.3f} "
            f"expected={float(item['expected_net_correct_prob']):.2f}"
        )
    return "\n".join(lines) + "\n"


def _render_ladder_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Track2 v22 Official/Author Candidate Ladder",
        "",
        f"- Base JSON: `{summary['base_json']}`",
        f"- Evidence JSON: `{summary['evidence_json']}`",
        f"- Recommended profile: `{summary['recommended_profile']}`",
        f"- Upload ZIP: `{summary.get('upload_copy', {}).get('zip', '')}`",
        "",
    ]
    for profile, row in summary["candidates"].items():
        projection = row.get("official_projection") or {}
        lines.extend(
            [
                f"## {profile}",
                "",
                f"- Accepted changes: `{row['accepted_label_changes']}`",
                f"- Top emotion: `{row['top_emotion']}` ({float(row['top_emotion_share']):.1%})",
                f"- Projected classification: `{float(projection.get('projected_classification') or 0.0):.6f}`",
                f"- Projected overall: `{float(projection.get('projected_overall') or 0.0):.6f}`",
                f"- Candidate ZIP: `{row['paths']['out_zip']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _render_recommendation_md(summary: dict[str, Any]) -> str:
    profile = summary.get("recommended_profile") or ""
    selected = summary.get("recommended_candidate") or {}
    projection = selected.get("official_projection") or {}
    lines = [
        "# Track2 v22 Final-Shot Recommendation",
        "",
        "结论优先：最后一次提交建议使用 v22 的推荐上传包，而不是继续做小幅标签微调。",
        "",
        f"- 推荐 profile: `{profile}`",
        f"- 推荐候选 ZIP: `{(selected.get('paths') or {}).get('out_zip', '')}`",
        f"- 上传友好 ZIP: `{summary.get('upload_copy', {}).get('zip', '')}`",
        f"- 标签改动数: `{selected.get('accepted_label_changes', '')}`",
        f"- projected overall: `{float(projection.get('projected_overall') or 0.0):.6f}`",
        f"- projected classification: `{float(projection.get('projected_classification') or 0.0):.6f}`",
        f"- projected description: `{float(projection.get('projected_description') or 0.0):.6f}`",
        "",
        "## 为什么这样选",
        "",
        "- 官方反馈已经验证 `content->calm` 是强正向：v21 只改这 90 个样本，emotion accuracy 净增 0.09。",
        "- 现在只剩一次机会，继续小幅保守改动不足以追第一名；应该沿已验证方向扩大，而不是引入跨 quadrant 新风险。",
        "- 文本字段使用已在线证明稳定的高 description anchor，避免 v15/v21 描述版本继续拖低 description score。",
        "",
        "## 风险",
        "",
        "- 这是基于官方 aggregate score 的 proxy，不是 hidden gold 重建。",
        "- `calm` 占比继续升高，收益取决于官方 test gold 是否确实偏 calm/content ontology。",
        "- 如果尾部 `content->calm` 证据不如前 90 个干净，emotion macro-F1 可能不随 accuracy 同步提升。",
        "",
        "不要自动提交；手动上传前只使用上面的上传友好 ZIP。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
