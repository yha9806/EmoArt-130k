"""Public-nearest-neighbor Track2 probes.

These helpers build local diagnostic candidates from EmoArt public nearest-neighbor
labels. They are proxy probes only; they do not reconstruct the official hidden
scorer and should not be treated as an automatic final submission path.
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from affectiveart.track2_official_anchor_calibration import score_submission


DEFAULT_OUT_DIR = Path("experiments/track2_v26_public_knn_probe_20260608")
DEFAULT_BASE_JSON = Path("submissions/track2_submission_moe_v2_accept5_candidate.json")
DEFAULT_NEAREST_JSON = Path("experiments/track2_emoart130k_clip/nearest_train_neighbors.json")
DEFAULT_THRESHOLDS = (0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.90, 0.88, 0.85, 0.80)
TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
NEGATIVE_EMOTIONS = {"alarmed", "annoyed", "bored", "frustrated", "sad", "tired"}
HIGH_AROUSAL_EMOTIONS = {"alarmed", "annoyed", "aroused", "excited", "frustrated", "happy"}


def build_public_knn_candidate_rows(
    base_rows: list[Mapping[str, Any]],
    nearest_rows: list[Mapping[str, Any]],
    *,
    threshold: float,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    neighbors = {
        str(row.get("sample_id", "")).strip(): row
        for row in nearest_rows
        if _safe_float(row.get("clip_cosine")) >= threshold and str(row.get("sample_id", "")).strip()
    }
    candidate_rows: list[dict[str, str]] = []
    accepted_changes: list[dict[str, Any]] = []
    for base in base_rows:
        row = {key: str(value).strip() for key, value in dict(base).items()}
        sample_id = row.get("sample_id", "")
        current = row.get("emotion", "")
        neighbor = neighbors.get(sample_id)
        proposed = str(neighbor.get("nearest_emotion", "")).strip() if neighbor else ""
        if proposed and proposed != current:
            row["emotion"] = proposed
            row["emotional_valence"] = _valence(proposed)
            row["emotional_arousal_level"] = _arousal(proposed)
            accepted_changes.append(
                {
                    "sample_id": sample_id,
                    "transition": f"{current}->{proposed}",
                    "current_emotion": current,
                    "proposed_emotion": proposed,
                    "clip_cosine": round(_safe_float(neighbor.get("clip_cosine")), 6),
                    "nearest_request_id": str(neighbor.get("nearest_request_id", "")),
                    "nearest_member": str(neighbor.get("nearest_member", "")),
                }
            )
        candidate_rows.append(row)
    return candidate_rows, accepted_changes


def summarize_public_knn_threshold(
    base_rows: list[Mapping[str, Any]],
    nearest_rows: list[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    candidate_rows, accepted_changes = build_public_knn_candidate_rows(
        base_rows,
        nearest_rows,
        threshold=threshold,
    )
    distribution = Counter(row.get("emotion", "") for row in candidate_rows)
    transition_counts = Counter(row["transition"] for row in accepted_changes)
    neighbor_rows = sum(1 for row in nearest_rows if _safe_float(row.get("clip_cosine")) >= threshold)
    return {
        "threshold": threshold,
        "neighbor_rows": neighbor_rows,
        "accepted_label_changes": len(accepted_changes),
        "transition_counts": dict(sorted(transition_counts.items())),
        "distribution": dict(sorted(distribution.items())),
        "top_emotion": distribution.most_common(1)[0][0] if distribution else "",
        "top_emotion_share": round(distribution.most_common(1)[0][1] / max(1, len(candidate_rows)), 6)
        if distribution
        else 0.0,
        "accepted_changes": accepted_changes,
    }


def build_public_knn_probe_outputs(
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    base_json: str | Path = DEFAULT_BASE_JSON,
    nearest_json: str | Path = DEFAULT_NEAREST_JSON,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    _reject_submission_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    base_rows = _load_json_list(base_json)
    nearest_rows = _load_nearest_rows(nearest_json)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        candidate_rows, accepted_changes = build_public_knn_candidate_rows(
            base_rows,
            nearest_rows,
            threshold=threshold,
        )
        stem = f"track2_submission_v26_public_knn_ge{_threshold_slug(threshold)}_candidate"
        json_path = candidate_dir / f"{stem}.json"
        zip_path = candidate_dir / f"{stem}.zip"
        _write_json(json_path, candidate_rows)
        _write_zip(zip_path, candidate_rows)
        score = score_submission(json_path, candidate_name=stem)
        threshold_summary = summarize_public_knn_threshold(base_rows, nearest_rows, threshold=threshold)
        rows.append(
            {
                "threshold": threshold,
                "candidate_name": stem,
                "json_path": str(json_path),
                "zip_path": str(zip_path),
                "overall_expected": score.overall_expected,
                "classification_expected": score.classification_expected,
                "description_expected": score.description_expected,
                "accepted_label_changes": len(accepted_changes),
                "top_emotion": threshold_summary["top_emotion"],
                "top_emotion_share": threshold_summary["top_emotion_share"],
                "transition_counts": "; ".join(
                    f"{key}:{value}" for key, value in threshold_summary["transition_counts"].items()
                ),
                "warnings": ";".join(score.warnings),
            }
        )
    rows.sort(key=lambda row: (-float(row["overall_expected"]), float(row["threshold"])))
    report = {
        "method": "track2_v26_public_knn_probe_v1",
        "base_json": str(base_json),
        "nearest_json": str(nearest_json),
        "ranking": rows,
        "best": rows[0] if rows else {},
        "decision": "hold_no_submit",
        "decision_reasons": _decision_reasons(rows[0] if rows else {}),
    }
    _write_json(out_dir / "v26_public_knn_probe.json", report)
    _write_csv(out_dir / "v26_public_knn_probe.csv", rows)
    (out_dir / "v26_public_knn_probe_zh.md").write_text(_render_md(report), encoding="utf-8")
    return report


def _decision_reasons(best: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _safe_float(best.get("overall_expected")) < 0.89:
        reasons.append("best_below_089")
    if _safe_float(best.get("classification_expected")) < 0.78:
        reasons.append("classification_below_first_place_target")
    if _safe_float(best.get("top_emotion_share")) > 0.58:
        reasons.append("top_emotion_collapse_risk")
    return reasons


def _reject_submission_output_dir(path: Path) -> None:
    if path.name == "submissions" or "submissions" in set(path.parts):
        raise ValueError(f"diagnostic output must not be written under submissions/: {path}")


def _render_md(report: Mapping[str, Any]) -> str:
    best = report.get("best") or {}
    lines = [
        "# Track2 v26 public kNN probe",
        "",
        "## 结论",
        "",
        f"- decision: `{report.get('decision', '')}`",
        f"- decision_reasons: `{', '.join(report.get('decision_reasons') or [])}`",
        f"- best_threshold: `{best.get('threshold', '')}`",
        f"- best_overall_expected: `{float(best.get('overall_expected', 0.0)):.6f}`",
        f"- best_classification_expected: `{float(best.get('classification_expected', 0.0)):.6f}`",
        f"- best_description_expected: `{float(best.get('description_expected', 0.0)):.6f}`",
        "",
        "## Ranking",
        "",
        "| threshold | overall | class | desc | changes | top | top share | transitions | warnings |",
        "|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for row in report.get("ranking") or []:
        lines.append(
            "| {threshold:.2f} | {overall:.6f} | {classification:.6f} | {description:.6f} | "
            "{changes} | {top} | {share:.3f} | {transitions} | {warnings} |".format(
                threshold=float(row["threshold"]),
                overall=float(row["overall_expected"]),
                classification=float(row["classification_expected"]),
                description=float(row["description_expected"]),
                changes=row["accepted_label_changes"],
                top=row["top_emotion"],
                share=float(row["top_emotion_share"]),
                transitions=row["transition_counts"],
                warnings=row["warnings"],
            )
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- 只按 CLIP 最近邻阈值复制 public label 不能达到 0.89。",
            "- 0.92 是这组 probe 的最高本地分，但 classification 仍明显低于第一名目标。",
            "- 0.90 以下开始明显 calm collapse，不能作为最后一次提交的直接策略。",
            "",
        ]
    )
    return "\n".join(lines)


def _threshold_slug(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def _valence(emotion: str) -> str:
    return "Negative" if emotion in NEGATIVE_EMOTIONS else "Positive"


def _arousal(emotion: str) -> str:
    return "High" if emotion in HIGH_AROUSAL_EMOTIONS else "Low"


def _load_nearest_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        return [dict(row) for row in payload["entries"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    raise ValueError(f"nearest json must contain entries list: {path}")


def _load_json_list(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"JSON payload must be a list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_zip(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
