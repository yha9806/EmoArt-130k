"""Track2 local signal audit utilities.

This module audits local candidate and model-signal frontiers. Scores for
unsubmitted candidates are proxy estimates, not official hidden-label results.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from affectiveart.track2_official_anchor_calibration import score_submission


DEFAULT_OUT_DIR = Path("experiments/track2_v25_signal_audit_20260608")
DEFAULT_BASE_JSON = Path("submissions/track2_submission_moe_v2_accept5_candidate.json")
DEFAULT_MODEL_PREDICTIONS = {
    "clip": Path("experiments/track2_emoart130k_clip/predictions.json"),
    "siglip2": Path("experiments/track2_emoart130k_siglip2/predictions.json"),
    "dinov2": Path("experiments/track2_emoart130k_dinov2/predictions.json"),
    "ensemble": Path("experiments/track2_ensemble/predictions.json"),
}
DEFAULT_CANDIDATE_GLOB = "submissions/track2_submission*_candidate.json"


def summarize_model_predictions(
    base_rows: list[Mapping[str, Any]],
    model_entries: Mapping[str, list[Mapping[str, Any]]],
    *,
    high_confidence: float = 0.8,
    high_margin: float = 0.4,
) -> dict[str, Any]:
    base_by_id = {str(row.get("sample_id", "")).strip(): str(row.get("emotion", "")).strip() for row in base_rows}
    per_model: dict[str, dict[str, Any]] = {}
    normalized_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, entries in sorted(model_entries.items()):
        normalized = {
            str(row.get("sample_id", "")).strip(): {
                "emotion": str(row.get("emotion", "")).strip(),
                "confidence": _safe_float(row.get("confidence")),
                "margin": _safe_float(row.get("margin")),
            }
            for row in entries
            if str(row.get("sample_id", "")).strip()
        }
        normalized_by_model[model_name] = normalized
        predictions = [row["emotion"] for sample_id, row in normalized.items() if sample_id in base_by_id]
        distribution = Counter(predictions)
        changes = [
            (sample_id, base_by_id[sample_id], row["emotion"], row["confidence"], row["margin"])
            for sample_id, row in normalized.items()
            if sample_id in base_by_id and row["emotion"] and row["emotion"] != base_by_id[sample_id]
        ]
        high_changes = [
            item
            for item in changes
            if float(item[3]) >= high_confidence and float(item[4]) >= high_margin
        ]
        transition_counts = Counter(f"{before}->{after}" for _, before, after, _, _ in changes)
        per_model[model_name] = {
            "rows": len(predictions),
            "changes_vs_base": len(changes),
            "high_confidence_changes": len(high_changes),
            "distribution": dict(sorted(distribution.items())),
            "top_emotion": distribution.most_common(1)[0][0] if distribution else "",
            "top_emotion_share": round(distribution.most_common(1)[0][1] / max(1, len(predictions)), 6)
            if distribution
            else 0.0,
            "transition_counts": dict(sorted(transition_counts.items())),
        }
    return {
        "models": per_model,
        "three_model_exact_agreement": _summarize_three_model_agreement(base_by_id, normalized_by_model),
    }


def rank_candidate_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_overall: float = 0.89,
) -> dict[str, Any]:
    normalized = [
        {
            **dict(row),
            "overall_expected": round(_safe_float(row.get("overall_expected")), 6),
            "classification_expected": round(_safe_float(row.get("classification_expected")), 6),
            "description_expected": round(_safe_float(row.get("description_expected")), 6),
        }
        for row in rows
    ]
    ranked = sorted(normalized, key=lambda row: (-float(row["overall_expected"]), str(row.get("candidate_name", ""))))
    best = ranked[0] if ranked else {}
    return {
        "target_overall": target_overall,
        "candidate_count": len(ranked),
        "above_086": sum(1 for row in ranked if float(row["overall_expected"]) >= 0.86),
        "above_target": sum(1 for row in ranked if float(row["overall_expected"]) >= target_overall),
        "best": best,
        "ranking": ranked,
    }


def choose_v25_direction(
    *,
    best_overall: float,
    target_overall: float,
    best_top_emotion_share: float,
    three_model_agreement_changed_count: int,
    max_safe_top_emotion_share: float = 0.58,
) -> dict[str, Any]:
    reasons: list[str] = []
    if best_overall < target_overall:
        reasons.append("best_candidate_below_target")
    if best_top_emotion_share > max_safe_top_emotion_share:
        reasons.append("top_emotion_collapse")
    if three_model_agreement_changed_count <= 0:
        reasons.append("no_model_agreement_changes")
    decision = "candidate_pool_sufficient" if not reasons else "needs_new_classification_signal"
    return {
        "decision": decision,
        "target_overall": round(target_overall, 6),
        "best_overall": round(best_overall, 6),
        "best_top_emotion_share": round(best_top_emotion_share, 6),
        "three_model_agreement_changed_count": int(three_model_agreement_changed_count),
        "reasons": reasons,
    }


def scan_candidate_pool(
    *,
    candidate_glob: str = DEFAULT_CANDIDATE_GLOB,
    target_overall: float = 0.89,
) -> dict[str, Any]:
    candidate_paths = sorted(Path(".").glob(candidate_glob))
    rows: list[dict[str, Any]] = []
    for path in candidate_paths:
        try:
            score = score_submission(path, candidate_name=path.stem)
        except Exception as exc:  # pragma: no cover - defensive diagnostics for dirty experiment dirs
            rows.append(
                {
                    "candidate_name": path.stem,
                    "json_path": str(path),
                    "overall_expected": 0.0,
                    "classification_expected": 0.0,
                    "description_expected": 0.0,
                    "warnings": f"score_error:{type(exc).__name__}",
                }
            )
            continue
        rows.append(
            {
                "candidate_name": score.candidate_name,
                "json_path": score.json_path,
                "calibration_kind": score.calibration_kind,
                "overall_expected": score.overall_expected,
                "classification_expected": score.classification_expected,
                "description_expected": score.description_expected,
                "label_changes_vs_anchor": score.label_changes_vs_anchor,
                "text_changed_rows_vs_anchor": score.text_changed_rows_vs_anchor,
                "transition_counts": "; ".join(f"{key}:{value}" for key, value in score.transition_counts.items()),
                "warnings": ";".join(score.warnings),
            }
        )
    return rank_candidate_scores(rows, target_overall=target_overall)


def build_v25_signal_audit_outputs(
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    base_json: str | Path = DEFAULT_BASE_JSON,
    model_prediction_paths: Mapping[str, str | Path] = DEFAULT_MODEL_PREDICTIONS,
    candidate_glob: str = DEFAULT_CANDIDATE_GLOB,
    target_overall: float = 0.89,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_rows = _load_json_list(base_json)
    model_entries = {
        name: _load_prediction_entries(path)
        for name, path in model_prediction_paths.items()
        if Path(path).exists()
    }
    signal_summary = summarize_model_predictions(base_rows, model_entries)
    candidate_pool = scan_candidate_pool(candidate_glob=candidate_glob, target_overall=target_overall)
    best_top_share = _best_candidate_top_emotion_share(candidate_pool.get("best", {}))
    decision = choose_v25_direction(
        best_overall=float(candidate_pool.get("best", {}).get("overall_expected", 0.0) or 0.0),
        target_overall=target_overall,
        best_top_emotion_share=best_top_share,
        three_model_agreement_changed_count=int(
            signal_summary["three_model_exact_agreement"].get("changed_count", 0)
        ),
    )
    report = {
        "method": "track2_v25_signal_audit_v1",
        "base_json": str(base_json),
        "model_prediction_paths": {name: str(path) for name, path in model_prediction_paths.items()},
        "candidate_glob": candidate_glob,
        "target_overall": target_overall,
        "candidate_pool": candidate_pool,
        "model_signal_summary": signal_summary,
        "decision": decision,
        "interpretation": (
            "Existing candidate pool is exhausted if above_target=0; final-shot work must add a new "
            "classification signal rather than only recombining old labels."
        ),
    }
    _write_json(out_dir / "v25_signal_audit.json", report)
    _write_csv(out_dir / "v25_candidate_pool_ranking.csv", list(candidate_pool.get("ranking", [])))
    _write_csv(out_dir / "v25_model_summary.csv", _model_summary_rows(signal_summary))
    (out_dir / "v25_signal_audit_zh.md").write_text(_render_v25_audit_md(report), encoding="utf-8")
    return report


def _summarize_three_model_agreement(
    base_by_id: Mapping[str, str],
    normalized_by_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    core_names = [name for name in ("clip", "siglip2", "dinov2") if name in normalized_by_model]
    if len(core_names) < 3:
        return {"changed_count": 0, "transition_counts": {}, "top_candidates": []}
    sample_ids = sorted(set(base_by_id).intersection(*(set(normalized_by_model[name]) for name in core_names)))
    changed: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        emotions = [str(normalized_by_model[name][sample_id].get("emotion", "")) for name in core_names]
        if len(set(emotions)) != 1:
            continue
        proposed = emotions[0]
        current = base_by_id[sample_id]
        if not proposed or proposed == current:
            continue
        confidences = [_safe_float(normalized_by_model[name][sample_id].get("confidence")) for name in core_names]
        margins = [_safe_float(normalized_by_model[name][sample_id].get("margin")) for name in core_names]
        changed.append(
            {
                "sample_id": sample_id,
                "current_emotion": current,
                "proposed_emotion": proposed,
                "transition": f"{current}->{proposed}",
                "mean_confidence": round(sum(confidences) / len(confidences), 6),
                "min_confidence": round(min(confidences), 6),
                "mean_margin": round(sum(margins) / len(margins), 6),
                "min_margin": round(min(margins), 6),
            }
        )
    changed.sort(
        key=lambda row: (
            -float(row["mean_confidence"]),
            -float(row["min_confidence"]),
            -float(row["mean_margin"]),
            str(row["sample_id"]),
        )
    )
    transitions = Counter(str(row["transition"]) for row in changed)
    return {
        "changed_count": len(changed),
        "transition_counts": dict(sorted(transitions.items())),
        "top_candidates": changed[:50],
    }


def _best_candidate_top_emotion_share(best_row: Mapping[str, Any]) -> float:
    path = str(best_row.get("json_path", ""))
    if not path or not Path(path).exists():
        return 0.0
    rows = _load_json_list(path)
    dist = Counter(str(row.get("emotion", "")).strip() for row in rows)
    return round(dist.most_common(1)[0][1] / max(1, len(rows)), 6) if dist else 0.0


def _model_summary_rows(signal_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, summary in sorted((signal_summary.get("models") or {}).items()):
        rows.append(
            {
                "model": name,
                "rows": summary.get("rows", 0),
                "changes_vs_base": summary.get("changes_vs_base", 0),
                "high_confidence_changes": summary.get("high_confidence_changes", 0),
                "top_emotion": summary.get("top_emotion", ""),
                "top_emotion_share": summary.get("top_emotion_share", 0.0),
            }
        )
    return rows


def _render_v25_audit_md(report: Mapping[str, Any]) -> str:
    pool = report["candidate_pool"]
    best = pool.get("best") or {}
    decision = report["decision"]
    agreement = report["model_signal_summary"]["three_model_exact_agreement"]
    lines = [
        "# Track2 v25 signal audit",
        "",
        "## 结论",
        "",
        f"- decision: `{decision['decision']}`",
        f"- reasons: `{', '.join(decision['reasons']) if decision['reasons'] else 'none'}`",
        f"- best_existing_candidate: `{best.get('candidate_name', '')}`",
        f"- best_existing_overall: `{float(best.get('overall_expected', 0.0)):.6f}`",
        f"- best_existing_classification: `{float(best.get('classification_expected', 0.0)):.6f}`",
        f"- best_existing_description: `{float(best.get('description_expected', 0.0)):.6f}`",
        f"- candidates_above_0.86: `{pool.get('above_086', 0)}`",
        f"- candidates_above_target_0.89: `{pool.get('above_target', 0)}`",
        f"- three_model_exact_agreement_changes: `{agreement.get('changed_count', 0)}`",
        "",
        "## 模型信号",
        "",
        "| model | rows | changes | high-conf changes | top emotion | top share |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in _model_summary_rows(report["model_signal_summary"]):
        lines.append(
            f"| {row['model']} | {row['rows']} | {row['changes_vs_base']} | "
            f"{row['high_confidence_changes']} | {row['top_emotion']} | {float(row['top_emotion_share']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 三模型一致变化 Top Transitions",
            "",
        ]
    )
    for transition, count in sorted(agreement.get("transition_counts", {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{transition}`: `{count}`")
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- 现有候选池没有达到本地 0.89 gate；继续组合旧候选没有冲第一依据。",
            "- 三模型一致变化主要用于发现新分类候选，但它本身受 public calm prior 影响，不能无门槛全量套用。",
            "- 下一步若继续冲分，应训练/校准一个不塌缩的 classification v26/v27，然后重新跑 v23/v25 gate。",
            "",
        ]
    )
    return "\n".join(lines)


def _load_prediction_entries(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        return [dict(row) for row in payload["entries"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    raise ValueError(f"prediction file must contain entries list: {path}")


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


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
