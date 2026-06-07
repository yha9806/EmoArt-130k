from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import linalg


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as payload:
        ids = [_decode_id(value) for value in payload["ids"].tolist()]
        features = np.asarray(payload["features"], dtype=np.float64)
    if features.ndim != 2 or len(ids) != features.shape[0]:
        raise ValueError(f"invalid feature cache shape: {path}")
    return {"ids": ids, "features": features, "path": str(path)}


def load_shortlist_sample_ids(path: str | Path | None, *, max_candidates: int = 0) -> list[str]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("review_queue") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("shortlist must be a list or contain review_queue")
    sample_ids = []
    for row in rows:
        if isinstance(row, dict) and row.get("sample_id"):
            sample_ids.append(str(row["sample_id"]))
    if max_candidates > 0:
        sample_ids = sample_ids[:max_candidates]
    return sample_ids


def greedy_hybrid_fid_search(
    *,
    current_ids: list[str],
    current_features: np.ndarray,
    candidate_ids: list[str],
    candidate_features: np.ndarray,
    reference_features: np.ndarray,
    candidate_pool: Iterable[str] | None = None,
    candidate_label: str = "candidate",
    max_replacements: int = 20,
    min_improvement: float = 1e-6,
) -> dict[str, Any]:
    current_matrix = np.asarray(current_features, dtype=np.float64)
    candidate_matrix = np.asarray(candidate_features, dtype=np.float64)
    reference_matrix = np.asarray(reference_features, dtype=np.float64)
    current_index = {sample_id: index for index, sample_id in enumerate(current_ids)}
    candidate_index = {sample_id: index for index, sample_id in enumerate(candidate_ids)}
    common_ids = [sample_id for sample_id in current_ids if sample_id in candidate_index]
    pool = list(candidate_pool or common_ids)
    pool = [sample_id for sample_id in pool if sample_id in current_index and sample_id in candidate_index]
    pool = sorted(dict.fromkeys(pool))

    reference_stats = frechet_stats(reference_matrix)
    working = current_matrix.copy()
    baseline_fid = frechet_distance(frechet_stats(working), reference_stats)
    best_fid = baseline_fid
    selected: list[dict[str, Any]] = []
    rejected_steps: list[dict[str, Any]] = []
    remaining = set(pool)
    for step in range(1, max(0, max_replacements) + 1):
        best_option: dict[str, Any] | None = None
        for sample_id in sorted(remaining):
            trial = working.copy()
            trial[current_index[sample_id]] = candidate_matrix[candidate_index[sample_id]]
            fid_value = frechet_distance(frechet_stats(trial), reference_stats)
            improvement = best_fid - fid_value
            option = {
                "sample_id": sample_id,
                "candidate_label": candidate_label,
                "fid_like": round(float(fid_value), 6),
                "improvement": round(float(improvement), 6),
            }
            if best_option is None or (improvement, sample_id) > (float(best_option["improvement"]), str(best_option["sample_id"])):
                best_option = option
        if best_option is None:
            break
        if float(best_option["improvement"]) <= min_improvement:
            rejected_steps.append(
                {
                    "step": step,
                    "reason": "no_remaining_candidate_improves_package_fid_like",
                    "best_non_improving_option": best_option,
                }
            )
            break
        sample_id = str(best_option["sample_id"])
        working[current_index[sample_id]] = candidate_matrix[candidate_index[sample_id]]
        remaining.remove(sample_id)
        best_fid = float(best_option["fid_like"])
        selected.append({"step": step, **best_option})

    return {
        "method": {
            "name": "track1_greedy_inception_hybrid_fid_search_v1",
            "warning": (
                "Local Inception feature search only. It optimizes a local FID-like proxy and does not check AAS, "
                "caption correctness, or official hidden evaluator settings."
            ),
            "candidate_label": candidate_label,
            "max_replacements": max_replacements,
            "min_improvement": min_improvement,
        },
        "summary": {
            "candidate_pool_count": len(pool),
            "selected_replacement_count": len(selected),
            "baseline_fid_like": round(float(baseline_fid), 6),
            "best_fid_like": round(float(best_fid), 6),
            "total_improvement": round(float(baseline_fid - best_fid), 6),
        },
        "selected_replacements": selected,
        "stop_conditions": rejected_steps,
    }


def write_hybrid_search_reports(
    report: dict[str, Any],
    *,
    json_path: str | Path,
    md_path: str | Path,
    replacement_manifest_path: str | Path | None = None,
    candidate_image_dir: str | Path | None = None,
) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_hybrid_search_markdown(report), encoding="utf-8")
    if replacement_manifest_path:
        _write_probe_manifest(report, Path(replacement_manifest_path), candidate_image_dir=candidate_image_dir)


def render_hybrid_search_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Track1 Greedy Inception Hybrid FID Search",
        "",
        "这是本地 Inception FID-like 子集搜索，不是官方评分器，也不是最终替换清单。",
        "",
        "## Summary",
        "",
        f"- candidate pool: `{summary['candidate_pool_count']}`",
        f"- selected replacements: `{summary['selected_replacement_count']}`",
        f"- baseline fid_like: `{summary['baseline_fid_like']}`",
        f"- best fid_like: `{summary['best_fid_like']}`",
        f"- total improvement: `{summary['total_improvement']}`",
        "",
        "## Selected Replacements",
        "",
    ]
    if not report.get("selected_replacements"):
        lines.append("- none")
    for row in report.get("selected_replacements", []):
        lines.append(
            f"{row['step']}. `{row['sample_id']}` candidate=`{row['candidate_label']}` "
            f"fid_like=`{row['fid_like']}` improvement=`{row['improvement']}`"
        )
    lines.extend(["", "## Stop Conditions", ""])
    if not report.get("stop_conditions"):
        lines.append("- max replacements reached or no candidates were provided")
    for row in report.get("stop_conditions", []):
        option = row.get("best_non_improving_option", {})
        lines.append(
            f"- step `{row['step']}`: {row['reason']}; best_non_improving=`{option.get('sample_id')}` "
            f"improvement=`{option.get('improvement')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def frechet_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    return features.mean(axis=0), np.cov(features, rowvar=False)


def frechet_distance(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> float:
    mu1, sigma1 = left
    mu2, sigma2 = right
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return max(0.0, float(value))


def _write_probe_manifest(report: dict[str, Any], path: Path, *, candidate_image_dir: str | Path | None) -> None:
    candidate_image_dir = Path(candidate_image_dir) if candidate_image_dir else None
    accepted = []
    for row in report.get("selected_replacements", []):
        sample_id = str(row["sample_id"])
        accepted.append(
            {
                "sample_id": sample_id,
                "decision": "accept",
                "candidate_image": str(candidate_image_dir / f"{sample_id}.jpg") if candidate_image_dir else "",
                "local_fid_like_improvement": row["improvement"],
                "final_status": "probe_only_not_final_human_unconfirmed",
            }
        )
    payload = {
        "method": "track1_greedy_inception_hybrid_fid_probe_manifest_v1",
        "warning": "Probe-only manifest for local package FID-like sanity. Not a final replacement manifest.",
        "accepted_replacements": accepted,
        "summary": report["summary"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _decode_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


__all__ = [
    "frechet_distance",
    "frechet_stats",
    "greedy_hybrid_fid_search",
    "load_feature_cache",
    "load_shortlist_sample_ids",
    "render_hybrid_search_markdown",
    "write_hybrid_search_reports",
]
