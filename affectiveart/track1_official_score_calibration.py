from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


LOCAL_TO_OFFICIAL_SCALES = {
    "conservative": 0.25,
    "expected": 0.50,
    "optimistic": 1.00,
}


def load_official_score_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("results") or payload.get("rows") or payload if isinstance(payload, dict) else payload
        return [_normalise_official_score_row(row) for row in rows]
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_local_fid_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("packages", payload) if isinstance(payload, dict) else payload
    output = []
    for row in rows:
        package = str(row.get("package") or row.get("name") or "").strip()
        fid_like = _float_or_none(row.get("fid_like"))
        if package and fid_like is not None:
            output.append({**dict(row), "package": package, "fid_like": fid_like})
    return output


def fit_fid_score_model(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    points = []
    for row in rows:
        fid = _float_or_none(row.get("official_fid") or row.get("FID") or row.get("fid"))
        fid_score = _float_or_none(row.get("official_fid_score") or row.get("FID Score") or row.get("fid_score"))
        if fid is not None and fid_score is not None:
            points.append((fid, fid_score))
    if len(points) < 2:
        raise ValueError("at least two official FID/FID Score rows are required")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        raise ValueError("official FID rows must contain varied FID values")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denom
    intercept = y_mean - slope * x_mean
    residuals = [y - (intercept + slope * x) for x, y in points]
    ss_res = sum(value * value for value in residuals)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "kind": "linear_fid_to_fid_score",
        "count": len(points),
        "slope": round(slope, 8),
        "intercept": round(intercept, 8),
        "r2": round(r2, 6),
        "fid_min": round(min(xs), 6),
        "fid_max": round(max(xs), 6),
        "fid_score_min": round(min(ys), 6),
        "fid_score_max": round(max(ys), 6),
    }


def fit_local_to_official_fid_model(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    points = []
    for row in rows:
        local_fid_like = _float_or_none(row.get("local_fid_like"))
        official_fid = _float_or_none(row.get("official_fid") or row.get("FID") or row.get("fid"))
        if local_fid_like is not None and official_fid is not None:
            points.append((local_fid_like, official_fid))
    if len(points) < 2:
        return {
            "kind": "insufficient_own_anchors",
            "count": len(points),
            "proxy_direction": "unknown",
            "warning": "At least two own submissions with local_fid_like and official_fid are required.",
        }
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        return {
            "kind": "degenerate_own_anchor_fit",
            "count": len(points),
            "proxy_direction": "unknown",
            "warning": "Own local_fid_like anchors do not vary.",
        }
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denom
    intercept = y_mean - slope * x_mean
    residuals = [y - (intercept + slope * x) for x, y in points]
    ss_res = sum(value * value for value in residuals)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    if slope > 0:
        proxy_direction = "aligned"
    elif slope < 0:
        proxy_direction = "anti_correlated"
    else:
        proxy_direction = "flat"
    return {
        "kind": "linear_local_fid_like_to_official_fid",
        "count": len(points),
        "slope": round(slope, 8),
        "intercept": round(intercept, 8),
        "r2": round(r2, 6),
        "local_fid_like_min": round(min(xs), 6),
        "local_fid_like_max": round(max(xs), 6),
        "official_fid_min": round(min(ys), 6),
        "official_fid_max": round(max(ys), 6),
        "proxy_direction": proxy_direction,
    }


def predict_fid_score(official_fid: float, model: dict[str, Any]) -> float:
    raw = float(model["intercept"]) + float(model["slope"]) * official_fid
    return round(min(1.0, max(0.0, raw)), 6)


def predict_official_fid_from_local(local_fid_like: float, model: dict[str, Any]) -> float:
    raw = float(model["intercept"]) + float(model["slope"]) * local_fid_like
    return round(max(0.0, raw), 6)


def official_overall(fid_score: float, aas: float) -> float:
    return round(0.5 * float(fid_score) + 0.5 * float(aas), 6)


def calibrate_track1_packages(
    official_rows: list[dict[str, Any]],
    local_fid_rows: list[dict[str, Any]],
    *,
    anchor_package: str,
    aas_assumption: float | None = None,
) -> dict[str, Any]:
    fid_score_model = fit_fid_score_model(official_rows)
    local_to_official_model = fit_local_to_official_fid_model(official_rows)
    anchor = _find_anchor(official_rows, anchor_package)
    if aas_assumption is not None:
        anchor["official_aas"] = float(aas_assumption)
    observed_by_package = _observed_official_by_package(official_rows)
    package_rows = []
    for row in local_fid_rows:
        package = str(row["package"])
        fid_like = float(row["fid_like"])
        observed = observed_by_package.get(package)
        if observed:
            fid_score = float(observed["official_fid_score"])
            aas = float(observed["official_aas"])
            overall = official_overall(fid_score, aas)
            package_rows.append(
                {
                    "package": package,
                    "local_fid_like": round(fid_like, 6),
                    "local_delta_vs_anchor": round(fid_like - float(anchor["local_fid_like"]), 6),
                    "scenarios": {
                        "observed": {
                            "projected_official_fid": round(float(observed["official_fid"]), 6),
                            "projected_fid_score": round(fid_score, 6),
                            "projected_aas": round(aas, 6),
                            "projected_overall": overall,
                            "submission_id": observed["submission_id"],
                        }
                    },
                    "overall_lower": overall,
                    "overall_expected": overall,
                    "overall_upper": overall,
                    "fid_score_expected": round(fid_score, 6),
                    "official_fid_expected": round(float(observed["official_fid"]), 6),
                    "projection_method": "observed_official",
                    "ranking_note": "observed_official",
                }
            )
            continue
        if local_to_official_model.get("kind") == "linear_local_fid_like_to_official_fid":
            projected_fid = predict_official_fid_from_local(fid_like, local_to_official_model)
            projected_fid_score = predict_fid_score(projected_fid, fid_score_model)
            projected_overall = official_overall(projected_fid_score, float(anchor["official_aas"]))
            package_rows.append(
                {
                    "package": package,
                    "local_fid_like": round(fid_like, 6),
                    "local_delta_vs_anchor": round(fid_like - float(anchor["local_fid_like"]), 6),
                    "scenarios": {
                        "own_anchor_fit": {
                            "projected_official_fid": projected_fid,
                            "projected_fid_score": projected_fid_score,
                            "projected_aas": round(float(anchor["official_aas"]), 6),
                            "projected_overall": projected_overall,
                        }
                    },
                    "overall_lower": projected_overall,
                    "overall_expected": projected_overall,
                    "overall_upper": projected_overall,
                    "fid_score_expected": projected_fid_score,
                    "official_fid_expected": projected_fid,
                    "projection_method": "own_anchor_fit",
                    "ranking_note": _ranking_note_from_official_fid(projected_fid, float(anchor["official_fid"])),
                }
            )
            continue
        local_delta = fid_like - float(anchor["local_fid_like"])
        scenarios = {}
        overall_values = []
        for name, scale in LOCAL_TO_OFFICIAL_SCALES.items():
            projected_fid = float(anchor["official_fid"]) + scale * local_delta
            projected_fid_score = predict_fid_score(projected_fid, fid_score_model)
            projected_overall = official_overall(projected_fid_score, float(anchor["official_aas"]))
            scenarios[name] = {
                "local_to_official_delta_scale": scale,
                "projected_official_fid": round(projected_fid, 6),
                "projected_fid_score": projected_fid_score,
                "projected_aas": round(float(anchor["official_aas"]), 6),
                "projected_overall": projected_overall,
            }
            overall_values.append(projected_overall)
        package_rows.append(
            {
                "package": package,
                "local_fid_like": round(fid_like, 6),
                "local_delta_vs_anchor": round(local_delta, 6),
                "scenarios": scenarios,
                "overall_lower": round(min(overall_values), 6),
                "overall_expected": scenarios["expected"]["projected_overall"],
                "overall_upper": round(max(overall_values), 6),
                "fid_score_expected": scenarios["expected"]["projected_fid_score"],
                "official_fid_expected": scenarios["expected"]["projected_official_fid"],
                "projection_method": "anchor_delta_heuristic",
                "ranking_note": _ranking_note(local_delta),
            }
        )
    package_rows.sort(key=lambda item: (-float(item["overall_expected"]), float(item["local_fid_like"]), str(item["package"])))
    return {
        "method": "track1_local_shadow_score_calibration_v1",
        "warning": (
            "LOCAL SHADOW SCORE ONLY: this is not the official Codabench evaluator or an official score. "
            "With one own local-to-official anchor, package scores are directional and low-confidence."
        ),
        "fid_score_model": fid_score_model,
        "local_to_official_fid_model": local_to_official_model,
        "anchor": anchor,
        "calibration_confidence": _calibration_confidence(official_rows),
        "packages": package_rows,
        "leaderboard_context": _leaderboard_context(official_rows, anchor),
    }


def write_calibration_reports(report: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")


def merge_local_anchor_metadata(
    official_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged_by_id = {
        str(row.get("submission_id") or row.get("id") or ""): dict(row)
        for row in official_rows
        if row.get("submission_id") or row.get("id")
    }
    ordered_ids = list(merged_by_id)
    for anchor in anchor_rows:
        submission_id = str(anchor.get("submission_id") or anchor.get("id") or "")
        if not submission_id:
            continue
        if submission_id not in merged_by_id:
            merged_by_id[submission_id] = dict(anchor)
            ordered_ids.append(submission_id)
            continue
        merged = dict(merged_by_id[submission_id])
        for key in ("local_package", "local_fid_like", "source_type", "notes"):
            if anchor.get(key) not in (None, ""):
                merged[key] = anchor[key]
        merged_by_id[submission_id] = merged
    return [merged_by_id[submission_id] for submission_id in ordered_ids]


def build_scorer_reproducibility_audit(official_rows: list[dict[str, Any]]) -> dict[str, Any]:
    formula_errors = []
    formula_rows = []
    for row in official_rows:
        official = _float_or_none(row.get("official_overall"))
        fid_score = _float_or_none(row.get("official_fid_score"))
        aas = _float_or_none(row.get("official_aas"))
        if None in (official, fid_score, aas):
            continue
        reproduced = 0.5 * float(fid_score) + 0.5 * float(aas)
        error = reproduced - float(official)
        formula_errors.append(error)
        formula_rows.append(
            {
                "submission_id": str(row.get("submission_id") or row.get("id") or ""),
                "participant": row.get("participant", ""),
                "official_overall": official,
                "reproduced_overall": round(reproduced, 10),
                "error": round(error, 12),
            }
        )

    fid_score_model = fit_fid_score_model(official_rows)
    fid_residuals = _fid_score_residuals(official_rows, fid_score_model)
    local_model = fit_local_to_official_fid_model(official_rows)
    own_anchor_rows = _own_local_official_anchor_rows(official_rows)
    return {
        "method": "track1_scorer_reproducibility_audit_v1",
        "official_formula_reproduction": {
            "count": len(formula_errors),
            "max_abs_error": round(max((abs(value) for value in formula_errors), default=0.0), 12),
            "mean_abs_error": round(mean(abs(value) for value in formula_errors), 12) if formula_errors else 0.0,
            "rows": formula_rows,
        },
        "fid_score_reproduction": {
            **fid_score_model,
            "mae": round(mean(abs(row["residual"]) for row in fid_residuals), 8) if fid_residuals else 0.0,
            "rmse": round((mean(row["residual"] ** 2 for row in fid_residuals)) ** 0.5, 8) if fid_residuals else 0.0,
            "max_abs_residual": round(max((abs(row["residual"]) for row in fid_residuals), default=0.0), 8),
            "worst_rows": sorted(fid_residuals, key=lambda row: abs(float(row["residual"])), reverse=True)[:5],
        },
        "local_proxy_reproduction": {
            "own_anchor_count": len(own_anchor_rows),
            "model_kind": local_model.get("kind"),
            "proxy_direction": local_model.get("proxy_direction", "unknown"),
            "readiness": _local_proxy_readiness(local_model, len(own_anchor_rows)),
            "reason": _local_proxy_readiness_reason(local_model, len(own_anchor_rows)),
            "model": local_model,
            "own_anchor_rows": own_anchor_rows,
        },
    }


def write_scorer_reproducibility_audit(report: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_reproducibility_audit_md(report), encoding="utf-8")


def _find_anchor(rows: list[dict[str, Any]], anchor_package: str) -> dict[str, Any]:
    for row in rows:
        local_package = str(row.get("local_package") or row.get("package") or "").strip()
        if local_package != anchor_package:
            continue
        official_fid = _float_or_none(row.get("official_fid"))
        official_fid_score_value = _float_or_none(row.get("official_fid_score"))
        official_aas_value = _float_or_none(row.get("official_aas"))
        local_fid_like = _float_or_none(row.get("local_fid_like"))
        if None not in (official_fid, official_fid_score_value, official_aas_value, local_fid_like):
            return {
                "anchor_package": anchor_package,
                "participant": row.get("participant", ""),
                "submission_id": str(row.get("submission_id") or row.get("id") or ""),
                "file_name": row.get("file_name", ""),
                "official_overall": _float_or_none(row.get("official_overall")),
                "official_fid": official_fid,
                "official_fid_score": official_fid_score_value,
                "official_aas": official_aas_value,
                "local_fid_like": local_fid_like,
            }
    raise ValueError(f"no complete local official anchor found for package: {anchor_package}")


def _normalise_official_score_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    if "scores" not in row:
        return dict(row)
    scores = {
        str(score.get("column_key")): _float_or_none(score.get("score"))
        for score in row.get("scores") or []
        if isinstance(score, dict) and score.get("column_key")
    }
    return {
        "participant": row.get("owner") or row.get("participant") or "",
        "submission_id": str(row.get("id") or row.get("pk") or row.get("submission_id") or ""),
        "file_name": row.get("filename") or row.get("file_name") or "",
        "date": row.get("created_when") or row.get("date") or "",
        "official_overall": scores.get("track1_overall"),
        "official_fid": scores.get("fid"),
        "official_fid_score": scores.get("fid_score"),
        "official_aas": scores.get("aas"),
        "content_alignment": scores.get("content_alignment"),
        "style_alignment": scores.get("style_alignment"),
        "attribute_alignment": scores.get("attribute_alignment"),
    }


def _fid_score_residuals(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        fid = _float_or_none(row.get("official_fid") or row.get("FID") or row.get("fid"))
        fid_score = _float_or_none(row.get("official_fid_score") or row.get("FID Score") or row.get("fid_score"))
        if fid is None or fid_score is None:
            continue
        predicted = float(model["intercept"]) + float(model["slope"]) * fid
        residual = fid_score - predicted
        output.append(
            {
                "submission_id": str(row.get("submission_id") or row.get("id") or ""),
                "participant": row.get("participant", ""),
                "official_fid": round(fid, 6),
                "official_fid_score": round(fid_score, 10),
                "predicted_fid_score": round(predicted, 10),
                "residual": round(residual, 10),
            }
        )
    return output


def _own_local_official_anchor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        local_fid_like = _float_or_none(row.get("local_fid_like"))
        official_fid = _float_or_none(row.get("official_fid") or row.get("FID") or row.get("fid"))
        if local_fid_like is None or official_fid is None:
            continue
        output.append(
            {
                "local_package": str(row.get("local_package") or row.get("package") or ""),
                "submission_id": str(row.get("submission_id") or row.get("id") or ""),
                "official_fid": round(official_fid, 6),
                "official_fid_score": _float_or_none(row.get("official_fid_score")),
                "official_aas": _float_or_none(row.get("official_aas")),
                "local_fid_like": round(local_fid_like, 6),
            }
        )
    return output


def _local_proxy_readiness(model: dict[str, Any], own_anchor_count: int) -> str:
    if own_anchor_count < 3:
        return "insufficient_own_anchors"
    if model.get("proxy_direction") != "aligned":
        return "proxy_direction_unstable"
    return "ready_for_directional_ranking"


def _local_proxy_readiness_reason(model: dict[str, Any], own_anchor_count: int) -> str:
    if own_anchor_count < 3:
        return "至少需要 3 个自有提交锚点，并且每个锚点都要同时保留本地包特征和官方组件分数。"
    if model.get("proxy_direction") != "aligned":
        return "本地 proxy 方向没有和官方 FID 对齐，因此不能安全用于包排序。"
    return "锚点数量足够做方向性排序，但这仍然不是官方评测器。"


def _render_reproducibility_audit_md(report: dict[str, Any]) -> str:
    formula = report["official_formula_reproduction"]
    fid_fit = report["fid_score_reproduction"]
    local = report["local_proxy_reproduction"]
    lines = [
        "# Track1 Scorer Reproducibility Audit",
        "",
        "这是本地评分器复现性审计，不是官方隐藏评测器。",
        "",
        "## 结论",
        "",
        f"- 官方总分公式复现样本数：`{formula['count']}`",
        f"- 官方总分公式最大绝对误差：`{formula['max_abs_error']}`",
        f"- FID 到 FID Score 拟合 R2：`{fid_fit['r2']}`",
        f"- FID 到 FID Score 拟合 RMSE：`{fid_fit['rmse']}`",
        f"- 本地 local/official 锚点数：`{local['own_anchor_count']}`",
        f"- 本地 proxy 方向：`{local['proxy_direction']}`",
        f"- 本地 proxy 就绪度：`{local['readiness']}`",
        f"- 原因：{local['reason']}",
        "",
        "## 本地锚点",
        "",
        "| package | submission | local fid_like | official FID | official FID Score | official AAS |",
        "|---|---|---:|---:|---:|---:|",
    ]
    anchors = local.get("own_anchor_rows") or []
    if not anchors:
        lines.append("| none |  |  |  |  |  |")
    for row in anchors:
        lines.append(
            f"| `{row.get('local_package', '')}` | `{row.get('submission_id', '')}` | "
            f"{row.get('local_fid_like', '')} | {row.get('official_fid', '')} | "
            f"{row.get('official_fid_score', '')} | {row.get('official_aas', '')} |"
        )
    lines += [
        "",
        "## FID Score 拟合最差残差",
        "",
        "| participant | submission | FID | actual FID Score | predicted | residual |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in fid_fit.get("worst_rows") or []:
        lines.append(
            f"| `{row.get('participant', '')}` | `{row.get('submission_id', '')}` | "
            f"{row.get('official_fid', '')} | {row.get('official_fid_score', '')} | "
            f"{row.get('predicted_fid_score', '')} | {row.get('residual', '')} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "- 官方 `overall = 0.5 * FID Score + 0.5 * AAS` 可以精确复现。",
        "- FID Score 对 FID 的公开映射可以高置信近似，但不是官方归一化函数源码。",
        "- 本地 proxy 目前不能当作可复现官方评分器；它只能做提交前风险排序。",
        "- 要让本地 scorer 真正可校准，至少还需要 3 个以上自有提交锚点，并且每个锚点要保留本地包特征。",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _observed_official_by_package(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    observed = {}
    for row in rows:
        package = str(row.get("local_package") or row.get("package") or "").strip()
        official_fid = _float_or_none(row.get("official_fid"))
        official_fid_score_value = _float_or_none(row.get("official_fid_score"))
        official_aas_value = _float_or_none(row.get("official_aas"))
        if package and None not in (official_fid, official_fid_score_value, official_aas_value):
            observed[package] = {
                "submission_id": str(row.get("submission_id") or row.get("id") or ""),
                "official_fid": official_fid,
                "official_fid_score": official_fid_score_value,
                "official_aas": official_aas_value,
            }
    return observed


def _calibration_confidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    own_anchor_count = 0
    for row in rows:
        if _float_or_none(row.get("local_fid_like")) is not None and _float_or_none(row.get("official_fid")) is not None:
            own_anchor_count += 1
    if own_anchor_count >= 3:
        level = "medium"
    elif own_anchor_count >= 2:
        level = "low_medium"
    else:
        level = "low"
    return {
        "level": level,
        "own_local_official_anchor_count": own_anchor_count,
        "reason": "Track1 has too few own submissions with both local fid_like and official component scores.",
    }


def _leaderboard_context(rows: list[dict[str, Any]], anchor: dict[str, Any]) -> dict[str, Any]:
    scored = []
    for row in rows:
        overall = _float_or_none(row.get("official_overall"))
        if overall is None:
            fid_score = _float_or_none(row.get("official_fid_score"))
            aas = _float_or_none(row.get("official_aas"))
            if fid_score is not None and aas is not None:
                overall = official_overall(fid_score, aas)
        if overall is None:
            continue
        scored.append(
            {
                "participant": row.get("participant", ""),
                "submission_id": str(row.get("submission_id") or row.get("id") or ""),
                "official_overall": round(overall, 6),
                "official_fid": _float_or_none(row.get("official_fid")),
                "official_fid_score": _float_or_none(row.get("official_fid_score")),
                "official_aas": _float_or_none(row.get("official_aas")),
            }
        )
    scored.sort(key=lambda row: (-float(row["official_overall"]), str(row["participant"])))
    best = scored[0] if scored else {}
    anchor_overall = _float_or_none(anchor.get("official_overall"))
    if anchor_overall is None:
        anchor_overall = official_overall(float(anchor["official_fid_score"]), float(anchor["official_aas"]))
    return {
        "row_count": len(scored),
        "best_public_row": best,
        "anchor_official_overall": round(anchor_overall, 6),
        "gap_to_best_public_overall": round(float(best.get("official_overall", anchor_overall)) - anchor_overall, 6)
        if best
        else 0.0,
    }


def _ranking_note(local_delta: float) -> str:
    if local_delta < -0.5:
        return "local_proxy_better_than_anchor"
    if local_delta > 0.5:
        return "local_proxy_worse_than_anchor"
    return "local_proxy_similar_to_anchor"


def _ranking_note_from_official_fid(projected_fid: float, anchor_fid: float) -> str:
    delta = projected_fid - anchor_fid
    if delta < -0.5:
        return "projected_official_fid_better_than_anchor"
    if delta > 0.5:
        return "projected_official_fid_worse_than_anchor"
    return "projected_official_fid_similar_to_anchor"


def _render_md(report: dict[str, Any]) -> str:
    anchor = report["anchor"]
    confidence = report["calibration_confidence"]
    context = report["leaderboard_context"]
    model = report["fid_score_model"]
    local_model = report.get("local_to_official_fid_model", {})
    lines = [
        "# Track1 Local Shadow Score Calibration",
        "",
        "> This is a local shadow calibration, not the official Codabench evaluator.",
        "",
        "## Summary",
        "",
        f"- Calibration confidence: `{confidence['level']}`",
        f"- Own local/official anchors: `{confidence['own_local_official_anchor_count']}`",
        f"- Anchor package: `{anchor['anchor_package']}`",
        f"- Anchor official: FID `{anchor['official_fid']}`, FID Score `{anchor['official_fid_score']}`, AAS `{anchor['official_aas']}`",
        f"- Anchor local fid_like: `{anchor['local_fid_like']}`",
        f"- Best public overall in ledger: `{context.get('best_public_row', {}).get('official_overall', '')}`",
        f"- Gap to best public overall: `{context['gap_to_best_public_overall']}`",
        "",
        "## FID To FID Score Fit",
        "",
        f"- Rows: `{model['count']}`",
        f"- slope: `{model['slope']}`",
        f"- intercept: `{model['intercept']}`",
        f"- r2: `{model['r2']}`",
        "",
        "## Local Proxy To Official FID",
        "",
        f"- model: `{local_model.get('kind', '')}`",
        f"- own anchors: `{local_model.get('count', 0)}`",
        f"- proxy direction: `{local_model.get('proxy_direction', 'unknown')}`",
    ]
    if "slope" in local_model:
        lines += [
            f"- slope: `{local_model['slope']}`",
            f"- intercept: `{local_model['intercept']}`",
            f"- r2: `{local_model['r2']}`",
        ]
    warning = local_model.get("warning")
    if warning:
        lines.append(f"- warning: {warning}")
    lines += [
        "",
        "## Package Ranking",
        "",
        "| package | local fid_like | expected official FID | expected overall | lower | upper | expected FID Score | method | note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["packages"]:
        lines.append(
            f"| `{row['package']}` | {row['local_fid_like']:.6f} | {row.get('official_fid_expected', 0):.6f} | "
            f"{row['overall_expected']:.6f} | "
            f"{row['overall_lower']:.6f} | {row['overall_upper']:.6f} | {row['fid_score_expected']:.6f} | "
            f"{row.get('projection_method', '')} | "
            f"{row['ranking_note']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Use this to rank candidate packages before spending Codabench submissions.",
        "- Do not treat the expected score as an official score.",
        "- Public leaderboard rows calibrate the FID-score scale only; they are not our hidden-test labels.",
        "- More own submissions with component scores are required before this can become a medium-confidence scorer.",
    ]
    return "\n".join(lines).rstrip() + "\n"


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
