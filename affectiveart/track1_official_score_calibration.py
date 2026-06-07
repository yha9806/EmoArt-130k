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
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        return [dict(row) for row in rows]
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
