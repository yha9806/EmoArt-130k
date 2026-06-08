from __future__ import annotations

import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


P1_LIGHT_VERSION = "track1_p1_light_proxy_v2"

POSTER_TERMS = (
    "poster",
    "propaganda",
    "typography",
    "cyrillic",
    "slogan",
    "lettering",
    "headline",
)


def parse_package_spec(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError(f"package spec must be NAME=SUBMISSION_JSON=IMAGE_DIR: {spec}")
    return parts[0].strip(), Path(parts[1].strip()), Path(parts[2].strip())


def load_route_index(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    route_path = Path(path)
    if not route_path.exists():
        return {}
    payload = json.loads(route_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("routes") or payload.get("rows") or []
    else:
        rows = []
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("sample_id"):
            continue
        item = dict(row)
        item["reference_style"] = _reference_style(item)
        index[str(item["sample_id"])] = item
    return index


def load_submission_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Track1 submission JSON must be a list")
    return [dict(row) for row in payload if isinstance(row, dict)]


def load_known_placeholder_sha256s(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    placeholder_path = Path(path)
    if not placeholder_path.exists():
        return {}
    payload = json.loads(placeholder_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw = (
            payload.get("known_placeholders")
            or payload.get("known_placeholder_sha256s")
            or payload.get("blocked_hashes")
            or {}
        )
        if isinstance(raw, dict):
            return {str(key): str(value) for key, value in raw.items()}
        if isinstance(raw, list):
            return {str(value): "known_placeholder" for value in raw}
    return {}


def load_replacement_sample_ids(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    manifest_path = Path(path)
    if not manifest_path.exists():
        return set()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = (
            payload.get("accepted_replacements")
            or payload.get("replacements")
            or payload.get("decisions")
            or payload.get("rows")
            or []
        )
    else:
        rows = []
    sample_ids: set[str] = set()
    for row in rows:
        if isinstance(row, str):
            sample_ids.add(row)
        elif isinstance(row, dict) and row.get("sample_id"):
            decision = str(row.get("decision") or row.get("status") or "accept").lower()
            if "reject" in decision or "hold" in decision or "rerun" in decision:
                continue
            sample_ids.add(str(row["sample_id"]))
    return sample_ids


def evaluate_package(
    package_name: str,
    submission_rows: list[dict[str, Any]],
    image_dir: str | Path,
    *,
    baseline_rows: list[dict[str, Any]],
    baseline_image_dir: str | Path,
    route_index: dict[str, dict[str, Any]] | None = None,
    known_placeholder_sha256s: dict[str, str] | set[str] | None = None,
    replacement_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    image_dir = Path(image_dir)
    baseline_image_dir = Path(baseline_image_dir)
    route_index = route_index or {}
    known_placeholders = _normalise_placeholder_map(known_placeholder_sha256s)
    replacement_sample_ids = replacement_sample_ids or set()
    baseline_by_id = {
        str(row.get("sample_id")): dict(row)
        for row in baseline_rows
        if row.get("sample_id")
    }

    rows = []
    for submission_row in submission_rows:
        sample_id = str(submission_row.get("sample_id") or "")
        if not sample_id:
            continue
        candidate_path = _row_image_path(submission_row, image_dir)
        baseline_path = _row_image_path(baseline_by_id.get(sample_id, submission_row), baseline_image_dir)
        candidate_hash = _sha256(candidate_path)
        baseline_hash = _sha256(baseline_path)
        actual_hash_changed = bool(candidate_hash and candidate_hash != baseline_hash)
        manifest_changed = sample_id in replacement_sample_ids if replacement_sample_ids else actual_hash_changed
        changed = actual_hash_changed
        route = route_index.get(sample_id, {})
        stats = _safe_image_stats(candidate_path)
        placeholder_type = known_placeholders.get(candidate_hash, "")
        caption = str(route.get("caption") or "")
        row = {
            "sample_id": sample_id,
            "path": str(candidate_path),
            "baseline_path": str(baseline_path),
            "changed": changed,
            "actual_hash_changed": actual_hash_changed,
            "manifest_changed": manifest_changed,
            "manifest_actual_gap": actual_hash_changed and not manifest_changed,
            "manifest_nonmaterial_change": manifest_changed and not actual_hash_changed,
            "exists": bool(stats.get("exists")),
            "image_error": str(stats.get("image_error") or ""),
            "sha256": candidate_hash,
            "known_placeholder_type": placeholder_type,
            "reference_style": _reference_style(route),
            "caption": caption,
            "poster_like": _is_poster_like(caption, route),
            "hard_constraint_count": len(route.get("hard_constraints") or []),
            "reference_asset_count": len(route.get("reference_assets") or []),
            "border_score": stats.get("border_score"),
            "edge_density": stats.get("edge_density"),
            "luma_stddev": stats.get("luma_stddev"),
            "file_size_bytes": stats.get("file_size_bytes"),
        }
        rows.append(row)

    summary = _package_summary(package_name, rows)
    return {
        "version": P1_LIGHT_VERSION,
        "package": package_name,
        "summary": summary,
        "rows": rows,
    }


def evaluate_packages(
    package_specs: Iterable[tuple[str, list[dict[str, Any]], str | Path] | tuple[str, list[dict[str, Any]], str | Path, set[str]]],
    *,
    baseline_rows: list[dict[str, Any]],
    baseline_image_dir: str | Path,
    route_index: dict[str, dict[str, Any]] | None = None,
    known_placeholder_sha256s: dict[str, str] | set[str] | None = None,
) -> dict[str, Any]:
    packages = []
    for spec in package_specs:
        name, rows, image_dir, replacement_ids = _normalise_package_spec(spec)
        packages.append(
            evaluate_package(
                name,
                rows,
                image_dir,
                baseline_rows=baseline_rows,
                baseline_image_dir=baseline_image_dir,
                route_index=route_index,
                known_placeholder_sha256s=known_placeholder_sha256s,
                replacement_sample_ids=replacement_ids,
            )
        )
    ranked = sorted(packages, key=_package_sort_key)
    recommended = ranked[0]["package"] if ranked else ""
    return {
        "version": P1_LIGHT_VERSION,
        "summary": {
            "package_count": len(packages),
            "recommended_package": recommended,
        },
        "packages": ranked,
    }


def _normalise_package_spec(
    spec: tuple[str, list[dict[str, Any]], str | Path] | tuple[str, list[dict[str, Any]], str | Path, set[str]],
) -> tuple[str, list[dict[str, Any]], str | Path, set[str]]:
    if len(spec) == 3:
        name, rows, image_dir = spec
        return name, rows, image_dir, set()
    name, rows, image_dir, replacement_ids = spec
    return name, rows, image_dir, set(replacement_ids)


def write_p1_light_reports(report: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")


def _package_summary(package_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [row for row in rows if row.get("changed")]
    actual_changed = [row for row in rows if row.get("actual_hash_changed")]
    manifest_changed = [row for row in rows if row.get("manifest_changed")]
    manifest_missing_actual_changed = [row for row in rows if row.get("manifest_actual_gap")]
    manifest_nonmaterial_changed = [row for row in rows if row.get("manifest_nonmaterial_change")]
    changed_styles = Counter(str(row.get("reference_style") or "unknown") for row in changed)
    known_placeholder_rows = [row for row in rows if row.get("known_placeholder_type")]
    invalid_rows = [row for row in rows if not row.get("exists") or row.get("image_error")]
    poster_like_count = sum(1 for row in changed if row.get("poster_like"))
    border_heavy = [
        row for row in changed
        if row.get("border_score") is not None and float(row["border_score"]) >= 0.35
    ]
    max_style_count = max(changed_styles.values()) if changed_styles else 0
    max_style_concentration = max_style_count / len(changed) if changed else 0.0
    status = _status(
        changed_count=len(changed),
        placeholder_count=len(known_placeholder_rows),
        invalid_count=len(invalid_rows),
        max_style_concentration=max_style_concentration,
        border_heavy_count=len(border_heavy),
    )
    return {
        "package": package_name,
        "status": status,
        "sample_count": len(rows),
        "changed_sample_count": len(changed),
        "actual_changed_sample_count": len(actual_changed),
        "manifest_changed_sample_count": len(manifest_changed),
        "manifest_actual_gap_count": len(manifest_missing_actual_changed),
        "manifest_missing_actual_changed_samples": [
            str(row["sample_id"]) for row in manifest_missing_actual_changed
        ],
        "manifest_nonmaterial_change_count": len(manifest_nonmaterial_changed),
        "manifest_nonmaterial_changed_samples": [
            str(row["sample_id"]) for row in manifest_nonmaterial_changed
        ],
        "known_placeholder_hit_count": len(known_placeholder_rows),
        "known_placeholder_samples": [str(row["sample_id"]) for row in known_placeholder_rows],
        "invalid_image_count": len(invalid_rows),
        "invalid_image_samples": [str(row["sample_id"]) for row in invalid_rows],
        "changed_style_counts": dict(changed_styles.most_common()),
        "changed_poster_like_count": poster_like_count,
        "changed_border_heavy_count": len(border_heavy),
        "max_changed_style_concentration": round(max_style_concentration, 4),
        "p1_light_score": _p1_light_score(
            status=status,
            changed_count=len(changed),
            placeholder_count=len(known_placeholder_rows),
            invalid_count=len(invalid_rows),
            max_style_concentration=max_style_concentration,
            border_heavy_count=len(border_heavy),
        ),
    }


def _status(
    *,
    changed_count: int,
    placeholder_count: int,
    invalid_count: int,
    max_style_concentration: float,
    border_heavy_count: int,
) -> str:
    if placeholder_count:
        return "reject_known_placeholder"
    if invalid_count:
        return "reject_invalid_image"
    if changed_count >= 20 and max_style_concentration >= 0.65:
        return "hold_template_concentration"
    if changed_count >= 20 and border_heavy_count / max(changed_count, 1) >= 0.45:
        return "hold_border_template_risk"
    return "candidate_ok_for_next_gate"


def _p1_light_score(
    *,
    status: str,
    changed_count: int,
    placeholder_count: int,
    invalid_count: int,
    max_style_concentration: float,
    border_heavy_count: int,
) -> float:
    if status.startswith("reject"):
        return 0.0
    score = 1.0
    if status.startswith("hold"):
        score -= 0.25
    if changed_count:
        score -= min(0.25, max(0.0, max_style_concentration - 0.35) * 0.25)
        score -= min(0.20, (border_heavy_count / changed_count) * 0.20)
    score -= min(0.5, placeholder_count * 0.1 + invalid_count * 0.1)
    return round(max(0.0, min(1.0, score)), 4)


def _package_sort_key(report: dict[str, Any]) -> tuple[int, int, float, int, str]:
    summary = report["summary"]
    status = str(summary["status"])
    if status.startswith("reject"):
        status_rank = 2
    elif status.startswith("hold"):
        status_rank = 1
    else:
        status_rank = 0
    no_op_rank = 1 if int(summary.get("changed_sample_count", 0)) == 0 else 0
    return (
        status_rank,
        no_op_rank,
        -float(summary.get("p1_light_score", 0.0)),
        -int(summary.get("changed_sample_count", 0)),
        str(report.get("package") or ""),
    )


def _normalise_placeholder_map(value: dict[str, str] | set[str] | None) -> dict[str, str]:
    if not value:
        return {}
    if isinstance(value, set):
        return {str(item): "known_placeholder" for item in value}
    return {str(key): str(item) for key, item in value.items()}


def _row_image_path(row: dict[str, Any], image_dir: Path) -> Path:
    raw_path = Path(str(row.get("path") or f"{row.get('sample_id', '')}.jpg"))
    return image_dir / raw_path.name


def _safe_image_stats(path: Path) -> dict[str, Any]:
    try:
        return _image_stats(path)
    except Exception as exc:  # pragma: no cover - defensive local artifact guard
        return {"exists": path.exists(), "image_error": str(exc)}


def _image_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    with Image.open(path) as image:
        image = image.convert("RGB")
        gray = image.convert("L").resize((256, 256), Image.Resampling.LANCZOS)
        pixels = _pixels(gray)
        mean = sum(pixels) / len(pixels)
        variance = sum((pixel - mean) ** 2 for pixel in pixels) / len(pixels)
        width, height = image.size
        edge_density = _edge_density(gray)
        border_score = _border_score(gray)
    return {
        "exists": True,
        "width": width,
        "height": height,
        "file_size_bytes": path.stat().st_size,
        "luma_mean": round(mean, 3),
        "luma_stddev": round(variance ** 0.5, 3),
        "edge_density": round(edge_density, 4),
        "border_score": round(border_score, 4),
    }


def _edge_density(gray: Image.Image) -> float:
    width, height = gray.size
    data = _pixels(gray)
    edges = 0
    total = 0
    for y in range(1, height):
        row = y * width
        prev_row = (y - 1) * width
        for x in range(1, width):
            diff = abs(data[row + x] - data[row + x - 1]) + abs(data[row + x] - data[prev_row + x])
            if diff > 50:
                edges += 1
            total += 1
    return edges / max(total, 1)


def _border_score(gray: Image.Image) -> float:
    width, height = gray.size
    crop = 16
    data = _pixels(gray)
    border = []
    center = []
    for y in range(height):
        for x in range(width):
            value = data[y * width + x]
            if x < crop or y < crop or x >= width - crop or y >= height - crop:
                border.append(value)
            elif crop * 3 < x < width - crop * 3 and crop * 3 < y < height - crop * 3:
                center.append(value)
    border_mean = sum(border) / max(len(border), 1)
    center_mean = sum(center) / max(len(center), 1)
    return abs(border_mean - center_mean) / 255.0


def _pixels(image: Image.Image) -> list[int]:
    if hasattr(image, "get_flattened_data"):
        return list(image.get_flattened_data())
    return list(image.getdata())


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return sha256(path.read_bytes()).hexdigest()


def _reference_style(route: dict[str, Any]) -> str:
    for key in ("reference_style", "style_family", "style", "artistic_style"):
        value = route.get(key)
        if value:
            return str(value)
    for note in route.get("reference_asset_notes") or []:
        match = re.search(r"style=([^;]+)", str(note))
        if match:
            return match.group(1).strip()
    return "unknown"


def _is_poster_like(caption: str, route: dict[str, Any]) -> bool:
    lower = caption.lower()
    if any(term in lower for term in POSTER_TERMS):
        return True
    style = _reference_style(route).lower()
    return "socialist realism" in style


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track1 P1-Light Proxy（本地轻量评分器）",
        "",
        "这是本地 gate 报告，不是官方隐藏评测器的完整模拟。",
        "它只用于在最后两次提交机会前排除明显坏包，并帮助选择下一轮进入视觉复核的候选包。",
        "",
    ]
    summary = report.get("summary", {})
    if "package_count" in summary:
        lines.extend(
            [
                "## 总结",
                "",
                f"- 比较包数量：`{summary.get('package_count', 0)}`",
                f"- 当前推荐进入下一道 gate 的包：`{summary.get('recommended_package', '')}`",
                "",
                "## 包级结果",
                "",
                "| 包 | 状态 | actual 变更 | manifest 变更 | actual/manifest 缺口 | placeholder 命中 | 最高风格集中度 | P1-light 分数 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for package in report.get("packages", []):
            item = package.get("summary", {})
            lines.append(
                f"| `{package.get('package', '')}` | `{item.get('status', '')}` | "
                f"{item.get('actual_changed_sample_count', item.get('changed_sample_count', 0))} | "
                f"{item.get('manifest_changed_sample_count', 0)} | "
                f"{item.get('manifest_actual_gap_count', 0)} | "
                f"{item.get('known_placeholder_hit_count', 0)} | "
                f"{item.get('max_changed_style_concentration', 0)} | {item.get('p1_light_score', 0)} |"
            )
    else:
        lines.extend(_package_markdown(report))
    return "\n".join(lines).rstrip() + "\n"


def _package_markdown(report: dict[str, Any]) -> list[str]:
    summary = report.get("summary", {})
    lines = [
        "## 总结",
        "",
        f"- 包：`{report.get('package', '')}`",
        f"- 状态：`{summary.get('status', '')}`",
        f"- 样本数：`{summary.get('sample_count', 0)}`",
        f"- actual hash 变更样本：`{summary.get('actual_changed_sample_count', summary.get('changed_sample_count', 0))}`",
        f"- manifest 标记变更样本：`{summary.get('manifest_changed_sample_count', 0)}`",
        f"- actual/manifest 缺口样本：`{summary.get('manifest_actual_gap_count', 0)}`",
        f"- 已知 placeholder 命中：`{summary.get('known_placeholder_hit_count', 0)}`",
        f"- 无效图片：`{summary.get('invalid_image_count', 0)}`",
        f"- 最高替换风格集中度：`{summary.get('max_changed_style_concentration', 0)}`",
        f"- P1-light 分数：`{summary.get('p1_light_score', 0)}`",
        "",
        "## 替换样本风格分布",
        "",
    ]
    style_counts = summary.get("changed_style_counts") or {}
    if not style_counts:
        lines.append("- 无")
    else:
        for style, count in style_counts.items():
            lines.append(f"- `{style}`: `{count}`")
    return lines
