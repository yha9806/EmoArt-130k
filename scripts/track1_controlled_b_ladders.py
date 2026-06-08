#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track1_candidate_package import build_candidate_package  # noqa: E402
from affectiveart.track1_controlled_b import (  # noqa: E402
    ControlledBCandidate,
    filter_current_identical_candidates,
    filter_placeholder_candidates,
    select_controlled_ladder,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "track1_v5_smoke_20260608" / "controlled_b_ladders_20260608"
DEFAULT_V3_SUBMISSION = ROOT / "submissions" / "track1_candidate_v3_gate7_20260606" / "submission.json"
DEFAULT_V3_IMAGES = ROOT / "submissions" / "track1_candidate_v3_gate7_20260606" / "images"
DEFAULT_CURRENT_IMAGES = ROOT / "submissions" / "track1" / "images"
DEFAULT_FULL1000_IMAGES = (
    ROOT / "submissions" / "track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605" / "images"
)
DEFAULT_V5_ACCEPT48 = (
    ROOT
    / "experiments"
    / "track1_v5_smoke_20260608"
    / "human_accept48_20260608"
    / "track1_v5_smoke_human_accept48_replacement_manifest.json"
)
DEFAULT_SHORTLIST = ROOT / "experiments" / "track1_v4_fid_distribution_champion_20260607" / "distribution_shortlist_top80.json"


CODE_LIKE_PLACEHOLDER_SHA256S = {
    # track1_0730/0735/0740/0772 provider JSON/code screenshots discovered in the 2026-06-08 ladder audit.
    "a7680246fd497c3d6b68a457dd2044527faef8a67d54f32316d4d6c59495884f",
    "49a8805c4cf91e9772a56197233bffd69231fc4de794162961d296953044ab33",
    "c98f1b8f5f976b443b2def008d1264fdbbe457c565b6159d61cda211f0fd55bb",
    "d856211752c3b35dcfd722399495f3d6f34f0609b1236fcbce65d3a09e9f44c9",
    # Re-encoded versions from the first unsafe Controlled-B build. Kept here so reruns cannot recycle them.
    "e167af1b65049732841d017e826f31339b3db52435f5890e75282376c22ad2eb",
    "28614dfdcf8012ec7628be7146df711cb8a68613a55b9163de22e336b237c02f",
    "3a986e3ad84b8cf330271ea2adcd1c0a462d9de27a6db6cccd9f1109165b473d",
}
CURRENT_RESCUE_SAMPLE_IDS = {"track1_0735", "track1_0740"}


PACKAGE_DIFF_SOURCES = [
    (
        "safe_subset_package_diff",
        ROOT / "submissions" / "track1_candidate_v2_safe_subset_20260606" / "images",
        85_000,
        "safe_subset",
    ),
    (
        "strict_subset_package_diff",
        ROOT / "submissions" / "track1_candidate_v2_strict_subset_20260606" / "images",
        82_000,
        "strict_subset",
    ),
    (
        "pilot100_human24_package_diff",
        ROOT / "submissions" / "track1_candidate_pilot100_human24_20260603" / "images",
        78_000,
        "human_review",
    ),
    (
        "scaled_high_precision_package_diff",
        ROOT / "submissions" / "track1_candidate_scaled_high_precision_20260602" / "images",
        76_000,
        "human_review",
    ),
    (
        "scaled_expanded_package_diff",
        ROOT / "submissions" / "track1_candidate_scaled_expanded_20260603" / "images",
        74_000,
        "human_review",
    ),
    (
        "human_v25_wave2_package_diff",
        ROOT / "submissions" / "track1_candidate_human_v25_wave2_redteam_20260514" / "images",
        72_000,
        "human_review",
    ),
    (
        "human_v23_fullbank_package_diff",
        ROOT / "submissions" / "track1_candidate_human_v23_fullbank_override_20260514" / "images",
        70_000,
        "human_review",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Track1 controlled-B ladder packages on the v3_gate7 base.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--v3-submission-json", type=Path, default=DEFAULT_V3_SUBMISSION)
    parser.add_argument("--v3-image-dir", type=Path, default=DEFAULT_V3_IMAGES)
    parser.add_argument("--current-image-dir", type=Path, default=DEFAULT_CURRENT_IMAGES)
    parser.add_argument("--full1000-image-dir", type=Path, default=DEFAULT_FULL1000_IMAGES)
    parser.add_argument("--v5-accept48-manifest", type=Path, default=DEFAULT_V5_ACCEPT48)
    parser.add_argument("--shortlist-json", type=Path, default=DEFAULT_SHORTLIST)
    parser.add_argument("--targets", type=int, nargs="+", default=[80, 120, 160])
    parser.add_argument("--submissions-dir", type=Path, default=ROOT / "submissions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = collect_candidates(
        v5_accept48_manifest=args.v5_accept48_manifest,
        shortlist_json=args.shortlist_json,
        v3_image_dir=args.v3_image_dir,
        current_image_dir=args.current_image_dir,
    )
    placeholder_filtered, blocked_placeholders = filter_placeholder_candidates(candidates, CODE_LIKE_PLACEHOLDER_SHA256S)
    filtered, blocked_current = filter_current_identical_candidates(
        placeholder_filtered,
        args.current_image_dir,
        allow_current_rescue_sample_ids=CURRENT_RESCUE_SAMPLE_IDS,
    )
    current_rescues = [
        candidate
        for candidate in filtered
        if candidate.sample_id in CURRENT_RESCUE_SAMPLE_IDS and _same_as_current(candidate, args.current_image_dir)
    ]
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.experiment_dir / "controlled_b_candidate_pool.json"
    pool_path.write_text(
        json.dumps(
            {
                "candidate_count_before_current_filter": len(candidates),
                "candidate_count_after_placeholder_filter": len(placeholder_filtered),
                "candidate_count_after_current_filter": len(filtered),
                "blocked_placeholder_count": len(blocked_placeholders),
                "blocked_current_identical_count": len(blocked_current),
                "current_rescue_sample_ids": sorted(CURRENT_RESCUE_SAMPLE_IDS),
                "current_rescue_count": len(current_rescues),
                "source_counts": Counter(candidate.source for candidate in filtered),
                "blocked_placeholders": [candidate.to_manifest_row() for candidate in blocked_placeholders],
                "blocked_current_identical": [candidate.to_manifest_row() for candidate in blocked_current],
                "current_rescues": [candidate.to_manifest_row() for candidate in current_rescues],
                "candidates": [candidate.to_manifest_row() for candidate in sorted(filtered, key=_sort_key, reverse=True)],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summaries: list[dict[str, Any]] = []
    for target in sorted(args.targets):
        selected = select_controlled_ladder(
            filtered,
            target_count=target,
            required_sample_ids=CURRENT_RESCUE_SAMPLE_IDS,
        )
        manifest_path = args.experiment_dir / f"controlled_b{target}_replacement_manifest.json"
        manifest_payload = {
            "manifest_name": f"track1_controlled_b{target}_on_v3_20260608",
            "policy": (
                "v3_gate7 base; known provider/JSON placeholder candidates filtered; "
                "old-current-identical candidates filtered except explicit v3-placeholder rescues; full 1000-image package"
            ),
            "target_replacement_count": target,
            "accepted_count": len(selected),
            "current_rescue_sample_ids": sorted(CURRENT_RESCUE_SAMPLE_IDS),
            "accepted_replacements": [candidate.to_manifest_row() for candidate in selected],
            "source_counts": Counter(candidate.source for candidate in selected),
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        out_name = f"track1_candidate_v5_controlled_b{target}_on_v3_20260608"
        out_dir = args.submissions_dir / out_name
        out_zip = args.submissions_dir / f"{out_name}.zip"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if out_zip.exists():
            out_zip.unlink()
        build_summary = build_candidate_package(
            baseline_submission_json=args.v3_submission_json,
            baseline_image_dir=args.v3_image_dir,
            replacement_manifest=manifest_path,
            out_dir=out_dir,
            out_zip=out_zip,
        )
        lineage = audit_lineage(
            candidate_image_dir=out_dir / "images",
            current_image_dir=args.current_image_dir,
            v3_image_dir=args.v3_image_dir,
            full1000_image_dir=args.full1000_image_dir,
        )
        zip_summary = summarize_zip(out_zip)
        summary = {
            "target": target,
            "manifest": str(manifest_path.relative_to(ROOT)),
            "out_dir": str(out_dir.relative_to(ROOT)),
            "out_zip": str(out_zip.relative_to(ROOT)),
            "build_summary": build_summary,
            "lineage": lineage,
            "zip_summary": zip_summary,
            "source_counts": Counter(candidate.source for candidate in selected),
        }
        summaries.append(summary)
        (args.experiment_dir / f"controlled_b{target}_lineage.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    report_path = args.experiment_dir / "controlled_b_ladders_summary_zh.md"
    report_path.write_text(render_markdown_report(summaries, pool_path), encoding="utf-8")
    html_path = args.experiment_dir / "controlled_b_ladders_review_zh.html"
    html_path.write_text(render_html_report(summaries), encoding="utf-8")
    print(
        json.dumps(
            {
                "pool": str(pool_path.relative_to(ROOT)),
                "report": str(report_path.relative_to(ROOT)),
                "html": str(html_path.relative_to(ROOT)),
                "packages": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def collect_candidates(
    *,
    v5_accept48_manifest: Path,
    shortlist_json: Path,
    v3_image_dir: Path,
    current_image_dir: Path,
) -> list[ControlledBCandidate]:
    candidates: list[ControlledBCandidate] = []
    candidates.extend(_manifest_candidates(v5_accept48_manifest, source="v5_human_accept48", priority=100_000))
    for source, package_dir, priority, tier in PACKAGE_DIFF_SOURCES:
        candidates.extend(
            _package_diff_candidates(
                source=source,
                package_dir=package_dir,
                v3_image_dir=v3_image_dir,
                priority=priority,
                tier=tier,
            )
        )
    candidates.extend(_shortlist_candidates(shortlist_json))
    return [candidate for candidate in candidates if Path(candidate.candidate_image).exists()]


def _manifest_candidates(path: Path, *, source: str, priority: int) -> list[ControlledBCandidate]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("accepted_replacements") or payload.get("decisions") or []
    candidates: list[ControlledBCandidate] = []
    for index, row in enumerate(rows):
        if row.get("decision") not in {None, "accept"}:
            continue
        sample_id = str(row.get("sample_id") or "")
        candidate_image = str(row.get("candidate_image") or row.get("image") or "")
        if not sample_id or not candidate_image:
            continue
        candidates.append(
            ControlledBCandidate(
                sample_id=sample_id,
                candidate_image=candidate_image,
                source=source,
                priority=priority,
                score=float(len(rows) - index),
                tier=str(row.get("tier") or row.get("candidate_strategy") or ""),
                caption=str(row.get("caption") or ""),
                metadata={"source_manifest": str(path.relative_to(ROOT)) if path.is_absolute() else str(path)},
            )
        )
    return candidates


def _package_diff_candidates(
    *,
    source: str,
    package_dir: Path,
    v3_image_dir: Path,
    priority: int,
    tier: str,
) -> list[ControlledBCandidate]:
    if not package_dir.exists():
        return []
    candidates: list[ControlledBCandidate] = []
    for candidate_image in sorted(package_dir.glob("track1_*.jpg")):
        sample_id = candidate_image.stem
        v3_image = v3_image_dir / candidate_image.name
        if not v3_image.exists() or _sha256(candidate_image) == _sha256(v3_image):
            continue
        candidates.append(
            ControlledBCandidate(
                sample_id=sample_id,
                candidate_image=str(candidate_image),
                source=source,
                priority=priority,
                score=0.0,
                tier=tier,
            )
        )
    return candidates


def _shortlist_candidates(path: Path) -> list[ControlledBCandidate]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[ControlledBCandidate] = []
    for key, priority in (("review_queue", 60_000), ("positive_watchlist", 50_000)):
        for row in payload.get(key, []):
            sample_id = str(row.get("sample_id") or "")
            candidate_image = str(row.get("candidate_image") or "")
            if not sample_id or not candidate_image:
                continue
            delta = float(row.get("delta") or 0.0)
            candidates.append(
                ControlledBCandidate(
                    sample_id=sample_id,
                    candidate_image=candidate_image,
                    source=f"v4_distribution_{key}",
                    priority=priority,
                    score=delta,
                    tier=str(row.get("review_tier") or key),
                    caption=str(row.get("caption") or ""),
                    metadata={
                        "delta": delta,
                        "distribution_delta": row.get("distribution_delta"),
                        "perceptual_delta": row.get("perceptual_delta"),
                        "candidate_score": row.get("candidate_score"),
                        "current_score": row.get("current_score"),
                    },
                )
            )
    return candidates


def audit_lineage(
    *,
    candidate_image_dir: Path,
    current_image_dir: Path,
    v3_image_dir: Path,
    full1000_image_dir: Path,
) -> dict[str, int]:
    candidate_hashes = _dir_hashes(candidate_image_dir)
    current_hashes = _dir_hashes(current_image_dir)
    v3_hashes = _dir_hashes(v3_image_dir)
    full1000_hashes = _dir_hashes(full1000_image_dir)
    return {
        "count": len(candidate_hashes),
        "same_as_current": _same_count(candidate_hashes, current_hashes),
        "same_as_v3_gate7": _same_count(candidate_hashes, v3_hashes),
        "same_as_full1000_no_fallback": _same_count(candidate_hashes, full1000_hashes),
        "changed_vs_current": _diff_count(candidate_hashes, current_hashes),
        "changed_vs_v3_gate7": _diff_count(candidate_hashes, v3_hashes),
    }


def summarize_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    return {
        "zip_size_bytes": path.stat().st_size,
        "zip_sha256": _sha256(path),
        "zip_entries": len(names),
        "zip_images": sum(1 for name in names if name.startswith("images/") and name.endswith(".jpg")),
        "has_submission_json": "submission.json" in names,
    }


def render_markdown_report(summaries: list[dict[str, Any]], pool_path: Path) -> str:
    lines = [
        "# Track1 Controlled-B Ladder Summary",
        "",
        "- 基底：`submissions/track1_candidate_v3_gate7_20260606/`",
        "- 策略：1000 张完整提交包，仅替换 selected 样本；过滤 provider/JSON 占位图。",
        "- current 回流策略：默认排除与旧 current 完全相同的候选图；仅 `track1_0735`、`track1_0740` 允许作为 v3 占位图救援。",
        f"- Candidate pool: `{pool_path.relative_to(ROOT)}`",
        "",
        "| package | replacements | same_as_current | same_as_v3 | same_as_full1000 | zip MB | sha256 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        lineage = summary["lineage"]
        zip_summary = summary["zip_summary"]
        lines.append(
            "| `{out_zip}` | {target} | {same_current} | {same_v3} | {same_full} | {mb:.2f} | `{sha}` |".format(
                out_zip=summary["out_zip"],
                target=summary["target"],
                same_current=lineage["same_as_current"],
                same_v3=lineage["same_as_v3_gate7"],
                same_full=lineage["same_as_full1000_no_fallback"],
                mb=zip_summary["zip_size_bytes"] / 1024 / 1024,
                sha=zip_summary["zip_sha256"],
            )
        )
    lines.extend(["", "## Source Counts", ""])
    for summary in summaries:
        lines.append(f"### B{summary['target']}")
        for source, count in sorted(summary["source_counts"].items()):
            lines.append(f"- `{source}`: `{count}`")
        lines.append("")
    return "\n".join(lines)


def render_html_report(summaries: list[dict[str, Any]]) -> str:
    rows = []
    for summary in summaries:
        lineage = summary["lineage"]
        zip_summary = summary["zip_summary"]
        rows.append(
            "<tr>"
            f"<td>B{summary['target']}</td>"
            f"<td>{summary['out_zip']}</td>"
            f"<td>{lineage['same_as_current']}</td>"
            f"<td>{lineage['same_as_v3_gate7']}</td>"
            f"<td>{lineage['changed_vs_v3_gate7']}</td>"
            f"<td>{zip_summary['zip_size_bytes'] / 1024 / 1024:.2f} MB</td>"
            f"<td><code>{zip_summary['zip_sha256']}</code></td>"
            "</tr>"
        )
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Track1 Controlled-B Ladder</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #d8dee4; padding: 8px; text-align: left; vertical-align: top; }
    th { background: #f6f8fa; }
    code { word-break: break-all; }
  </style>
</head>
<body>
  <h1>Track1 Controlled-B Ladder</h1>
  <p>这些都是 1000 张完整提交包，基底为 v3_gate7。已过滤 provider/JSON 占位图；same_as_current 只允许 0735/0740 这种 v3 占位图救援。</p>
  <table>
    <thead>
      <tr><th>包</th><th>ZIP</th><th>same_as_current</th><th>same_as_v3</th><th>changed_vs_v3</th><th>大小</th><th>SHA256</th></tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
""".replace("{rows}", "\n".join(rows))


def _dir_hashes(path: Path) -> dict[str, str]:
    return {image.stem: _sha256(image) for image in sorted(path.glob("*.jpg"))}


def _same_count(left: dict[str, str], right: dict[str, str]) -> int:
    return sum(1 for sample_id, digest in left.items() if right.get(sample_id) == digest)


def _diff_count(left: dict[str, str], right: dict[str, str]) -> int:
    return sum(1 for sample_id, digest in left.items() if right.get(sample_id) != digest)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _same_as_current(candidate: ControlledBCandidate, current_image_dir: Path) -> bool:
    current_path = current_image_dir / f"{candidate.sample_id}.jpg"
    candidate_path = Path(candidate.candidate_image)
    return current_path.exists() and candidate_path.exists() and _sha256(current_path) == _sha256(candidate_path)


def _sort_key(candidate: ControlledBCandidate) -> tuple[int, float, str]:
    return (candidate.priority, candidate.score, candidate.sample_id)


if __name__ == "__main__":
    raise SystemExit(main())
