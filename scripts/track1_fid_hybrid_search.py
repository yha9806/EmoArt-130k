#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_fid_hybrid_search import (  # noqa: E402
    greedy_hybrid_fid_search,
    load_feature_cache,
    load_shortlist_sample_ids,
    write_hybrid_search_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Greedily search Track1 hybrid replacements with local Inception FID-like features.")
    parser.add_argument("--current-features", required=True, type=Path)
    parser.add_argument("--candidate-features", required=True, type=Path)
    parser.add_argument("--reference-features", required=True, type=Path)
    parser.add_argument("--shortlist-json", type=Path)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--candidate-image-dir", type=Path)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--max-replacements", type=int, default=20)
    parser.add_argument("--min-improvement", type=float, default=1e-6)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--replacement-manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    current = load_feature_cache(_abs(args.current_features))
    candidate = load_feature_cache(_abs(args.candidate_features))
    reference = load_feature_cache(_abs(args.reference_features))
    pool = load_shortlist_sample_ids(_abs(args.shortlist_json) if args.shortlist_json else None, max_candidates=args.max_candidates)
    report = greedy_hybrid_fid_search(
        current_ids=current["ids"],
        current_features=current["features"],
        candidate_ids=candidate["ids"],
        candidate_features=candidate["features"],
        reference_features=reference["features"],
        candidate_pool=pool or None,
        candidate_label=args.candidate_label,
        max_replacements=args.max_replacements,
        min_improvement=args.min_improvement,
    )
    write_hybrid_search_reports(
        report,
        json_path=_abs(args.out_json),
        md_path=_abs(args.out_md),
        replacement_manifest_path=_abs(args.replacement_manifest) if args.replacement_manifest else None,
        candidate_image_dir=_abs(args.candidate_image_dir) if args.candidate_image_dir else None,
    )
    print("WARNING: local Inception FID-like proxy only; this is not the official Track1 evaluator.")
    print(args.out_json)
    print(args.out_md)
    if args.replacement_manifest:
        print(args.replacement_manifest)
    return 0


def _abs(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
