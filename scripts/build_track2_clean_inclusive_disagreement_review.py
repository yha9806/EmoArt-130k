#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_disagreement_review import write_disagreement_review_outputs


def validate_input_paths(parser: argparse.ArgumentParser, paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            parser.error(f"missing required input: {path}")


def load_high_similarity_sample_ids(path: str | Path) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("unique_sample_ids"), list):
        return {str(sample_id) for sample_id in payload["unique_sample_ids"] if sample_id}
    rows = payload.get("entries", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError(f"expected list or object with entries: {path}")
    ids: set[str] = set()
    for row in rows:
        if isinstance(row, dict) and row.get("sample_id"):
            ids.add(str(row["sample_id"]))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track2 clean/inclusive disagreement HTML review packet.")
    parser.add_argument("--current-json", type=Path, default=Path("submissions/track2_submission.json"))
    parser.add_argument(
        "--clean-predictions-json",
        type=Path,
        default=Path(
            "experiments/track2_public_style_distillation_20260603/"
            "siglip2_cached_logreg_v1/clean_predictions.json"
        ),
    )
    parser.add_argument(
        "--inclusive-predictions-json",
        type=Path,
        default=Path(
            "experiments/track2_public_style_distillation_20260603/"
            "siglip2_cached_logreg_v1/inclusive_predictions.json"
        ),
    )
    parser.add_argument(
        "--high-similarity-json",
        type=Path,
        default=Path("experiments/track2_public_style_distillation_20260603/public_leak_exclusion_manifest.json"),
    )
    parser.add_argument("--image-zip", type=Path, default=Path("data/raw/Track2_testset.zip"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/track2_final_push_20260603/clean_inclusive_disagreement"),
    )
    args = parser.parse_args()

    validate_input_paths(
        parser,
        [
            args.current_json,
            args.clean_predictions_json,
            args.inclusive_predictions_json,
            args.high_similarity_json,
            args.image_zip,
        ],
    )
    try:
        high_similarity_sample_ids = load_high_similarity_sample_ids(args.high_similarity_json)
        report = write_disagreement_review_outputs(
            current_json=args.current_json,
            clean_predictions_json=args.clean_predictions_json,
            inclusive_predictions_json=args.inclusive_predictions_json,
            image_zip=args.image_zip,
            out_dir=args.out_dir,
            high_similarity_sample_ids=high_similarity_sample_ids,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        parser.error(f"invalid input: {str(exc) or exc.__class__.__name__}")

    print(
        json.dumps(
            {
                "row_count": report["row_count"],
                "outputs": report["outputs"],
                "category_counts": report["category_counts"],
                "high_similarity_count": report["high_similarity_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
