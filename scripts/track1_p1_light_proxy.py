from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from affectiveart.track1_p1_light_proxy import (  # noqa: E402
    evaluate_packages,
    load_known_placeholder_sha256s,
    load_replacement_sample_ids,
    load_route_index,
    load_submission_rows,
    parse_package_spec,
    write_p1_light_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Track1 P1-light package proxy gates over existing candidate packages."
    )
    parser.add_argument("--baseline-submission-json", required=True)
    parser.add_argument("--baseline-image-dir", required=True)
    parser.add_argument("--routes-json")
    parser.add_argument("--known-placeholder-json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="Package spec as NAME=SUBMISSION_JSON=IMAGE_DIR. Repeat for multiple packages.",
    )
    parser.add_argument(
        "--replacement-manifest",
        action="append",
        default=[],
        help="Optional replacement manifest as NAME=MANIFEST_JSON. Repeat for packages with known accepted replacements.",
    )
    args = parser.parse_args()
    if not args.package:
        raise SystemExit("at least one --package is required")

    baseline_rows = load_submission_rows(args.baseline_submission_json)
    route_index = load_route_index(args.routes_json)
    known_placeholders = load_known_placeholder_sha256s(args.known_placeholder_json)
    manifest_by_name = _parse_manifest_specs(args.replacement_manifest)
    specs = []
    for raw_spec in args.package:
        name, submission_json, image_dir = parse_package_spec(raw_spec)
        specs.append((name, load_submission_rows(submission_json), image_dir, manifest_by_name.get(name, set())))

    report = evaluate_packages(
        specs,
        baseline_rows=baseline_rows,
        baseline_image_dir=args.baseline_image_dir,
        route_index=route_index,
        known_placeholder_sha256s=known_placeholders,
    )
    write_p1_light_reports(report, args.out_json, args.out_md)
    print(args.out_md)


def _parse_manifest_specs(specs: list[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for spec in specs:
        parts = spec.split("=", 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(f"manifest spec must be NAME=MANIFEST_JSON: {spec}")
        result[parts[0].strip()] = load_replacement_sample_ids(parts[1].strip())
    return result


if __name__ == "__main__":
    main()
