#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.track2_multimodal_description import (
    DESCRIPTION_REWRITE_MODEL,
    DESCRIPTION_REWRITE_RESPONSE_SCHEMA,
    append_rewrite_decision,
    build_prompt,
    load_rewrite_decisions,
    load_track2_rows,
    normalize_image_for_gemini,
    ordered_rows_for_ids,
    write_rewrite_candidate_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/apply Gemini multimodal Track2 description rewrites.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run Gemini and append JSONL rewrite decisions")
    run_parser.add_argument("--source-json", type=Path, required=True)
    run_parser.add_argument("--image-dir", type=Path, required=True)
    run_parser.add_argument("--decisions-jsonl", type=Path, required=True)
    run_parser.add_argument("--model", default=DESCRIPTION_REWRITE_MODEL)
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--sample-ids-file", type=Path, default=None)
    run_parser.add_argument("--max-side", type=int, default=1024)
    run_parser.add_argument("--keychain-service", default="affectiveart-gemini-api-key")
    run_parser.add_argument("--keychain-account", default="gemini")

    apply_parser = subparsers.add_parser("apply", help="Apply accepted rewrite decisions to a candidate JSON")
    apply_parser.add_argument("--source-json", type=Path, required=True)
    apply_parser.add_argument("--decisions-jsonl", type=Path, required=True)
    apply_parser.add_argument("--out-json", type=Path, required=True)
    apply_parser.add_argument("--out-zip", type=Path, required=True)
    apply_parser.add_argument("--report-json", type=Path, required=True)
    apply_parser.add_argument("--report-md", type=Path, required=True)
    apply_parser.add_argument("--min-confidence", type=float, default=0.74)
    apply_parser.add_argument("--min-dimension-score", type=float, default=0.72)

    args = parser.parse_args()
    if args.command == "run":
        run_gemini_rewrites(args)
        return
    if args.command == "apply":
        report = write_rewrite_candidate_outputs(
            source_json=args.source_json,
            decisions_jsonl=args.decisions_jsonl,
            out_json=args.out_json,
            out_zip=args.out_zip,
            report_json=args.report_json,
            report_md=args.report_md,
            min_confidence=args.min_confidence,
            min_dimension_score=args.min_dimension_score,
        )
        print(json.dumps({key: report[key] for key in ["changed_rows", "changed_fields", "classification_label_changes"]}, indent=2))


def run_gemini_rewrites(args: argparse.Namespace) -> None:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed in this Python environment") from exc

    api_key = os.environ.get("GEMINI_API_KEY") or _read_keychain_password(
        args.keychain_service,
        args.keychain_account,
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set and Keychain lookup returned no key")

    rows = load_track2_rows(args.source_json)
    sample_ids = _read_sample_ids(args.sample_ids_file)
    queue = ordered_rows_for_ids(rows, sample_ids)
    completed = set(load_rewrite_decisions(args.decisions_jsonl))
    client = genai.Client(api_key=api_key)

    reviewed = 0
    for row in queue:
        if args.limit and reviewed >= args.limit:
            break
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in completed:
            continue
        image_path = args.image_dir / f"{sample_id}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"missing image for {sample_id}: {image_path}")
        image_bytes, mime_type = normalize_image_for_gemini(image_path, max_side=args.max_side)
        response = client.models.generate_content(
            model=args.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                build_prompt(row),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DESCRIPTION_REWRITE_RESPONSE_SCHEMA,
                temperature=0.15,
            ),
        )
        raw = json.loads(response.text or "{}")
        payload = {
            "sample_id": sample_id,
            "model": args.model,
            **raw,
        }
        append_rewrite_decision(args.decisions_jsonl, payload)
        completed.add(sample_id)
        reviewed += 1
        print(
            f"{sample_id}: rewrite={payload.get('rewrite_needed')} "
            f"conf={float(payload.get('confidence', 0) or 0):.2f} "
            f"reason={str(payload.get('reason', ''))[:90]}",
            flush=True,
        )
    print(json.dumps({"reviewed": reviewed, "decisions_jsonl": str(args.decisions_jsonl)}, indent=2))


def _read_sample_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip().split(",", 1)[0].strip()
        if value and not value.startswith("#"):
            values.append(value)
    return values


def _read_keychain_password(service: str, account: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


if __name__ == "__main__":
    main()
