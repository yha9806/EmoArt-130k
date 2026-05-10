#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_compiler_gate import compile_vulca_prompt
from affectiveart.track1_challenger import effective_create_returncode, merge_run_summary_rows
from affectiveart.track1_vulca import select_track1_tradition


DEFAULT_VULCA_SRC = Path("/Users/yhryzy/dev/vulca/.worktrees/caption-fidelity-content-lock-v1/src")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Track1 challengers with 130k-aware Vulca prompt packets.")
    parser.add_argument(
        "--packets-jsonl",
        type=Path,
        default=Path("experiments/track1_130k_compiler_gate_v1/prompt_packets_with_refs.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/track1_130k_compiler_gate_v1/challengers"),
    )
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--vlm-model", default="gemini/gemini-3-flash-preview")
    parser.add_argument("--vulca-src", type=Path, default=DEFAULT_VULCA_SRC)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = set(args.sample_id)
    packets = []
    for line in args.packets_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        packet = json.loads(line)
        if selected and packet["sample_id"] not in selected:
            continue
        packets.append(packet)
    if args.limit:
        packets = packets[: args.limit]

    image_dir = args.out_dir / "images"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for packet in packets:
        sample_id = packet["sample_id"]
        image_path = image_dir / f"{sample_id}.png"
        json_path = args.out_dir / f"{sample_id}.create.json"
        stderr_path = args.out_dir / f"{sample_id}.stderr.txt"
        if image_path.exists() and json_path.exists() and not args.force:
            rows.append({"sample_id": sample_id, "image": str(image_path), "json": str(json_path), "cached": True})
            continue
        prompt = compile_vulca_prompt(packet)
        env = build_env(args.vulca_src, args.vlm_model)
        cmd = [
            sys.executable,
            "-m",
            "vulca.cli",
            "create",
            prompt,
            "--tradition",
            select_track1_tradition(packet["caption"]),
            "--provider",
            args.provider,
            "--mode",
            "strict",
            "--content-lock",
            "--output-is-artwork-itself",
            "--output",
            str(image_path),
            "--json",
        ]
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        json_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        effective_returncode = effective_create_returncode(completed.returncode, json_path, image_path)
        rows.append(
            {
                "sample_id": sample_id,
                "returncode": effective_returncode,
                "process_returncode": completed.returncode,
                "image": str(image_path),
                "json": str(json_path),
                "stderr_path": str(stderr_path),
                "stderr_tail": completed.stderr[-2000:],
            }
        )

    summary_path = args.out_dir / "run_summary.json"
    if selected and summary_path.exists():
        existing_rows = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = merge_run_summary_rows(existing_rows, rows)
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary_path)
    return 0


def build_env(vulca_src: Path, vlm_model: str) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{vulca_src}{os.pathsep}{ROOT}"
        if not existing
        else f"{vulca_src}{os.pathsep}{ROOT}{os.pathsep}{existing}"
    )
    env["VULCA_VLM_MODEL"] = vlm_model
    if env.get("GEMINI_API_KEY") and not env.get("GOOGLE_API_KEY"):
        env["GOOGLE_API_KEY"] = env["GEMINI_API_KEY"]
    env.pop("VULCA_API_URL", None)
    return env


if __name__ == "__main__":
    raise SystemExit(main())
