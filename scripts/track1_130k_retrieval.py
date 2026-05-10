#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_130k_retrieval import select_references


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach EmoArt-130k references to Track1 prompt packets.")
    parser.add_argument(
        "--packets-jsonl",
        type=Path,
        default=Path("experiments/track1_130k_compiler_gate_v1/prompt_packets.jsonl"),
    )
    parser.add_argument(
        "--compiler-jsonl",
        type=Path,
        default=Path("experiments/track1_130k_compiler_gate_v1/emoart130k_compiler_train.jsonl"),
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("experiments/track1_130k_compiler_gate_v1/prompt_packets_with_refs.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    references = [
        json.loads(line)
        for line in args.compiler_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for line in args.packets_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            packet = json.loads(line)
            packet["retrieved_references"] = [
                ref.__dict__ for ref in select_references(packet, references, top_k=args.top_k)
            ]
            fh.write(json.dumps(packet, ensure_ascii=False) + "\n")
            written += 1
    print(args.out_jsonl)
    print(f"packets: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
