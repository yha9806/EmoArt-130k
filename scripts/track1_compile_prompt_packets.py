#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.challenge import _read_track1_samples
from affectiveart.track1_prompt_packet import compile_and_rank_packets


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile Track1 captions into structured prompt packets.")
    parser.add_argument("--track1-zip", type=Path, default=Path("data/raw/Track1_testset.zip"))
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("experiments/track1_130k_compiler_gate_v1/prompt_packets.jsonl"),
    )
    parser.add_argument(
        "--risk-json",
        type=Path,
        default=Path("experiments/track1_130k_compiler_gate_v1/prompt_packet_risk_rank.json"),
    )
    args = parser.parse_args()

    _format, rows = _read_track1_samples(args.track1_zip)
    packets = compile_and_rank_packets(rows)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for packet in packets:
            fh.write(json.dumps(packet, ensure_ascii=False) + "\n")
    args.risk_json.write_text(json.dumps(packets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.out_jsonl)
    print(args.risk_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
