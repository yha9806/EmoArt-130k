#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.emoart130k import iter_emoart_examples


DEFAULT_ROOT = Path("/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export EmoArt-130k compiler training JSONL.")
    parser.add_argument("--annotation-json", type=Path, default=DEFAULT_ROOT / "Annotation.json")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("experiments/track1_130k_compiler_gate_v1/emoart130k_compiler_train.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    written = 0
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for example in iter_emoart_examples(args.annotation_json, data_root=args.data_root):
            if args.limit and written >= args.limit:
                break
            row = example.to_compiler_training_row()
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[example.emotion] += 1
            written += 1

    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary = {
        "rows": written,
        "emotion_distribution": dict(sorted(counts.items())),
        "out_jsonl": str(args.out_jsonl),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.out_jsonl)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
