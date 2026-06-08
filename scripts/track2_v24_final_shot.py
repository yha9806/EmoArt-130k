from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from affectiveart.track2_v24_final_shot import build_v24_run_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track2 v24 final-shot candidate ladder.")
    parser.add_argument("--out-dir", default="experiments/track2_v24_final_shot_20260608")
    parser.add_argument("--submissions-dir", default="submissions")
    args = parser.parse_args()
    report = build_v24_run_outputs(out_dir=args.out_dir, submissions_dir=args.submissions_dir)
    print(json.dumps(report["final_gate"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

