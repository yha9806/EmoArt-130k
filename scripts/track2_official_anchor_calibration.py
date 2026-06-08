from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from affectiveart.track2_official_anchor_calibration import (
    DEFAULT_OUT_DIR,
    default_candidate_paths,
    write_scoreboard_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track2 official-anchor calibrated scorer.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    board = write_scoreboard_outputs(candidates=default_candidate_paths(), out_dir=Path(args.out_dir))
    print(
        json.dumps(
            {
                "method": board["method"],
                "out_dir": str(args.out_dir),
                "top_candidate": board["ranking"][0]["candidate_name"] if board["ranking"] else "",
                "top_expected": board["ranking"][0]["overall_expected"] if board["ranking"] else None,
                "blocking_residual_count": board["anchor_residual_summary"]["blocking_residual_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
