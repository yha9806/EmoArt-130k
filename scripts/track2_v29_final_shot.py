from __future__ import annotations

import argparse
import json
from pathlib import Path

from affectiveart.track2_official_anchor_calibration import score_submission
from affectiveart.track2_v29_final_shot import load_track2_rows, run_v29_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track2 v29 FAB-G description and rerank final-shot sweep.")
    parser.add_argument("--base-json", default="submissions/track2_submission_v21_calmshift90_candidate.json")
    parser.add_argument("--out-dir", default="experiments/track2_v29_fabg_description_qwen_rerank_20260608")
    parser.add_argument("--candidate-prefix", default="submissions/track2_submission_v29")
    parser.add_argument("--label-change-credit", type=float, default=0.00022)
    args = parser.parse_args()

    rows = load_track2_rows(args.base_json)
    base_score = score_submission(args.base_json, candidate_name="v29_base")
    profiles = run_v29_sweep(
        rows,
        candidate_prefix=Path(args.candidate_prefix),
        out_dir=Path(args.out_dir),
        base_classification_expected=base_score.classification_expected,
        label_change_credit=args.label_change_credit,
    )
    print(json.dumps([profile.__dict__ for profile in profiles], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
