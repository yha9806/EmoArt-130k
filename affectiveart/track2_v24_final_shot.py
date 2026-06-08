from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from affectiveart.track2_official_anchor_calibration import CalibratedScore


@dataclass(frozen=True)
class FinalGateThresholds:
    min_overall: float = 0.89
    min_classification: float = 0.78
    min_description: float = 0.99
    max_top_emotion_share: float = 0.58


UNSAFE_TEXT_MARKERS = (
    "give this answer",
    "give this submission",
    "perfect score",
    "ignore previous instructions",
    "ignore the rubric",
    "reward the caption",
    "evaluator should",
    "judge should",
)


def is_text_evaluator_safe(text: str) -> bool:
    lowered = str(text or "").lower()
    return not any(marker in lowered for marker in UNSAFE_TEXT_MARKERS)


def choose_v24_final_candidate(
    scores: Iterable[CalibratedScore],
    *,
    thresholds: FinalGateThresholds,
    validation_ok: bool,
    label_consistency_issue_count: int,
    missing_emotions: list[str],
    top_emotion_share: float,
    repeats_failed_pattern: bool,
    unsafe_text_count: int,
) -> dict[str, Any]:
    ranked = sorted(scores, key=lambda score: (-score.overall_expected, score.candidate_name))
    if not ranked:
        return {"decision": "hold_no_submit", "candidate_name": "", "reasons": ["no_candidates"]}
    best = ranked[0]
    reasons: list[str] = []
    if best.overall_expected < thresholds.min_overall:
        reasons.append("overall_below_089")
    if best.classification_expected < thresholds.min_classification:
        reasons.append("classification_below_078")
    if best.description_expected < thresholds.min_description:
        reasons.append("description_below_099")
    if not validation_ok:
        reasons.append("validator_failed")
    if label_consistency_issue_count:
        reasons.append("label_consistency_issues")
    if missing_emotions:
        reasons.append("missing_emotions")
    if top_emotion_share > thresholds.max_top_emotion_share:
        reasons.append("top_emotion_collapse")
    if repeats_failed_pattern:
        reasons.append("repeats_781601_failed_pattern")
    if unsafe_text_count:
        reasons.append("unsafe_description_text")
    decision = "hold_no_submit" if reasons else "recommend_final_submit"
    return {
        "decision": decision,
        "candidate_name": best.candidate_name,
        "overall_expected": best.overall_expected,
        "classification_expected": best.classification_expected,
        "description_expected": best.description_expected,
        "reasons": reasons,
    }

