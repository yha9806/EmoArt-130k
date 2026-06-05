from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_moe_specialist_ensemble import expected_label_for_emotion


ACCEPT_SAFE = "accept_safe"
ACCEPT_CROSS_MICRO = "accept_cross_micro"
HOLD_REVIEW = "hold_review"
BLOCK = "block"

V6_LADDERS = ("v6_safe_sameq", "v6_cross_micro", "v6_desc_plus")


@dataclass(frozen=True)
class V6Delta:
    sample_id: str
    current_emotion: str
    proposed_emotion: str
    current_valence: str = ""
    current_arousal: str = ""
    proposed_valence: str = ""
    proposed_arousal: str = ""
    transition: str = ""
    same_quadrant: bool = False
    cross_quadrant_risk: bool = False
    supporting_source_count: int = 0
    supporting_family_count: int = 0
    supporting_sources: tuple[str, ...] = ()
    supporting_families: tuple[str, ...] = ()
    evidence_score: float = 0.0
    gemini35_prefers_proposed: bool = False
    gemini35_fit_margin: float = 0.0
    gemini35_objection: bool = False
    vulca_objection: bool = False
    public_reference_support: bool = False
    public_reference_contradiction: bool = False
    human_confirmed: bool = False
    hard96_net_gain: int = 0
    hard96_net_loss: int = 0
    malformed: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectorThresholds:
    sameq_min_score: float = 8.0
    sameq_min_sources: int = 3
    sameq_min_families: int = 2
    cross_min_score: float = 9.0
    cross_min_sources: int = 4
    cross_min_families: int = 3
    cross_min_fit_margin: float = 0.30
    cross_min_hard96_gain: int = 1


@dataclass(frozen=True)
class SelectorDecision:
    sample_id: str
    proposed_emotion: str
    transition: str
    decision: str
    ladder: str
    reason_codes: tuple[str, ...]
    evidence_score: float
    same_quadrant: bool
    cross_quadrant_risk: bool
    proposed_valence: str
    proposed_arousal: str
    source_count: int
    family_count: int


def load_evidence_matrix(path: str | Path) -> list[V6Delta]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [_delta_from_row(row) for row in csv.DictReader(handle)]


def select_v6_deltas(
    deltas: list[V6Delta],
    thresholds: SelectorThresholds | None = None,
) -> list[SelectorDecision]:
    thresholds = thresholds or SelectorThresholds()
    decisions = [_select_one(delta, thresholds) for delta in deltas]

    accepted_by_sample: dict[str, list[int]] = {}
    for index, decision in enumerate(decisions):
        if decision.decision in {ACCEPT_SAFE, ACCEPT_CROSS_MICRO}:
            accepted_by_sample.setdefault(decision.sample_id, []).append(index)

    for indexes in accepted_by_sample.values():
        if len(indexes) <= 1:
            continue
        for index in indexes:
            decision = decisions[index]
            reasons = _append_reason(decision.reason_codes, "proposal_conflict")
            decisions[index] = replace(
                decision,
                decision=HOLD_REVIEW,
                ladder="",
                reason_codes=reasons,
            )

    return decisions


def build_candidate_rows(
    baseline_rows: list[dict[str, Any]],
    decisions: list[SelectorDecision],
    *,
    ladder: str,
) -> list[dict[str, Any]]:
    selected = {
        decision.sample_id: decision
        for decision in decisions
        if decision.ladder == ladder and decision.decision in {ACCEPT_SAFE, ACCEPT_CROSS_MICRO}
    }
    output: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        row = dict(baseline)
        decision = selected.get(str(row.get("sample_id", "")).strip())
        if decision is not None:
            row["emotion"] = decision.proposed_emotion
            row["emotional_valence"] = decision.proposed_valence
            row["emotional_arousal_level"] = decision.proposed_arousal
        output.append({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})
    return output


def write_v6_outputs(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("write_v6_outputs is implemented in Task 3")


def _delta_from_row(row: dict[str, Any]) -> V6Delta:
    raw = dict(row)
    sample_id = _text(row.get("sample_id"))
    current_emotion = _emotion(row.get("current_emotion"))
    proposed_emotion = _emotion(row.get("proposed_emotion"))
    malformed = any(not item for item in (sample_id, current_emotion, proposed_emotion))

    current_valence, current_arousal = _labels_for(
        current_emotion,
        row.get("current_valence"),
        row.get("current_arousal"),
    )
    proposed_valence, proposed_arousal = _labels_for(
        proposed_emotion,
        row.get("proposed_valence"),
        row.get("proposed_arousal"),
    )
    same_quadrant = _optional_bool(row.get("same_quadrant"))
    if same_quadrant is None:
        same_quadrant = bool(
            current_valence
            and current_arousal
            and current_valence == proposed_valence
            and current_arousal == proposed_arousal
        )

    transition = ""
    if current_emotion or proposed_emotion:
        transition = f"{current_emotion}->{proposed_emotion}"

    return V6Delta(
        sample_id=sample_id,
        current_emotion=current_emotion,
        current_valence=current_valence,
        current_arousal=current_arousal,
        proposed_emotion=proposed_emotion,
        proposed_valence=proposed_valence,
        proposed_arousal=proposed_arousal,
        transition=transition,
        same_quadrant=same_quadrant,
        cross_quadrant_risk=not same_quadrant,
        supporting_source_count=_int(row.get("supporting_source_count")),
        supporting_family_count=_int(row.get("supporting_family_count")),
        supporting_sources=_split_list(row.get("supporting_sources")),
        supporting_families=_split_list(row.get("supporting_families")),
        evidence_score=_float(row.get("evidence_score")),
        gemini35_prefers_proposed=_bool(row.get("gemini35_prefers_proposed")),
        gemini35_fit_margin=_float(row.get("gemini35_fit_margin")),
        gemini35_objection=_bool(row.get("gemini35_objection")),
        vulca_objection=_bool(row.get("vulca_objection")),
        public_reference_support=_bool(row.get("public_reference_support")),
        public_reference_contradiction=_bool(row.get("public_reference_contradiction")),
        human_confirmed=_bool(row.get("human_confirmed")),
        hard96_net_gain=_int(row.get("hard96_net_gain")),
        hard96_net_loss=_int(row.get("hard96_net_loss")),
        malformed=malformed,
        raw=raw,
    )


def _select_one(delta: V6Delta, thresholds: SelectorThresholds) -> SelectorDecision:
    reasons: list[str] = []
    decision = HOLD_REVIEW
    ladder = ""

    if delta.malformed:
        reasons.append("missing_required_delta_field")
    if delta.public_reference_contradiction:
        reasons.append("public_reference_contradiction")
    if delta.gemini35_objection:
        reasons.append("gemini35_objection")
    if delta.vulca_objection:
        reasons.append("vulca_objection")
    if delta.hard96_net_loss > 0:
        reasons.append("hard96_net_loss")
    if _dangerous_positive_low_to_negative(delta):
        reasons.append("dangerous_positive_low_to_negative")

    if not reasons:
        if delta.same_quadrant:
            reasons.extend(_sameq_threshold_reasons(delta, thresholds))
            if not reasons:
                decision = ACCEPT_SAFE
                ladder = "v6_safe_sameq"
                reasons.append("same_quadrant_multi_source_support")
        else:
            reasons.extend(_cross_threshold_reasons(delta, thresholds))
            if not reasons:
                decision = ACCEPT_CROSS_MICRO
                ladder = "v6_cross_micro"
                reasons.append("cross_quadrant_arbitrated_micro_gain")
    elif not delta.same_quadrant and not delta.malformed:
        for reason in _cross_threshold_reasons(delta, thresholds):
            if reason not in reasons:
                reasons.append(reason)

    return SelectorDecision(
        sample_id=delta.sample_id,
        proposed_emotion=delta.proposed_emotion,
        transition=delta.transition,
        decision=decision,
        ladder=ladder,
        reason_codes=tuple(reasons or ["insufficient_evidence"]),
        evidence_score=delta.evidence_score,
        same_quadrant=delta.same_quadrant,
        cross_quadrant_risk=delta.cross_quadrant_risk,
        proposed_valence=delta.proposed_valence,
        proposed_arousal=delta.proposed_arousal,
        source_count=delta.supporting_source_count,
        family_count=delta.supporting_family_count,
    )


def _sameq_threshold_reasons(delta: V6Delta, thresholds: SelectorThresholds) -> list[str]:
    reasons: list[str] = []
    if delta.evidence_score < thresholds.sameq_min_score:
        reasons.append("low_evidence_score")
    if delta.supporting_source_count < thresholds.sameq_min_sources:
        reasons.append("insufficient_source_count")
    if delta.supporting_family_count < thresholds.sameq_min_families:
        reasons.append("insufficient_family_count")
    return reasons


def _cross_threshold_reasons(delta: V6Delta, thresholds: SelectorThresholds) -> list[str]:
    reasons: list[str] = []
    if not delta.gemini35_prefers_proposed or delta.gemini35_fit_margin < thresholds.cross_min_fit_margin:
        reasons.append("missing_cross_arbitration")
    if delta.hard96_net_gain < thresholds.cross_min_hard96_gain:
        reasons.append("missing_hard96_gain")
    if delta.evidence_score < thresholds.cross_min_score:
        reasons.append("low_evidence_score")
    if delta.supporting_source_count < thresholds.cross_min_sources:
        reasons.append("insufficient_source_count")
    if delta.supporting_family_count < thresholds.cross_min_families:
        reasons.append("insufficient_family_count")
    return reasons


def _dangerous_positive_low_to_negative(delta: V6Delta) -> bool:
    return (
        delta.current_emotion in {"calm", "content", "glad"}
        and delta.current_valence == "Positive"
        and delta.current_arousal == "Low"
        and delta.proposed_valence == "Negative"
    )


def _labels_for(emotion: str, valence: Any, arousal: Any) -> tuple[str, str]:
    clean_valence = _text(valence)
    clean_arousal = _text(arousal)
    if clean_valence and clean_arousal:
        return clean_valence, clean_arousal
    if not emotion:
        return clean_valence, clean_arousal
    try:
        expected_valence, expected_arousal = expected_label_for_emotion(emotion)
    except ValueError:
        return clean_valence, clean_arousal
    return clean_valence or expected_valence, clean_arousal or expected_arousal


def _text(value: Any) -> str:
    return str(value or "").strip()


def _emotion(value: Any) -> str:
    return _text(value).lower()


def _optional_bool(value: Any) -> bool | None:
    text = _text(value).lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _bool(value: Any) -> bool:
    return _optional_bool(value) is True


def _int(value: Any) -> int:
    try:
        return int(float(_text(value)))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(_text(value))
    except (TypeError, ValueError):
        return 0.0


def _split_list(value: Any) -> tuple[str, ...]:
    text = _text(value)
    if not text:
        return ()
    return tuple(item.strip() for item in text.replace(",", ";").split(";") if item.strip())


def _append_reason(reasons: tuple[str, ...], reason: str) -> tuple[str, ...]:
    if reason in reasons:
        return reasons
    return (*reasons, reason)
