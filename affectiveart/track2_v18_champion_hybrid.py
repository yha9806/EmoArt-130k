from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_EMOTIONS, TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_audit import strict_track2_label_issues
from affectiveart.track2_v17_classification_calibration import (
    DEFAULT_DUPLICATE_SOURCES as V17_DEFAULT_DUPLICATE_SOURCES,
    DEFAULT_PREDICTION_SOURCES as V17_DEFAULT_PREDICTION_SOURCES,
    FORMAL_SUBMISSION_NAMES,
    apply_v17_changes,
    build_evidence_rows,
    load_csv_rows,
    load_duplicate_rows,
    load_prediction_sources,
    load_track2_rows,
    parse_official_failed_transition_counts,
)


DEFAULT_EXPERIMENT_DIR = Path("experiments/track2_v18_champion_hybrid_20260607")
DEFAULT_LEADERBOARD = Path("experiments/track2_official_results_20260606/track2_public_leaderboard_20260606.csv")
DEFAULT_OFFICIAL_SCORES = Path(
    "experiments/track2_official_results_20260606/track2_known_official_exact_scores_from_ledger_20260606.csv"
)
DEFAULT_PAIRWISE_DIFFS = Path(
    "experiments/track2_official_results_20260606/track2_my_submission_pairwise_diffs_20260606.csv"
)
DEFAULT_BASE_CANDIDATES = {
    "779605": Path("submissions/track2_submission_moe_v2_accept5_candidate.json"),
    "782683": Path("submissions/track2_submission_v12_stable_probe_candidate.json"),
    "v15_desc_expand300": Path("submissions/track2_submission_v15_desc_expand300_candidate.json"),
}
DEFAULT_OUT_JSON = Path("submissions/track2_submission_v18_champion_hybrid_candidate.json")
DEFAULT_OUT_ZIP = Path("submissions/track2_submission_v18_champion_hybrid_candidate.zip")
NO_AUTO_SUBMIT_POLICY = True


@dataclass(frozen=True)
class OfficialAnchor:
    submission_id: str
    file_name: str
    overall: float
    classification: float
    description: float


@dataclass(frozen=True)
class ChampionTargets:
    first_participant: str
    current_participant: str
    first_overall: float
    current_overall: float
    first_classification: float
    current_classification: float
    first_description: float
    current_description: float
    first_emotion_accuracy: float
    current_emotion_accuracy: float
    first_emotion_macro_f1: float
    current_emotion_macro_f1: float
    first_visual_grounding: float
    current_visual_grounding: float
    first_attribute_specificity: float
    current_attribute_specificity: float
    first_overall_caption: float
    current_overall_caption: float


def load_official_anchors(path: str | Path) -> dict[str, OfficialAnchor]:
    anchors: dict[str, OfficialAnchor] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            submission_id = str(row.get("submission_id", "")).strip()
            if not submission_id:
                continue
            anchors[submission_id] = OfficialAnchor(
                submission_id=submission_id,
                file_name=str(row.get("file_name", "")).strip(),
                overall=_safe_float(row.get("official_overall")),
                classification=_safe_float(row.get("official_classification")),
                description=_safe_float(row.get("official_description")),
            )
    return anchors


def load_champion_targets(path: str | Path, participant: str = "vulcaart") -> ChampionTargets:
    rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"leaderboard is empty: {path}")
    first = min(rows, key=lambda row: _safe_int(row.get("#"), default=999999))
    current = next((row for row in rows if str(row.get("Participant", "")).strip() == participant), None)
    if current is None:
        raise ValueError(f"participant not found in leaderboard: {participant}")
    return ChampionTargets(
        first_participant=str(first.get("Participant", "")).strip(),
        current_participant=str(current.get("Participant", "")).strip(),
        first_overall=_safe_float(first.get("Overall Score")),
        current_overall=_safe_float(current.get("Overall Score")),
        first_classification=_safe_float(first.get("Classification Score")),
        current_classification=_safe_float(current.get("Classification Score")),
        first_description=_safe_float(first.get("Description Score")),
        current_description=_safe_float(current.get("Description Score")),
        first_emotion_accuracy=_safe_float(first.get("Emotion Accuracy")),
        current_emotion_accuracy=_safe_float(current.get("Emotion Accuracy")),
        first_emotion_macro_f1=_safe_float(first.get("Emotion Macro F1")),
        current_emotion_macro_f1=_safe_float(current.get("Emotion Macro F1")),
        first_visual_grounding=_safe_float(first.get("Visual Grounding")),
        current_visual_grounding=_safe_float(current.get("Visual Grounding")),
        first_attribute_specificity=_safe_float(first.get("Attribute Specificity")),
        current_attribute_specificity=_safe_float(current.get("Attribute Specificity")),
        first_overall_caption=_safe_float(first.get("Overall Caption")),
        current_overall_caption=_safe_float(current.get("Overall Caption")),
    )


def choose_v18_base(
    anchors: dict[str, OfficialAnchor],
    available: dict[str, Path],
) -> dict[str, str]:
    usable = [
        (anchor.overall, anchor.classification, anchor.description, submission_id, path)
        for submission_id, anchor in anchors.items()
        for key, path in available.items()
        if key == submission_id
    ]
    if not usable:
        fallback = available.get("v15_desc_expand300")
        if fallback is None:
            raise ValueError("no usable v18 base candidates")
        return {"submission_id": "v15_desc_expand300", "base_json": str(fallback), "reason": "fallback_v15_desc_expand300"}
    _, _, _, submission_id, path = max(usable)
    return {"submission_id": submission_id, "base_json": str(path), "reason": "highest_exact_official_overall"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
