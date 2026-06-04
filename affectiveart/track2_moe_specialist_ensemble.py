from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
from typing import Any
import zipfile

from affectiveart.challenge import TRACK2_JSON_EMOTIONS
from affectiveart.track2_audit import (
    HIGH_AROUSAL_EMOTIONS,
    LOW_AROUSAL_EMOTIONS,
    NEGATIVE_EMOTIONS,
    POSITIVE_EMOTIONS,
    compute_track2_distribution,
)
from affectiveart.track2_visual_audit import find_track2_image_member


VALID_TRACK2_EMOTIONS = TRACK2_JSON_EMOTIONS
SINGLE_SOURCE_SPECIALIST_ROLES = frozenset(
    {"boundary", "tail", "va", "description", "specialist"}
)
DRY_RUN_OUTPUT_FILENAMES = {
    "json": "track2_moe_specialist_dry_run_report.json",
    "markdown": "track2_moe_specialist_dry_run_report.md",
    "html": "html_review/track2_moe_specialist_dry_run_review.html",
}


@dataclass(frozen=True)
class GateThresholds:
    min_supporting_sources: int = 2
    min_support_confidence: float = 0.50
    high_confidence: float = 0.86
    min_margin: float = 0.12
    min_macro_f1_gain: float = 0.015
    min_hardcase_macro_f1_gain: float = 0.030
    max_accuracy_drop: float = 0.010
    max_valence_accuracy_drop: float = 0.005
    max_arousal_accuracy_drop: float = 0.005
    max_top_emotion_share_delta: float = 0.020


def expected_label_for_emotion(emotion: str) -> tuple[str, str]:
    normalized = str(emotion).strip().lower()
    if normalized in POSITIVE_EMOTIONS:
        valence = "Positive"
    elif normalized in NEGATIVE_EMOTIONS:
        valence = "Negative"
    else:
        raise ValueError(f"unsupported Track2 emotion: {emotion}")

    if normalized in HIGH_AROUSAL_EMOTIONS:
        arousal = "High"
    elif normalized in LOW_AROUSAL_EMOTIONS:
        arousal = "Low"
    else:
        raise ValueError(f"unsupported Track2 emotion: {emotion}")

    return valence, arousal


def normalize_expert_entries(payload: Any, source: str, role: str) -> list[dict[str, Any]]:
    rows = _extract_payload_rows(payload)
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        emotion = str(row.get("emotion", "")).strip().lower()
        if emotion not in VALID_TRACK2_EMOTIONS:
            continue
        confidence = _safe_float(row.get("confidence"))

        normalized_rows.append(
            {
                "sample_id": sample_id,
                "source": str(source),
                "role": str(role),
                "emotion": emotion,
                "confidence": confidence,
                "margin": _safe_float(row.get("margin")),
                "top3": _normalize_top3(
                    row.get("top3"),
                    fallback=emotion,
                    fallback_probability=confidence,
                ),
            }
        )
    return normalized_rows


def build_gate_decision(
    current_row: dict[str, Any],
    expert_rows: list[dict[str, Any]],
    *,
    thresholds: GateThresholds | None = None,
    description_audit: dict[str, Any] | None = None,
    high_similarity_public_reference: bool = False,
) -> dict[str, Any]:
    thresholds = thresholds or GateThresholds()
    sample_id = str(current_row.get("sample_id", "")).strip()
    current_emotion = str(current_row.get("emotion", "")).strip().lower()
    relevant_rows = _relevant_expert_rows(sample_id, expert_rows)
    evidence = _summarize_evidence(relevant_rows)
    proposed_emotion = _choose_proposal(current_emotion, evidence)

    if not proposed_emotion:
        return _decision_row(
            current_row,
            relevant_rows,
            "keep_current",
            current_emotion,
            ["no_supported_change"],
        )

    reasons: list[str] = []
    support = evidence.get(proposed_emotion, [])
    supporting_source_count = _supporting_source_count(support)
    quality_supporting_source_count = _supporting_source_count(
        _quality_support_rows(support, thresholds)
    )
    strong_opposition = _strong_opposition(
        proposed_emotion,
        evidence,
        thresholds,
    )
    if supporting_source_count >= thresholds.min_supporting_sources:
        if quality_supporting_source_count >= thresholds.min_supporting_sources:
            reasons.append(f"supported_by_{quality_supporting_source_count}_sources")
        else:
            reasons.append("support_below_quality_bar")
    elif _has_single_high_confidence_source(
        proposed_emotion,
        evidence,
        thresholds,
    ):
        if strong_opposition:
            reasons.append("insufficient_independent_support")
        elif _has_single_high_confidence_specialist_source(
            proposed_emotion,
            evidence,
            thresholds,
        ):
            reasons.append("single_high_confidence_source_without_strong_opposition")
        else:
            reasons.append("single_source_not_specialist")
    else:
        reasons.append("insufficient_independent_support")
    if strong_opposition:
        reasons.append("strong_opposition")

    if _description_verdict(description_audit) == "contradiction":
        reasons.append("description_contradiction")
    if high_similarity_public_reference:
        reasons.append("high_similarity_requires_explicit_review")

    blocking_reasons = {
        "insufficient_independent_support",
        "single_source_not_specialist",
        "strong_opposition",
        "support_below_quality_bar",
        "description_contradiction",
        "high_similarity_requires_explicit_review",
    }
    decision = "hold" if blocking_reasons.intersection(reasons) else "accept_change"
    proposed_valence, proposed_arousal = expected_label_for_emotion(proposed_emotion)
    return _decision_row(
        current_row,
        relevant_rows,
        decision,
        proposed_emotion,
        reasons,
        proposed_valence=proposed_valence,
        proposed_arousal=proposed_arousal,
    )


def build_dry_run_report(
    current_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    *,
    queue_sample_ids,
    high_similarity_sample_ids=None,
    description_audit_by_id=None,
) -> dict[str, Any]:
    queued_ids = _normalized_unique_sample_ids(queue_sample_ids)
    high_similarity_ids = set(
        _normalized_unique_sample_ids(high_similarity_sample_ids or set())
    )
    description_audits = _normalize_sample_id_mapping(description_audit_by_id)
    current_by_id = {
        str(row.get("sample_id", "")).strip(): row
        for row in current_rows
        if isinstance(row, dict) and str(row.get("sample_id", "")).strip()
    }

    decisions: list[dict[str, Any]] = []
    missing_current_sample_ids: list[str] = []
    for sample_id in queued_ids:
        if sample_id not in current_by_id:
            missing_current_sample_ids.append(sample_id)
            continue
        decisions.append(
            build_gate_decision(
                current_by_id[sample_id],
                expert_rows,
                description_audit=description_audits.get(sample_id),
                high_similarity_public_reference=sample_id in high_similarity_ids,
            )
        )

    decision_counter = Counter(row["decision"] for row in decisions)
    decision_counts = {
        decision: decision_counter.get(decision, 0)
        for decision in ("accept_change", "hold", "keep_current")
    }
    decision_counts.update(
        {
            decision: count
            for decision, count in decision_counter.items()
            if decision not in decision_counts
        }
    )
    accepted_transition_counts = Counter(
        f"{row['current_emotion']}->{row['proposed_emotion']}"
        for row in decisions
        if row["decision"] == "accept_change"
    )
    return {
        "method": "track2_moe_specialist_dry_run_v1",
        "row_count": len(decisions),
        "decision_counts": dict(decision_counts),
        "accepted_transition_counts": dict(accepted_transition_counts),
        "baseline_distribution": compute_track2_distribution(current_rows),
        "missing_current_sample_ids": missing_current_sample_ids,
        "formal_submission_overwritten": False,
        "candidate_json_written": False,
        "candidate_zip_written": False,
        "rows": decisions,
    }


def write_dry_run_outputs(
    report: dict[str, Any],
    *,
    image_zip: str | Path,
    out_dir: str | Path,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    _validate_dry_run_out_dir(out_dir)
    html_dir = out_dir / "html_review"
    assets_dir = html_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        key: str(out_dir / filename)
        for key, filename in DRY_RUN_OUTPUT_FILENAMES.items()
    }
    rows = [dict(row) for row in report.get("rows", []) if isinstance(row, dict)]
    asset_by_sample_id = _extract_review_assets(rows, image_zip, assets_dir)
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id in asset_by_sample_id:
            row["image_asset"] = asset_by_sample_id[sample_id]

    output_report = dict(report)
    output_report["outputs"] = outputs
    output_report["rows"] = rows

    Path(outputs["json"]).write_text(
        json.dumps(output_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(outputs["markdown"]).write_text(
        render_dry_run_markdown(output_report, outputs),
        encoding="utf-8",
    )
    Path(outputs["html"]).write_text(
        render_dry_run_html(output_report),
        encoding="utf-8",
    )
    return outputs


def render_dry_run_markdown(
    report: dict[str, Any],
    outputs: dict[str, str],
) -> str:
    rows = _dry_run_rows(report)
    lines = [
        "# Track2 MoE Specialist Dry-Run",
        "",
        f"- rows: {report.get('row_count', len(rows))}",
        f"- formal_submission_overwritten: {bool(report.get('formal_submission_overwritten', False))}",
        f"- candidate_json_written: {bool(report.get('candidate_json_written', False))}",
        f"- candidate_zip_written: {bool(report.get('candidate_zip_written', False))}",
        f"- HTML review: {outputs.get('html', '')}",
        "",
        "## Decision Counts",
    ]

    decision_counts = report.get("decision_counts", {})
    if isinstance(decision_counts, dict) and decision_counts:
        for decision, count in sorted(decision_counts.items()):
            lines.append(f"- {decision}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Accepted Transitions"])
    transitions = report.get("accepted_transition_counts", {})
    if isinstance(transitions, dict) and transitions:
        for transition, count in sorted(transitions.items()):
            lines.append(f"- {transition}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Rows"])
    if not rows:
        lines.append("- none")
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        lines.extend(
            [
                "",
                f"### {sample_id}",
                f"- decision: {row.get('decision', '')}",
                (
                    "- current: "
                    f"{row.get('current_emotion', '')} / "
                    f"{row.get('current_valence', '')} / "
                    f"{row.get('current_arousal', '')}"
                ),
                (
                    "- proposed: "
                    f"{row.get('proposed_emotion', '')} / "
                    f"{row.get('proposed_valence', '')} / "
                    f"{row.get('proposed_arousal', '')}"
                ),
                f"- reasons: {', '.join(_string_list(row.get('reasons'))) or 'none'}",
            ]
        )

    return "\n".join(lines) + "\n"


def render_dry_run_html(report: dict[str, Any]) -> str:
    rows = _dry_run_rows(report)
    cards = "\n".join(_render_dry_run_card(row) for row in rows)
    decision_counts = report.get("decision_counts", {})
    count_pills = ""
    if isinstance(decision_counts, dict):
        count_pills = "\n".join(
            f'<span class="pill">{_escape(decision)}: {_escape(count)}</span>'
            for decision, count in sorted(decision_counts.items())
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track2 MoE Specialist Dry-Run</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #52606d;
      --line: #d7dee8;
      --panel: #ffffff;
      --page: #f5f7fa;
      --accent: #0f766e;
      --hold: #925a14;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--page);
    }}
    header {{
      padding: 20px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 20px auto 36px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 26px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 5px 9px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.2;
    }}
    .notice {{
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .card {{
      display: grid;
      grid-template-columns: minmax(180px, 260px) minmax(0, 1fr);
      gap: 18px;
      margin: 0 0 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px;
    }}
    .artwork {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #eef2f6;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0 12px;
    }}
    .decision {{
      border-color: color-mix(in srgb, var(--accent) 35%, var(--line));
      color: var(--accent);
      font-weight: 650;
    }}
    .decision.hold {{
      border-color: color-mix(in srgb, var(--hold) 35%, var(--line));
      color: var(--hold);
    }}
    .labels {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 0 0 12px;
    }}
    .label-box {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
    }}
    .label-title {{
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .triplet {{
      margin: 0;
      font-size: 15px;
      line-height: 1.35;
    }}
    .reasons {{
      margin: 0 0 12px 18px;
      padding: 0;
    }}
    .reasons li {{
      margin: 4px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-top: 1px solid var(--line);
      padding: 7px 6px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 650;
      background: #f7f9fc;
    }}
    @media (max-width: 760px) {{
      .card, .labels {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Track2 MoE Specialist Dry-Run</h1>
    <div class="summary">
      <span class="pill">Rows: {_escape(report.get('row_count', len(rows)))}</span>
      {count_pills}
    </div>
    <p class="notice">Dry-run review only. No submission JSON or ZIP is written.</p>
  </header>
  <main>
    {cards}
  </main>
</body>
</html>
"""
    return html_text


def _normalized_unique_sample_ids(values) -> list[str]:
    normalized_ids = [
        str(value).strip()
        for value in (values or [])
        if str(value).strip()
    ]
    return sorted(dict.fromkeys(normalized_ids))


def _normalize_sample_id_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        str(sample_id).strip(): payload
        for sample_id, payload in value.items()
        if str(sample_id).strip()
    }


def _relevant_expert_rows(
    sample_id: str,
    expert_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    relevant_rows: list[dict[str, Any]] = []
    for row in expert_rows or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("sample_id", "")).strip() != sample_id:
            continue
        normalized = _normalize_gate_evidence_row(row)
        if normalized:
            relevant_rows.append(normalized)
    return relevant_rows


def _normalize_gate_evidence_row(row: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = str(row.get("sample_id", "")).strip()
    raw_source = row.get("source", "")
    source = "" if raw_source is None else str(raw_source).strip()
    emotion = str(row.get("emotion", "")).strip().lower()
    if not sample_id or not source or emotion not in VALID_TRACK2_EMOTIONS:
        return None
    confidence = _safe_float(row.get("confidence"))
    return {
        "sample_id": sample_id,
        "source": source,
        "role": str(row.get("role", "")).strip(),
        "emotion": emotion,
        "confidence": confidence,
        "margin": _safe_float(row.get("margin")),
        "top3": _normalize_top3(
            row.get("top3"),
            fallback=emotion,
            fallback_probability=confidence,
        ),
    }


def _summarize_evidence(
    expert_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {}
    for row in expert_rows:
        emotion = str(row.get("emotion", "")).strip().lower()
        if emotion in VALID_TRACK2_EMOTIONS:
            evidence.setdefault(emotion, []).append(row)
    return evidence


def _choose_proposal(
    current_emotion: str,
    evidence: dict[str, list[dict[str, Any]]],
) -> str:
    candidates = [emotion for emotion in evidence if emotion != current_emotion]
    if not candidates:
        return ""
    return sorted(
        candidates,
        key=lambda emotion: (
            -_supporting_source_count(evidence[emotion]),
            -sum(_safe_float(row.get("confidence")) for row in evidence[emotion]),
            emotion,
        ),
    )[0]


def _supporting_source_count(rows: list[dict[str, Any]]) -> int:
    return len({_source_key(row) for row in rows})


def _source_key(row: dict[str, Any]) -> str:
    return str(row.get("source", "")).strip()


def _has_single_high_confidence_source(
    proposed_emotion: str,
    evidence: dict[str, list[dict[str, Any]]],
    thresholds: GateThresholds,
) -> bool:
    support = evidence.get(proposed_emotion, [])
    if _supporting_source_count(support) != 1:
        return False
    return any(
        _safe_float(row.get("confidence")) >= thresholds.high_confidence
        and _safe_float(row.get("margin")) >= thresholds.min_margin
        for row in support
    )


def _has_single_high_confidence_specialist_source(
    proposed_emotion: str,
    evidence: dict[str, list[dict[str, Any]]],
    thresholds: GateThresholds,
) -> bool:
    support = evidence.get(proposed_emotion, [])
    if _supporting_source_count(support) != 1:
        return False
    return any(
        _safe_float(row.get("confidence")) >= thresholds.high_confidence
        and _safe_float(row.get("margin")) >= thresholds.min_margin
        and _is_single_source_specialist_role(row)
        for row in support
    )


def _is_single_source_specialist_role(row: dict[str, Any]) -> bool:
    return (
        str(row.get("role", "")).strip().lower()
        in SINGLE_SOURCE_SPECIALIST_ROLES
    )


def _quality_support_rows(
    rows: list[dict[str, Any]],
    thresholds: GateThresholds,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _safe_float(row.get("confidence")) >= thresholds.min_support_confidence
        and _safe_float(row.get("margin")) >= thresholds.min_margin
    ]


def _strong_opposition(
    proposed_emotion: str,
    evidence: dict[str, list[dict[str, Any]]],
    thresholds: GateThresholds,
) -> bool:
    for emotion, rows in evidence.items():
        if emotion == proposed_emotion:
            continue
        for row in rows:
            if (
                _safe_float(row.get("confidence")) >= thresholds.high_confidence
                and _safe_float(row.get("margin")) >= thresholds.min_margin
            ):
                return True
    return False


def _description_verdict(description_audit: dict[str, Any] | None) -> str:
    if not isinstance(description_audit, dict):
        return ""
    return str(description_audit.get("verdict", "")).strip().lower()


def _decision_row(
    current_row: dict[str, Any],
    expert_rows: list[dict[str, Any]],
    decision: str,
    proposed_emotion: str,
    reasons: list[str],
    *,
    proposed_valence: str | None = None,
    proposed_arousal: str | None = None,
) -> dict[str, Any]:
    if proposed_valence is None or proposed_arousal is None:
        proposed_valence, proposed_arousal = expected_label_for_emotion(
            proposed_emotion
        )
    return {
        "sample_id": str(current_row.get("sample_id", "")).strip(),
        "decision": decision,
        "current_emotion": str(current_row.get("emotion", "")).strip().lower(),
        "current_valence": str(current_row.get("emotional_valence", "")).strip(),
        "current_arousal": str(
            current_row.get("emotional_arousal_level", "")
        ).strip(),
        "proposed_emotion": proposed_emotion,
        "proposed_valence": proposed_valence,
        "proposed_arousal": proposed_arousal,
        "reasons": list(reasons),
        "expert_evidence": sorted(
            expert_rows,
            key=lambda row: (
                str(row.get("source", "")),
                str(row.get("emotion", "")),
                str(row.get("role", "")),
            ),
        ),
    }


def _extract_payload_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("entries", "predictions", "rows"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _normalize_top3(
    value: Any,
    fallback: str,
    fallback_probability: float = 0.0,
) -> list[dict[str, float | str]]:
    entries = value if isinstance(value, list) else [value]
    normalized: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for entry in entries:
        emotion, probability = _top3_entry(entry)
        if emotion in VALID_TRACK2_EMOTIONS and emotion not in seen:
            normalized.append(
                {
                    "emotion": emotion,
                    "probability": _safe_float(probability),
                }
            )
            seen.add(emotion)
    if not normalized:
        fallback_emotion = str(fallback).strip().lower()
        if fallback_emotion not in VALID_TRACK2_EMOTIONS:
            return []
        normalized.append(
            {
                "emotion": fallback_emotion,
                "probability": _safe_float(fallback_probability),
            }
        )
    return normalized[:3]


def _top3_entry(value: Any) -> tuple[str, Any]:
    if isinstance(value, dict):
        for key in ("emotion", "label", "class"):
            if value.get(key):
                return str(value[key]).strip().lower(), _top3_probability(value)
        return "", None
    return str(value).strip().lower(), None


def _top3_probability(value: dict[str, Any]) -> Any:
    for key in ("probability", "prob", "confidence", "score"):
        if key in value:
            return value[key]
    return None


def _extract_review_assets(
    rows: list[dict[str, Any]],
    image_zip: str | Path,
    assets_dir: Path,
) -> dict[str, str]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    asset_by_sample_id: dict[str, str] = {}
    seen_sample_ids: set[str] = set()
    with zipfile.ZipFile(image_zip) as zf:
        names = zf.namelist()
        for row in rows:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id or sample_id in seen_sample_ids:
                continue
            seen_sample_ids.add(sample_id)
            member = find_track2_image_member(names, sample_id)
            if not member:
                continue
            target = assets_dir / _safe_asset_filename(sample_id)
            with zf.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            asset_by_sample_id[sample_id] = f"assets/{target.name}"
    return asset_by_sample_id


def _safe_asset_filename(sample_id: str) -> str:
    raw_sample_id = str(sample_id).strip()
    safe_stem = "".join(
        character
        if character.isascii()
        and (character.isalnum() or character in {"_", "-"})
        else "_"
        for character in raw_sample_id
    )
    if not safe_stem:
        safe_stem = "sample"
    suffix = hashlib.sha256(raw_sample_id.encode("utf-8")).hexdigest()
    return f"{safe_stem}-{suffix}.jpg"


def _validate_dry_run_out_dir(out_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    submissions_dir = (repo_root / "submissions").resolve(strict=False)
    resolved_out_dir = out_dir.expanduser().resolve(strict=False)
    try:
        resolved_out_dir.relative_to(submissions_dir)
    except ValueError:
        return
    raise ValueError("dry-run output must not be under submissions/")


def _dry_run_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _render_dry_run_card(row: dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    decision = str(row.get("decision", "")).strip()
    decision_class = " hold" if decision == "hold" else ""
    image_src = str(row.get("image_asset") or f"assets/{_safe_asset_filename(sample_id)}")
    reasons = _string_list(row.get("reasons"))
    reason_items = "\n".join(f"<li>{_escape(reason)}</li>" for reason in reasons)
    if not reason_items:
        reason_items = "<li>none</li>"

    evidence = row.get("expert_evidence", [])
    evidence_rows = ""
    if isinstance(evidence, list):
        evidence_rows = "\n".join(
            _render_evidence_row(expert_row)
            for expert_row in evidence
            if isinstance(expert_row, dict)
        )
    if not evidence_rows:
        evidence_rows = '<tr><td colspan="6">No expert evidence</td></tr>'

    return f"""<section class="card" id="{_escape(sample_id)}">
  <img class="artwork" src="{_escape(image_src)}" alt="{_escape(sample_id)}">
  <div>
    <h2>{_escape(sample_id)}</h2>
    <div class="meta">
      <span class="pill decision{decision_class}">{_escape(decision)}</span>
    </div>
    <div class="labels">
      <div class="label-box">
        <p class="label-title">Current label</p>
        <p class="triplet">{_escape(row.get('current_emotion', ''))} / {_escape(row.get('current_valence', ''))} / {_escape(row.get('current_arousal', ''))}</p>
      </div>
      <div class="label-box">
        <p class="label-title">Proposed label</p>
        <p class="triplet">{_escape(row.get('proposed_emotion', ''))} / {_escape(row.get('proposed_valence', ''))} / {_escape(row.get('proposed_arousal', ''))}</p>
      </div>
    </div>
    <ul class="reasons">
      {reason_items}
    </ul>
    <table>
      <thead>
        <tr>
          <th>Source</th>
          <th>Role</th>
          <th>Emotion</th>
          <th>Confidence</th>
          <th>Margin</th>
          <th>Top 3</th>
        </tr>
      </thead>
      <tbody>
        {evidence_rows}
      </tbody>
    </table>
  </div>
</section>"""


def _render_evidence_row(row: dict[str, Any]) -> str:
    return f"""<tr>
  <td>{_escape(row.get('source', ''))}</td>
  <td>{_escape(row.get('role', ''))}</td>
  <td>{_escape(row.get('emotion', ''))}</td>
  <td>{_escape(_format_number(row.get('confidence')))}</td>
  <td>{_escape(_format_number(row.get('margin')))}</td>
  <td>{_escape(_format_top3(row.get('top3')))}</td>
</tr>"""


def _format_top3(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    entries: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        emotion = str(item.get("emotion", "")).strip()
        probability = _format_number(item.get("probability"))
        if emotion and probability:
            entries.append(f"{emotion} {probability}")
        elif emotion:
            entries.append(emotion)
    return ", ".join(entries)


def _format_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    number = _safe_float(value)
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if str(value or "").strip():
        return [str(value).strip()]
    return []


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


__all__ = [
    "DRY_RUN_OUTPUT_FILENAMES",
    "GateThresholds",
    "build_dry_run_report",
    "build_gate_decision",
    "expected_label_for_emotion",
    "normalize_expert_entries",
    "render_dry_run_html",
    "render_dry_run_markdown",
    "write_dry_run_outputs",
]
