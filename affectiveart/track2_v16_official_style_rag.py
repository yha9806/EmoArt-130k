from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from affectiveart.challenge import TRACK2_JSON_EMOTIONS, TRACK2_JSON_SUBMISSION_KEYS
from affectiveart.track2_audit import HIGH_AROUSAL_EMOTIONS, NEGATIVE_EMOTIONS, strict_track2_label_issues
from affectiveart.track2_multimodal_description import normalize_image_for_gemini


DEFAULT_PREDICTION_SOURCES = {
    "siglip2": Path("experiments/track2_emoart130k_siglip2/predictions.json"),
    "clip": Path("experiments/track2_emoart130k_clip/predictions.json"),
    "dinov2": Path("experiments/track2_emoart130k_dinov2/predictions.json"),
    "gemini35_specialist": Path(
        "experiments/track2_moe_specialist_ensemble_20260603/dry_run_v2/gemini35_vlm_specialist_predictions.json"
    ),
}
DEFAULT_DUPLICATE_SOURCES = [
    Path("experiments/track2_emoart130k_clip/deep_duplicate_audit_20260511/track2_deep_duplicate_top10_audit.json"),
    Path("experiments/track2_emoart130k_clip/overlap_reference/ge095_all/ge095_public_neighbor_audit.json"),
]
DEFAULT_CALIBRATION_REPORT = Path(
    "experiments/track2_v16_official_style_rag_20260607/calibration/official_counterfactual_report.json"
)
V16_RAG_MODEL = "gemini-3.5-flash"
FORMAL_SUBMISSION_NAMES = {"track2_submission.json", "track2_submission.zip"}
V16_RAG_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string"},
        "emotion": {"type": "string"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "visual_evidence": {"type": "string"},
        "boundary_analysis": {"type": "string"},
    },
    "required": [
        "decision",
        "emotion",
        "confidence",
        "reason",
        "visual_evidence",
        "boundary_analysis",
    ],
}
V16_RAG_PROMPT = """You are auditing one AffectiveArt Track2 emotion label proposal.

Use only visual evidence from the artwork and the supplied evidence summary.

Task:
- Decide whether to keep the current label or change to the proposed label.
- The target label set is exactly:
  aroused, excited, happy, alarmed, annoyed, frustrated, sad, bored, tired, content, calm, glad.
- Be especially conservative on calm/content boundary cases because a previous bulk same-quadrant submission reduced the official classification score.
- Do not write instructions to an evaluator or scorer.

Return exactly one JSON object:
- decision: one of keep_current, use_proposed, use_other
- emotion: the final chosen emotion label
- confidence: decimal from 0.0 to 1.0
- reason: concise explanation
- visual_evidence: concrete visible cues
- boundary_analysis: why the chosen label is better than the alternative

Current submission row:
{row_json}

Queue evidence:
{queue_json}
"""

@dataclass(frozen=True)
class OfficialSubmissionScore:
    submission_id: str
    overall: float
    classification: float
    description: float


def official_delta_summary(
    *,
    anchor: OfficialSubmissionScore,
    candidate: OfficialSubmissionScore,
    changed_rows: int,
) -> dict[str, Any]:
    changed = max(1, int(changed_rows))
    classification_delta = float(candidate.classification) - float(anchor.classification)
    return {
        "anchor_submission_id": anchor.submission_id,
        "candidate_submission_id": candidate.submission_id,
        "classification_delta": classification_delta,
        "classification_delta_per_changed_row": classification_delta / changed,
        "overall_delta": float(candidate.overall) - float(anchor.overall),
        "description_delta": float(candidate.description) - float(anchor.description),
        "changed_rows": int(changed_rows),
        "batch_verdict": "negative_official_evidence" if classification_delta < 0 else "nonnegative_official_evidence",
    }


def classify_transition_risk(
    *,
    transition: str,
    evidence_sources: set[str],
    official_failed_transition_count: int = 0,
) -> str:
    del transition
    if official_failed_transition_count >= 10 and evidence_sources <= {"same_quadrant_batch"}:
        return "blocked_by_failed_official_batch"
    if "exact_public_duplicate" in evidence_sources:
        return "allow_exact_duplicate"
    if {"rag_teacher", "multibackbone_consensus"} <= evidence_sources:
        return "allow_strong_consensus"
    return "hold_needs_more_evidence"


def load_official_scores(path: str | Path) -> dict[str, OfficialSubmissionScore]:
    scores: dict[str, OfficialSubmissionScore] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            submission_id = str(row.get("submission_id", "")).strip()
            if not submission_id:
                continue
            scores[submission_id] = OfficialSubmissionScore(
                submission_id=submission_id,
                overall=_safe_float(row.get("official_overall")),
                classification=_safe_float(row.get("official_classification")),
                description=_safe_float(row.get("official_description")),
            )
    return scores


def load_pairwise_diff_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "candidate_a": str(row.get("candidate_a", "")).strip(),
                    "candidate_b": str(row.get("candidate_b", "")).strip(),
                    "candidate_a_submission_id": _extract_submission_id(row.get("candidate_a")),
                    "candidate_b_submission_id": _extract_submission_id(row.get("candidate_b")),
                    "emotion_label_changes": int(_safe_float(row.get("emotion_label_changes"))),
                    "top_emotion_transitions": str(row.get("top_emotion_transitions", "")).strip(),
                    "transition_counts": parse_transition_counts(row.get("top_emotion_transitions", "")),
                }
            )
    return rows


def parse_transition_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in str(value or "").split(";"):
        item = part.strip()
        if not item or ":" not in item:
            continue
        transition, count_text = item.rsplit(":", 1)
        transition = transition.strip()
        if "->" not in transition:
            continue
        counts[transition] = int(_safe_float(count_text))
    return counts


def build_official_counterfactual_report(
    *,
    official_scores: dict[str, OfficialSubmissionScore],
    pairwise_diff_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not official_scores:
        raise ValueError("official scores are empty")
    anchor = max(official_scores.values(), key=lambda item: (item.classification, item.overall, item.submission_id))
    failed_batches: list[dict[str, Any]] = []
    failed_transition_counts: Counter[str] = Counter()

    for row in pairwise_diff_rows:
        candidate_id = str(row.get("candidate_a_submission_id", ""))
        anchor_id = str(row.get("candidate_b_submission_id", ""))
        if candidate_id not in official_scores or anchor_id not in official_scores:
            continue
        summary = official_delta_summary(
            anchor=official_scores[anchor_id],
            candidate=official_scores[candidate_id],
            changed_rows=int(row.get("emotion_label_changes") or 0),
        )
        summary["candidate_a"] = row.get("candidate_a", "")
        summary["candidate_b"] = row.get("candidate_b", "")
        summary["transition_counts"] = row.get("transition_counts", {})
        if summary["batch_verdict"] == "negative_official_evidence":
            failed_batches.append(summary)
            for transition, count in dict(row.get("transition_counts", {})).items():
                failed_transition_counts[str(transition)] += int(count)

    return {
        "method": "track2_v16_official_counterfactual_calibration_v1",
        "anchor_submission_id": anchor.submission_id,
        "anchor_classification": anchor.classification,
        "official_score_count": len(official_scores),
        "pairwise_diff_count": len(pairwise_diff_rows),
        "known_failed_batches": failed_batches,
        "failed_transition_counts": dict(sorted(failed_transition_counts.items())),
        "policy": (
            "Official aggregate results show that the 781601 same-quadrant batch reduced classification. "
            "v16 therefore blocks failed high-count boundary transitions unless stronger per-sample evidence is present."
        ),
    }


def build_rag_queue_rows(
    *,
    current_rows: list[dict[str, Any]],
    prediction_rows: dict[str, Any],
    failed_transition_counts: dict[str, int] | None = None,
    duplicate_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    failed_counts = failed_transition_counts or {}
    duplicate_by_id = _index_rows_by_sample_id(duplicate_rows or [])
    output: list[dict[str, Any]] = []

    for row in current_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        current = _canonical_emotion(row.get("emotion"))
        if not sample_id or not current:
            continue
        proposals = _proposal_scores_for_sample(
            sample_id=sample_id,
            current=current,
            prediction_rows=prediction_rows,
            duplicate_rows=duplicate_by_id.get(sample_id, []),
        )
        for proposed, payload in proposals.items():
            if proposed == current:
                continue
            transition = f"{current}->{proposed}"
            sources = set(payload["sources"])
            risk_gate = classify_transition_risk(
                transition=transition,
                evidence_sources=sources,
                official_failed_transition_count=int(failed_counts.get(transition, 0)),
            )
            priority = _queue_priority(
                sources=sources,
                risk_gate=risk_gate,
                confidence=float(payload["confidence"]),
                same_valence=_valence(current) == _valence(proposed),
                same_arousal=_arousal(current) == _arousal(proposed),
            )
            output.append(
                {
                    "sample_id": sample_id,
                    "current_emotion": current,
                    "proposed_emotion": proposed,
                    "transition": transition,
                    "risk_gate": risk_gate,
                    "priority_score": round(priority, 6),
                    "confidence": round(float(payload["confidence"]), 6),
                    "sources": ",".join(sorted(sources)),
                    "public_neighbor_labels": ",".join(sorted(payload["public_neighbor_labels"])),
                    "same_valence": _valence(current) == _valence(proposed),
                    "same_arousal": _arousal(current) == _arousal(proposed),
                    "reason": payload["reason"],
                }
            )
    return sorted(output, key=lambda item: (-float(item["priority_score"]), item["sample_id"], item["proposed_emotion"]))


def write_rag_queue_outputs(
    *,
    base_json: str | Path,
    out_dir: str | Path,
    limit: int = 160,
    prediction_sources: dict[str, str | Path] | None = None,
    duplicate_sources: list[str | Path] | None = None,
    calibration_report: str | Path | None = DEFAULT_CALIBRATION_REPORT,
) -> dict[str, Any]:
    rows = _load_json_list(Path(base_json))
    predictions = load_prediction_sources(prediction_sources or DEFAULT_PREDICTION_SOURCES)
    duplicates = load_duplicate_rows(duplicate_sources or DEFAULT_DUPLICATE_SOURCES)
    failed_counts = load_failed_transition_counts(calibration_report)
    queue = build_rag_queue_rows(
        current_rows=rows,
        prediction_rows=predictions,
        failed_transition_counts=failed_counts,
        duplicate_rows=duplicates,
    )[: max(0, int(limit))]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"rag_teacher_queue_top{len(queue)}"
    _write_json(out_dir / f"{prefix}.json", queue)
    _write_queue_csv(out_dir / f"{prefix}.csv", queue)
    (out_dir / f"{prefix}.md").write_text(render_rag_queue_markdown(queue), encoding="utf-8")
    (out_dir / f"sample_ids_top{len(queue)}.txt").write_text(
        "\n".join(str(row["sample_id"]) for row in queue) + ("\n" if queue else ""),
        encoding="utf-8",
    )
    summary = {
        "method": "track2_v16_rag_teacher_queue_v1",
        "base_json": str(base_json),
        "limit": int(limit),
        "queue_count": len(queue),
        "risk_gate_counts": dict(Counter(str(row["risk_gate"]) for row in queue)),
        "transition_counts": dict(Counter(str(row["transition"]) for row in queue)),
        "top_rows": queue[:20],
        "formal_submission_overwritten": False,
    }
    _write_json(out_dir / "rag_queue_summary.json", summary)
    return summary


def should_accept_v16_decision(
    decision: dict[str, Any],
    *,
    min_confidence: float = 0.86,
) -> bool:
    current = _canonical_emotion(decision.get("current_emotion"))
    proposed = _canonical_emotion(
        decision.get("proposed_emotion")
        or decision.get("teacher_emotion")
        or decision.get("emotion")
    )
    if not current or not proposed or current == proposed:
        return False
    confidence = _safe_float(decision.get("confidence") or decision.get("teacher_confidence"))
    if confidence < min_confidence:
        return False
    risk_gate = str(decision.get("risk_gate", "")).strip()
    if risk_gate == "blocked_by_failed_official_batch":
        return False
    evidence_sources = _source_set(decision.get("evidence_sources") or decision.get("sources"))
    if risk_gate == "allow_exact_duplicate" or "exact_public_duplicate" in evidence_sources:
        return True
    if risk_gate == "allow_strong_consensus":
        return True
    if {"rag_teacher", "multibackbone_consensus"} <= evidence_sources:
        return True
    return False


def load_v16_decisions_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    by_sample: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id:
                by_sample[sample_id] = dict(row)
    return list(by_sample.values())


def apply_v16_decisions_to_rows(
    rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    max_changes: int = 40,
    min_confidence: float = 0.86,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = [{key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS} for row in rows]
    decisions_by_id = {str(row.get("sample_id", "")): dict(row) for row in decisions if str(row.get("sample_id", "")).strip()}
    accepted: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for row in output:
        sample_id = str(row.get("sample_id", "")).strip()
        decision = decisions_by_id.get(sample_id)
        if decision is None:
            continue
        decision.setdefault("current_emotion", row.get("emotion", ""))
        proposed = _canonical_emotion(
            decision.get("proposed_emotion")
            or decision.get("teacher_emotion")
            or decision.get("emotion")
        )
        if len(accepted) >= max_changes:
            rejection_counts["max_changes_reached"] += 1
            continue
        if not should_accept_v16_decision(decision, min_confidence=min_confidence):
            rejection_counts[_rejection_reason(decision, current=str(row.get("emotion", "")), proposed=proposed)] += 1
            continue
        before = _label_triplet(row)
        row["emotion"] = proposed
        row["emotional_valence"] = _valence(proposed)
        row["emotional_arousal_level"] = _arousal(proposed)
        after = _label_triplet(row)
        if before != after:
            accepted.append(
                {
                    "sample_id": sample_id,
                    "transition": f"{before[0]}->{after[0]}",
                    "before": {
                        "emotion": before[0],
                        "emotional_valence": before[1],
                        "emotional_arousal_level": before[2],
                    },
                    "after": {
                        "emotion": after[0],
                        "emotional_valence": after[1],
                        "emotional_arousal_level": after[2],
                    },
                    "confidence": _safe_float(decision.get("confidence") or decision.get("teacher_confidence")),
                    "risk_gate": str(decision.get("risk_gate", "")),
                    "evidence_sources": sorted(_source_set(decision.get("evidence_sources") or decision.get("sources"))),
                    "reason": str(decision.get("reason", "")),
                }
            )

    distribution = Counter(str(row.get("emotion", "")) for row in output)
    label_issues = [issue for row in output for issue in strict_track2_label_issues(row)]
    report = {
        "method": "track2_v16_rag_consensus_candidate_v1",
        "row_count": len(output),
        "decision_count": len(decisions),
        "accepted_label_changes": len(accepted),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "transition_counts": dict(Counter(item["transition"] for item in accepted)),
        "cross_quadrant_changes": sum(
            1 for item in accepted if not _same_quadrant(item["before"]["emotion"], item["after"]["emotion"])
        ),
        "distribution": dict(sorted(distribution.items())),
        "missing_emotions": sorted(TRACK2_JSON_EMOTIONS - set(distribution)),
        "top_emotion": distribution.most_common(1)[0][0] if distribution else "",
        "top_emotion_share": (distribution.most_common(1)[0][1] / len(output)) if output else 0.0,
        "label_consistency_issue_count": len(label_issues),
        "accepted_changes": accepted,
        "formal_submission_overwritten": False,
    }
    return output, report


def write_v16_candidate_outputs(
    *,
    base_json: str | Path,
    decisions_jsonl: str | Path,
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
    max_changes: int = 40,
    min_confidence: float = 0.86,
) -> dict[str, Any]:
    out_json_path = Path(out_json)
    out_zip_path = Path(out_zip)
    _assert_safe_v16_candidate_path(out_json_path)
    _assert_safe_v16_candidate_path(out_zip_path)
    rows = _load_json_list(Path(base_json))
    decisions = load_v16_decisions_jsonl(decisions_jsonl)
    candidate_rows, report = apply_v16_decisions_to_rows(
        rows,
        decisions,
        max_changes=max_changes,
        min_confidence=min_confidence,
    )
    report["base_json"] = str(base_json)
    report["decisions_jsonl"] = str(decisions_jsonl)
    report["out_json"] = str(out_json)
    report["out_zip"] = str(out_zip)
    _write_json(out_json_path, candidate_rows)
    _write_zip(out_zip_path, out_json_path)
    _write_json(Path(report_json), report)
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text(render_v16_candidate_markdown(report), encoding="utf-8")
    return report


def render_v16_candidate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v16 RAG Consensus Candidate",
        "",
        f"- Method: `{report['method']}`",
        f"- Base JSON: `{report.get('base_json', '')}`",
        f"- Candidate JSON: `{report.get('out_json', '')}`",
        f"- Candidate ZIP: `{report.get('out_zip', '')}`",
        f"- Decisions: {report['decision_count']}",
        f"- Accepted label changes: {report['accepted_label_changes']}",
        f"- Cross-quadrant changes: {report['cross_quadrant_changes']}",
        f"- Label consistency issues: {report['label_consistency_issue_count']}",
        f"- Missing emotions: {', '.join(report['missing_emotions']) or 'none'}",
        f"- Top emotion: {report['top_emotion']} ({float(report['top_emotion_share']):.1%})",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        "",
        "## Rejections",
        "",
    ]
    if report.get("rejection_counts"):
        for key, value in sorted(dict(report["rejection_counts"]).items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    lines.extend(["", "## Accepted Changes", ""])
    if report.get("accepted_changes"):
        for item in report["accepted_changes"][:80]:
            lines.append(
                "- "
                f"{item['sample_id']}: {item['transition']}; "
                f"conf={float(item['confidence']):.2f}; gate={item['risk_gate']}; "
                f"{item.get('reason', '')}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def run_rag_teacher(
    *,
    base_json: str | Path,
    image_dir: str | Path,
    queue_json: str | Path,
    decisions_jsonl: str | Path,
    model: str = V16_RAG_MODEL,
    limit: int = 160,
    max_side: int = 1024,
    keychain_service: str = "affectiveart-gemini-api-key",
    keychain_account: str = "gemini",
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed in this Python environment") from exc

    api_key = os.environ.get("GEMINI_API_KEY") or _read_keychain_password(keychain_service, keychain_account)
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set and Keychain lookup returned no key")

    base_rows = {str(row.get("sample_id", "")): row for row in _load_json_list(Path(base_json))}
    queue_rows = _load_json_list(Path(queue_json))
    completed = {str(row.get("sample_id", "")) for row in load_v16_decisions_jsonl(decisions_jsonl)}
    client = genai.Client(api_key=api_key)

    reviewed = 0
    for queue_row in queue_rows:
        if limit and reviewed >= limit:
            break
        sample_id = str(queue_row.get("sample_id", "")).strip()
        if not sample_id or sample_id in completed:
            continue
        base_row = base_rows.get(sample_id)
        if base_row is None:
            raise ValueError(f"queue sample missing from base JSON: {sample_id}")
        image_path = _find_image(Path(image_dir), sample_id)
        image_bytes, mime_type = normalize_image_for_gemini(image_path, max_side=max_side)
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                build_rag_teacher_prompt(base_row, queue_row),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=V16_RAG_RESPONSE_SCHEMA,
                temperature=0.10,
            ),
        )
        raw = json.loads(response.text or "{}")
        teacher_emotion = _canonical_emotion(raw.get("emotion"))
        proposed = _canonical_emotion(queue_row.get("proposed_emotion"))
        decision = {
            "sample_id": sample_id,
            "model": model,
            "current_emotion": _canonical_emotion(queue_row.get("current_emotion")),
            "proposed_emotion": proposed if teacher_emotion == proposed else teacher_emotion,
            "queue_proposed_emotion": proposed,
            "risk_gate": str(queue_row.get("risk_gate", "")),
            "evidence_sources": sorted(_source_set(queue_row.get("sources")) | {"rag_teacher"}),
            "confidence": _safe_float(raw.get("confidence")),
            "teacher_decision": str(raw.get("decision", "")).strip(),
            "reason": str(raw.get("reason", "")).strip(),
            "visual_evidence": str(raw.get("visual_evidence", "")).strip(),
            "boundary_analysis": str(raw.get("boundary_analysis", "")).strip(),
        }
        if decision["teacher_decision"] == "keep_current":
            decision["proposed_emotion"] = decision["current_emotion"]
        append_jsonl(decisions_jsonl, decision)
        completed.add(sample_id)
        reviewed += 1
        print(
            f"{sample_id}: {decision['current_emotion']}->{decision['proposed_emotion']} "
            f"conf={decision['confidence']:.2f} gate={decision['risk_gate']}",
            flush=True,
        )
    return {"reviewed": reviewed, "decisions_jsonl": str(decisions_jsonl)}


def build_rag_teacher_prompt(base_row: dict[str, Any], queue_row: dict[str, Any]) -> str:
    row_json = json.dumps({key: base_row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS}, ensure_ascii=False, indent=2)
    queue_json = json.dumps(queue_row, ensure_ascii=False, indent=2)
    return V16_RAG_PROMPT.format(row_json=row_json, queue_json=queue_json)


def load_prediction_sources(paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source, path_value in paths.items():
        path = Path(path_value)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("entries", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            emotion = _canonical_emotion(row.get("target_emotion") or row.get("emotion"))
            if not sample_id or not emotion:
                continue
            entry = merged.setdefault(sample_id, {"predictions": []})
            entry["predictions"].append(
                {
                    "source": source,
                    "emotion": emotion,
                    "confidence": _safe_float(row.get("confidence")),
                    "margin": _safe_float(row.get("margin")),
                    "decision": str(row.get("decision", "")).strip(),
                    "rationale": str(row.get("rationale", "")).strip(),
                    "knn_emotion": _canonical_emotion(row.get("knn_emotion")),
                    "knn_confidence": _safe_float(row.get("knn_confidence")),
                }
            )
    return merged


def load_duplicate_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
        if isinstance(entries, list):
            rows.extend(dict(row) for row in entries if isinstance(row, dict))
    return rows


def load_failed_transition_counts(path: str | Path | None) -> dict[str, int]:
    if path is None:
        return {}
    report_path = Path(path)
    if not report_path.exists():
        return {}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in dict(payload.get("failed_transition_counts", {})).items()}


def render_rag_queue_markdown(queue: list[dict[str, Any]]) -> str:
    lines = [
        "# Track2 v16 RAG Teacher Queue",
        "",
        f"- Queue rows: {len(queue)}",
        f"- Risk gates: `{dict(Counter(str(row['risk_gate']) for row in queue))}`",
        "",
        "| sample | transition | gate | priority | confidence | sources | public labels |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in queue[:120]:
        lines.append(
            "| "
            f"{row['sample_id']} | {row['transition']} | {row['risk_gate']} | "
            f"{float(row['priority_score']):.3f} | {float(row['confidence']):.3f} | "
            f"{row['sources']} | {row['public_neighbor_labels']} |"
        )
    return "\n".join(lines) + "\n"


def write_official_counterfactual_outputs(
    *,
    official_scores_csv: str | Path,
    pairwise_diffs_csv: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_official_counterfactual_report(
        official_scores=load_official_scores(official_scores_csv),
        pairwise_diff_rows=load_pairwise_diff_rows(pairwise_diffs_csv),
    )
    (out_dir / "official_counterfactual_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "official_counterfactual_report.md").write_text(
        render_official_counterfactual_markdown(report),
        encoding="utf-8",
    )
    return report


def render_official_counterfactual_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 v16 Official Counterfactual Calibration",
        "",
        f"- Method: `{report['method']}`",
        f"- Anchor submission: `{report['anchor_submission_id']}`",
        f"- Anchor classification: {float(report['anchor_classification']):.6f}",
        f"- Official score rows: {report['official_score_count']}",
        f"- Pairwise diff rows: {report['pairwise_diff_count']}",
        "",
        "## Policy",
        "",
        str(report["policy"]),
        "",
        "## Known Failed Batches",
        "",
    ]
    failed = list(report.get("known_failed_batches", []))
    if not failed:
        lines.append("- none")
    for batch in failed:
        lines.append(
            "- "
            f"{batch['candidate_submission_id']} vs {batch['anchor_submission_id']}: "
            f"class delta {float(batch['classification_delta']):.6f}, "
            f"changed rows {int(batch['changed_rows'])}, "
            f"per-row {float(batch['classification_delta_per_changed_row']):.8f}"
        )
    lines.extend(["", "## Failed Transition Counts", ""])
    counts = dict(report.get("failed_transition_counts", {}))
    if not counts:
        lines.append("- none")
    for transition, count in counts.items():
        lines.append(f"- {transition}: {count}")
    return "\n".join(lines) + "\n"


def _extract_submission_id(value: Any) -> str:
    match = re.match(r"^(\d{6,})", str(value or "").strip())
    return match.group(1) if match else ""


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _proposal_scores_for_sample(
    *,
    sample_id: str,
    current: str,
    prediction_rows: dict[str, Any],
    duplicate_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    raw = prediction_rows.get(sample_id, {})
    if "target_emotion" in raw:
        emotion = _canonical_emotion(raw.get("target_emotion"))
        if emotion:
            scores[emotion] = {
                "score": _safe_float(raw.get("confidence")),
                "confidence": _safe_float(raw.get("confidence")),
                "sources": set(str(item) for item in raw.get("sources", [])),
                "public_neighbor_labels": set(),
                "reason": str(raw.get("reason", "")),
            }
    for prediction in raw.get("predictions", []) if isinstance(raw, dict) else []:
        emotion = _canonical_emotion(prediction.get("emotion"))
        if not emotion:
            continue
        source = str(prediction.get("source", "model"))
        confidence = _safe_float(prediction.get("confidence"))
        margin = max(0.0, _safe_float(prediction.get("margin")))
        entry = scores.setdefault(
            emotion,
            {"score": 0.0, "confidence": 0.0, "sources": set(), "public_neighbor_labels": set(), "reason": ""},
        )
        entry["score"] += confidence + 0.35 * margin
        entry["confidence"] = max(float(entry["confidence"]), confidence)
        entry["sources"].add(source)
        if _canonical_emotion(prediction.get("knn_emotion")) == emotion and _safe_float(prediction.get("knn_confidence")) >= 0.55:
            entry["sources"].add("public_neighbor_agreement")
            entry["score"] += 0.25
        if str(prediction.get("decision", "")).lower() == "change":
            entry["sources"].add("rag_teacher")
            entry["score"] += 0.35
        rationale = str(prediction.get("rationale") or prediction.get("reason") or "").strip()
        if rationale and rationale not in str(entry.get("reason", "")):
            entry["reason"] = "; ".join(part for part in [str(entry.get("reason", "")), rationale] if part)
    for emotion, entry in scores.items():
        model_sources = {source for source in entry["sources"] if source in {"siglip2", "clip", "dinov2"}}
        if len(model_sources) >= 2:
            entry["sources"].add("multibackbone_consensus")

    for duplicate in duplicate_rows:
        emotion = _canonical_emotion(duplicate.get("public_emotion") or duplicate.get("nearest_emotion"))
        if not emotion:
            continue
        entry = scores.setdefault(
            emotion,
            {"score": 0.0, "confidence": 0.0, "sources": set(), "public_neighbor_labels": set(), "reason": ""},
        )
        clip_cosine = _safe_float(duplicate.get("clip_cosine") or duplicate.get("top1_clip_cosine"))
        label = str(duplicate.get("public_emotion") or duplicate.get("nearest_emotion") or emotion)
        entry["public_neighbor_labels"].add(label)
        if clip_cosine >= 0.985 or duplicate.get("suspect_duplicate") is True:
            entry["sources"].add("exact_public_duplicate")
            entry["score"] += 4.0
            entry["confidence"] = max(float(entry["confidence"]), clip_cosine)
        elif clip_cosine >= 0.95:
            entry["sources"].add("public_neighbor_agreement")
            entry["score"] += 1.0
            entry["confidence"] = max(float(entry["confidence"]), clip_cosine)

    # Keep only proposals that have evidence beyond the current label.
    return {
        emotion: entry
        for emotion, entry in scores.items()
        if emotion != current and (entry["sources"] or float(entry["score"]) > 0)
    }


def _queue_priority(
    *,
    sources: set[str],
    risk_gate: str,
    confidence: float,
    same_valence: bool,
    same_arousal: bool,
) -> float:
    priority = float(confidence)
    if "exact_public_duplicate" in sources:
        priority += 4.0
    if "multibackbone_consensus" in sources:
        priority += 2.5
    if "rag_teacher" in sources:
        priority += 2.0
    if "public_neighbor_agreement" in sources:
        priority += 1.0
    if same_valence and same_arousal:
        priority += 0.35
    if risk_gate == "blocked_by_failed_official_batch":
        priority -= 5.0
    return priority


def _canonical_emotion(value: Any) -> str:
    emotion = str(value or "").strip().lower()
    if emotion == "contentment":
        emotion = "content"
    return emotion if emotion in TRACK2_JSON_EMOTIONS else ""


def _valence(emotion: str) -> str:
    return "Negative" if emotion in NEGATIVE_EMOTIONS else "Positive"


def _arousal(emotion: str) -> str:
    return "High" if emotion in HIGH_AROUSAL_EMOTIONS else "Low"


def _index_rows_by_sample_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            grouped.setdefault(sample_id, []).append(row)
    return grouped


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _assert_safe_v16_candidate_path(path: Path) -> None:
    if path.name in FORMAL_SUBMISSION_NAMES:
        raise ValueError(f"refusing to write formal Track2 submission path: {path}")
    if path.suffix not in {".json", ".zip"}:
        raise ValueError(f"candidate output must be JSON or ZIP: {path}")
    if not path.stem.startswith("track2_submission_v16_") or not path.stem.endswith("_candidate"):
        raise ValueError(f"candidate output must be a v16 side-path candidate: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_zip(zip_path: Path, json_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, "submission.json")


def _write_queue_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "current_emotion",
        "proposed_emotion",
        "transition",
        "risk_gate",
        "priority_score",
        "confidence",
        "sources",
        "public_neighbor_labels",
        "same_valence",
        "same_arousal",
        "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _source_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(part).strip() for part in value if str(part).strip()}
    return set()


def _label_triplet(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("emotion", "")),
        str(row.get("emotional_valence", "")),
        str(row.get("emotional_arousal_level", "")),
    )


def _same_quadrant(left: str, right: str) -> bool:
    return _valence(left) == _valence(right) and _arousal(left) == _arousal(right)


def _rejection_reason(decision: dict[str, Any], *, current: str, proposed: str) -> str:
    if not _canonical_emotion(current):
        return "missing_current_emotion"
    if not proposed:
        return "missing_proposed_emotion"
    if _canonical_emotion(current) == proposed:
        return "kept_current"
    if _safe_float(decision.get("confidence") or decision.get("teacher_confidence")) < 0.86:
        return "low_confidence"
    if str(decision.get("risk_gate", "")).strip() == "blocked_by_failed_official_batch":
        return "blocked_by_failed_official_batch"
    return "weak_evidence_gate"


def _find_image(image_dir: Path, sample_id: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = image_dir / f"{sample_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing image for {sample_id} under {image_dir}")


def _read_keychain_password(service: str, account: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Track2 v16 official-style RAG classification tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser("calibrate", help="Build official counterfactual calibration report")
    calibrate.add_argument("--official-scores", type=Path, required=True)
    calibrate.add_argument("--pairwise-diffs", type=Path, required=True)
    calibrate.add_argument("--out-dir", type=Path, required=True)
    build_queue = subparsers.add_parser("build-queue", help="Build v16 RAG teacher queue")
    build_queue.add_argument("--base-json", type=Path, required=True)
    build_queue.add_argument("--out-dir", type=Path, required=True)
    build_queue.add_argument("--limit", type=int, default=160)
    build_queue.add_argument("--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT)
    run_teacher = subparsers.add_parser("run-rag-teacher", help="Run Gemini RAG teacher decisions")
    run_teacher.add_argument("--base-json", type=Path, required=True)
    run_teacher.add_argument("--image-dir", type=Path, required=True)
    run_teacher.add_argument("--queue-json", type=Path, required=True)
    run_teacher.add_argument("--decisions-jsonl", type=Path, required=True)
    run_teacher.add_argument("--model", default=V16_RAG_MODEL)
    run_teacher.add_argument("--limit", type=int, default=160)
    run_teacher.add_argument("--max-side", type=int, default=1024)
    run_teacher.add_argument("--keychain-service", default="affectiveart-gemini-api-key")
    run_teacher.add_argument("--keychain-account", default="gemini")
    apply_parser = subparsers.add_parser("apply", help="Apply accepted v16 RAG decisions to a candidate")
    apply_parser.add_argument("--base-json", type=Path, required=True)
    apply_parser.add_argument("--decisions-jsonl", type=Path, required=True)
    apply_parser.add_argument("--out-json", type=Path, required=True)
    apply_parser.add_argument("--out-zip", type=Path, required=True)
    apply_parser.add_argument("--report-json", type=Path, required=True)
    apply_parser.add_argument("--report-md", type=Path, required=True)
    apply_parser.add_argument("--max-changes", type=int, default=40)
    apply_parser.add_argument("--min-confidence", type=float, default=0.86)

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        report = write_official_counterfactual_outputs(
            official_scores_csv=args.official_scores,
            pairwise_diffs_csv=args.pairwise_diffs,
            out_dir=args.out_dir,
        )
        print(
            json.dumps(
                {
                    "anchor_submission_id": report["anchor_submission_id"],
                    "failed_batch_count": len(report["known_failed_batches"]),
                    "failed_transition_count": len(report["failed_transition_counts"]),
                },
                indent=2,
            )
        )
    elif args.command == "run-rag-teacher":
        summary = run_rag_teacher(
            base_json=args.base_json,
            image_dir=args.image_dir,
            queue_json=args.queue_json,
            decisions_jsonl=args.decisions_jsonl,
            model=args.model,
            limit=args.limit,
            max_side=args.max_side,
            keychain_service=args.keychain_service,
            keychain_account=args.keychain_account,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "apply":
        report = write_v16_candidate_outputs(
            base_json=args.base_json,
            decisions_jsonl=args.decisions_jsonl,
            out_json=args.out_json,
            out_zip=args.out_zip,
            report_json=args.report_json,
            report_md=args.report_md,
            max_changes=args.max_changes,
            min_confidence=args.min_confidence,
        )
        print(
            json.dumps(
                {
                    "accepted_label_changes": report["accepted_label_changes"],
                    "cross_quadrant_changes": report["cross_quadrant_changes"],
                    "label_consistency_issue_count": report["label_consistency_issue_count"],
                },
                indent=2,
            )
        )
    elif args.command == "build-queue":
        summary = write_rag_queue_outputs(
            base_json=args.base_json,
            out_dir=args.out_dir,
            limit=args.limit,
            calibration_report=args.calibration_report,
        )
        print(
            json.dumps(
                {
                    "queue_count": summary["queue_count"],
                    "risk_gate_counts": summary["risk_gate_counts"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
