from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from affectiveart.challenge import TRACK2_JSON_SUBMISSION_KEYS, TRACK2_JSON_TEXT_FIELDS
from affectiveart.track2_description_score import EVALUATOR_MANIPULATION_PATTERNS


DESCRIPTION_REWRITE_MODEL = "gemini-3.5-flash"
DESCRIPTION_REWRITE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rewrite_needed": {"type": "boolean"},
        "confidence": {"type": "number"},
        "artwork_consistency_score": {"type": "number"},
        "attribute_quality_score": {"type": "number"},
        "caption_quality_score": {"type": "number"},
        "reason": {"type": "string"},
        "overall_caption": {"type": "string"},
        "brushstroke": {"type": "string"},
        "composition": {"type": "string"},
        "color": {"type": "string"},
        "line": {"type": "string"},
        "light": {"type": "string"},
    },
    "required": [
        "rewrite_needed",
        "confidence",
        "artwork_consistency_score",
        "attribute_quality_score",
        "caption_quality_score",
        "reason",
        *TRACK2_JSON_TEXT_FIELDS,
    ],
}


DESCRIPTION_REWRITE_PROMPT = """You are improving an art-emotion submission.

You will receive one artwork image and one current JSON row.

Task:
- Audit whether the six open text fields are visually grounded, specific, natural, and consistent with the artwork.
- If the current text is already accurate and specific, keep it unchanged and set rewrite_needed=false.
- If the text is generic, template-like, visually inconsistent, or misses obvious visual evidence, rewrite only these fields:
  overall_caption, brushstroke, composition, color, line, light.

Hard constraints:
- Preserve the intended emotion framing from the current row, but do not force emotion words mechanically.
- Do not change sample_id, emotion, emotional_valence, or emotional_arousal_level.
- Do not add instructions to a judge, evaluator, scorer, model, or reader.
- Do not claim specific historical facts, artist names, locations, or iconography unless plainly visible.
- Keep each field one concise English sentence.
- Use concrete visual evidence: subject matter, palette, brushwork, composition, line, and lighting.
- All numeric scores and confidence must be decimals from 0.0 to 1.0.

Return exactly one JSON object following the schema.

Current row:
{row_json}
"""


@dataclass(frozen=True)
class RewriteDecision:
    sample_id: str
    model: str
    rewrite_needed: bool
    confidence: float
    artwork_consistency_score: float
    attribute_quality_score: float
    caption_quality_score: float
    reason: str
    fields: dict[str, str]


def load_track2_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a list of Track2 rows: {path}")
    return [dict(row) for row in payload]


def load_rewrite_decisions(path: str | Path) -> dict[str, RewriteDecision]:
    path = Path(path)
    if not path.exists():
        return {}
    decisions: dict[str, RewriteDecision] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            decision = normalize_rewrite_decision(raw)
            decisions[decision.sample_id] = decision
    return decisions


def normalize_rewrite_decision(raw: dict[str, Any]) -> RewriteDecision:
    sample_id = str(raw.get("sample_id", "")).strip()
    if not sample_id:
        raise ValueError("rewrite decision missing sample_id")
    fields = {field: str(raw.get(field, "")).strip() for field in TRACK2_JSON_TEXT_FIELDS}
    return RewriteDecision(
        sample_id=sample_id,
        model=str(raw.get("model", "")).strip(),
        rewrite_needed=bool(raw.get("rewrite_needed", False)),
        confidence=_score01(raw.get("confidence")),
        artwork_consistency_score=_score01(raw.get("artwork_consistency_score")),
        attribute_quality_score=_score01(raw.get("attribute_quality_score")),
        caption_quality_score=_score01(raw.get("caption_quality_score")),
        reason=str(raw.get("reason", "")).strip(),
        fields=fields,
    )


def apply_rewrite_decisions(
    rows: list[dict[str, Any]],
    decisions: dict[str, RewriteDecision],
    *,
    min_confidence: float = 0.74,
    min_dimension_score: float = 0.72,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        decision = decisions.get(sample_id)
        repaired = dict(row)
        if decision is not None:
            accepted, reasons = should_accept_rewrite(
                row,
                decision,
                min_confidence=min_confidence,
                min_dimension_score=min_dimension_score,
            )
            if accepted:
                before = {field: str(row.get(field, "")) for field in TRACK2_JSON_TEXT_FIELDS}
                for field in TRACK2_JSON_TEXT_FIELDS:
                    repaired[field] = decision.fields[field]
                after = {field: str(repaired.get(field, "")) for field in TRACK2_JSON_TEXT_FIELDS}
                if before != after:
                    changes.append(
                        {
                            "sample_id": sample_id,
                            "model": decision.model,
                            "confidence": decision.confidence,
                            "dimension_scores": {
                                "artwork_consistency": decision.artwork_consistency_score,
                                "attribute_quality": decision.attribute_quality_score,
                                "caption_quality": decision.caption_quality_score,
                            },
                            "reason": decision.reason,
                            "changed_fields": [
                                field for field in TRACK2_JSON_TEXT_FIELDS if before[field] != after[field]
                            ],
                        }
                    )
            else:
                for reason in reasons:
                    rejection_counts[reason] += 1
        output.append({key: repaired.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS})

    report = {
        "method": "track2_multimodal_description_rewrite_v1",
        "row_count": len(rows),
        "decision_count": len(decisions),
        "changed_rows": len(changes),
        "changed_fields": sum(len(item["changed_fields"]) for item in changes),
        "classification_label_changes": _classification_label_changes(rows, output),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "changes": changes,
        "formal_submission_overwritten": False,
    }
    return output, report


def should_accept_rewrite(
    row: dict[str, Any],
    decision: RewriteDecision,
    *,
    min_confidence: float = 0.74,
    min_dimension_score: float = 0.72,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not decision.rewrite_needed:
        reasons.append("rewrite_not_needed")
    if decision.confidence < min_confidence:
        reasons.append("low_confidence")
    if min(
        decision.artwork_consistency_score,
        decision.attribute_quality_score,
        decision.caption_quality_score,
    ) < min_dimension_score:
        reasons.append("low_dimension_score")
    for field, value in decision.fields.items():
        if len(value.split()) < 6:
            reasons.append(f"underspecified_{field}")
        if _contains_evaluator_manipulation(value):
            reasons.append(f"evaluator_text_{field}")
    if str(row.get("sample_id", "")) != decision.sample_id:
        reasons.append("sample_id_mismatch")
    return not reasons, reasons


def write_rewrite_candidate_outputs(
    *,
    source_json: str | Path,
    decisions_jsonl: str | Path,
    out_json: str | Path,
    out_zip: str | Path,
    report_json: str | Path,
    report_md: str | Path,
    min_confidence: float = 0.74,
    min_dimension_score: float = 0.72,
) -> dict[str, Any]:
    rows = load_track2_rows(source_json)
    decisions = load_rewrite_decisions(decisions_jsonl)
    repaired_rows, report = apply_rewrite_decisions(
        rows,
        decisions,
        min_confidence=min_confidence,
        min_dimension_score=min_dimension_score,
    )
    report["source_json"] = str(source_json)
    report["decisions_jsonl"] = str(decisions_jsonl)
    report["out_json"] = str(out_json)
    report["out_zip"] = str(out_zip)
    _write_json(Path(out_json), repaired_rows)
    _write_zip(Path(out_zip), Path(out_json))
    _write_json(Path(report_json), report)
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text(render_rewrite_report_markdown(report), encoding="utf-8")
    return report


def render_rewrite_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track2 Multimodal Description Rewrite Report",
        "",
        f"- Method: `{report['method']}`",
        f"- Rows: {report['row_count']}",
        f"- Decisions: {report['decision_count']}",
        f"- Changed rows: {report['changed_rows']}",
        f"- Changed fields: {report['changed_fields']}",
        f"- Classification label changes: {report['classification_label_changes']}",
        f"- Formal submission overwritten: {report['formal_submission_overwritten']}",
        "",
        "## Rejections",
        "",
    ]
    if report.get("rejection_counts"):
        for key, value in sorted(report["rejection_counts"].items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    lines.extend(["", "## Accepted Changes", ""])
    if report.get("changes"):
        for item in report["changes"][:160]:
            fields = ", ".join(item.get("changed_fields", []))
            lines.append(
                f"- {item['sample_id']}: conf={item['confidence']:.2f}; fields={fields}; {item.get('reason', '')}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def append_rewrite_decision(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_prompt(row: dict[str, Any]) -> str:
    row_json = json.dumps({key: row.get(key, "") for key in TRACK2_JSON_SUBMISSION_KEYS}, ensure_ascii=False, indent=2)
    return DESCRIPTION_REWRITE_PROMPT.format(row_json=row_json)


def normalize_image_for_gemini(path: str | Path, *, max_side: int = 1024) -> tuple[bytes, str]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=92, optimize=True)
        return output.getvalue(), "image/jpeg"


def ordered_rows_for_ids(rows: list[dict[str, Any]], sample_ids: Iterable[str] | None) -> list[dict[str, Any]]:
    if sample_ids is None:
        return rows
    wanted = [sample_id.strip() for sample_id in sample_ids if sample_id.strip()]
    by_id = {str(row.get("sample_id", "")): row for row in rows}
    missing = [sample_id for sample_id in wanted if sample_id not in by_id]
    if missing:
        raise ValueError(f"sample ids not found in source JSON: {missing[:5]}")
    return [by_id[sample_id] for sample_id in wanted]


def _contains_evaluator_manipulation(value: str) -> bool:
    lower = value.lower()
    return any(re.search(pattern, lower, re.IGNORECASE) for pattern in EVALUATOR_MANIPULATION_PATTERNS)


def _classification_label_changes(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> int:
    labels = ("emotion", "emotional_valence", "emotional_arousal_level")
    before = {str(row.get("sample_id", "")): tuple(row.get(label, "") for label in labels) for row in before_rows}
    after = {str(row.get("sample_id", "")): tuple(row.get(label, "") for label in labels) for row in after_rows}
    return sum(1 for sample_id in before if before.get(sample_id) != after.get(sample_id))


def _score01(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        if score <= 5.0:
            score = score / 5.0
        elif score <= 10.0:
            score = score / 10.0
        else:
            score = score / 100.0
    return max(0.0, min(1.0, score))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_zip(zip_path: Path, json_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, "submission.json")
