from __future__ import annotations

from typing import Any


def compile_vulca_prompt(packet: dict[str, Any]) -> str:
    requirements = packet.get("hard_requirements", []) or []
    allowed_text = packet.get("allowed_text", []) or []
    forbidden = packet.get("forbidden_artifacts", []) or []
    required_surface = packet.get("caption_required_surface_features", []) or []
    allowed_surface = packet.get("allowed_surface_features", []) or []
    unrequested_surface = packet.get("unrequested_physical_artifact_features", []) or []
    caption = _provider_safe_caption(str(packet.get("caption", "")))
    lines = [
        "NON-NEGOTIABLE CONTENT REQUIREMENTS",
        f"Caption: {caption}",
    ]
    for item in requirements:
        lines.append(f"- Must include: {item}")
    lines.extend(
        [
            "",
            "ARTIFACT BOUNDARY REQUIREMENT",
            f"- Artwork category: {packet.get('artwork_category', 'artwork')}",
            "- The output must be the artwork surface itself, not a gallery photograph, wall mockup, catalog page, display scene, or framed product shot.",
        ]
    )
    if allowed_text:
        lines.append("- Allow text-like marks only for: " + ", ".join(allowed_text))
    else:
        lines.append("- Do not add readable or pseudo-readable text.")
    if forbidden:
        lines.append("- Forbidden artifacts: " + ", ".join(forbidden))
    lines.extend(
        [
            "",
            "SURFACE FEATURE CONTROL",
        ]
    )
    if required_surface:
        lines.append("- Required surface features from caption: " + ", ".join(required_surface))
    else:
        lines.append("- Required surface features from caption: none")
        if packet.get("artwork_category") == "poster":
            lines.append(
                "- Poster may include internal printed margins, typography blocks, or graphic border lines as part of the poster design."
            )
            lines.append(
                "- Do not add an external frame, photo mat, wall display, drop shadow, catalog mockup, or product-photo presentation."
            )
        else:
            lines.append(
                "- No extra border, aged paper, folded paper, mat, shadow, or physical display treatment unless the caption explicitly asks for it."
            )
    if allowed_surface:
        lines.append("- Allowed surface treatment: " + ", ".join(allowed_surface))
    if packet.get("artwork_category") == "album_leaf":
        lines.append(
            "- Album leaf borders must be printed or painted internal page design only, not an external frame."
        )
        lines.append(
            "- Do not surround the album leaf with an outer gray or black frame, photo mat, shadow, or display mount."
        )
    if unrequested_surface:
        lines.append(
            "- Do not add unrequested physical artifact features: "
            + ", ".join(unrequested_surface)
        )
    lines.extend(
        [
            "",
            "GENERATION PRIORITY",
            "1. Exact caption content and named objects.",
            "2. Artwork surface/category boundary.",
            "3. Requested artistic style and emotional atmosphere.",
            "4. External references are used only by separate scoring or review steps, not as content requirements in this prompt.",
        ]
    )
    return "\n".join(lines).strip()


def _provider_safe_caption(caption: str) -> str:
    return caption.replace(
        "anti-fascist caricature scenes",
        "non-graphic symbolic anti-fascist caricature panels",
    )


def decision_from_scores(
    *,
    sample_id: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    min_margin: float = 0.12,
) -> dict[str, Any]:
    baseline_score = _score_audit(baseline)
    candidate_score = _score_audit(candidate)
    margin = round(candidate_score - baseline_score, 4)
    if candidate.get("caption_fidelity") != "pass":
        return {
            "sample_id": sample_id,
            "decision": "reject",
            "margin": margin,
            "reason": "candidate caption fidelity is not pass",
        }
    if baseline.get("caption_fidelity") == "pass" and margin < min_margin:
        return {
            "sample_id": sample_id,
            "decision": "reject",
            "margin": margin,
            "reason": "candidate is not a clear win over a passing baseline",
        }
    if margin >= min_margin:
        return {
            "sample_id": sample_id,
            "decision": "accept",
            "margin": margin,
            "reason": "candidate is a clear win",
        }
    return {
        "sample_id": sample_id,
        "decision": "reject",
        "margin": margin,
        "reason": "not a clear win",
    }


def _score_audit(row: dict[str, Any]) -> float:
    fidelity = {"pass": 1.0, "partial": 0.55, "fail": 0.0}.get(
        str(row.get("caption_fidelity", "")),
        0.0,
    )
    style = float(row.get("style_fidelity", 0.0))
    quality = float(row.get("visual_quality", 0.0))
    emotion = float(row.get("emotional_atmosphere", row.get("emotion", 0.0)) or 0.0)
    return round(fidelity * 0.45 + style * 0.2 + quality * 0.25 + emotion * 0.1, 4)


def summarize_decisions(rows: list[dict[str, Any]]) -> dict[str, int]:
    accepted = sum(1 for row in rows if row.get("decision") == "accept")
    rejected = sum(1 for row in rows if row.get("decision") == "reject")
    held = sum(1 for row in rows if row.get("decision") == "hold")
    return {"total": len(rows), "accepted": accepted, "rejected": rejected, "held": held}
