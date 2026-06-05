from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from affectiveart.track1_reference_pack import build_reference_guided_prompt, build_reference_pack
from affectiveart.track1_reference_text_bank import load_reference_text_bank


def build_moe_prompt_packet(
    *,
    rank: int,
    route: dict[str, Any],
    contract: dict[str, Any],
    reference_text_packet: dict[str, Any] | None = None,
    provider_prompt_path: str | Path | None = None,
    distribution_route: dict[str, Any] | None = None,
    candidate_strategy: str = "legacy",
) -> dict[str, Any]:
    sample_id = str(route["sample_id"])
    caption = str(contract["caption"])
    variant = _variant_for_route(route)
    pack = build_reference_pack(sample_id, caption)
    provider_prompt = build_reference_guided_prompt(sample_id, caption, pack, variant=variant)
    if reference_text_packet:
        provider_prompt += "\n\n" + render_provider_text_section(reference_text_packet, caption=caption)
    safety_context = render_neutral_historical_context(caption)
    if safety_context:
        provider_prompt += "\n\n" + safety_context
    if distribution_route and candidate_strategy != "legacy":
        provider_prompt = _soften_template_phrases(provider_prompt, strategy=candidate_strategy)
        provider_prompt += "\n\n" + render_distribution_strategy_section(
            distribution_route,
            candidate_strategy=candidate_strategy,
            caption=caption,
        )
    reference_assets = _clean_reference_assets((distribution_route or {}).get("reference_assets", []))
    provider_prompt += "\n\nASPECT AND CANVAS PLAN\n"
    aspect = dict(contract.get("aspect_plan") or {})
    if aspect.get("prompt_directive"):
        directive = str(aspect["prompt_directive"])
        if distribution_route and candidate_strategy != "legacy":
            directive = _soften_template_phrases(directive, strategy=candidate_strategy)
        provider_prompt += f"- {directive}\n"
    provider_prompt += f"- Aspect label: {aspect.get('label', '')}; target canvas: {aspect.get('width')}x{aspect.get('height')}.\n"
    provider_prompt = _strip_provider_metadata(provider_prompt, sample_id=sample_id)
    return {
        "rank": rank,
        "sample_id": sample_id,
        "caption": caption,
        "variant": variant,
        "candidate_strategy": candidate_strategy,
        "provider_prompt": provider_prompt,
        "provider_prompt_path": str(provider_prompt_path) if provider_prompt_path else "",
        "review_metadata": {
            "sample_id": sample_id,
            "family_id": str((distribution_route or {}).get("family_id") or ""),
            "candidate_strategy": candidate_strategy,
            "recommended_model": route.get("recommended_model", ""),
            "fallback_model": route.get("fallback_model", ""),
            "candidate_count": int(route.get("candidate_count") or 0),
            "primary_expert": route.get("primary_expert", ""),
            "support_experts": list(route.get("support_experts", [])),
            "priority_bucket": route.get("priority_bucket", ""),
            "queue_score": route.get("queue_score", 0),
            "reference_level": route.get("reference_level", ""),
            "reference_asset_count": len(reference_assets),
        },
        "reference_assets": reference_assets,
        "aspect_plan": aspect,
        "reference_contract": contract.get("reference_contract", {}),
        "text_contract": contract.get("text_contract", {}),
        "relation_contract": contract.get("relation_contract", {}),
        "fid_risk": list(contract.get("fid_risk", [])),
    }


def build_moe_prompt_packets(
    routes: Iterable[dict[str, Any]],
    *,
    contracts: dict[str, dict[str, Any]],
    reference_text_bank: dict[str, dict[str, Any]] | None,
    out_dir: str | Path,
    limit: int | None = None,
    distribution_routes: dict[str, dict[str, Any]] | list[dict[str, Any]] | None = None,
    max_strategies_per_sample: int | None = None,
    strategy_mode: str = "auto",
) -> list[dict[str, Any]]:
    out_dir = _safe_output_dir(out_dir)
    packets: list[dict[str, Any]] = []
    route_by_sample = _distribution_routes_by_sample(distribution_routes)
    mode = _validate_strategy_mode(strategy_mode)
    if mode == "legacy":
        route_by_sample = {}
    generation_routes = [route for route in _validate_route_rows(list(routes)) if int(route.get("candidate_count") or 0) > 0]
    _validate_contract_map(contracts)
    missing_contracts = [str(route["sample_id"]) for route in generation_routes if str(route["sample_id"]) not in contracts]
    if missing_contracts:
        raise ValueError(f"missing contracts for sample_id: {', '.join(missing_contracts)}")
    if mode == "distribution":
        missing_distribution = [str(route["sample_id"]) for route in generation_routes if str(route["sample_id"]) not in route_by_sample]
        if missing_distribution:
            raise ValueError(f"missing distribution routes for sample_id: {', '.join(missing_distribution)}")

    prompt_dir = out_dir / "provider_prompts_top"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    rank = 0
    for route in generation_routes:
        sample_id = str(route["sample_id"])
        contract = contracts[sample_id]
        distribution_route = route_by_sample.get(sample_id)
        strategies = _candidate_strategies(distribution_route, max_strategies_per_sample=max_strategies_per_sample)
        for candidate_strategy in strategies:
            if limit is not None and limit > 0 and len(packets) >= limit:
                return packets
            rank += 1
            suffix = "" if candidate_strategy == "legacy" else f"_{candidate_strategy}"
            prompt_path = prompt_dir / f"{rank:02d}_{sample_id}{suffix}.txt"
            packet = build_moe_prompt_packet(
                rank=rank,
                route=route,
                contract=contract,
                reference_text_packet=(reference_text_bank or {}).get(sample_id),
                provider_prompt_path=prompt_path,
                distribution_route=distribution_route,
                candidate_strategy=candidate_strategy,
            )
            prompt_path.write_text(packet["provider_prompt"].rstrip() + "\n", encoding="utf-8")
            packets.append(packet)
    return packets


def render_provider_text_section(packet: dict[str, Any], *, caption: str) -> str:
    text = packet.get("text", {})
    refs = packet.get("references", {})
    modes = [str(mode) for mode in text.get("modes", [])]
    lines = [
        "TEXT AND FACTUAL ANCHORS",
        f"- Text required: {bool(text.get('required'))}; modes: {', '.join(modes)}.",
    ]
    text_bank = _provider_text_bank(packet, caption=caption)
    if text_bank:
        lines.append("Allowed text cues:")
        lines.extend(f"- {item}" for item in text_bank)
    directives = _provider_text_directives(text)
    if directives:
        lines.append("Text directives:")
        lines.extend(f"- {item}" for item in directives)
    must_avoid = [str(item) for item in text.get("must_avoid", []) if str(item)]
    if must_avoid:
        lines.append("Text must avoid:")
        lines.extend(f"- {item}" for item in must_avoid)
    anchors = [str(item) for item in refs.get("factual_anchors", []) if str(item)]
    if anchors:
        lines.append("Factual anchors:")
        lines.extend(f"- {item}" for item in anchors)
    return "\n".join(lines).strip()


def render_neutral_historical_context(caption: str) -> str:
    text = caption.lower()
    block_prone_terms = (
        "propaganda",
        "soviet",
        "kremlin",
        "wartime",
        "war",
        "soldier",
        "military",
        "tank",
        "rifle",
        "flag",
        "surrender",
    )
    if not any(term in text for term in block_prone_terms):
        return ""
    return "\n".join(
        [
            "NEUTRAL HISTORICAL ART CONTEXT",
            "- Treat this as a neutral historical artwork reconstruction for an art dataset challenge, not political advocacy, persuasion, recruitment, or a real-world call to action.",
            "- Depict flags, slogans, uniforms, vehicles, emblems, and architecture only as caption-required visual attributes of a historical artwork surface.",
            "- Avoid explicit injury, gore, hatred, extremist praise, contemporary political claims, or instructions for real-world action.",
        ]
    )


def render_distribution_strategy_section(
    route: dict[str, Any],
    *,
    candidate_strategy: str,
    caption: str,
) -> str:
    hard_constraints = _clean_provider_list(route.get("hard_constraints", []))
    composition_hints = _clean_provider_list(route.get("composition_hints", []))
    medium_options = _clean_provider_list(route.get("medium_options", []))
    style_freedom = _clean_provider_list(route.get("style_freedom", []))

    lines = ["DISTRIBUTION-AWARE ART DIRECTION"]
    if candidate_strategy == "aas_safe":
        lines.append("- Keep the caption requirements and named symbols strict; use style variation only after the core scene is clear.")
        if hard_constraints:
            lines.append("- Preserve: " + "; ".join(hard_constraints) + ".")
    elif candidate_strategy == "reference_style":
        lines.append(
            "- Match the reference family through medium, crop, density, color, brushwork, texture, and lighting while preserving caption logic."
        )
    elif candidate_strategy == "fid_diverse":
        lines.append(
            "- Avoid repeating a generic centered portrait-poster template; choose a less common crop, depth, figure scale, or viewpoint when the caption permits."
        )
        lines.append("- Keep required caption content, but let the artwork read as a natural historical artwork rather than a uniform generated layout.")
    else:
        lines.append("- Preserve caption content while varying artwork surface, crop, and painterly handling.")

    if composition_hints:
        lines.append("- Composition cues: " + "; ".join(composition_hints[:6]) + ".")
    if medium_options:
        lines.append("- Medium choices: " + "; ".join(medium_options[:6]) + ".")
    if style_freedom:
        lines.append("- Allowed variation: " + "; ".join(style_freedom[:6]) + ".")
    if "poster" in caption.lower() and candidate_strategy != "fid_diverse":
        lines.append("- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.")
    if _clean_reference_assets(route.get("reference_assets", [])):
        lines.append(
            "- Use the attached reference board only for official style-family, medium, landmark, symbol, and composition cues; do not copy unrelated exact objects."
        )
    return "\n".join(lines).strip()


def write_moe_prompt_packet_reports(rows: list[dict[str, Any]], *, out_dir: str | Path) -> None:
    out_dir = _safe_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(rows)
    (out_dir / "track1_moe_prompt_packets.json").write_text(
        json.dumps({"summary": summary, "packets": rows}, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "track1_moe_prompt_packets.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_csv(rows, out_dir / "track1_moe_prompt_packets.csv")
    (out_dir / "track1_moe_prompt_packets_zh.md").write_text(_render_md(rows, summary), encoding="utf-8")


def load_routes(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise ValueError("routes JSON must be a list or an object with 'rows'")
    return _validate_route_rows(rows)


def load_distribution_routes(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("routes"), list):
        rows = payload["routes"]
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise ValueError("distribution routes JSON must be a list or an object with 'routes' or 'rows'")
    return _distribution_routes_by_sample(rows)


def load_contracts(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise ValueError("contracts JSON must be a list or an object with 'rows'")
    contract_map = {str(row["sample_id"]): dict(row) for row in _validate_contract_rows(rows)}
    return contract_map


def load_reference_text_packets(path: str | Path | None) -> dict[str, dict[str, Any]]:
    return load_reference_text_bank(path)


def _variant_for_route(route: dict[str, Any]) -> str:
    expert = str(route.get("primary_expert") or "")
    support = set(str(item) for item in route.get("support_experts", []))
    if expert == "poster_expert":
        return "moe_poster_logic_tight_defect_targeted_logic_anchor_v1"
    if expert == "scroll_expert":
        return "moe_flat_scroll_scan_logic_anchor_v1"
    if "logic_expert" in support:
        return "moe_defect_targeted_logic_anchor_v1"
    return "moe_reference_guided_v1"


def _provider_text_bank(packet: dict[str, Any], *, caption: str) -> list[str]:
    text = packet.get("text", {})
    modes = set(str(mode) for mode in text.get("modes", []))
    lower = caption.lower()
    cues: list[str] = []
    if "document" in modes:
        cues.extend(["АКТ О КАПИТУЛЯЦИИ", "ПОЛНАЯ КАПИТУЛЯЦИЯ"])
        cues.append("Use a document title, simplified body lines, a signature line, and a seal.")
    elif "cyrillic" in modes:
        if "anniversary" in lower or "academy" in lower:
            cues.extend(["220 ЛЕТ", "АКАДЕМИЯ НАУК"])
        elif "border" in lower or "frontier" in lower:
            cues.append("СЛАВА ПОГРАНИЧНИКАМ!")
        elif "victory" in lower or "wartime" in lower or "battle" in lower:
            cues.extend(["ПОБЕДА БУДЕТ ЗА НАМИ!", "ЗА РОДИНУ!"])
        else:
            cues.append("Use one short caption-appropriate Cyrillic headline.")
    if "calligraphy" in modes or "seal" in modes:
        cues.extend(
            [
                "Use brush calligraphy marks only where requested by the caption.",
                "Keep seals as small red seal marks integrated into the artwork surface.",
            ]
        )
    return _unique(cues)


def _provider_text_directives(text: dict[str, Any]) -> list[str]:
    directives = [str(item) for item in text.get("directives", []) if str(item)]
    return [
        item
        for item in directives
        if "future" not in item.lower()
        and "reference" not in item.lower()
        and "sample id" not in item.lower()
    ] + ["Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits."]


def _strip_provider_metadata(prompt: str, *, sample_id: str) -> str:
    forbidden_lines = (
        "EXPERT ROUTING CONSTRAINTS",
        "Recommended generation model:",
        "Generate ",
        "Primary expert workflow:",
        "Support checks before acceptance:",
        "fallback:",
        "Reference queries for future grounding:",
    )
    kept: list[str] = []
    skip_reference_queries = False
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Variant:"):
            if not kept or kept[-1] != "Produce only the image.":
                kept.append("Produce only the image.")
            continue
        if stripped == "Reference queries for future grounding:":
            skip_reference_queries = True
            continue
        if skip_reference_queries:
            if stripped.startswith("- "):
                continue
            skip_reference_queries = False
        if any(stripped.startswith(prefix) for prefix in forbidden_lines):
            continue
        if sample_id and sample_id in line:
            line = line.replace(sample_id, "")
        kept.append(line.rstrip())
    text = "\n".join(kept).strip()
    text = text.replace("gemini-3-pro-image", "").replace("gemini-3.1-flash-image", "")
    return text


def _distribution_routes_by_sample(
    distribution_routes: dict[str, dict[str, Any]] | list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if not distribution_routes:
        return {}
    if isinstance(distribution_routes, dict):
        if all(isinstance(value, dict) for value in distribution_routes.values()):
            return {str(key): dict(value) for key, value in distribution_routes.items()}
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in distribution_routes:
        if isinstance(row, dict) and row.get("sample_id"):
            result[str(row["sample_id"])] = dict(row)
    return result


def _validate_strategy_mode(strategy_mode: str) -> str:
    mode = str(strategy_mode or "auto")
    if mode not in {"auto", "legacy", "distribution"}:
        raise ValueError(f"strategy_mode must be one of auto, legacy, distribution: {mode}")
    return mode


def _validate_route_rows(rows: list[Any]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"route row {index} must be an object")
        if not row.get("sample_id"):
            raise ValueError(f"route row {index} missing required sample_id")
        if "candidate_count" not in row:
            raise ValueError(f"route row {index} missing required candidate_count")
        try:
            candidate_count = int(row["candidate_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"route row {index} has invalid candidate_count") from exc
        if candidate_count < 0:
            raise ValueError(f"route row {index} has invalid candidate_count")
        normalized = dict(row)
        normalized["sample_id"] = str(row["sample_id"])
        normalized["candidate_count"] = candidate_count
        validated.append(normalized)
    return validated


def _validate_contract_rows(rows: list[Any]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"contract row {index} must be an object")
        if not row.get("sample_id"):
            raise ValueError(f"contract row {index} missing required sample_id")
        if not row.get("caption"):
            raise ValueError(f"contract row {index} missing required caption")
        normalized = dict(row)
        normalized["sample_id"] = str(row["sample_id"])
        normalized["caption"] = str(row["caption"])
        validated.append(normalized)
    return validated


def _validate_contract_map(contracts: dict[str, dict[str, Any]]) -> None:
    for sample_id, contract in contracts.items():
        if not isinstance(contract, dict):
            raise ValueError(f"contract for sample_id {sample_id} must be an object")
        if not contract.get("caption"):
            raise ValueError(f"contract for sample_id {sample_id} missing required caption")


def _candidate_strategies(
    distribution_route: dict[str, Any] | None,
    *,
    max_strategies_per_sample: int | None,
) -> list[str]:
    if not distribution_route:
        return ["legacy"]
    strategies = [
        str(item.get("strategy") if isinstance(item, dict) else item)
        for item in distribution_route.get("candidate_strategies", [])
        if str(item.get("strategy") if isinstance(item, dict) else item)
    ]
    if not strategies:
        strategies = ["aas_safe", "reference_style", "fid_diverse"]
    if max_strategies_per_sample is not None and max_strategies_per_sample > 0:
        strategies = strategies[:max_strategies_per_sample]
    return strategies


def _soften_template_phrases(prompt: str, *, strategy: str) -> str:
    if strategy == "legacy":
        return prompt
    replacements = {
        "portrait poster canvas": "artwork surface",
        "Portrait poster canvas": "Artwork surface",
        "front-facing": "clearly legible",
        "Front-facing": "Clearly legible",
        "flat printed poster": "period printed artwork",
        "Flat printed poster": "Period printed artwork",
        "medium-distance figures": "varied figure scale",
        "Medium-distance figures": "Varied figure scale",
        "fewer, larger": "clear, selective",
        "Fewer, larger": "Clear, selective",
        "fewer/larger text blocks": "clear selective text blocks",
        "Fewer/larger text blocks": "Clear selective text blocks",
        "internal poster margins": "natural artwork edges",
        "Internal poster margins": "Natural artwork edges",
        "graphic poster composition": "composed artwork",
        "Graphic poster composition": "Composed artwork",
    }
    text = prompt
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def _clean_provider_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if "/" in text or "\\" in text:
            continue
        result.append(text.replace("_", " "))
    return _unique(result)


def _clean_reference_assets(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return _unique(str(value).strip() for value in values if str(value).strip())


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_budget_by_sample: dict[str, int] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id and sample_id not in source_budget_by_sample:
            source_budget_by_sample[sample_id] = int(row["review_metadata"].get("candidate_count") or 0)
    return {
        "total": len(rows),
        "candidate_budget": len(rows),
        "source_candidate_budget": sum(source_budget_by_sample.values()),
        "models": _counts(row["review_metadata"]["recommended_model"] for row in rows),
        "experts": _counts(row["review_metadata"]["primary_expert"] for row in rows),
    }


def _safe_output_dir(out_dir: str | Path) -> Path:
    path = Path(out_dir).expanduser()
    resolved = path.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    protected_images = (repo_root / "submissions" / "track1" / "images").resolve()
    protected_files = {
        (repo_root / "submissions" / "track1_submission.json").resolve(),
        (repo_root / "submissions" / "track1_submission.zip").resolve(),
    }
    if resolved == protected_images or protected_images in resolved.parents:
        raise ValueError(f"protected output path rejected: {resolved}")
    report_paths = {
        (resolved / "track1_moe_prompt_packets.json").resolve(),
        (resolved / "track1_moe_prompt_packets.jsonl").resolve(),
        (resolved / "track1_moe_prompt_packets.csv").resolve(),
        (resolved / "track1_moe_prompt_packets_zh.md").resolve(),
    }
    if resolved in protected_files or report_paths.intersection(protected_files):
        raise ValueError(f"protected output path rejected: {resolved}")
    return resolved


def _render_md(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Track1 Clean MoE Prompt Packets",
        "",
        "这是一份干净 provider prompt 审核包。候选数、模型名、expert route、reference queries 只保存在 metadata，不进入 Gemini prompt。",
        "",
        "## 摘要",
        "",
        f"- 样本数：{summary['total']}",
        f"- 候选图预算：{summary['candidate_budget']}",
        f"- 模型：{summary['models']}",
        f"- 专家：{summary['experts']}",
        "",
        "## 审核清单",
        "",
    ]
    for row in rows:
        meta = row["review_metadata"]
        lines.extend(
            [
                f"### {row['rank']:02d}. `{row['sample_id']}`",
                "",
                f"- 官方 caption：{row['caption']}",
                f"- 生成 metadata：model=`{meta['recommended_model']}` candidates=`{meta['candidate_count']}` expert=`{meta['primary_expert']}` support={', '.join(meta['support_experts']) or 'none'}",
                f"- Provider prompt：`{row['provider_prompt_path']}`",
                "",
                "<details><summary>Clean provider prompt</summary>",
                "",
                "```text",
                row["provider_prompt"],
                "```",
                "",
                "</details>",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "sample_id",
                "recommended_model",
                "candidate_count",
                "primary_expert",
                "support_experts",
                "provider_prompt_path",
                "caption",
            ],
        )
        writer.writeheader()
        for row in rows:
            meta = row["review_metadata"]
            writer.writerow(
                {
                    "rank": row["rank"],
                    "sample_id": row["sample_id"],
                    "recommended_model": meta["recommended_model"],
                    "candidate_count": meta["candidate_count"],
                    "primary_expert": meta["primary_expert"],
                    "support_experts": ";".join(meta["support_experts"]),
                    "provider_prompt_path": row["provider_prompt_path"],
                    "caption": row["caption"],
                }
            )


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _unique(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
