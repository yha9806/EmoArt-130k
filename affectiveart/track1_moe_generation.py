from __future__ import annotations

import argparse
import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image

from affectiveart.track1_vulca import select_track1_tradition


DEFAULT_FLASH_IMAGE_MODEL = "gemini-3.1-flash-image-preview"


def load_moe_packets(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    return [dict(row) for row in payload.get("packets", payload.get("rows", []))]


def resolve_image_model(
    model_label: str,
    *,
    flash_image_model: str = DEFAULT_FLASH_IMAGE_MODEL,
    pro_image_model: str | None = None,
    image_model_override: str | None = None,
) -> str:
    if image_model_override:
        return image_model_override
    label = (model_label or "").strip()
    if not label:
        return flash_image_model
    if label == "gemini-3.1-flash-image":
        return flash_image_model
    if label == "gemini-3.1-flash-image-preview":
        return label
    if label == "gemini-3-pro-image":
        return pro_image_model or flash_image_model
    return label


def build_generation_jobs(
    packets: Iterable[dict[str, Any]],
    *,
    out_dir: str | Path,
    sample_ids: Iterable[str] | None = None,
    limit: int | None = None,
    max_candidates_per_sample: int | None = None,
    flash_image_model: str = DEFAULT_FLASH_IMAGE_MODEL,
    pro_image_model: str | None = None,
    image_model_override: str | None = None,
) -> list[dict[str, Any]]:
    out_dir = Path(out_dir)
    prompt_dir = out_dir / "prompts"
    image_dir = out_dir / "images"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    selected = set(sample_ids or [])
    filtered = [packet for packet in packets if not selected or str(packet.get("sample_id")) in selected]
    if limit is not None and limit > 0:
        filtered = filtered[:limit]

    jobs: list[dict[str, Any]] = []
    for packet in filtered:
        sample_id = str(packet["sample_id"])
        meta = dict(packet.get("review_metadata") or {})
        aspect = dict(packet.get("aspect_plan") or {})
        requested_count = int(meta.get("candidate_count") or 1)
        candidate_count = requested_count
        if max_candidates_per_sample is not None and max_candidates_per_sample > 0:
            candidate_count = min(candidate_count, max_candidates_per_sample)
        width = int(aspect.get("width") or 1024)
        height = int(aspect.get("height") or 1024)
        model_label = str(meta.get("recommended_model") or "")
        model = resolve_image_model(
            model_label,
            flash_image_model=flash_image_model,
            pro_image_model=pro_image_model,
            image_model_override=image_model_override,
        )
        candidate_strategy = _candidate_strategy(packet)
        for candidate_index in range(1, candidate_count + 1):
            stem = f"{sample_id}{candidate_strategy}_c{candidate_index:02d}"
            prompt_path = prompt_dir / f"{stem}.txt"
            image_path = image_dir / f"{stem}.png"
            provider_prompt = str(packet["provider_prompt"])
            prompt_path.write_text(provider_prompt.rstrip() + "\n", encoding="utf-8")
            jobs.append(
                {
                    "rank": packet.get("rank"),
                    "sample_id": sample_id,
                    "candidate_index": candidate_index,
                    "caption": packet.get("caption", ""),
                    "provider_prompt": provider_prompt,
                    "prompt_path": str(prompt_path),
                    "image_path": str(image_path),
                    "metadata_path": str(image_path.with_suffix(".json")),
                    "requested_model_label": model_label,
                    "model": model,
                    "width": width,
                    "height": height,
                    "aspect_label": aspect.get("label", ""),
                    "candidate_strategy": candidate_strategy.removeprefix("_") or "",
                    "primary_expert": meta.get("primary_expert", ""),
                    "status": "planned",
                }
            )
    return jobs


def run_generation_jobs(
    jobs: Iterable[dict[str, Any]],
    *,
    provider_factory: Callable[[str], Any] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    return asyncio.run(_run_generation_jobs_async(list(jobs), provider_factory=provider_factory, force=force))


async def _run_generation_jobs_async(
    jobs: list[dict[str, Any]],
    *,
    provider_factory: Callable[[str], Any] | None,
    force: bool,
) -> list[dict[str, Any]]:
    if provider_factory is None:
        from vulca.providers.gemini import GeminiImageProvider

        provider_factory = lambda model: GeminiImageProvider(model=model)

    providers: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for job in jobs:
        row = dict(job)
        image_path = Path(str(row["image_path"]))
        metadata_path = Path(str(row["metadata_path"]))
        if image_path.exists() and metadata_path.exists() and not force:
            row["status"] = "cached"
            rows.append(row)
            print(f"{row['sample_id']} c{row['candidate_index']:02d}: cached {image_path}", flush=True)
            continue
        image_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            model = str(row["model"])
            if model not in providers:
                providers[model] = provider_factory(model)
            provider = providers[model]
            result = await provider.generate(
                str(row["provider_prompt"]),
                tradition=select_track1_tradition(str(row.get("caption") or "")),
                width=int(row.get("width") or 1024),
                height=int(row.get("height") or 1024),
                raw_prompt=True,
            )
            _save_result_as_png(result.image_b64, image_path)
            metadata = {
                "sample_id": row["sample_id"],
                "candidate_index": row["candidate_index"],
                "caption": row.get("caption", ""),
                "requested_model_label": row.get("requested_model_label", ""),
                "image_model": model,
                "requested_width": int(row.get("width") or 1024),
                "requested_height": int(row.get("height") or 1024),
                "provider_metadata": getattr(result, "metadata", {}),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            row["status"] = "generated"
            print(f"{row['sample_id']} c{row['candidate_index']:02d}: generated {image_path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - manifest should capture provider failures.
            row["status"] = "error"
            row["error"] = str(exc)
            error_path = image_path.with_suffix(".error.txt")
            error_path.write_text(str(exc) + "\n", encoding="utf-8")
            row["error_path"] = str(error_path)
            print(f"{row['sample_id']} c{row['candidate_index']:02d}: error {exc}", flush=True)
        rows.append(row)
    return rows


def write_generation_manifest(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = _summary(rows)
    path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Track1 candidates from clean MoE provider prompt packets.")
    parser.add_argument("--packets-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples before candidate expansion.")
    parser.add_argument("--max-candidates-per-sample", type=int, default=0)
    parser.add_argument("--flash-image-model", default=DEFAULT_FLASH_IMAGE_MODEL)
    parser.add_argument("--pro-image-model", default="")
    parser.add_argument("--image-model-override", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def generate_from_args(args: argparse.Namespace) -> int:
    packets = load_moe_packets(args.packets_jsonl)
    jobs = build_generation_jobs(
        packets,
        out_dir=args.out_dir,
        sample_ids=args.sample_id,
        limit=args.limit or None,
        max_candidates_per_sample=args.max_candidates_per_sample or None,
        flash_image_model=args.flash_image_model,
        pro_image_model=args.pro_image_model or None,
        image_model_override=args.image_model_override or None,
    )
    if args.dry_run:
        rows = [dict(job, status="planned") for job in jobs]
    else:
        rows = run_generation_jobs(jobs, force=args.force)
    write_generation_manifest(rows, args.out_dir / "candidate_manifest.json")
    print(args.out_dir / "candidate_manifest.json")
    return 0 if all(row.get("status") in {"planned", "generated", "cached"} for row in rows) else 1


def _save_result_as_png(image_b64: str, path: Path) -> None:
    data = base64.b64decode(image_b64)
    with Image.open(BytesIO(data)) as image:
        image.convert("RGB").save(path, format="PNG")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"total": len(rows)}
    for status in ("planned", "generated", "cached", "error"):
        summary[status] = sum(1 for row in rows if row.get("status") == status)
    summary["samples"] = len({str(row.get("sample_id")) for row in rows})
    summary["models"] = _counts(str(row.get("model") or "") for row in rows)
    return summary


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _candidate_strategy(packet: dict[str, Any]) -> str:
    value = str(packet.get("candidate_strategy") or "").strip()
    if not value or value == "legacy":
        return ""
    safe = "".join(character if character.isalnum() else "_" for character in value)
    safe = "_".join(part for part in safe.split("_") if part)
    return f"_{safe}" if safe else ""
