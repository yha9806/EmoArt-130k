#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
import tarfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageFile
from scipy import linalg
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local InceptionV3 FID-like sanity check for Track1 packages."
    )
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--reference-style", action="append", default=[])
    parser.add_argument("--reference-limit", type=int, default=2048)
    parser.add_argument("--reference-seed", type=int, default=20260603)
    parser.add_argument(
        "--fid-feature-dim",
        type=int,
        default=512,
        help="Project Inception pool features to this dimension before Fréchet distance. Use 2048 for full pool features.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        help="Optional directory to save projected reference and package features as NPZ files.",
    )
    parser.add_argument(
        "--package",
        action="append",
        required=True,
        help="Package spec NAME=SUBMISSION_JSON=IMAGE_DIR",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    device = choose_device(args.device)
    model = build_feature_model(device)
    transform = build_transform()
    package_specs = parse_package_specs(args.package)
    reference_items = sample_reference_items(
        args.reference_root,
        styles=args.reference_style,
        limit=args.reference_limit,
        seed=args.reference_seed,
    )
    reference_features = encode_reference_items(
        reference_items,
        model=model,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
    )
    print(f"encoded reference features: {reference_features.shape}", flush=True)
    projector = build_projector(reference_features.shape[1], args.fid_feature_dim, args.reference_seed)
    reference_features = project_features(reference_features, projector)
    if args.feature_cache_dir:
        save_feature_cache(
            args.feature_cache_dir,
            "reference",
            [f"{tar_path.name}::{member_name}" for tar_path, member_name in reference_items],
            reference_features,
        )
    reference_stats = frechet_stats(reference_features)
    packages = []
    for package_name, submission_json, image_dir in package_specs:
        image_paths = package_image_paths(submission_json, image_dir)
        print(f"encoding package {package_name}: {len(image_paths)} images", flush=True)
        features = encode_image_paths(
            image_paths,
            model=model,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
        )
        print(f"encoded package {package_name}: {features.shape}", flush=True)
        features = project_features(features, projector)
        if args.feature_cache_dir:
            save_feature_cache(args.feature_cache_dir, package_name, [path.stem for path in image_paths], features)
        stats = frechet_stats(features)
        packages.append(
            {
                "package": package_name,
                "submission_json": str(submission_json),
                "image_dir": str(image_dir),
                "image_count": len(image_paths),
                "feature_shape": list(features.shape),
                "fid_like": round(float(frechet_distance(stats, reference_stats)), 6),
            }
        )
    packages.sort(key=lambda item: item["fid_like"])
    report = {
        "method": {
            "name": "local_inception_v3_fid_like_sanity",
            "note": (
                "Not the official hidden evaluator. Uses torchvision InceptionV3 pool features, "
                "torchvision-style resize/normalization, sampled local EmoArt reference images, "
                "and an optional deterministic Gaussian random projection before Fréchet distance."
            ),
            "device": str(device),
            "batch_size": args.batch_size,
            "reference_root": str(args.reference_root),
            "reference_styles": args.reference_style or "all_tar_styles",
            "reference_limit": args.reference_limit,
            "reference_seed": args.reference_seed,
            "reference_count": int(reference_features.shape[0]),
            "reference_feature_shape": list(reference_features.shape),
            "fid_feature_dim": args.fid_feature_dim,
            "projection": (
                "deterministic Gaussian random projection seeded by reference_seed when "
                "fid_feature_dim < Inception feature dim"
            ),
        },
        "packages": packages,
        "recommended_by_fid_like": packages[0]["package"] if packages else "",
        "wall_seconds": round(time.time() - started, 2),
    }
    write_report(report, args.out_json, args.out_md)
    print(json.dumps(report["packages"], indent=2, ensure_ascii=False))
    return 0


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def build_feature_model(device: torch.device) -> torch.nn.Module:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def parse_package_specs(specs: Iterable[str]) -> list[tuple[str, Path, Path]]:
    result = []
    for spec in specs:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"package spec must be NAME=SUBMISSION_JSON=IMAGE_DIR: {spec}")
        name, submission_json, image_dir = parts
        result.append((name, Path(submission_json), Path(image_dir)))
    return result


def package_image_paths(submission_json: Path, image_dir: Path) -> list[Path]:
    rows = json.loads(submission_json.read_text(encoding="utf-8"))
    paths = []
    for row in rows:
        name = Path(str(row.get("path", ""))).name
        if name:
            paths.append(image_dir / name)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing package images: {missing[:3]}")
    return paths


def sample_reference_items(
    reference_root: Path,
    *,
    styles: list[str],
    limit: int,
    seed: int,
) -> list[tuple[Path, str]]:
    tar_paths = reference_tar_paths(reference_root, styles)
    members_by_tar = []
    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r:gz") as archive:
            members = [
                member.name
                for member in archive.getmembers()
                if member.isfile() and member.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
        if members:
            members_by_tar.append((tar_path, sorted(members)))
    if not members_by_tar:
        raise FileNotFoundError("no reference images found")
    all_items = [(tar_path, member) for tar_path, members in members_by_tar for member in members]
    rng = random.Random(seed)
    if limit > 0 and len(all_items) > limit:
        all_items = rng.sample(all_items, limit)
    all_items.sort(key=lambda item: stable_key(str(item[0]), item[1]))
    return all_items


def reference_tar_paths(reference_root: Path, styles: list[str]) -> list[Path]:
    if not styles:
        return sorted(reference_root.glob("*.tar.gz"))
    result = []
    for style in styles:
        path = reference_root / f"{style}.tar.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        result.append(path)
    return result


def encode_reference_items(
    items: list[tuple[Path, str]],
    *,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    features = []
    batch = []
    for tar_path, member_names in group_items_by_tar(items):
        selected = set(member_names)
        with tarfile.open(tar_path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or member.name not in selected:
                    continue
                try:
                    image = load_tar_image_from_archive(archive, member)
                except Exception:
                    continue
                batch.append(transform(image))
                if len(batch) >= batch_size:
                    features.append(encode_batch(batch, model=model, device=device))
                    batch = []
    if batch:
        features.append(encode_batch(batch, model=model, device=device))
    if not features:
        raise RuntimeError("failed to encode reference images")
    return np.concatenate(features, axis=0)


def encode_image_paths(
    paths: list[Path],
    *,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    features = []
    batch = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        batch.append(transform(image))
        if len(batch) >= batch_size:
            features.append(encode_batch(batch, model=model, device=device))
            batch = []
    if batch:
        features.append(encode_batch(batch, model=model, device=device))
    return np.concatenate(features, axis=0)


def encode_batch(
    batch: list[torch.Tensor],
    *,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.stack(batch).to(device)
    with torch.inference_mode():
        output = model(tensor)
    if isinstance(output, tuple):
        output = output[0]
    return output.detach().float().cpu().numpy()


def load_tar_image(tar_path: Path, member_name: str) -> Image.Image:
    with tarfile.open(tar_path, "r:gz") as archive:
        return load_tar_image_from_archive(archive, member_name)


def load_tar_image_from_archive(archive: tarfile.TarFile, member: str | tarfile.TarInfo) -> Image.Image:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(member_name)
    data = extracted.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def group_items_by_tar(items: list[tuple[Path, str]]) -> list[tuple[Path, list[str]]]:
    grouped: dict[Path, list[str]] = {}
    for tar_path, member_name in items:
        grouped.setdefault(tar_path, []).append(member_name)
    return sorted(grouped.items(), key=lambda item: str(item[0]))


def frechet_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    return features.mean(axis=0), np.cov(features, rowvar=False)


def build_projector(input_dim: int, output_dim: int, seed: int) -> np.ndarray | None:
    if output_dim <= 0 or output_dim >= input_dim:
        return None
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / math.sqrt(output_dim), size=(input_dim, output_dim)).astype(np.float64)


def project_features(features: np.ndarray, projector: np.ndarray | None) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if projector is None:
        return features
    return features @ projector


def save_feature_cache(cache_dir: Path, name: str, ids: list[str], features: np.ndarray) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    path = cache_dir / f"{safe_name}_features.npz"
    np.savez_compressed(path, ids=np.asarray(ids), features=np.asarray(features, dtype=np.float32))
    return path


def frechet_distance(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> float:
    mu1, sigma1 = left
    mu2, sigma2 = right
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return max(0.0, float(value))


def stable_key(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()


def write_report(report: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Track1 Inception FID-Like Sanity Check",
        "",
        "This is not the official hidden evaluator. It uses local torchvision InceptionV3 features, torchvision-style resize/normalization, sampled local EmoArt reference images, and optional deterministic Gaussian projection.",
        "",
        "## Method",
        "",
        f"- Device: `{report['method']['device']}`",
        f"- Reference styles: `{report['method']['reference_styles']}`",
        f"- Reference count: `{report['method']['reference_count']}`",
        f"- Reference seed: `{report['method']['reference_seed']}`",
        f"- FID feature dim: `{report['method']['fid_feature_dim']}`",
        "",
        "## Packages",
        "",
    ]
    best = report.get("recommended_by_fid_like", "")
    for package in report["packages"]:
        marker = " best" if package["package"] == best else ""
        lines.append(
            f"- `{package['package']}` fid_like=`{package['fid_like']}` images=`{package['image_count']}`{marker}"
        )
    lines.extend(["", f"Wall seconds: `{report['wall_seconds']}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
