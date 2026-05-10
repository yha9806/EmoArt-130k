#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.challenge import _read_track2_samples
from affectiveart.jepa_backbones import lookup_backbone
from affectiveart.track2_supervised import (
    PublicExample,
    filter_embedding_rows_by_request_id,
    filter_public_examples,
    iter_public_examples,
    load_excluded_public_request_ids,
    write_supervised_candidate,
)


DEFAULT_PUBLIC_ROOT = Path("/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k")
DEFAULT_ANNOTATION = DEFAULT_PUBLIC_ROOT / "Annotation.json"
DEFAULT_TRACK2_ZIP = Path("data/raw/Track2_testset.zip")
DEFAULT_CURRENT_JSON = Path("submissions/track2_submission_gemini_reviewed.json")
DEFAULT_OUT_DIR = Path("experiments/track2_emoart130k_clip")
SEED = 20260508


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a CLIP-based supervised Track2 classifier on public EmoArt-130k labels."
    )
    parser.add_argument("--annotation-json", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--track2-zip", type=Path, default=DEFAULT_TRACK2_ZIP)
    parser.add_argument("--current-json", type=Path, default=DEFAULT_CURRENT_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--backend", choices=["clip", "hf"], default="clip")
    parser.add_argument("--model", default="ViT-B/32")
    parser.add_argument(
        "--allow-static-video-proxy",
        action="store_true",
        help="Allow video backbones by repeating still images as frames. Use only for diagnostics.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit-train", type=int, default=0, help="Stratified train subset size; 0 means full.")
    parser.add_argument("--limit-test", type=int, default=0, help="Only encode first N test samples; 0 means full.")
    parser.add_argument("--max-per-class", type=int, default=0, help="Class cap before optional limit; 0 means no cap.")
    parser.add_argument("--style-filter", action="append", default=[], help="Only use this public style; repeatable.")
    parser.add_argument(
        "--exclude-public-overlap-json",
        type=Path,
        default=DEFAULT_OUT_DIR / "suspected_public_test_overlap.json",
        help="JSON audit file with public request_ids to exclude from training/kNN evidence.",
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--skip-candidate", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.84)
    parser.add_argument("--min-margin", type=float, default=0.20)
    parser.add_argument("--min-knn-confidence", type=float, default=0.55)
    args = parser.parse_args()
    validate_backbone_for_static_track2(args.model, allow_static_video_proxy=args.allow_static_video_proxy)

    t0 = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    excluded_request_ids = load_excluded_public_request_ids(args.exclude_public_overlap_json)
    if excluded_request_ids:
        print(
            f"excluding {len(excluded_request_ids)} suspected public/test overlap examples "
            f"from {args.exclude_public_overlap_json}"
        )

    examples = list(iter_public_examples(args.annotation_json, data_root=args.public_root))
    examples = filter_public_examples(examples, excluded_request_ids)
    if args.style_filter:
        wanted_styles = set(args.style_filter)
        examples = [example for example in examples if example.style in wanted_styles]
    selected_examples = select_examples(
        examples,
        rng=rng,
        max_per_class=args.max_per_class or None,
        limit=args.limit_train or None,
    )
    print(f"public examples: {len(examples)} selected: {len(selected_examples)}")
    print(f"selected label distribution: {dict(sorted(Counter(e.emotion for e in selected_examples).items()))}")

    import torch

    device = resolve_device(args.device, torch)
    print(f"loading {args.backend} {args.model} on {device}")
    encoder = build_encoder(args.backend, args.model, device)

    train_cache = args.out_dir / f"train_{args.backend}_{cache_slug(args.model)}_{selection_tag(args)}.npz"
    train_npz = load_or_encode_public(
        train_cache,
        selected_examples,
        encoder=encoder,
        batch_size=args.batch_size,
        force=args.force_embeddings,
    )
    raw_train_count = len(train_npz["ids"])
    train_npz = filter_embedding_rows_by_request_id(train_npz, excluded_request_ids)
    filtered_train_count = len(train_npz["ids"])
    if raw_train_count != filtered_train_count:
        print(f"filtered cached train embeddings: {raw_train_count} -> {filtered_train_count}")

    test_cache = args.out_dir / f"test_{args.backend}_{cache_slug(args.model)}_{args.limit_test or 'full'}.npz"
    test_npz = load_or_encode_test(
        test_cache,
        args.track2_zip,
        encoder=encoder,
        batch_size=args.batch_size,
        limit=args.limit_test or None,
        force=args.force_embeddings,
    )

    metrics, predictions = train_and_predict(
        train_embeddings=train_npz["embeddings"],
        train_labels=train_npz["labels"],
        test_embeddings=test_npz["embeddings"],
        test_ids=test_npz["ids"],
        current_json=args.current_json if args.current_json.exists() else None,
        val_size=args.val_size,
        logreg_c=args.logreg_c,
        knn_k=args.knn_k,
        seed=args.seed,
    )

    predictions_path = args.out_dir / "predictions.json"
    metrics_path = args.out_dir / "metrics.json"
    payload = {
        "backend": args.backend,
        "model": args.model,
        "selection": selection_tag(args),
        "entries": predictions,
    }
    predictions_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics["wall_seconds"] = round(time.perf_counter() - t0, 1)
    metrics["predictions_json"] = str(predictions_path)
    metrics["excluded_public_overlap_count"] = len(excluded_request_ids)
    metrics["exclude_public_overlap_json"] = (
        str(args.exclude_public_overlap_json) if args.exclude_public_overlap_json else ""
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {predictions_path}")
    print(f"wrote {metrics_path}")
    print(f"holdout macro_f1: {metrics.get('holdout_macro_f1')}")
    print(f"test top distribution: {metrics['test_distribution']}")

    if not args.skip_candidate and args.current_json.exists() and not args.limit_test:
        report = write_supervised_candidate(
            args.current_json,
            predictions_path,
            out_json="submissions/track2_submission_supervised_candidate.json",
            out_zip="submissions/track2_submission_supervised_candidate.zip",
            report_json="submissions/track2_supervised_candidate_report.json",
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            min_knn_confidence=args.min_knn_confidence,
        )
        print("wrote submissions/track2_submission_supervised_candidate.json")
        print("wrote submissions/track2_submission_supervised_candidate.zip")
        print(f"supervised candidate changed_rows: {report['changed_rows']}")


def resolve_device(requested: str, torch_module) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_encoder(backend: str, model_name: str, device: str):
    if backend == "clip":
        return ClipImageEncoder.from_pretrained(model_name, device=device)
    if backend == "hf":
        return HFImageEncoder.from_pretrained(model_name, device=device)
    raise ValueError(f"unsupported backend: {backend}")


def validate_backbone_for_static_track2(model_name: str, *, allow_static_video_proxy: bool) -> None:
    try:
        backbone = lookup_backbone(model_name)
    except KeyError:
        return
    if backbone.modality == "video" and not allow_static_video_proxy:
        raise ValueError(
            f"{backbone.name} is a video backbone. Static Track2 images should not be passed "
            "through a video proxy unless --allow-static-video-proxy is explicit."
        )


class ClipImageEncoder:
    def __init__(self, model, preprocess, *, device: str):
        self.model = model
        self.preprocess = preprocess
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name: str, *, device: str):
        import clip  # type: ignore[import-not-found]

        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        return cls(model, preprocess, device=device)

    def encode_images(self, images: list[Image.Image]) -> list[np.ndarray]:
        import torch

        tensors = [self.preprocess(image) for image in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return [row.detach().cpu().numpy().astype("float32") for row in features]


class HFImageEncoder:
    def __init__(self, model, processor, *, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name: str, *, device: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        if device == "mps":
            model = model.to(dtype=torch.float32)
        return cls(model, processor, device=device)

    def encode_images(self, images: list[Image.Image]) -> list[np.ndarray]:
        import torch

        try:
            inputs = self.processor(images=images, return_tensors="pt")
        except TypeError as exc:
            if "videos" not in str(exc):
                raise
            inputs = self.processor(videos=static_video_batch(images, frame_count=64), return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(**inputs)
                if not hasattr(features, "norm"):
                    if getattr(features, "pooler_output", None) is not None:
                        features = features.pooler_output
                    elif getattr(features, "last_hidden_state", None) is not None:
                        features = features.last_hidden_state[:, 0]
                    else:
                        raise TypeError(f"unsupported get_image_features output: {type(features)!r}")
            else:
                outputs = self.model(**inputs)
                if getattr(outputs, "pooler_output", None) is not None:
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0]
            features = features / features.norm(dim=-1, keepdim=True)
        return [row.detach().cpu().numpy().astype("float32") for row in features]


def static_video_batch(images: list[Image.Image], *, frame_count: int) -> list[list[Image.Image]]:
    return [[image] * frame_count for image in images]


def cache_slug(model_name: str) -> str:
    return model_name.lower().replace("/", "_").replace("-", "_")


def selection_tag(args) -> str:
    parts = []
    if args.max_per_class:
        parts.append(f"max{args.max_per_class}pc")
    if args.limit_train:
        parts.append(f"limit{args.limit_train}")
    if not parts:
        parts.append("full")
    parts.append(f"seed{args.seed}")
    return "_".join(parts)


def select_examples(
    examples: list[PublicExample],
    *,
    rng: random.Random,
    max_per_class: int | None,
    limit: int | None,
) -> list[PublicExample]:
    by_label: dict[str, list[PublicExample]] = defaultdict(list)
    for example in examples:
        by_label[example.emotion].append(example)
    selected: list[PublicExample] = []
    for label in sorted(by_label):
        label_examples = list(by_label[label])
        rng.shuffle(label_examples)
        if max_per_class is not None:
            label_examples = label_examples[:max_per_class]
        selected.extend(label_examples)
    if limit is not None and len(selected) > limit:
        selected = stratified_limit(selected, limit=limit, rng=rng)
    rng.shuffle(selected)
    return selected


def stratified_limit(examples: list[PublicExample], *, limit: int, rng: random.Random) -> list[PublicExample]:
    by_label: dict[str, list[PublicExample]] = defaultdict(list)
    for example in examples:
        by_label[example.emotion].append(example)
    labels = sorted(by_label)
    selected: list[PublicExample] = []
    base = max(1, limit // max(1, len(labels)))
    for label in labels:
        label_examples = list(by_label[label])
        rng.shuffle(label_examples)
        selected.extend(label_examples[: min(base, len(label_examples))])
    selected_ids = {example.request_id for example in selected}
    remaining_pool = [
        example
        for label in labels
        for example in by_label[label]
        if example.request_id not in selected_ids
    ]
    rng.shuffle(remaining_pool)
    selected.extend(remaining_pool[: max(0, limit - len(selected))])
    return selected[:limit]


def load_or_encode_public(
    cache_path: Path,
    examples: list[PublicExample],
    *,
    encoder,
    batch_size: int,
    force: bool,
) -> dict[str, np.ndarray]:
    if cache_path.exists() and not force:
        print(f"loading {cache_path}")
        return dict(np.load(cache_path, allow_pickle=True))
    encoded = encode_public_examples(examples, encoder=encoder, batch_size=batch_size)
    np.savez_compressed(cache_path, **encoded)
    print(f"wrote {cache_path}")
    return encoded


def load_or_encode_test(
    cache_path: Path,
    track2_zip: Path,
    *,
    encoder,
    batch_size: int,
    limit: int | None,
    force: bool,
) -> dict[str, np.ndarray]:
    if cache_path.exists() and not force:
        print(f"loading {cache_path}")
        return dict(np.load(cache_path, allow_pickle=True))
    encoded = encode_test_images(track2_zip, encoder=encoder, batch_size=batch_size, limit=limit)
    np.savez_compressed(cache_path, **encoded)
    print(f"wrote {cache_path}")
    return encoded


def encode_public_examples(
    examples: list[PublicExample],
    *,
    encoder,
    batch_size: int,
) -> dict[str, np.ndarray]:
    by_tar: dict[str, list[tuple[int, PublicExample]]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_tar[example.tar_path].append((index, example))
    embeddings: list[np.ndarray] = [None] * len(examples)  # type: ignore[list-item]
    failures: list[dict[str, str]] = []
    for tar_index, tar_path in enumerate(sorted(by_tar), start=1):
        grouped = by_tar[tar_path]
        print(f"[{tar_index}/{len(by_tar)}] {Path(tar_path).name}: {len(grouped)} images", flush=True)
        targets: dict[str, list[tuple[int, PublicExample]]] = defaultdict(list)
        for index, example in grouped:
            targets[example.member].append((index, example))
        with tarfile.open(tar_path, "r:gz") as tf:
            batch_images = []
            batch_indices = []
            remaining = len(targets)
            for member in tf:
                if remaining <= 0:
                    break
                if not member.isfile() or member.name not in targets:
                    continue
                matches = targets.pop(member.name)
                remaining -= 1
                try:
                    file_obj = tf.extractfile(member)
                    if file_obj is None:
                        raise FileNotFoundError(member.name)
                    image = Image.open(BytesIO(file_obj.read())).convert("RGB")
                    for index, _example in matches:
                        batch_images.append(image.copy())
                        batch_indices.append(index)
                except Exception as exc:
                    for _index, example in matches:
                        failures.append({"request_id": example.request_id, "error": str(exc)})
                    continue
                if len(batch_images) >= batch_size:
                    flush_batch(encoder, batch_images, batch_indices, embeddings)
                    batch_images, batch_indices = [], []
            if batch_images:
                flush_batch(encoder, batch_images, batch_indices, embeddings)
    keep = [i for i, emb in enumerate(embeddings) if emb is not None]
    kept_examples = [examples[i] for i in keep]
    return {
        "ids": np.array([example.request_id for example in kept_examples], dtype=object),
        "labels": np.array([example.emotion for example in kept_examples], dtype=object),
        "styles": np.array([example.style for example in kept_examples], dtype=object),
        "members": np.array([example.member for example in kept_examples], dtype=object),
        "embeddings": np.stack([embeddings[i] for i in keep]).astype("float32"),
        "failures_json": np.array(json.dumps(failures, ensure_ascii=False), dtype=object),
    }


def encode_test_images(
    track2_zip: Path,
    *,
    encoder,
    batch_size: int,
    limit: int | None,
) -> dict[str, np.ndarray]:
    _format_version, rows = _read_track2_samples(track2_zip)
    if limit is not None:
        rows = rows[:limit]
    embeddings: list[np.ndarray] = []
    sample_ids: list[str] = []
    image_files: list[str] = []
    failures: list[dict[str, str]] = []
    with zipfile.ZipFile(track2_zip) as zf:
        batch_images = []
        pending_ids: list[str] = []
        pending_files: list[str] = []
        for row in rows:
            try:
                image = Image.open(BytesIO(zf.read(row["image_file"]))).convert("RGB")
                batch_images.append(image)
                pending_ids.append(row["sample_id"])
                pending_files.append(row["image_file"])
            except Exception as exc:
                failures.append({"sample_id": row.get("sample_id", ""), "error": str(exc)})
                continue
            if len(batch_images) >= batch_size:
                batch_embeddings = encoder.encode_images(batch_images)
                embeddings.extend(batch_embeddings)
                sample_ids.extend(pending_ids)
                image_files.extend(pending_files)
                batch_images, pending_ids, pending_files = [], [], []
        if batch_images:
            batch_embeddings = encoder.encode_images(batch_images)
            embeddings.extend(batch_embeddings)
            sample_ids.extend(pending_ids)
            image_files.extend(pending_files)
    return {
        "ids": np.array(sample_ids, dtype=object),
        "image_files": np.array(image_files, dtype=object),
        "embeddings": np.stack(embeddings).astype("float32"),
        "failures_json": np.array(json.dumps(failures, ensure_ascii=False), dtype=object),
    }


def flush_batch(encoder, images, indices, embeddings) -> None:
    batch_embeddings = encoder.encode_images(images)
    for index, embedding in zip(indices, batch_embeddings):
        embeddings[index] = embedding


def train_and_predict(
    *,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_ids: np.ndarray,
    current_json: Path | None,
    val_size: float,
    logreg_c: float,
    knn_k: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedShuffleSplit

    x = np.asarray(train_embeddings, dtype="float32")
    y = np.asarray(train_labels).astype(str)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=logreg_c)
    metrics: dict = {
        "train_count": int(len(y)),
        "train_distribution": dict(sorted(Counter(y).items())),
        "logreg_c": logreg_c,
        "knn_k": knn_k,
    }
    can_holdout = len(set(y)) > 1 and min(Counter(y).values()) >= 2 and 0.0 < val_size < 0.5
    if can_holdout:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(x, y))
        clf.fit(x[train_idx], y[train_idx])
        val_pred = clf.predict(x[val_idx])
        metrics["holdout_accuracy"] = round(float(accuracy_score(y[val_idx], val_pred)), 4)
        metrics["holdout_macro_f1"] = round(float(f1_score(y[val_idx], val_pred, average="macro")), 4)
        metrics["holdout_weighted_f1"] = round(float(f1_score(y[val_idx], val_pred, average="weighted")), 4)
    clf.fit(x, y)

    probabilities = clf.predict_proba(test_embeddings)
    classes = np.asarray(clf.classes_).astype(str)
    knn_votes = compute_knn_votes(x, y, test_embeddings, k=knn_k)
    current = load_current_emotions(current_json) if current_json is not None else {}
    entries: list[dict] = []
    for index, sample_id in enumerate(np.asarray(test_ids).astype(str)):
        order = np.argsort(probabilities[index])[::-1]
        top = order[0]
        second = order[1] if len(order) > 1 else order[0]
        top3 = [
            {"emotion": str(classes[i]), "probability": round(float(probabilities[index][i]), 6)}
            for i in order[:3]
        ]
        vote = knn_votes[index]
        entry = {
            "sample_id": sample_id,
            "current": current.get(sample_id, ""),
            "emotion": str(classes[top]),
            "confidence": round(float(probabilities[index][top]), 6),
            "margin": round(float(probabilities[index][top] - probabilities[index][second]), 6),
            "top3": top3,
            "knn_emotion": vote["emotion"],
            "knn_confidence": vote["confidence"],
            "knn_top3": vote["top3"],
        }
        entries.append(entry)
    metrics["test_count"] = len(entries)
    metrics["test_distribution"] = dict(sorted(Counter(entry["emotion"] for entry in entries).items()))
    metrics["current_distribution"] = dict(sorted(Counter(entry["current"] for entry in entries if entry["current"]).items()))
    metrics["classifier_knn_agreement"] = round(
        sum(1 for entry in entries if entry["emotion"] == entry["knn_emotion"]) / max(1, len(entries)),
        4,
    )
    return metrics, entries


def compute_knn_votes(train_embeddings: np.ndarray, labels: np.ndarray, test_embeddings: np.ndarray, *, k: int) -> list[dict]:
    labels = np.asarray(labels).astype(str)
    k = min(k, len(labels))
    votes: list[dict] = []
    for start in range(0, len(test_embeddings), 64):
        sims = np.asarray(test_embeddings[start : start + 64], dtype="float32") @ train_embeddings.T
        top_indices = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        for row_index, indices in enumerate(top_indices):
            row_sims = sims[row_index, indices]
            order = np.argsort(row_sims)[::-1]
            ranked = indices[order]
            counts: Counter[str] = Counter(labels[i] for i in ranked)
            weighted: Counter[str] = Counter()
            for i in ranked:
                weighted[labels[i]] += max(0.0, float(sims[row_index, i]))
            ordered_labels = sorted(counts, key=lambda label: (counts[label], weighted[label]), reverse=True)
            top3 = [
                {
                    "emotion": label,
                    "votes": int(counts[label]),
                    "similarity_sum": round(float(weighted[label]), 6),
                }
                for label in ordered_labels[:3]
            ]
            votes.append(
                {
                    "emotion": ordered_labels[0],
                    "confidence": round(float(counts[ordered_labels[0]] / k), 6),
                    "top3": top3,
                }
            )
    return votes


def load_current_emotions(path: Path) -> dict[str, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("sample_id", "")): str(row.get("emotion", "")) for row in rows if isinstance(row, dict)}


if __name__ == "__main__":
    main()
