# Vulca JEPA Model Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, repeatable JEPA-focused experiment lane for Vulca that tests whether JEPA-style visual representations improve image structure auditing, Track2 emotion evidence, and Track1 generated-image fidelity checks.

**Architecture:** Keep challenge submissions stable while adding a separate experiment layer. Reuse the existing Track2 embedding/classification pipeline for image backbones, add a small model registry for JEPA/DINO/SigLIP/CLIP variants, and add Vulca-specific audit reports that compare generated or labeled artwork against caption, structure, and emotion evidence. Treat the output as an evaluation lane for deciding how JEPA helps Vulca, not as an automatic replacement for the current challenge submissions.

**Tech Stack:** Python 3.14, NumPy, scikit-learn, PIL, Transformers/Hugging Face, existing `affectiveart` modules, local EmoArt-130k files, existing Track1/Track2 submissions and experiments.

---

## Current Baseline

Existing usable artifacts:

- Track2 final candidate: `submissions/track2_submission.zip`
- Track2 strict ensemble report: `submissions/track2_ensemble_strict_candidate_report.json`
- Track2 full model runs:
  - `experiments/track2_emoart130k_clip/metrics.json`
  - `experiments/track2_emoart130k_siglip2/metrics.json`
  - `experiments/track2_emoart130k_dinov2/metrics.json`
- I-JEPA smoke artifacts:
  - `experiments/track2_backend_smoke_ijepa/metrics.json`
  - `experiments/track2_backend_smoke_ijepa/predictions.json`
- Track1 Vulca generation/audit artifacts:
  - `submissions/track1_submission.zip`
  - `submissions/track1_vulca_live_audit_report.json`

Known findings:

- SigLIP2 is currently the strongest Track2 single backbone by local holdout macro-F1.
- DINOv2 is useful as a structural vision baseline but did not beat SigLIP2 for Track2 emotion labels.
- I-JEPA is research-interesting, but `facebook/ijepa_vith16_1k` was too slow for local full Track2 on MPS.
- `facebook/ijepa_vith14_1k` is lower input resolution, not lower parameter count; it is still ViT-H scale.
- V-JEPA should not be used as a static Track2 classifier by repeating a still image into video frames; that does not test the model's core dynamic prediction capability.

## Intended Product Upgrade

This project should not try to make JEPA replace Gemini/SigLIP for all labels. The useful product role is narrower:

1. Use JEPA/DINO-family embeddings to audit visual structure and subject preservation.
2. Use SigLIP/CLIP-family embeddings to audit text-image semantic alignment.
3. Use Vulca-EMNLP style checks to audit caption, attribute, and emotion consistency.
4. Use disagreements between these systems to find cases where Vulca over-emphasizes style/culture while losing the literal user request.

## File Structure

Create:

- `affectiveart/jepa_backbones.py`
  - Model registry for local experiment backbones.
  - Distinguishes image-safe backbones from video-only backbones.
  - Encodes known constraints such as gated access, expected speed, and whether a model is recommended for Track2.

- `affectiveart/vulca_jepa_audit.py`
  - Pure functions for comparing embedding-model predictions with current submission labels.
  - Pure functions for selecting samples for human review.
  - Report builder for Track1 and Track2 JEPA/Vulca audit summaries.

- `scripts/vulca_jepa_experiment.py`
  - CLI entrypoint for smoke runs and report generation.
  - Reads existing metrics/predictions where possible.
  - Starts new embedding runs only when explicitly requested.

- `tests/test_jepa_backbones.py`
  - Unit tests for model registry and backend selection.

- `tests/test_vulca_jepa_audit.py`
  - Unit tests for disagreement scoring and review sample selection.

- `docs/vulca_jepa_experiment_report.md`
  - Human-readable report produced from local results.

Modify:

- `scripts/track2_clip_emoart130k.py`
  - Import model registry metadata from `affectiveart/jepa_backbones.py`.
  - Refuse video-only backbones unless `--allow-static-video-proxy` is passed.
  - Keep existing CLIP/HF behavior intact.

Do not modify:

- `submissions/track1_submission.zip`
- `submissions/track2_submission.zip`

Those are challenge submission artifacts. Experiments should write to `experiments/vulca_jepa_*` and docs only.

---

### Task 1: Model Registry

**Files:**
- Create: `affectiveart/jepa_backbones.py`
- Test: `tests/test_jepa_backbones.py`

- [ ] **Step 1: Write the failing registry tests**

Create `tests/test_jepa_backbones.py`:

```python
import unittest

from affectiveart.jepa_backbones import (
    BACKBONES,
    image_backbones,
    lookup_backbone,
    recommended_track2_backbones,
)


class JepaBackboneRegistryTest(unittest.TestCase):
    def test_registry_contains_image_jepa_and_video_jepa_metadata(self) -> None:
        ijepa = lookup_backbone("ijepa-vith16-1k")
        self.assertEqual(ijepa.model_id, "facebook/ijepa_vith16_1k")
        self.assertEqual(ijepa.modality, "image")
        self.assertFalse(ijepa.recommended_for_track2_full)

        vjepa = lookup_backbone("vjepa2-vitl")
        self.assertEqual(vjepa.modality, "video")
        self.assertFalse(vjepa.static_image_safe)

    def test_recommended_track2_backbones_exclude_video_models(self) -> None:
        names = [backbone.name for backbone in recommended_track2_backbones()]
        self.assertIn("siglip2-base-patch16-224", names)
        self.assertIn("dinov2-base", names)
        self.assertNotIn("vjepa2-vitl", names)

    def test_image_backbones_include_ijepa_variants(self) -> None:
        names = [backbone.name for backbone in image_backbones()]
        self.assertIn("ijepa-vith16-1k", names)
        self.assertIn("ijepa-vith14-1k", names)

    def test_lookup_raises_for_unknown_name(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown backbone"):
            lookup_backbone("missing-model")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_jepa_backbones.py' -v
```

Expected result:

```text
ModuleNotFoundError: No module named 'affectiveart.jepa_backbones'
```

- [ ] **Step 3: Implement the registry**

Create `affectiveart/jepa_backbones.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    model_id: str
    backend: str
    modality: str
    family: str
    static_image_safe: bool
    recommended_for_track2_full: bool
    local_notes: str


BACKBONES: tuple[BackboneSpec, ...] = (
    BackboneSpec(
        name="clip-vit-b-32",
        model_id="ViT-B/32",
        backend="clip",
        modality="image",
        family="clip",
        static_image_safe=True,
        recommended_for_track2_full=True,
        local_notes="Fast baseline; useful for leakage audit and retrieval sanity checks.",
    ),
    BackboneSpec(
        name="siglip2-base-patch16-224",
        model_id="google/siglip2-base-patch16-224",
        backend="hf",
        modality="image",
        family="siglip",
        static_image_safe=True,
        recommended_for_track2_full=True,
        local_notes="Current strongest Track2 single backbone in local holdout.",
    ),
    BackboneSpec(
        name="dinov2-base",
        model_id="facebook/dinov2-base",
        backend="hf",
        modality="image",
        family="dino",
        static_image_safe=True,
        recommended_for_track2_full=True,
        local_notes="Good structural vision baseline; weaker than SigLIP2 on Track2 emotion labels.",
    ),
    BackboneSpec(
        name="ijepa-vith16-1k",
        model_id="facebook/ijepa_vith16_1k",
        backend="hf",
        modality="image",
        family="jepa",
        static_image_safe=True,
        recommended_for_track2_full=False,
        local_notes="ViT-H 448px I-JEPA; smoke works but local full Track2 is too slow on MPS.",
    ),
    BackboneSpec(
        name="ijepa-vith14-1k",
        model_id="facebook/ijepa_vith14_1k",
        backend="hf",
        modality="image",
        family="jepa",
        static_image_safe=True,
        recommended_for_track2_full=False,
        local_notes="ViT-H 224px I-JEPA; lower resolution, not lower parameter count.",
    ),
    BackboneSpec(
        name="vjepa2-vitl",
        model_id="facebook/vjepa2-vitl-fpc64-256",
        backend="hf",
        modality="video",
        family="jepa",
        static_image_safe=False,
        recommended_for_track2_full=False,
        local_notes="Video model; repeated still frames are not a meaningful Track2 signal.",
    ),
)


def lookup_backbone(name: str) -> BackboneSpec:
    for backbone in BACKBONES:
        if backbone.name == name or backbone.model_id == name:
            return backbone
    raise KeyError(f"unknown backbone: {name}")


def image_backbones() -> list[BackboneSpec]:
    return [backbone for backbone in BACKBONES if backbone.modality == "image"]


def recommended_track2_backbones() -> list[BackboneSpec]:
    return [backbone for backbone in BACKBONES if backbone.recommended_for_track2_full]
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_jepa_backbones.py' -v
```

Expected result:

```text
Ran 4 tests
OK
```

- [ ] **Step 5: Commit**

Run:

```bash
git add affectiveart/jepa_backbones.py tests/test_jepa_backbones.py
git commit -m "feat: add JEPA backbone registry"
```

---

### Task 2: Guard Video JEPA Usage in Track2 Runner

**Files:**
- Modify: `scripts/track2_clip_emoart130k.py`
- Test: `tests/test_track2_embedding_script.py`

- [ ] **Step 1: Add failing tests for video-model guardrails**

Append to `tests/test_track2_embedding_script.py`:

```python
import unittest

from scripts.track2_clip_emoart130k import validate_backbone_for_static_track2


class Track2BackboneGuardTest(unittest.TestCase):
    def test_rejects_video_backbone_without_static_proxy_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "video backbone"):
            validate_backbone_for_static_track2("vjepa2-vitl", allow_static_video_proxy=False)

    def test_allows_video_backbone_when_proxy_flag_is_explicit(self) -> None:
        validate_backbone_for_static_track2("vjepa2-vitl", allow_static_video_proxy=True)

    def test_allows_image_backbone_without_proxy_flag(self) -> None:
        validate_backbone_for_static_track2("ijepa-vith16-1k", allow_static_video_proxy=False)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_track2_embedding_script.py' -v
```

Expected result:

```text
ImportError: cannot import name 'validate_backbone_for_static_track2'
```

- [ ] **Step 3: Add the CLI flag and validation function**

Modify `scripts/track2_clip_emoart130k.py`.

Add imports near the existing `affectiveart.track2_supervised` import:

```python
from affectiveart.jepa_backbones import lookup_backbone
```

Add an argument near the existing model arguments:

```python
parser.add_argument(
    "--allow-static-video-proxy",
    action="store_true",
    help="Allow video backbones by repeating still images as frames. Use only for diagnostics.",
)
```

Call the guard immediately after parsing arguments:

```python
validate_backbone_for_static_track2(args.model, allow_static_video_proxy=args.allow_static_video_proxy)
```

Add this function near `build_encoder`:

```python
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
```

- [ ] **Step 4: Run the embedding script tests**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_track2_embedding_script.py' -v
```

Expected result:

```text
OK
```

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/track2_clip_emoart130k.py tests/test_track2_embedding_script.py
git commit -m "chore: guard video JEPA usage for static Track2"
```

---

### Task 3: Vulca JEPA Audit Core

**Files:**
- Create: `affectiveart/vulca_jepa_audit.py`
- Test: `tests/test_vulca_jepa_audit.py`

- [ ] **Step 1: Write failing tests for audit scoring**

Create `tests/test_vulca_jepa_audit.py`:

```python
import unittest

from affectiveart.vulca_jepa_audit import (
    build_disagreement_report,
    select_vulca_jepa_review_samples,
)


class VulcaJepaAuditTest(unittest.TestCase):
    def test_selects_high_confidence_structure_disagreements(self) -> None:
        rows = [
            {
                "sample_id": "track2_0001",
                "current": "content",
                "emotion": "calm",
                "confidence": 0.40,
                "knn_emotion": "calm",
                "knn_confidence": 1.00,
                "model_vote_count": 3,
                "overall_caption": "A serene mountain landscape with a small boat on still water.",
            },
            {
                "sample_id": "track2_0002",
                "current": "sad",
                "emotion": "sad",
                "confidence": 0.85,
                "knn_emotion": "sad",
                "knn_confidence": 1.00,
                "model_vote_count": 3,
                "overall_caption": "A grieving figure in a dark interior.",
            },
        ]

        selected = select_vulca_jepa_review_samples(rows, limit=1)

        self.assertEqual([row["sample_id"] for row in selected], ["track2_0001"])
        self.assertGreater(selected[0]["review_priority"], 0.0)

    def test_report_counts_emotion_disagreements_and_calm_content_cases(self) -> None:
        rows = [
            {"sample_id": "a", "current": "content", "emotion": "calm", "knn_emotion": "calm", "knn_confidence": 1.0},
            {"sample_id": "b", "current": "sad", "emotion": "sad", "knn_emotion": "sad", "knn_confidence": 1.0},
        ]

        report = build_disagreement_report(rows)

        self.assertEqual(report["row_count"], 2)
        self.assertEqual(report["model_current_disagreements"], 1)
        self.assertEqual(report["content_to_calm_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_vulca_jepa_audit.py' -v
```

Expected result:

```text
ModuleNotFoundError: No module named 'affectiveart.vulca_jepa_audit'
```

- [ ] **Step 3: Implement the audit functions**

Create `affectiveart/vulca_jepa_audit.py`:

```python
from __future__ import annotations

from collections import Counter
from typing import Any


CALM_WORDS = ("serene", "quiet", "still", "peaceful", "calm", "misty", "tranquil")


def select_vulca_jepa_review_samples(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        priority = _review_priority(row)
        if priority <= 0:
            continue
        enriched = dict(row)
        enriched["review_priority"] = round(priority, 6)
        scored.append(enriched)
    scored.sort(key=lambda row: (row["review_priority"], row.get("sample_id", "")), reverse=True)
    return scored[:limit]


def build_disagreement_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    current_vs_model = 0
    calm_content = 0
    knn_disagreements = 0
    target_counter: Counter[str] = Counter()
    for row in rows:
        current = str(row.get("current", ""))
        emotion = str(row.get("emotion", ""))
        knn_emotion = str(row.get("knn_emotion", emotion))
        if current and emotion and current != emotion:
            current_vs_model += 1
        if current == "content" and emotion == "calm":
            calm_content += 1
        if emotion and knn_emotion and emotion != knn_emotion:
            knn_disagreements += 1
        if emotion:
            target_counter[emotion] += 1
    return {
        "row_count": len(rows),
        "model_current_disagreements": current_vs_model,
        "content_to_calm_candidates": calm_content,
        "model_knn_disagreements": knn_disagreements,
        "model_distribution": dict(sorted(target_counter.items())),
    }


def _review_priority(row: dict[str, Any]) -> float:
    current = str(row.get("current", ""))
    emotion = str(row.get("emotion", ""))
    knn_emotion = str(row.get("knn_emotion", emotion))
    if not current or not emotion or current == emotion:
        return 0.0
    priority = 0.0
    if emotion == knn_emotion:
        priority += float(row.get("knn_confidence", 0.0))
    priority += 0.15 * int(row.get("model_vote_count", 0))
    caption = str(row.get("overall_caption", "")).lower()
    if current == "content" and emotion == "calm" and any(word in caption for word in CALM_WORDS):
        priority += 0.5
    return priority
```

- [ ] **Step 4: Run the audit tests**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_vulca_jepa_audit.py' -v
```

Expected result:

```text
OK
```

- [ ] **Step 5: Commit**

Run:

```bash
git add affectiveart/vulca_jepa_audit.py tests/test_vulca_jepa_audit.py
git commit -m "feat: add Vulca JEPA audit scoring"
```

---

### Task 4: Local Experiment CLI

**Files:**
- Create: `scripts/vulca_jepa_experiment.py`
- Test: `tests/test_vulca_jepa_audit.py`

- [ ] **Step 1: Add a failing test for report generation input shape**

Append to `tests/test_vulca_jepa_audit.py`:

```python
import json
import tempfile
from pathlib import Path

from scripts.vulca_jepa_experiment import load_prediction_entries


class VulcaJepaExperimentCliTest(unittest.TestCase):
    def test_load_prediction_entries_accepts_predictions_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            path.write_text(
                json.dumps({"entries": [{"sample_id": "track2_0001", "emotion": "calm"}]}),
                encoding="utf-8",
            )

            rows = load_prediction_entries(path)

        self.assertEqual(rows, [{"sample_id": "track2_0001", "emotion": "calm"}])
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_vulca_jepa_audit.py' -v
```

Expected result:

```text
ModuleNotFoundError: No module named 'scripts.vulca_jepa_experiment'
```

- [ ] **Step 3: Implement the experiment CLI**

Create `scripts/vulca_jepa_experiment.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affectiveart.jepa_backbones import BACKBONES
from affectiveart.vulca_jepa_audit import build_disagreement_report, select_vulca_jepa_review_samples


DEFAULT_PREDICTIONS = Path("experiments/track2_ensemble_strict/predictions.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local Vulca JEPA experiment reports.")
    parser.add_argument("--predictions-json", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-json", type=Path, default=Path("experiments/vulca_jepa_audit/report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/vulca_jepa_experiment_report.md"))
    parser.add_argument("--review-limit", type=int, default=40)
    args = parser.parse_args()

    rows = load_prediction_entries(args.predictions_json)
    report = build_disagreement_report(rows)
    selected = select_vulca_jepa_review_samples(rows, limit=args.review_limit)
    report["selected_review_samples"] = selected
    report["backbones"] = [backbone.__dict__ for backbone in BACKBONES]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown_report(report), encoding="utf-8")
    print(args.out_json)
    print(args.out_md)


def load_prediction_entries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("prediction file must be a list or an object with entries")
    return [dict(row) for row in rows if isinstance(row, dict)]


def render_markdown_report(report: dict) -> str:
    lines = [
        "# Vulca JEPA Experiment Report",
        "",
        "## Summary",
        "",
        f"- Row count: {report['row_count']}",
        f"- Model/current disagreements: {report['model_current_disagreements']}",
        f"- Content-to-calm candidates: {report['content_to_calm_candidates']}",
        f"- Model/KNN disagreements: {report['model_knn_disagreements']}",
        "",
        "## Review Samples",
        "",
    ]
    for row in report.get("selected_review_samples", []):
        lines.append(
            f"- `{row.get('sample_id')}` priority={row.get('review_priority')} "
            f"{row.get('current', '')} -> {row.get('emotion', '')}"
        )
    lines.extend(["", "## Backbone Registry", ""])
    for backbone in report.get("backbones", []):
        lines.append(
            f"- `{backbone['name']}`: {backbone['model_id']} "
            f"({backbone['modality']}, {backbone['family']}) - {backbone['local_notes']}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the CLI test**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_vulca_jepa_audit.py' -v
```

Expected result:

```text
OK
```

- [ ] **Step 5: Run the local report generator**

Run:

```bash
python3 scripts/vulca_jepa_experiment.py
```

Expected output:

```text
experiments/vulca_jepa_audit/report.json
docs/vulca_jepa_experiment_report.md
```

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/vulca_jepa_experiment.py tests/test_vulca_jepa_audit.py experiments/vulca_jepa_audit/report.json docs/vulca_jepa_experiment_report.md
git commit -m "feat: add Vulca JEPA experiment report"
```

---

### Task 5: Run Controlled JEPA Smoke Experiments

**Files:**
- Read: `scripts/track2_clip_emoart130k.py`
- Create or update: `experiments/vulca_jepa_smoke/manifest.json`
- Create or update: `docs/vulca_jepa_experiment_report.md`

- [ ] **Step 1: Run the known working I-JEPA smoke**

Run:

```bash
python3 scripts/track2_clip_emoart130k.py \
  --backend hf \
  --model facebook/ijepa_vith16_1k \
  --style-filter 'Abstract Art' \
  --limit-train 12 \
  --limit-test 3 \
  --device mps \
  --batch-size 4 \
  --skip-candidate \
  --force-embeddings \
  --out-dir experiments/vulca_jepa_smoke/ijepa_vith16_1k
```

Expected output includes:

```text
wrote experiments/vulca_jepa_smoke/ijepa_vith16_1k/predictions.json
wrote experiments/vulca_jepa_smoke/ijepa_vith16_1k/metrics.json
```

- [ ] **Step 2: Run the DINOv2 comparison smoke**

Run:

```bash
python3 scripts/track2_clip_emoart130k.py \
  --backend hf \
  --model facebook/dinov2-base \
  --style-filter 'Abstract Art' \
  --limit-train 12 \
  --limit-test 3 \
  --device mps \
  --batch-size 4 \
  --skip-candidate \
  --force-embeddings \
  --out-dir experiments/vulca_jepa_smoke/dinov2_base
```

Expected output includes:

```text
wrote experiments/vulca_jepa_smoke/dinov2_base/predictions.json
wrote experiments/vulca_jepa_smoke/dinov2_base/metrics.json
```

- [ ] **Step 3: Run the SigLIP2 comparison smoke**

Run:

```bash
python3 scripts/track2_clip_emoart130k.py \
  --backend hf \
  --model google/siglip2-base-patch16-224 \
  --style-filter 'Abstract Art' \
  --limit-train 12 \
  --limit-test 3 \
  --device mps \
  --batch-size 4 \
  --skip-candidate \
  --force-embeddings \
  --out-dir experiments/vulca_jepa_smoke/siglip2_base_patch16_224
```

Expected output includes:

```text
wrote experiments/vulca_jepa_smoke/siglip2_base_patch16_224/predictions.json
wrote experiments/vulca_jepa_smoke/siglip2_base_patch16_224/metrics.json
```

- [ ] **Step 4: Create the smoke manifest**

Create `experiments/vulca_jepa_smoke/manifest.json`:

```json
{
  "purpose": "Compare JEPA-style image representations against DINOv2 and SigLIP2 on a tiny Track2 smoke slice.",
  "runs": [
    {
      "name": "ijepa_vith16_1k",
      "predictions": "experiments/vulca_jepa_smoke/ijepa_vith16_1k/predictions.json",
      "metrics": "experiments/vulca_jepa_smoke/ijepa_vith16_1k/metrics.json"
    },
    {
      "name": "dinov2_base",
      "predictions": "experiments/vulca_jepa_smoke/dinov2_base/predictions.json",
      "metrics": "experiments/vulca_jepa_smoke/dinov2_base/metrics.json"
    },
    {
      "name": "siglip2_base_patch16_224",
      "predictions": "experiments/vulca_jepa_smoke/siglip2_base_patch16_224/predictions.json",
      "metrics": "experiments/vulca_jepa_smoke/siglip2_base_patch16_224/metrics.json"
    }
  ],
  "interpretation": "This smoke test checks compatibility and qualitative prediction behavior only. It is not a leaderboard estimate."
}
```

- [ ] **Step 5: Commit**

Run:

```bash
git add experiments/vulca_jepa_smoke docs/vulca_jepa_experiment_report.md
git commit -m "exp: add Vulca JEPA smoke comparison"
```

---

### Task 6: Decide Whether to Run Any Full JEPA Job

**Files:**
- Read: `experiments/vulca_jepa_smoke/manifest.json`
- Read: `experiments/track2_emoart130k_siglip2/metrics.json`
- Read: `experiments/track2_emoart130k_dinov2/metrics.json`
- Create or update: `docs/vulca_jepa_experiment_report.md`

- [ ] **Step 1: Check the smoke runtime**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for metrics_path in [
    Path("experiments/vulca_jepa_smoke/ijepa_vith16_1k/metrics.json"),
    Path("experiments/vulca_jepa_smoke/dinov2_base/metrics.json"),
    Path("experiments/vulca_jepa_smoke/siglip2_base_patch16_224/metrics.json"),
]:
    metrics = json.loads(metrics_path.read_text())
    print(metrics_path, metrics.get("wall_seconds"), metrics.get("test_distribution"))
PY
```

Expected output:

```text
experiments/vulca_jepa_smoke/ijepa_vith16_1k/metrics.json <seconds> <distribution>
experiments/vulca_jepa_smoke/dinov2_base/metrics.json <seconds> <distribution>
experiments/vulca_jepa_smoke/siglip2_base_patch16_224/metrics.json <seconds> <distribution>
```

- [ ] **Step 2: Apply the full-run gate**

Use this rule:

```text
Run a full I-JEPA Track2 job only if the I-JEPA smoke run finishes in less than 3x the DINOv2 smoke runtime and produces no encoder failures.
```

If the gate passes, run:

```bash
python3 scripts/track2_clip_emoart130k.py \
  --backend hf \
  --model facebook/ijepa_vith16_1k \
  --device mps \
  --batch-size 4 \
  --out-dir experiments/vulca_jepa_full/ijepa_vith16
```

Expected output includes:

```text
wrote experiments/vulca_jepa_full/ijepa_vith16/predictions.json
wrote experiments/vulca_jepa_full/ijepa_vith16/metrics.json
```

If the gate fails, write this exact conclusion into `docs/vulca_jepa_experiment_report.md`:

```markdown
## Full I-JEPA Gate

Full I-JEPA Track2 was not run because the local smoke runtime exceeded the 3x DINOv2 runtime gate or failed to complete cleanly. I-JEPA remains a research baseline for structure auditing, not a challenge submission backbone.
```

- [ ] **Step 3: Commit the decision**

Run:

```bash
git add docs/vulca_jepa_experiment_report.md experiments/vulca_jepa_full/ijepa_vith16 experiments/vulca_jepa_smoke
git commit -m "docs: record Vulca JEPA full-run decision"
```

If no full run was created, use:

```bash
git add docs/vulca_jepa_experiment_report.md experiments/vulca_jepa_smoke
git commit -m "docs: record Vulca JEPA full-run decision"
```

---

### Task 7: Add Track1 Vulca Generated-Image Audit Integration

**Files:**
- Modify: `affectiveart/vulca_jepa_audit.py`
- Test: `tests/test_vulca_jepa_audit.py`
- Create or update: `docs/vulca_jepa_experiment_report.md`

- [ ] **Step 1: Add a failing test for Track1 audit row scoring**

Append to `tests/test_vulca_jepa_audit.py`:

```python
from affectiveart.vulca_jepa_audit import score_track1_generation_risk


class Track1VulcaJepaAuditTest(unittest.TestCase):
    def test_flags_style_high_content_low_generation(self) -> None:
        row = {
            "sample_id": "track1_0002",
            "caption_fidelity_score": 0.42,
            "style_score": 0.91,
            "structure_score": 0.38,
            "caption": "A bamboo, orchid, and calligraphy composition.",
        }

        scored = score_track1_generation_risk(row)

        self.assertEqual(scored["risk_level"], "high")
        self.assertIn("style-content mismatch", scored["reasons"])
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_vulca_jepa_audit.py' -v
```

Expected result:

```text
ImportError: cannot import name 'score_track1_generation_risk'
```

- [ ] **Step 3: Implement the Track1 risk scorer**

Append to `affectiveart/vulca_jepa_audit.py`:

```python
def score_track1_generation_risk(row: dict[str, Any]) -> dict[str, Any]:
    caption_fidelity = float(row.get("caption_fidelity_score", 0.0))
    style_score = float(row.get("style_score", 0.0))
    structure_score = float(row.get("structure_score", 0.0))
    reasons: list[str] = []
    if style_score >= 0.85 and caption_fidelity < 0.60:
        reasons.append("style-content mismatch")
    if structure_score < 0.50:
        reasons.append("weak structure preservation")
    if caption_fidelity < 0.50:
        reasons.append("low caption fidelity")
    risk_level = "low"
    if len(reasons) >= 2:
        risk_level = "high"
    elif reasons:
        risk_level = "medium"
    scored = dict(row)
    scored["risk_level"] = risk_level
    scored["reasons"] = reasons
    return scored
```

- [ ] **Step 4: Run the tests**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_vulca_jepa_audit.py' -v
```

Expected result:

```text
OK
```

- [ ] **Step 5: Commit**

Run:

```bash
git add affectiveart/vulca_jepa_audit.py tests/test_vulca_jepa_audit.py docs/vulca_jepa_experiment_report.md
git commit -m "feat: add Track1 Vulca JEPA risk scoring"
```

---

### Task 8: Final Verification

**Files:**
- Read: all changed files
- Create or update: `docs/vulca_jepa_experiment_report.md`

- [ ] **Step 1: Run the focused tests**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_jepa_backbones.py' -v
python3 -m unittest discover -s tests -p 'test_vulca_jepa_audit.py' -v
python3 -m unittest discover -s tests -p 'test_track2_embedding_script.py' -v
```

Expected result:

```text
OK
OK
OK
```

- [ ] **Step 2: Run the full suite**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected result:

```text
OK
```

- [ ] **Step 3: Verify challenge submissions were not modified by this project**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission.json
tmpdir=$(mktemp -d)
unzip -q submissions/track1_submission.zip -d "$tmpdir"
python3 -m affectiveart.challenge validate-track1 "$tmpdir/submission.json" --check-files
rm -rf "$tmpdir"
unzip -l submissions/track2_submission.zip | grep submission.json
unzip -l submissions/track1_submission.zip | head
```

Expected result:

```text
OK
OK
submission.json
```

- [ ] **Step 4: Commit final report**

Run:

```bash
git add docs/vulca_jepa_experiment_report.md
git commit -m "docs: summarize Vulca JEPA experiment results"
```

---

## Interpretation Rules

Use these rules when reading results:

- If SigLIP2 wins emotion macro-F1 but DINO/I-JEPA finds better structural neighbors, use JEPA/DINO as audit signals, not primary emotion classifiers.
- If I-JEPA full runtime is too slow locally, keep it as a smoke-only research baseline until GPU/HF Jobs are available.
- If V-JEPA is evaluated on repeated still frames, label that run as a diagnostic only and keep it out of challenge claims.
- If a generated image scores high on style but low on caption fidelity or structure, treat that as Vulca over-stylization evidence.
- If three independent image backbones agree on `content -> calm` and the caption contains calm/serene cues, treat it as a high-priority manual review candidate.

## Success Criteria

The project is successful when:

- There is a tested model registry that documents which JEPA models are meaningful for static image work.
- Video JEPA is guarded behind an explicit diagnostic flag.
- There is a reproducible Vulca JEPA report CLI.
- The report explains why JEPA is or is not used in Track2 submission.
- Track1 and Track2 submission ZIPs remain valid.
- The final report gives a practical product direction: JEPA/DINO for visual structure auditing, SigLIP/CLIP for semantic alignment, Vulca-EMNLP for caption and emotion consistency.

## Self-Review

Spec coverage:

- Local JEPA experiment is covered by Tasks 1, 5, and 6.
- Vulca project upgrade is covered by Tasks 3, 4, and 7.
- Challenge submission safety is covered by Task 8.
- Video JEPA limitations are covered by Task 2 and the interpretation rules.

Placeholder scan:

- The plan avoids open-ended implementation notes.
- Every code-writing task includes exact file paths, code blocks, commands, and expected results.

Type consistency:

- `BackboneSpec` is defined in Task 1 and imported in later tasks.
- `select_vulca_jepa_review_samples`, `build_disagreement_report`, and `score_track1_generation_risk` are defined before use.
- CLI paths are explicit and write JEPA experiments under `experiments/vulca_jepa_*`, so they do not overwrite existing challenge submission ZIPs.
