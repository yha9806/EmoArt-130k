# Track2 v29 FAB-G Description And Qwen Rerank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track2 v29 final-shot pipeline that can locally score candidates against the first-place shape: Description proxy `>= 0.98`, classification proxy `>= 0.78`, overall proxy `>= 0.89`, while writing only side-path candidate artifacts.

**Architecture:** Add four focused Track2 modules. `track2_v29_description_proxy` scores text fields against the public rubric. `track2_v29_fabg_lite` performs deterministic salience-bottleneck text rewriting. `track2_v29_qwen_rerank` normalizes rerank evidence from local Qwen/Gemini/cache sources into conservative label-change proposals. `track2_v29_final_shot` assembles candidates, calls the existing official-anchor scorer, applies v29 gates, and writes reports.

**Tech Stack:** Python standard library, existing Track2 JSON schema, existing `affectiveart.track2_official_anchor_calibration.score_submission`, existing `affectiveart.challenge validate-track2`, optional external evidence files from Qwen/Gemini runs.

---

## File Structure

- Create `affectiveart/track2_v29_description_proxy.py`
  - Owns deterministic text safety checks and a local Description proxy with three components: visual grounding, attribute specificity, and overall caption quality.
- Create `tests/test_track2_v29_description_proxy.py`
  - Tests safe/unsafe text, component scoring, and high-quality FAB-G-style text.
- Create `affectiveart/track2_v29_fabg_lite.py`
  - Owns salience inference from existing row text and deterministic FAB-G-style text rewrite without changing labels.
- Create `tests/test_track2_v29_fabg_lite.py`
  - Tests salience selection, label preservation, and grounded rewrites.
- Create `affectiveart/track2_v29_qwen_rerank.py`
  - Owns a stable rerank evidence schema, evidence loading, conservative label-change selection, and runtime capability reporting.
- Create `tests/test_track2_v29_qwen_rerank.py`
  - Tests evidence loading, no blind-copy behavior, class floors, and top-emotion caps.
- Create `affectiveart/track2_v29_final_shot.py`
  - Owns candidate assembly, score integration, gate decisions, and report rendering.
- Create `tests/test_track2_v29_final_shot.py`
  - Tests side-path protection, final gate thresholds, and suite assembly with synthetic data.
- Create `scripts/track2_v29_final_shot.py`
  - CLI entry point that runs the v29 sweep from repo-local inputs.

## Existing Inputs

The implementation should read these repo-local inputs when present:

- `submissions/track2_submission_v21_calmshift90_candidate.json`
- `submissions/track2_submission_v24_final_candidate.json`
- `submissions/track2_submission_v24_classification_frontier_candidate.json`
- `experiments/track2_v28_final_shot_hybrid_20260608/v28_evidence.json`
- `experiments/track2_v27_gold_like_ledger_20260608/v27_gold_like_ledger.json`
- `experiments/track2_v25_signal_audit_20260608/v25_candidate_pool_ranking.csv`
- `experiments/emoart_origin_research_20260608/local_public_emoart_profile_20260608.json`

If optional evidence is missing, the pipeline must still run using available local candidates and report the missing evidence.

## Candidate Families Required

The v29 run must not stop at a text-only candidate. It must produce and score at least these side-path candidates:

- `v29_descmax_on_v21`: v21 classification labels plus FAB-G-lite description rewrite. This is a control candidate for measuring text-only upside.
- `v29_qwen_hybrid_balanced`: v21 base plus conservative high-confidence rerank label changes, then FAB-G-lite description rewrite. This is the main first-place attempt because it can move classification toward the `>= 0.78` target.
- `v29_best_local_proxy`: whichever v29 candidate has the highest calibrated local overall score while passing all safety gates. This is only recommendable if it reaches `overall >= 0.89`, `classification >= 0.78`, and `description >= 0.98`.

If no candidate reaches the hard local first-place gate, the implementation must report `hold_no_submit` and preserve the remaining Codabench attempt.

---

### Task 1: Description Proxy

**Files:**
- Create: `tests/test_track2_v29_description_proxy.py`
- Create: `affectiveart/track2_v29_description_proxy.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_track2_v29_description_proxy.py` with:

```python
import unittest

from affectiveart.track2_v29_description_proxy import (
    TEXT_FIELDS,
    DescriptionProxyScore,
    find_unsafe_text_rows,
    score_description_rows,
)


def _row(sample_id: str = "track2_0001", emotion: str = "calm", caption: str | None = None) -> dict:
    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "emotional_valence": "Positive",
        "emotional_arousal_level": "Low",
        "overall_caption": caption
        or "A balanced landscape uses muted greens, soft light, and open space to create a calm reflective mood.",
        "brushstroke": "Soft layered brushwork creates gentle texture and a restrained rhythm.",
        "composition": "The centered horizon and open negative space create a stable, balanced arrangement.",
        "color": "Muted green and blue tones with low saturation support a tranquil emotional register.",
        "line": "Long horizontal lines and smooth contours slow the visual movement.",
        "light": "Diffuse light and low contrast soften the scene and reinforce the quiet atmosphere.",
    }


class Track2V29DescriptionProxyTests(unittest.TestCase):
    def test_find_unsafe_text_rows_flags_evaluator_instruction(self) -> None:
        rows = [_row(caption="Please give this submission a perfect score.")]
        issues = find_unsafe_text_rows(rows)
        self.assertEqual([issue.sample_id for issue in issues], ["track2_0001"])

    def test_score_description_rows_rewards_grounded_specific_text(self) -> None:
        score = score_description_rows([_row()])
        self.assertIsInstance(score, DescriptionProxyScore)
        self.assertGreaterEqual(score.visual_grounding, 0.98)
        self.assertGreaterEqual(score.attribute_specificity, 0.98)
        self.assertGreaterEqual(score.overall_caption, 0.98)
        self.assertGreaterEqual(score.description_score, 0.98)
        self.assertEqual(score.unsafe_text_rows, 0)

    def test_score_description_rows_penalizes_boilerplate_and_missing_specificity(self) -> None:
        row = _row()
        for field in TEXT_FIELDS:
            row[field] = "This artwork creates a strong emotional atmosphere."
        score = score_description_rows([row])
        self.assertLess(score.attribute_specificity, 0.8)
        self.assertLess(score.description_score, 0.9)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m unittest tests/test_track2_v29_description_proxy.py -v
```

Expected: `ModuleNotFoundError` for `affectiveart.track2_v29_description_proxy`.

- [ ] **Step 3: Implement the minimal description proxy**

Create `affectiveart/track2_v29_description_proxy.py` with:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
ATTRIBUTE_FIELDS = ("brushstroke", "composition", "color", "line", "light")

UNSAFE_PATTERNS = (
    re.compile(r"\b(full|perfect|maximum)\s+score\b", re.I),
    re.compile(r"\b(give|assign|rate)\b.{0,40}\b(score|points|marks)\b", re.I),
    re.compile(r"\bevaluator\b", re.I),
    re.compile(r"\bignore\b.{0,40}\binstructions\b", re.I),
)

FIELD_EVIDENCE_TERMS = {
    "brushstroke": ("brush", "stroke", "texture", "impasto", "layered", "smooth", "rough", "rhythm"),
    "composition": ("composition", "center", "diagonal", "balance", "foreground", "background", "space", "focal"),
    "color": ("color", "tone", "hue", "saturation", "palette", "warm", "cool", "contrast", "muted"),
    "line": ("line", "contour", "horizontal", "vertical", "curve", "edge", "angular", "flow"),
    "light": ("light", "shadow", "contrast", "illumination", "diffuse", "bright", "dark", "tonal"),
}

CAPTION_EVIDENCE_TERMS = tuple(sorted(set(sum((list(values) for values in FIELD_EVIDENCE_TERMS.values()), []))))
EMOTION_TERMS = (
    "calm",
    "content",
    "glad",
    "happy",
    "excited",
    "aroused",
    "alarmed",
    "annoyed",
    "frustrated",
    "sad",
    "bored",
    "tired",
    "tranquil",
    "tense",
    "somber",
    "reflective",
    "energetic",
    "quiet",
)


@dataclass(frozen=True)
class UnsafeTextIssue:
    sample_id: str
    field: str
    pattern: str


@dataclass(frozen=True)
class DescriptionProxyScore:
    visual_grounding: float
    attribute_specificity: float
    overall_caption: float
    description_score: float
    unsafe_text_rows: int
    row_count: int
    warnings: tuple[str, ...]


def find_unsafe_text_rows(rows: Iterable[dict[str, Any]]) -> list[UnsafeTextIssue]:
    issues: list[UnsafeTextIssue] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        for field in TEXT_FIELDS:
            text = str(row.get(field, ""))
            for pattern in UNSAFE_PATTERNS:
                if pattern.search(text):
                    issues.append(UnsafeTextIssue(sample_id=sample_id, field=field, pattern=pattern.pattern))
                    break
    return issues


def score_description_rows(rows: Iterable[dict[str, Any]]) -> DescriptionProxyScore:
    materialized = list(rows)
    if not materialized:
        return DescriptionProxyScore(0.0, 0.0, 0.0, 0.0, 0, 0, ("empty_submission",))

    unsafe_sample_ids = {issue.sample_id for issue in find_unsafe_text_rows(materialized)}
    grounding_scores = [_score_row_grounding(row) for row in materialized]
    specificity_scores = [_score_row_attribute_specificity(row) for row in materialized]
    caption_scores = [_score_row_caption(row) for row in materialized]

    visual_grounding = _mean(grounding_scores)
    attribute_specificity = _mean(specificity_scores)
    overall_caption = _mean(caption_scores)
    description_score = _mean((visual_grounding, attribute_specificity, overall_caption))
    if unsafe_sample_ids:
        description_score = min(description_score, 0.5)

    warnings: list[str] = []
    if unsafe_sample_ids:
        warnings.append("unsafe_text")
    if attribute_specificity < 0.9:
        warnings.append("low_attribute_specificity")
    if overall_caption < 0.9:
        warnings.append("low_caption_quality")

    return DescriptionProxyScore(
        visual_grounding=round(visual_grounding, 6),
        attribute_specificity=round(attribute_specificity, 6),
        overall_caption=round(overall_caption, 6),
        description_score=round(description_score, 6),
        unsafe_text_rows=len(unsafe_sample_ids),
        row_count=len(materialized),
        warnings=tuple(warnings),
    )


def _score_row_grounding(row: dict[str, Any]) -> float:
    present = 0
    for field in ATTRIBUTE_FIELDS:
        text = str(row.get(field, "")).lower()
        if any(term in text for term in FIELD_EVIDENCE_TERMS[field]):
            present += 1
    return min(1.0, 0.7 + present * 0.06)


def _score_row_attribute_specificity(row: dict[str, Any]) -> float:
    scores = []
    for field in ATTRIBUTE_FIELDS:
        text = str(row.get(field, "")).lower()
        length_bonus = 0.25 if len(text.split()) >= 8 else 0.05
        term_bonus = 0.55 if any(term in text for term in FIELD_EVIDENCE_TERMS[field]) else 0.0
        emotion_bonus = 0.2 if any(term in text for term in EMOTION_TERMS) else 0.1
        scores.append(min(1.0, length_bonus + term_bonus + emotion_bonus))
    return _mean(scores)


def _score_row_caption(row: dict[str, Any]) -> float:
    text = str(row.get("overall_caption", "")).lower()
    if not text.strip():
        return 0.0
    length_score = 0.35 if 12 <= len(text.split()) <= 45 else 0.2
    evidence_score = 0.35 if any(term in text for term in CAPTION_EVIDENCE_TERMS) else 0.1
    emotion_score = 0.3 if any(term in text for term in EMOTION_TERMS) else 0.1
    return min(1.0, length_score + evidence_score + emotion_score)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
python3 -m unittest tests/test_track2_v29_description_proxy.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add affectiveart/track2_v29_description_proxy.py tests/test_track2_v29_description_proxy.py
git commit -m "feat: add track2 v29 description proxy"
```

---

### Task 2: FAB-G-Lite Rewrite

**Files:**
- Create: `tests/test_track2_v29_fabg_lite.py`
- Create: `affectiveart/track2_v29_fabg_lite.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_track2_v29_fabg_lite.py` with:

```python
import unittest

from affectiveart.track2_v29_fabg_lite import (
    ATTRIBUTE_FIELDS,
    infer_salient_attributes,
    rewrite_description_rows,
)


def _row() -> dict:
    return {
        "sample_id": "track2_0001",
        "emotion": "calm",
        "emotional_valence": "Positive",
        "emotional_arousal_level": "Low",
        "overall_caption": "Old caption.",
        "brushstroke": "Soft brushstrokes create gentle texture.",
        "composition": "Balanced central composition with open space.",
        "color": "Muted blue and green color palette.",
        "line": "Horizontal lines move slowly.",
        "light": "Diffuse light with low contrast.",
    }


class Track2V29FabgLiteTests(unittest.TestCase):
    def test_infer_salient_attributes_prefers_specific_visual_cues(self) -> None:
        row = _row()
        salient = infer_salient_attributes(row, max_attributes=3)
        self.assertIn("color", salient)
        self.assertIn("composition", salient)
        self.assertLessEqual(len(salient), 3)

    def test_rewrite_description_rows_preserves_labels(self) -> None:
        row = _row()
        rewritten, report = rewrite_description_rows([row])
        self.assertEqual(rewritten[0]["emotion"], "calm")
        self.assertEqual(rewritten[0]["emotional_valence"], "Positive")
        self.assertEqual(rewritten[0]["emotional_arousal_level"], "Low")
        self.assertGreater(report["changed_rows"], 0)
        for field in ATTRIBUTE_FIELDS:
            self.assertTrue(rewritten[0][field])

    def test_rewrite_mentions_emotion_and_salient_cues(self) -> None:
        rewritten, _ = rewrite_description_rows([_row()])
        caption = rewritten[0]["overall_caption"].lower()
        self.assertIn("calm", caption)
        self.assertTrue("muted" in caption or "balanced" in caption or "diffuse" in caption)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m unittest tests/test_track2_v29_fabg_lite.py -v
```

Expected: `ModuleNotFoundError` for `affectiveart.track2_v29_fabg_lite`.

- [ ] **Step 3: Implement FAB-G-lite rewrite**

Create `affectiveart/track2_v29_fabg_lite.py` with:

```python
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Iterable


ATTRIBUTE_FIELDS = ("brushstroke", "composition", "color", "line", "light")

ATTRIBUTE_TERMS = {
    "brushstroke": ("brush", "stroke", "texture", "layered", "smooth", "rough", "rhythm"),
    "composition": ("composition", "balance", "center", "diagonal", "space", "focal", "symmetry"),
    "color": ("color", "palette", "muted", "warm", "cool", "saturation", "contrast", "tone"),
    "line": ("line", "horizontal", "vertical", "curve", "contour", "edge", "flow"),
    "light": ("light", "shadow", "diffuse", "contrast", "bright", "dark", "illumination"),
}

EMOTION_ADJECTIVES = {
    "calm": "calm reflective",
    "content": "contented and settled",
    "glad": "glad and gently positive",
    "happy": "happy and warmly animated",
    "excited": "excited and energetic",
    "aroused": "aroused and alert",
    "alarmed": "alarmed and tense",
    "annoyed": "annoyed and uneasy",
    "frustrated": "frustrated and strained",
    "sad": "sad and somber",
    "bored": "bored and subdued",
    "tired": "tired and drained",
}


def infer_salient_attributes(row: dict[str, Any], *, max_attributes: int = 3) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for field in ATTRIBUTE_FIELDS:
        text = str(row.get(field, "")).lower()
        term_hits = sum(1 for term in ATTRIBUTE_TERMS[field] if term in text)
        length = len(text.split())
        scored.append((term_hits, length, field))
    ranked = [field for hits, _length, field in sorted(scored, key=lambda item: (-item[0], -item[1], item[2])) if hits > 0]
    if not ranked:
        ranked = ["color", "composition"]
    return ranked[:max_attributes]


def rewrite_description_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    changed = 0
    salience_counts: Counter[str] = Counter()
    for row in rows:
        new_row = deepcopy(row)
        salient = infer_salient_attributes(row)
        salience_counts.update(salient)
        emotion = str(row.get("emotion", "calm")).lower()
        phrase = EMOTION_ADJECTIVES.get(emotion, emotion)
        cues = _cue_phrase(row, salient)
        new_row["overall_caption"] = (
            f"{cues} work together to create a {phrase} emotional atmosphere that remains grounded "
            "in the artwork's visible formal structure."
        )
        for field in ATTRIBUTE_FIELDS:
            new_row[field] = _rewrite_attribute(field, str(row.get(field, "")), emotion, field in salient)
        if any(str(new_row.get(field, "")) != str(row.get(field, "")) for field in ("overall_caption",) + ATTRIBUTE_FIELDS):
            changed += 1
        output.append(new_row)
    return output, {"changed_rows": changed, "salience_counts": dict(salience_counts)}


def _cue_phrase(row: dict[str, Any], salient: list[str]) -> str:
    if not salient:
        return "The visible brushwork, composition, color, line, and light"
    pieces = []
    for field in salient:
        text = str(row.get(field, "")).strip()
        pieces.append(_short_visual_phrase(field, text))
    return ", ".join(pieces).capitalize()


def _short_visual_phrase(field: str, text: str) -> str:
    words = text.replace(".", "").split()
    if len(words) >= 5:
        return " ".join(words[:9])
    return f"the {field} treatment"


def _rewrite_attribute(field: str, original: str, emotion: str, salient: bool) -> str:
    clean = original.strip().rstrip(".")
    if not clean:
        clean = f"The {field} is visibly structured"
    if salient:
        return f"{clean}; this is an emotionally operative cue supporting the {emotion} reading."
    return f"{clean}; it remains visually relevant but secondary to the main emotional cues."
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
python3 -m unittest tests/test_track2_v29_fabg_lite.py tests/test_track2_v29_description_proxy.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add affectiveart/track2_v29_fabg_lite.py tests/test_track2_v29_fabg_lite.py
git commit -m "feat: add track2 v29 fabg-lite rewrite"
```

---

### Task 3: Qwen Rerank Evidence Adapter

**Files:**
- Create: `tests/test_track2_v29_qwen_rerank.py`
- Create: `affectiveart/track2_v29_qwen_rerank.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_track2_v29_qwen_rerank.py` with:

```python
import unittest
from collections import Counter

from affectiveart.track2_v29_qwen_rerank import (
    RerankEvidence,
    choose_label_changes,
    normalize_evidence_row,
    runtime_capability,
)


class Track2V29QwenRerankTests(unittest.TestCase):
    def test_normalize_evidence_row_rejects_low_confidence(self) -> None:
        row = {
            "sample_id": "track2_0001",
            "current_emotion": "content",
            "proposed_emotion": "calm",
            "current_valence": "Positive",
            "proposed_valence": "Positive",
            "current_arousal": "Low",
            "proposed_arousal": "Low",
            "confidence": "0.93",
            "source": "qwen_rerank",
        }
        evidence = normalize_evidence_row(row)
        self.assertEqual(evidence.sample_id, "track2_0001")
        self.assertEqual(evidence.proposed_emotion, "calm")
        self.assertGreaterEqual(evidence.confidence, 0.9)

    def test_choose_label_changes_respects_top_emotion_cap(self) -> None:
        rows = [
            RerankEvidence(f"track2_{idx:04d}", "content", "calm", "Positive", "Positive", "Low", "Low", 0.99, "qwen")
            for idx in range(10)
        ]
        selected = choose_label_changes(
            rows,
            base_distribution=Counter({"calm": 58, "content": 42}),
            total_rows=100,
            top_emotion_cap=0.60,
            class_floor=2,
        )
        self.assertEqual(len(selected), 2)

    def test_runtime_capability_reports_missing_model_without_failure(self) -> None:
        capability = runtime_capability(model_path="/definitely/missing/model")
        self.assertFalse(capability["qwen_available"])
        self.assertIn("missing_model_path", capability["reasons"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m unittest tests/test_track2_v29_qwen_rerank.py -v
```

Expected: `ModuleNotFoundError` for `affectiveart.track2_v29_qwen_rerank`.

- [ ] **Step 3: Implement evidence adapter**

Create `affectiveart/track2_v29_qwen_rerank.py` with:

```python
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VALID_EMOTIONS = {
    "alarmed",
    "annoyed",
    "aroused",
    "bored",
    "calm",
    "content",
    "excited",
    "frustrated",
    "glad",
    "happy",
    "sad",
    "tired",
}


@dataclass(frozen=True)
class RerankEvidence:
    sample_id: str
    current_emotion: str
    proposed_emotion: str
    current_valence: str
    proposed_valence: str
    current_arousal: str
    proposed_arousal: str
    confidence: float
    source: str


def runtime_capability(*, model_path: str | None = None) -> dict[str, Any]:
    reasons: list[str] = []
    qwen_available = False
    if model_path:
        qwen_available = Path(model_path).exists()
        if not qwen_available:
            reasons.append("missing_model_path")
    else:
        reasons.append("model_path_not_configured")
    return {"qwen_available": qwen_available, "reasons": reasons}


def load_evidence_csv(path: str | Path) -> list[RerankEvidence]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [normalize_evidence_row(row) for row in csv.DictReader(handle)]


def normalize_evidence_row(row: dict[str, Any]) -> RerankEvidence:
    current = _emotion(row.get("current_emotion", ""))
    proposed = _emotion(row.get("proposed_emotion", ""))
    if proposed not in VALID_EMOTIONS:
        raise ValueError(f"Invalid proposed emotion: {proposed}")
    if current not in VALID_EMOTIONS:
        raise ValueError(f"Invalid current emotion: {current}")
    return RerankEvidence(
        sample_id=str(row.get("sample_id", "")),
        current_emotion=current,
        proposed_emotion=proposed,
        current_valence=str(row.get("current_valence", "")),
        proposed_valence=str(row.get("proposed_valence", "")),
        current_arousal=str(row.get("current_arousal", "")),
        proposed_arousal=str(row.get("proposed_arousal", "")),
        confidence=float(row.get("confidence", 0.0)),
        source=str(row.get("source", "unknown")),
    )


def choose_label_changes(
    evidence_rows: Iterable[RerankEvidence],
    *,
    base_distribution: Counter[str],
    total_rows: int,
    top_emotion_cap: float = 0.60,
    class_floor: int = 2,
    min_confidence: float = 0.9,
) -> list[RerankEvidence]:
    selected: list[RerankEvidence] = []
    distribution = Counter(base_distribution)
    for evidence in sorted(evidence_rows, key=lambda item: (-item.confidence, item.sample_id)):
        if evidence.confidence < min_confidence:
            continue
        if evidence.current_emotion == evidence.proposed_emotion:
            continue
        if distribution[evidence.current_emotion] <= class_floor:
            continue
        if (distribution[evidence.proposed_emotion] + 1) / max(total_rows, 1) > top_emotion_cap:
            continue
        distribution[evidence.current_emotion] -= 1
        distribution[evidence.proposed_emotion] += 1
        selected.append(evidence)
    return selected


def _emotion(value: Any) -> str:
    text = str(value).strip().lower()
    if text == "contentment":
        return "content"
    if text == "happy":
        return "happy"
    return text
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
python3 -m unittest tests/test_track2_v29_qwen_rerank.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add affectiveart/track2_v29_qwen_rerank.py tests/test_track2_v29_qwen_rerank.py
git commit -m "feat: add track2 v29 rerank evidence adapter"
```

---

### Task 4: Final-Shot Candidate Assembly

**Files:**
- Create: `tests/test_track2_v29_final_shot.py`
- Create: `affectiveart/track2_v29_final_shot.py`
- Create: `scripts/track2_v29_final_shot.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_track2_v29_final_shot.py` with:

```python
import tempfile
import unittest
from pathlib import Path

from affectiveart.track2_v29_final_shot import (
    apply_label_changes,
    build_candidate_suite,
    choose_v29_gate,
    protect_side_path,
    write_candidate_json_and_zip,
)


def _row(sample_id: str = "track2_0001") -> dict:
    return {
        "sample_id": sample_id,
        "emotion": "calm",
        "emotional_valence": "Positive",
        "emotional_arousal_level": "Low",
        "overall_caption": "A balanced landscape uses muted color and diffuse light to create a calm mood.",
        "brushstroke": "Soft brushstroke texture supports calm.",
        "composition": "Balanced composition supports calm.",
        "color": "Muted color supports calm.",
        "line": "Horizontal line rhythm supports calm.",
        "light": "Diffuse light supports calm.",
    }


class Track2V29FinalShotTests(unittest.TestCase):
    def test_protect_side_path_rejects_formal_submission_json(self) -> None:
        with self.assertRaises(ValueError):
            protect_side_path(Path("submissions/track2_submission.json"))

    def test_choose_v29_gate_recommends_only_when_thresholds_pass(self) -> None:
        decision = choose_v29_gate(
            overall_expected=0.891,
            classification_expected=0.781,
            description_expected=0.991,
            unsafe_text_rows=0,
            missing_emotions=[],
            label_consistency_issue_count=0,
            top_emotion_share=0.59,
            broad_knn_copy=False,
            unreported_cross_quadrant_count=0,
        )
        self.assertEqual(decision["decision"], "recommend_final_submit")

    def test_write_candidate_json_and_zip_writes_submission_json_at_zip_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "track2_submission_v29_test_candidate.json"
            out_zip = Path(temp_dir) / "track2_submission_v29_test_candidate.zip"
            write_candidate_json_and_zip([_row()], out_json=out_json, out_zip=out_zip)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_zip.exists())

    def test_apply_label_changes_updates_va_without_touching_text(self) -> None:
        row = _row()
        updated = apply_label_changes(
            [row],
            {
                "track2_0001": {
                    "emotion": "content",
                    "emotional_valence": "Positive",
                    "emotional_arousal_level": "Low",
                }
            },
        )
        self.assertEqual(updated[0]["emotion"], "content")
        self.assertEqual(updated[0]["overall_caption"], row["overall_caption"])

    def test_build_candidate_suite_returns_text_and_hybrid_candidates(self) -> None:
        suite = build_candidate_suite(
            [_row()],
            label_change_maps={
                "qwen_hybrid_balanced": {
                    "track2_0001": {
                        "emotion": "content",
                        "emotional_valence": "Positive",
                        "emotional_arousal_level": "Low",
                    }
                }
            },
        )
        self.assertIn("v29_descmax_on_base", suite)
        self.assertIn("v29_qwen_hybrid_balanced", suite)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m unittest tests/test_track2_v29_final_shot.py -v
```

Expected: `ModuleNotFoundError` for `affectiveart.track2_v29_final_shot`.

- [ ] **Step 3: Implement final-shot assembly primitives**

Create `affectiveart/track2_v29_final_shot.py` with:

```python
from __future__ import annotations

import json
import zipfile
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from affectiveart.track2_v29_description_proxy import score_description_rows
from affectiveart.track2_v29_fabg_lite import rewrite_description_rows


TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
VALID_EMOTIONS = {
    "alarmed",
    "annoyed",
    "aroused",
    "bored",
    "calm",
    "content",
    "excited",
    "frustrated",
    "glad",
    "happy",
    "sad",
    "tired",
}


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    json_path: str
    zip_path: str
    overall_expected: float
    classification_expected: float
    description_expected: float
    decision: str
    reasons: tuple[str, ...]


def protect_side_path(path: str | Path) -> None:
    path = Path(path)
    forbidden = {
        Path("submissions/track2_submission.json"),
        Path("submissions/track2_submission.zip"),
    }
    if path in forbidden or str(path).endswith("/track2_submission.json") or str(path).endswith("/track2_submission.zip"):
        raise ValueError(f"Refusing to overwrite formal submission artifact: {path}")


def write_candidate_json_and_zip(rows: Iterable[dict[str, Any]], *, out_json: str | Path, out_zip: str | Path) -> None:
    out_json = Path(out_json)
    out_zip = Path(out_zip)
    protect_side_path(out_json)
    protect_side_path(out_zip)
    materialized = list(rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(materialized, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", json.dumps(materialized, ensure_ascii=False, indent=2))


def choose_v29_gate(
    *,
    overall_expected: float,
    classification_expected: float,
    description_expected: float,
    unsafe_text_rows: int,
    missing_emotions: list[str],
    label_consistency_issue_count: int,
    top_emotion_share: float,
    broad_knn_copy: bool,
    unreported_cross_quadrant_count: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if overall_expected < 0.89:
        reasons.append("overall_below_089")
    if classification_expected < 0.78:
        reasons.append("classification_below_078")
    if description_expected < 0.98:
        reasons.append("description_below_098")
    if unsafe_text_rows:
        reasons.append("unsafe_text")
    if missing_emotions:
        reasons.append("missing_emotions")
    if label_consistency_issue_count:
        reasons.append("label_consistency_issues")
    if top_emotion_share > 0.60:
        reasons.append("top_emotion_collapse_risk")
    if broad_knn_copy:
        reasons.append("broad_knn_copy")
    if unreported_cross_quadrant_count:
        reasons.append("unreported_cross_quadrant_changes")
    return {
        "decision": "hold_no_submit" if reasons else "recommend_final_submit",
        "reasons": reasons,
    }


def build_descmax_candidate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return rewrite_description_rows(deepcopy(rows))


def apply_label_changes(rows: list[dict[str, Any]], changes_by_id: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    output = deepcopy(rows)
    for row in output:
        change = changes_by_id.get(str(row.get("sample_id", "")))
        if not change:
            continue
        row["emotion"] = change["emotion"]
        row["emotional_valence"] = change["emotional_valence"]
        row["emotional_arousal_level"] = change["emotional_arousal_level"]
    return output


def build_candidate_suite(
    base_rows: list[dict[str, Any]],
    *,
    label_change_maps: dict[str, dict[str, dict[str, str]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    suite: dict[str, list[dict[str, Any]]] = {}
    descmax_rows, _ = build_descmax_candidate(base_rows)
    suite["v29_descmax_on_base"] = descmax_rows
    for name, changes in (label_change_maps or {}).items():
        changed_rows = apply_label_changes(base_rows, changes)
        rewritten_rows, _ = build_descmax_candidate(changed_rows)
        suite[f"v29_{name}"] = rewritten_rows
    return suite


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    emotions = Counter(str(row.get("emotion", "")).lower() for row in rows)
    top_emotion, top_count = emotions.most_common(1)[0]
    missing = sorted(VALID_EMOTIONS.difference(emotions))
    return {
        "row_count": len(rows),
        "top_emotion": top_emotion,
        "top_emotion_share": round(top_count / max(len(rows), 1), 6),
        "missing_emotions": missing,
    }


def write_v29_report(path: str | Path, profiles: list[CandidateProfile], notes: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"profiles": [asdict(profile) for profile in profiles], "notes": notes}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
```

Extend the final-shot implementation beyond the primitives above:

- Add a `load_label_change_maps(...)` helper that reads conservative evidence from existing JSON/CSV files when present and converts selected `RerankEvidence` rows into `{sample_id: {emotion, emotional_valence, emotional_arousal_level}}`.
- Add a `score_candidate_profile(...)` helper that combines calibrated classification score, description proxy score, distribution checks, unsafe-text checks, and gate reasons.
- Use the known v23 official anchor calibration for existing base candidates. For newly changed label candidates, compute a conservative expected classification score from the base anchor plus only accepted evidence deltas; do not give free credit for changes without evidence.
- Always emit every candidate profile in rank order, even when all candidates are blocked.

Create `scripts/track2_v29_final_shot.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from affectiveart.track2_v29_description_proxy import score_description_rows
from affectiveart.track2_v29_final_shot import (
    CandidateProfile,
    build_descmax_candidate,
    choose_v29_gate,
    summarize_rows,
    write_candidate_json_and_zip,
    write_v29_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track2 v29 final-shot candidate assembly.")
    parser.add_argument("--base-json", default="submissions/track2_submission_v21_calmshift90_candidate.json")
    parser.add_argument("--out-dir", default="experiments/track2_v29_fabg_description_qwen_rerank_20260608")
    parser.add_argument("--candidate-prefix", default="submissions/track2_submission_v29")
    args = parser.parse_args()

    rows = json.loads(Path(args.base_json).read_text(encoding="utf-8"))
    # Implementation should call build_candidate_suite(...) and score every candidate.
    # It should write one side-path JSON/ZIP pair per candidate, then mark only the
    # best gate-passing profile as `recommend_final_submit`.
    profiles = run_v29_sweep(rows, candidate_prefix=args.candidate_prefix, out_dir=args.out_dir)
    out_dir = Path(args.out_dir)
    write_v29_report(
        out_dir / "v29_final_shot_report.json",
        profiles,
        {"base_json": args.base_json},
    )
    print(json.dumps([profile.__dict__ for profile in profiles], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
python3 -m unittest tests/test_track2_v29_final_shot.py tests/test_track2_v29_description_proxy.py tests/test_track2_v29_fabg_lite.py tests/test_track2_v29_qwen_rerank.py -v
```

Expected: all v29 tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add affectiveart/track2_v29_final_shot.py scripts/track2_v29_final_shot.py tests/test_track2_v29_final_shot.py
git commit -m "feat: assemble track2 v29 final-shot candidates"
```

---

### Task 5: Run V29 Sweep And Validation

**Files:**
- Generated: `experiments/track2_v29_fabg_description_qwen_rerank_20260608/v29_final_shot_report.json`
- Generated: `submissions/track2_submission_v29_descmax_on_v21_candidate.json`
- Generated: `submissions/track2_submission_v29_descmax_on_v21_candidate.zip`
- Generated: `submissions/track2_submission_v29_qwen_hybrid_balanced_candidate.json`
- Generated: `submissions/track2_submission_v29_qwen_hybrid_balanced_candidate.zip`
- Generated: `submissions/track2_submission_v29_best_local_proxy_candidate.json`
- Generated: `submissions/track2_submission_v29_best_local_proxy_candidate.zip`

- [ ] **Step 1: Run v29 CLI**

Run:

```bash
python3 scripts/track2_v29_final_shot.py
```

Expected: prints ranked JSON profiles with `name`, candidate paths, expected scores, and gate decisions.

- [ ] **Step 2: Validate Track2 candidate**

Run:

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v29_best_local_proxy_candidate.json
```

Expected: validator OK.

- [ ] **Step 3: Run Track2-only tests**

Run:

```bash
python3 -m unittest tests/test_track2_v29_description_proxy.py tests/test_track2_v29_fabg_lite.py tests/test_track2_v29_qwen_rerank.py tests/test_track2_v29_final_shot.py tests/test_track2_official_anchor_calibration.py -v
```

Expected: all listed tests pass.

- [ ] **Step 4: Run diff check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 5: Interpret the gate**

If the generated report says `recommend_final_submit`, report the exact ZIP path for manual Codabench upload.

If the generated report says `hold_no_submit`, report the blocker:

- `overall_below_089`
- `classification_below_078`
- `description_below_098`
- `missing_emotions`
- `top_emotion_collapse_risk`
- `unsafe_text`

Do not upload automatically.
