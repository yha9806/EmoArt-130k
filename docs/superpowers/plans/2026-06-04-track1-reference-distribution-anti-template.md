# Track1 Reference Distribution Anti-Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track1 generation-planning layer that prevents batch-level poster template collapse while preserving caption fidelity, using reference-family routing, three candidate strategies, prompt linting, and distribution proxy reports.

**Architecture:** Add focused modules beside the existing Track1 pipeline: a reference-family bank, a distribution router, a prompt-template lint, and a distribution audit. Then update the MoE prompt packet builder to emit `aas_safe`, `reference_style`, and `fid_diverse` provider prompts while keeping all routing metadata out of Gemini prompts.

**Tech Stack:** Python 3.14, stdlib JSON/CSV/pathlib/dataclasses, Pillow for lightweight image stats, existing `affectiveart` modules, pytest/unittest, existing Gemini generation runner. Do not modify `submissions/track1_submission.json`, `submissions/track1_submission.zip`, or `submissions/track1/images/`.

---

## Source Spec

Use the design document:

- `docs/superpowers/specs/2026-06-04-track1-reference-distribution-anti-template-design.md`

The motivating observations are:

- MoE Top28 candidates: 90/93 `portrait_poster`, 93/93 provider prompts contain `front-facing`, 90/93 contain `flat printed poster`.
- Top28 reference assets: 16/31 landscape, 10/31 portrait, 5/31 square-ish.
- Official Track1 combines FID and AAS equally; leaderboard top teams have AAS near saturation, so FID is a decisive optimization target.

## Execution Phases And Stop Gates

This plan deliberately separates source implementation from image generation and package replacement.

1. Source-only phase:
   - Implement Tasks 1-6.
   - Run tests and prompt lint.
   - Do not call Gemini generation.
   - Do not write under `submissions/`.

2. Ten-sample smoke phase:
   - Generate at most 10 selected samples with three strategies.
   - Output only under `experiments/track1_reference_conditioned_pilot_20260603/moe_antitemplate_smoke10_20260604/`.
   - Stop if AAS-safe candidates lose required caption content or support surfaces.

3. Top28 phase:
   - Run Top28 only after the ten-sample smoke is visually acceptable.
   - Output only under `experiments/track1_reference_conditioned_pilot_20260603/moe_antitemplate_top28_candidates_20260604/`.
   - Compare against the current 90/93 `portrait_poster` baseline in review HTML.

4. Package phase:
   - Build no replacement package until human review identifies clear wins.
   - Protected paths must remain unchanged:
     - `submissions/track1_submission.json`
     - `submissions/track1_submission.zip`
     - `submissions/track1/images/`

## File Structure

- Create `affectiveart/track1_reference_family_bank.py`
  - Builds compact reference-family summaries from local images and optional metadata.
  - Provides deterministic aspect, medium, composition, and family hints.

- Create `scripts/track1_reference_family_bank.py`
  - CLI wrapper to build JSON/CSV/MD reports.

- Create `tests/test_track1_reference_family_bank.py`
  - Unit and CLI tests with small generated fixture images.

- Create `affectiveart/track1_distribution_router.py`
  - Converts a domain contract and optional reference-family bank into hard constraints, style freedom, selected visual family, and candidate strategy plan.

- Create `scripts/track1_distribution_router.py`
  - CLI wrapper to produce route JSON/CSV/MD from final contracts.

- Create `tests/test_track1_distribution_router.py`
  - Router tests for Kremlin, aviation, surrender document, scroll, and generic artwork cases.

- Create `affectiveart/track1_prompt_lint.py`
  - Batch-level prompt collapse detector.

- Create `scripts/track1_prompt_lint.py`
  - CLI wrapper to lint prompt packet JSON/JSONL and provider prompt directories.

- Create `tests/test_track1_prompt_lint.py`
  - Tests for phrase collapse, caption-required exemptions, and clean diverse prompt batches.

- Modify `affectiveart/track1_moe_prompt_packets.py`
  - Add three-strategy packet expansion: `aas_safe`, `reference_style`, `fid_diverse`.
  - Remove unnecessary template phrases from non-safe strategies.
  - Preserve metadata separation guarantees.

- Modify `scripts/track1_moe_prompt_packets.py`
  - Add `--distribution-routes-json`, `--strategy-mode`, and `--max-strategies-per-sample`.

- Modify `tests/test_track1_moe_prompt_packets.py`
  - Add tests for strategy expansion, lint-friendly prompts, and metadata isolation.

- Create `affectiveart/track1_distribution_audit.py`
  - Measures generated candidate diversity: aspect histogram, image stats, prompt phrase stats, and reference-family coverage.

- Create `scripts/track1_distribution_audit.py`
  - CLI wrapper to audit candidate manifests and prompt packets.

- Create `tests/test_track1_distribution_audit.py`
  - Image fixture tests for aspect and stats reporting.

- Create `scripts/track1_render_strategy_review_html.py`
  - Generates a review HTML with current, old candidate, three new strategies, and reference family panel.

- Create `tests/test_track1_strategy_review_html.py`
  - HTML smoke tests with local fixture assets.

---

### Task 1: Reference Family Bank

**Files:**
- Create: `affectiveart/track1_reference_family_bank.py`
- Create: `scripts/track1_reference_family_bank.py`
- Create: `tests/test_track1_reference_family_bank.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_track1_reference_family_bank.py`:

```python
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from affectiveart.track1_reference_family_bank import (
    classify_aspect,
    infer_reference_family,
    build_reference_family_bank,
    write_reference_family_reports,
)


class Track1ReferenceFamilyBankTest(unittest.TestCase):
    def test_classify_aspect_labels_landscape_portrait_and_squareish(self):
        self.assertEqual(classify_aspect(1600, 900)["label"], "landscape")
        self.assertEqual(classify_aspect(768, 1024)["label"], "portrait")
        self.assertEqual(classify_aspect(1000, 950)["label"], "square_ish")

    def test_infer_reference_family_uses_caption_and_path_terms(self):
        caption = "A Socialist Realism poster with Kremlin towers and blue searchlights."
        family = infer_reference_family("track1_0803", caption, "refs/kremlin_searchlight_01.jpg")
        self.assertEqual(family["family_id"], "kremlin_red_square")
        self.assertIn("searchlight", family["composition_hints"])

    def test_build_reference_family_bank_summarizes_fixture_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (1600, 900), (120, 50, 40)).save(root / "kremlin_landscape.jpg")
            Image.new("RGB", (768, 1024), (40, 80, 120)).save(root / "naval_poster.jpg")
            rows = [
                {
                    "sample_id": "track1_0803",
                    "caption": "A Socialist Realism propaganda poster with Kremlin tower and searchlights.",
                    "reference_path": str(root / "kremlin_landscape.jpg"),
                },
                {
                    "sample_id": "track1_0077",
                    "caption": "A Soviet naval propaganda poster with a sailor hoisting red and white flags.",
                    "reference_path": str(root / "naval_poster.jpg"),
                },
            ]
            bank = build_reference_family_bank(rows)

        self.assertEqual(bank["summary"]["total"], 2)
        self.assertEqual(bank["summary"]["aspect_labels"]["landscape"], 1)
        self.assertEqual(bank["summary"]["aspect_labels"]["portrait"], 1)
        self.assertIn("kremlin_red_square", bank["families"])
        self.assertIn("naval_aviation_vehicle", bank["families"])

    def test_write_reports_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (1600, 900), (120, 50, 40)).save(root / "kremlin.jpg")
            input_json = root / "references.json"
            input_json.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "sample_id": "track1_0803",
                                "caption": "A Kremlin tower with searchlights.",
                                "reference_path": str(root / "kremlin.jpg"),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out_json = root / "bank.json"
            out_csv = root / "bank.csv"
            out_md = root / "bank.md"
            bank = build_reference_family_bank(json.loads(input_json.read_text())["rows"])
            write_reference_family_reports(bank, json_path=out_json, csv_path=out_csv, md_path=out_md)

            script = Path(__file__).resolve().parents[1] / "scripts" / "track1_reference_family_bank.py"
            spec = importlib.util.spec_from_file_location("track1_reference_family_bank_cli", script)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            code = module.main(
                [
                    "--references-json",
                    str(input_json),
                    "--out-json",
                    str(out_json),
                    "--out-csv",
                    str(out_csv),
                    "--out-md",
                    str(out_md),
                ]
            )

        self.assertEqual(code, 0)
        self.assertTrue(out_json.exists())
        self.assertIn("kremlin_red_square", out_json.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_track1_reference_family_bank.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'affectiveart.track1_reference_family_bank'`.

- [ ] **Step 3: Implement the reference family bank module**

Create `affectiveart/track1_reference_family_bank.py`:

```python
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


FAMILY_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("kremlin_red_square", ("kremlin", "red square", "searchlight", "spasskaya"), ("red brick tower", "searchlight", "night scene")),
    ("battle_tank_cavalry", ("battle", "tank", "cavalry", "mounted", "soldiers", "infantry"), ("battle panorama", "diagonal motion", "crowd action")),
    ("naval_aviation_vehicle", ("naval", "sailor", "ship", "airplane", "pilot", "airmen", "train"), ("vehicle structure", "mechanical coherence", "open sky or water")),
    ("surrender_document_tableau", ("surrender", "document", "officer", "treaty", "declaration"), ("tableau", "document focal point", "group portrait")),
    ("agriculture_industry_worker", ("wheat", "peasants", "factory", "worker", "industrial", "forge", "harvest"), ("labor scene", "tools", "productive landscape")),
    ("commemorative_medal_institution", ("medal", "anniversary", "academy", "commemorative", "ribbon"), ("symbolic center", "institutional balance", "emblem")),
    ("scroll_album_paper_support", ("scroll", "album", "graph paper", "folded paper", "ruled page"), ("support surface", "paper texture", "mounting")),
)


def classify_aspect(width: int, height: int) -> dict[str, Any]:
    ratio = width / max(height, 1)
    if ratio >= 1.25:
        label = "landscape"
    elif ratio <= 0.8:
        label = "portrait"
    else:
        label = "square_ish"
    return {"width": width, "height": height, "ratio": round(ratio, 4), "label": label}


def infer_reference_family(sample_id: str, caption: str, reference_path: str = "") -> dict[str, Any]:
    text = f"{sample_id} {caption} {reference_path}".lower()
    for family_id, terms, hints in FAMILY_RULES:
        if any(term in text for term in terms):
            return {"family_id": family_id, "matched_terms": [term for term in terms if term in text], "composition_hints": list(hints)}
    return {"family_id": "generic_artwork", "matched_terms": [], "composition_hints": ["artwork surface", "caption-led composition"]}


def build_reference_family_bank(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        caption = str(row.get("caption") or "")
        reference_path = str(row.get("reference_path") or row.get("path") or "")
        image_info = _image_info(reference_path)
        aspect = classify_aspect(image_info["width"], image_info["height"]) if image_info["exists"] else {"label": "missing", "ratio": 0}
        family = infer_reference_family(sample_id, caption, reference_path)
        item = {
            "sample_id": sample_id,
            "caption": caption,
            "reference_path": reference_path,
            "exists": image_info["exists"],
            "aspect": aspect,
            "family_id": family["family_id"],
            "matched_terms": family["matched_terms"],
            "composition_hints": family["composition_hints"],
        }
        items.append(item)
        families[item["family_id"]].append(item)
    payload_families = {
        family_id: {
            "count": len(members),
            "aspect_labels": dict(Counter(str(member["aspect"]["label"]) for member in members)),
            "representatives": members[:6],
        }
        for family_id, members in sorted(families.items())
    }
    return {
        "summary": {
            "total": len(items),
            "existing": sum(1 for item in items if item["exists"]),
            "aspect_labels": dict(Counter(str(item["aspect"]["label"]) for item in items)),
            "families": {family_id: data["count"] for family_id, data in payload_families.items()},
        },
        "rows": items,
        "families": payload_families,
    }


def write_reference_family_reports(bank: dict[str, Any], *, json_path: str | Path, csv_path: str | Path, md_path: str | Path) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_id", "family_id", "aspect_label", "aspect_ratio", "reference_path", "exists"])
        writer.writeheader()
        for row in bank["rows"]:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "family_id": row["family_id"],
                    "aspect_label": row["aspect"]["label"],
                    "aspect_ratio": row["aspect"].get("ratio", 0),
                    "reference_path": row["reference_path"],
                    "exists": row["exists"],
                }
            )
    md_path.write_text(_render_md(bank), encoding="utf-8")


def load_reference_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    return [dict(row) for row in payload.get("rows", [])]


def _image_info(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    if not path_value or not path.exists():
        return {"exists": False, "width": 0, "height": 0}
    with Image.open(path) as image:
        width, height = image.size
    return {"exists": True, "width": width, "height": height}


def _render_md(bank: dict[str, Any]) -> str:
    lines = [
        "# Track1 Reference Family Bank",
        "",
        "Reference family summary for distribution-aware Track1 generation.",
        "",
        "## Summary",
        "",
        f"- Total references: {bank['summary']['total']}",
        f"- Existing references: {bank['summary']['existing']}",
        f"- Aspect labels: {bank['summary']['aspect_labels']}",
        f"- Families: {bank['summary']['families']}",
        "",
        "## Families",
        "",
    ]
    for family_id, data in bank["families"].items():
        lines.append(f"- `{family_id}` count={data['count']} aspects={data['aspect_labels']}")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Implement the CLI wrapper**

Create `scripts/track1_reference_family_bank.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_reference_family_bank import (  # noqa: E402
    build_reference_family_bank,
    load_reference_rows,
    write_reference_family_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Track1 reference family bank for distribution-aware generation.")
    parser.add_argument("--references-json", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bank = build_reference_family_bank(load_reference_rows(args.references_json))
    write_reference_family_reports(bank, json_path=args.out_json, csv_path=args.out_csv, md_path=args.out_md)
    print(json.dumps(bank["summary"], indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests**

Run:

```bash
python3 -m pytest tests/test_track1_reference_family_bank.py -q
```

Expected: `4 passed`.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add affectiveart/track1_reference_family_bank.py scripts/track1_reference_family_bank.py tests/test_track1_reference_family_bank.py
git commit -m "feat: add track1 reference family bank"
```

---

### Task 2: Distribution Router

**Files:**
- Create: `affectiveart/track1_distribution_router.py`
- Create: `scripts/track1_distribution_router.py`
- Create: `tests/test_track1_distribution_router.py`

- [ ] **Step 1: Write failing router tests**

Create `tests/test_track1_distribution_router.py`:

```python
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track1_distribution_router import (
    build_distribution_route,
    build_distribution_routes,
    write_distribution_route_reports,
)


def _contract(sample_id: str, caption: str, **overrides):
    row = {
        "sample_id": sample_id,
        "caption": caption,
        "aspect_plan": {"label": "portrait_poster", "width": 768, "height": 1024},
        "reference_contract": {"required": False, "categories": []},
        "text_contract": {"required": False, "modes": []},
        "relation_contract": {"required": False, "checks": []},
        "fid_risk": [],
    }
    row.update(overrides)
    return row


class Track1DistributionRouterTest(unittest.TestCase):
    def test_kremlin_route_has_hard_landmark_and_style_freedom(self):
        route = build_distribution_route(
            _contract(
                "track1_0803",
                "A Socialist Realism propaganda poster with the Soviet, American, and British flags above a Kremlin tower.",
                reference_contract={"required": True, "categories": ["landmark", "flags"]},
                text_contract={"required": True, "modes": ["cyrillic"]},
            )
        )

        self.assertEqual(route["family_id"], "kremlin_red_square")
        self.assertIn("landmark_reference", route["hard_constraints"])
        self.assertIn("aspect_variation_allowed", route["style_freedom"])
        self.assertEqual([s["strategy"] for s in route["candidate_strategies"]], ["aas_safe", "reference_style", "fid_diverse"])

    def test_scroll_route_does_not_allow_free_crop_of_support(self):
        route = build_distribution_route(
            _contract(
                "track1_0063",
                "A vertical hanging scroll with silk mounting, roller rods, calligraphy, and an ink landscape.",
            )
        )

        self.assertEqual(route["family_id"], "scroll_album_paper_support")
        self.assertIn("support_surface_required", route["hard_constraints"])
        self.assertIn("preserve_support_aspect", route["style_freedom"])
        self.assertNotIn("free_crop_allowed", route["style_freedom"])

    def test_generic_route_has_safe_fallback_family_and_strategies(self):
        route = build_distribution_route(
            _contract(
                "track1_0999",
                "A calm watercolor painting of a quiet room with soft light and delicate brushwork.",
            )
        )

        self.assertEqual(route["family_id"], "generic_artwork")
        self.assertIn("caption_content", route["hard_constraints"])
        self.assertEqual(len(route["candidate_strategies"]), 3)

    def test_build_routes_writes_reports_and_cli(self):
        contracts = [
            _contract("track1_0803", "A Kremlin tower with Soviet flags and searchlights."),
            _contract("track1_0077", "A Soviet naval poster with a sailor hoisting red and white flags."),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts_json = root / "contracts.json"
            out_json = root / "routes.json"
            out_csv = root / "routes.csv"
            out_md = root / "routes.md"
            contracts_json.write_text(json.dumps({"rows": contracts}), encoding="utf-8")
            routes = build_distribution_routes(contracts)
            write_distribution_route_reports(routes, json_path=out_json, csv_path=out_csv, md_path=out_md)

            script = Path(__file__).resolve().parents[1] / "scripts" / "track1_distribution_router.py"
            spec = importlib.util.spec_from_file_location("track1_distribution_router_cli", script)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            code = module.main(
                [
                    "--contract-json",
                    str(contracts_json),
                    "--out-json",
                    str(out_json),
                    "--out-csv",
                    str(out_csv),
                    "--out-md",
                    str(out_md),
                ]
            )

        self.assertEqual(code, 0)
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["total"], 2)
        self.assertIn("naval_aviation_vehicle", payload["summary"]["families"])
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_track1_distribution_router.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'affectiveart.track1_distribution_router'`.

- [ ] **Step 3: Implement `track1_distribution_router.py`**

Create `affectiveart/track1_distribution_router.py`:

```python
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from affectiveart.track1_reference_family_bank import infer_reference_family


def build_distribution_route(contract: dict[str, Any], reference_family_bank: dict[str, Any] | None = None) -> dict[str, Any]:
    sample_id = str(contract.get("sample_id") or "")
    caption = str(contract.get("caption") or "")
    family = infer_reference_family(sample_id, caption)
    hard_constraints = _hard_constraints(contract, caption)
    style_freedom = _style_freedom(contract, caption, hard_constraints)
    candidate_strategies = _candidate_strategies(hard_constraints, style_freedom)
    return {
        "sample_id": sample_id,
        "caption": caption,
        "family_id": family["family_id"],
        "matched_terms": family["matched_terms"],
        "composition_hints": family["composition_hints"],
        "hard_constraints": hard_constraints,
        "style_freedom": style_freedom,
        "candidate_strategies": candidate_strategies,
        "reference_family_summary": _family_summary(reference_family_bank, family["family_id"]),
        "source_aspect_plan": dict(contract.get("aspect_plan") or {}),
    }


def build_distribution_routes(contracts: Iterable[dict[str, Any]], reference_family_bank: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [build_distribution_route(contract, reference_family_bank=reference_family_bank) for contract in contracts if contract.get("sample_id")]


def write_distribution_route_reports(routes: list[dict[str, Any]], *, json_path: str | Path, csv_path: str | Path, md_path: str | Path) -> None:
    payload = {"summary": _summary(routes), "rows": routes}
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_id", "family_id", "strategy_count", "hard_constraints", "style_freedom"])
        writer.writeheader()
        for row in routes:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "family_id": row["family_id"],
                    "strategy_count": len(row["candidate_strategies"]),
                    "hard_constraints": ";".join(row["hard_constraints"]),
                    "style_freedom": ";".join(row["style_freedom"]),
                }
            )
    md_path.write_text(_render_md(payload), encoding="utf-8")


def load_contract_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    return [dict(row) for row in payload.get("rows", [])]


def load_reference_family_bank(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hard_constraints(contract: dict[str, Any], caption: str) -> list[str]:
    lower = caption.lower()
    constraints = ["caption_content", "artifact_boundary"]
    reference_categories = set(str(item) for item in (contract.get("reference_contract") or {}).get("categories", []))
    text_modes = set(str(item) for item in (contract.get("text_contract") or {}).get("modes", []))
    if reference_categories & {"landmark", "flags", "medal", "uniform", "document", "vehicle", "aircraft"}:
        constraints.append("landmark_reference" if "landmark" in reference_categories else "entity_reference")
    if text_modes or bool((contract.get("text_contract") or {}).get("required")):
        constraints.append("requested_text_only")
    if "scroll" in lower or "album" in lower or "graph paper" in lower or "folded paper" in lower:
        constraints.append("support_surface_required")
    if any(term in lower for term in ("saluting", "hoisting", "marching", "fleeing", "holding", "surrender", "train window")):
        constraints.append("relation_physics")
    return _unique(constraints)


def _style_freedom(contract: dict[str, Any], caption: str, hard_constraints: list[str]) -> list[str]:
    lower = caption.lower()
    freedom = ["medium_variation_allowed", "brushwork_variation_allowed", "lighting_variation_allowed", "composition_density_allowed"]
    if "support_surface_required" in hard_constraints:
        freedom.append("preserve_support_aspect")
    else:
        freedom.extend(["aspect_variation_allowed", "free_crop_allowed", "viewpoint_variation_allowed"])
    if "poster" in lower:
        freedom.append("poster_surface_allowed_not_template_required")
    return _unique(freedom)


def _candidate_strategies(hard_constraints: list[str], style_freedom: list[str]) -> list[dict[str, Any]]:
    return [
        {"strategy": "aas_safe", "constraint_strength": "high", "freedom_strength": "low"},
        {"strategy": "reference_style", "constraint_strength": "medium_high", "freedom_strength": "medium"},
        {"strategy": "fid_diverse", "constraint_strength": "medium", "freedom_strength": "high"},
    ]


def _family_summary(bank: dict[str, Any] | None, family_id: str) -> dict[str, Any]:
    if not bank:
        return {}
    return dict((bank.get("families") or {}).get(family_id) or {})


def _summary(routes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(routes),
        "families": dict(Counter(row["family_id"] for row in routes)),
        "strategy_budget": sum(len(row["candidate_strategies"]) for row in routes),
    }


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Track1 Distribution Router",
        "",
        f"- Total: {payload['summary']['total']}",
        f"- Families: {payload['summary']['families']}",
        f"- Strategy budget: {payload['summary']['strategy_budget']}",
        "",
    ]
    for row in payload["rows"][:80]:
        lines.append(f"- `{row['sample_id']}` family=`{row['family_id']}` strategies={','.join(s['strategy'] for s in row['candidate_strategies'])}")
    return "\n".join(lines).rstrip() + "\n"


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
```

- [ ] **Step 4: Implement the CLI wrapper**

Create `scripts/track1_distribution_router.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_distribution_router import (  # noqa: E402
    build_distribution_routes,
    load_contract_rows,
    load_reference_family_bank,
    write_distribution_route_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route Track1 contracts into distribution-aware visual families and candidate strategies.")
    parser.add_argument("--contract-json", required=True, type=Path)
    parser.add_argument("--reference-family-bank-json", type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    routes = build_distribution_routes(
        load_contract_rows(args.contract_json),
        reference_family_bank=load_reference_family_bank(args.reference_family_bank_json),
    )
    write_distribution_route_reports(routes, json_path=args.out_json, csv_path=args.out_csv, md_path=args.out_md)
    print(json.dumps({"total": len(routes)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m pytest tests/test_track1_distribution_router.py tests/test_track1_reference_family_bank.py -q
```

Expected: `8 passed`.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add affectiveart/track1_distribution_router.py scripts/track1_distribution_router.py tests/test_track1_distribution_router.py
git commit -m "feat: add track1 distribution router"
```

---

### Task 3: Prompt Template Lint

**Files:**
- Create: `affectiveart/track1_prompt_lint.py`
- Create: `scripts/track1_prompt_lint.py`
- Create: `tests/test_track1_prompt_lint.py`

- [ ] **Step 1: Write failing lint tests**

Create `tests/test_track1_prompt_lint.py`:

```python
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track1_prompt_lint import lint_prompt_batch, write_prompt_lint_reports


class Track1PromptLintTest(unittest.TestCase):
    def test_detects_front_facing_portrait_collapse(self):
        prompts = [
            {"sample_id": f"track1_{i:04d}", "provider_prompt": "front-facing flat printed poster portrait poster canvas"}
            for i in range(8)
        ] + [
            {"sample_id": "track1_9999", "provider_prompt": "loose watercolor landscape with asymmetrical crop"}
        ]

        report = lint_prompt_batch(prompts, threshold=0.6)

        self.assertEqual(report["summary"]["status"], "fail")
        self.assertGreaterEqual(report["phrases"]["front-facing"]["ratio"], 0.6)
        self.assertIn("front-facing", report["summary"]["failed_phrases"])

    def test_clean_diverse_batch_passes(self):
        prompts = [
            {"sample_id": "track1_0001", "provider_prompt": "wide landscape oil painting with deep crowd scene"},
            {"sample_id": "track1_0002", "provider_prompt": "vertical scroll with silk support and ink brushwork"},
            {"sample_id": "track1_0003", "provider_prompt": "square watercolor composition with soft light"},
        ]

        report = lint_prompt_batch(prompts, threshold=0.6)

        self.assertEqual(report["summary"]["status"], "pass")
        self.assertEqual(report["summary"]["failed_phrases"], [])

    def test_reports_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets = root / "packets.json"
            out_json = root / "lint.json"
            out_md = root / "lint.md"
            packets.write_text(
                json.dumps(
                    {
                        "packets": [
                            {"sample_id": "track1_0001", "provider_prompt": "front-facing portrait poster canvas"},
                            {"sample_id": "track1_0002", "provider_prompt": "front-facing portrait poster canvas"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = lint_prompt_batch(json.loads(packets.read_text())["packets"], threshold=0.6)
            write_prompt_lint_reports(report, json_path=out_json, md_path=out_md)

            script = Path(__file__).resolve().parents[1] / "scripts" / "track1_prompt_lint.py"
            spec = importlib.util.spec_from_file_location("track1_prompt_lint_cli", script)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            code = module.main(["--packets-json", str(packets), "--out-json", str(out_json), "--out-md", str(out_md), "--threshold", "0.6"])

        self.assertEqual(code, 2)
        self.assertIn("front-facing", out_json.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_track1_prompt_lint.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'affectiveart.track1_prompt_lint'`.

- [ ] **Step 3: Implement prompt lint module**

Create `affectiveart/track1_prompt_lint.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PHRASES = (
    "front-facing",
    "flat printed poster",
    "portrait poster canvas",
    "medium-distance figures",
    "fewer, larger",
    "internal poster margins",
    "graphic poster composition",
)


def lint_prompt_batch(rows: Iterable[dict[str, Any]], *, threshold: float = 0.6, phrases: Iterable[str] = DEFAULT_PHRASES) -> dict[str, Any]:
    items = [dict(row) for row in rows]
    total = len(items)
    phrase_report: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for phrase in phrases:
        lower = phrase.lower()
        sample_ids = [str(row.get("sample_id") or "") for row in items if lower in str(row.get("provider_prompt") or row.get("prompt") or "").lower()]
        ratio = len(sample_ids) / total if total else 0.0
        phrase_report[phrase] = {"count": len(sample_ids), "ratio": round(ratio, 6), "sample_ids": sample_ids[:50]}
        if total and ratio >= threshold:
            failed.append(phrase)
    return {
        "summary": {
            "total": total,
            "threshold": threshold,
            "status": "fail" if failed else "pass",
            "failed_phrases": failed,
        },
        "phrases": phrase_report,
    }


def write_prompt_lint_reports(report: dict[str, Any], *, json_path: str | Path, md_path: str | Path) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")


def load_prompt_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    return [dict(row) for row in payload.get("packets", payload.get("rows", []))]


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Track1 Prompt Template Lint",
        "",
        f"- Status: `{report['summary']['status']}`",
        f"- Total prompts: {report['summary']['total']}",
        f"- Threshold: {report['summary']['threshold']}",
        f"- Failed phrases: {report['summary']['failed_phrases']}",
        "",
        "## Phrase Counts",
        "",
    ]
    for phrase, data in report["phrases"].items():
        lines.append(f"- `{phrase}` count={data['count']} ratio={data['ratio']}")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Implement CLI wrapper**

Create `scripts/track1_prompt_lint.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_prompt_lint import lint_prompt_batch, load_prompt_rows, write_prompt_lint_reports  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lint Track1 provider prompts for batch-level template collapse.")
    parser.add_argument("--packets-json", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--allow-fail", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = lint_prompt_batch(load_prompt_rows(args.packets_json), threshold=args.threshold)
    write_prompt_lint_reports(report, json_path=args.out_json, md_path=args.out_md)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False, sort_keys=True))
    if report["summary"]["status"] == "fail" and not args.allow_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m pytest tests/test_track1_prompt_lint.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add affectiveart/track1_prompt_lint.py scripts/track1_prompt_lint.py tests/test_track1_prompt_lint.py
git commit -m "feat: add track1 prompt template lint"
```

---

### Task 4: Three-Strategy MoE Prompt Packets

**Files:**
- Modify: `affectiveart/track1_moe_prompt_packets.py`
- Modify: `scripts/track1_moe_prompt_packets.py`
- Modify: `tests/test_track1_moe_prompt_packets.py`

- [ ] **Step 1: Add failing tests to `tests/test_track1_moe_prompt_packets.py`**

Append these tests to `Track1MoePromptPacketsTest`:

```python
    def test_distribution_route_expands_three_candidate_strategies(self):
        contract = _contract()
        route = _route(candidate_count=3)
        distribution_route = {
            "sample_id": "track1_0803",
            "family_id": "kremlin_red_square",
            "hard_constraints": ["caption_content", "landmark_reference", "requested_text_only"],
            "style_freedom": ["aspect_variation_allowed", "brushwork_variation_allowed", "viewpoint_variation_allowed"],
            "composition_hints": ["red brick tower", "night scene", "searchlight"],
            "candidate_strategies": [
                {"strategy": "aas_safe", "constraint_strength": "high", "freedom_strength": "low"},
                {"strategy": "reference_style", "constraint_strength": "medium_high", "freedom_strength": "medium"},
                {"strategy": "fid_diverse", "constraint_strength": "medium", "freedom_strength": "high"},
            ],
        }

        rows = build_moe_prompt_packets(
            [route],
            contracts={contract["sample_id"]: contract},
            reference_text_bank={},
            distribution_routes={contract["sample_id"]: distribution_route},
            out_dir=Path(tempfile.mkdtemp()),
            max_strategies_per_sample=3,
        )

        self.assertEqual([row["candidate_strategy"] for row in rows], ["aas_safe", "reference_style", "fid_diverse"])
        self.assertIn("red brick tower", rows[1]["provider_prompt"])
        self.assertNotIn("portrait poster canvas", rows[2]["provider_prompt"].lower())
        self.assertEqual(rows[0]["review_metadata"]["family_id"], "kremlin_red_square")

    def test_default_behavior_remains_one_packet_without_distribution_routes(self):
        contract = _contract()
        route = _route(candidate_count=4)
        rows = build_moe_prompt_packets(
            [route],
            contracts={contract["sample_id"]: contract},
            reference_text_bank={},
            out_dir=Path(tempfile.mkdtemp()),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate_strategy"], "legacy")
```

- [ ] **Step 2: Run the targeted tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_track1_moe_prompt_packets.py::Track1MoePromptPacketsTest::test_distribution_route_expands_three_candidate_strategies -q
```

Expected: fail with `TypeError: build_moe_prompt_packets() got an unexpected keyword argument 'distribution_routes'`.

- [ ] **Step 3: Modify `build_moe_prompt_packet` signature**

In `affectiveart/track1_moe_prompt_packets.py`, change the function signature to:

```python
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
```

Add these fields to the returned packet:

```python
        "candidate_strategy": candidate_strategy,
```

Inside `review_metadata`, add:

```python
            "family_id": (distribution_route or {}).get("family_id", ""),
            "candidate_strategy": candidate_strategy,
```

- [ ] **Step 4: Add strategy prompt rendering helpers**

In `affectiveart/track1_moe_prompt_packets.py`, add:

```python
def render_distribution_strategy_section(distribution_route: dict[str, Any], *, candidate_strategy: str) -> str:
    if not distribution_route or candidate_strategy == "legacy":
        return ""
    hard = [str(item) for item in distribution_route.get("hard_constraints", []) if str(item)]
    freedom = [str(item) for item in distribution_route.get("style_freedom", []) if str(item)]
    hints = [str(item) for item in distribution_route.get("composition_hints", []) if str(item)]
    lines = [
        "DISTRIBUTION-AWARE STRATEGY",
        f"- Candidate strategy: {candidate_strategy}.",
        f"- Visual family: {distribution_route.get('family_id', '')}.",
    ]
    if hard:
        lines.append("Hard constraints:")
        lines.extend(f"- {item}" for item in hard)
    if hints:
        lines.append("Reference-family composition hints:")
        lines.extend(f"- {item}" for item in hints[:6])
    if freedom:
        lines.append("Allowed style freedom:")
        lines.extend(f"- {item}" for item in freedom)
    if candidate_strategy == "aas_safe":
        lines.append("Keep constraints strong; avoid unnecessary composition experimentation.")
    elif candidate_strategy == "reference_style":
        lines.append("Match the reference family through medium, crop, density, color, brushwork, and lighting while preserving hard constraints.")
    elif candidate_strategy == "fid_diverse":
        lines.append("Avoid repeating a generic front-facing portrait poster template; vary viewpoint, aspect, crop, figure scale, texture, and scene density when the caption permits.")
    return "\n".join(lines).strip()


def soften_prompt_for_strategy(prompt: str, *, candidate_strategy: str) -> str:
    if candidate_strategy in {"legacy", "aas_safe"}:
        return prompt
    replacements = {
        "Use a portrait poster canvas; keep the full printed poster front-facing and avoid left/right overflow.": "Use an artwork canvas whose aspect and crop fit the caption and reference family; avoid overflow.",
        "Make it front-facing and fully filled by the artwork surface.": "Make the output the artwork itself; choose a natural crop and viewpoint for the reference family.",
        "Fill the output with the flat printed poster artwork; internal poster margins are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.": "Keep the artwork surface itself; internal margins, natural crop, aging, or print texture are allowed when they fit the caption.",
        "Prefer medium-distance figures and clear silhouettes over close-up hands, ropes, weapons, or flag details.": "Use figure scale that fits the scene; simplify risky local mechanics without forcing every scene to medium distance.",
    }
    result = prompt
    for old, new in replacements.items():
        result = result.replace(old, new)
    if candidate_strategy == "fid_diverse":
        result = result.replace("Poster means a flat front-facing printed artwork; internal margins and printed border lines are allowed.", "Poster surface is allowed when requested; do not force a single generic portrait poster template.")
    return result
```

- [ ] **Step 5: Expand packets per distribution strategy**

Modify `build_moe_prompt_packets` signature:

```python
def build_moe_prompt_packets(
    routes: Iterable[dict[str, Any]],
    *,
    contracts: dict[str, dict[str, Any]],
    reference_text_bank: dict[str, dict[str, Any]] | None,
    out_dir: str | Path,
    limit: int | None = None,
    distribution_routes: dict[str, dict[str, Any]] | None = None,
    max_strategies_per_sample: int | None = None,
) -> list[dict[str, Any]]:
```

Inside the route loop, replace the single packet creation with:

```python
        distribution_route = (distribution_routes or {}).get(sample_id)
        strategy_rows = _strategies_for_distribution_route(distribution_route)
        if max_strategies_per_sample is not None and max_strategies_per_sample > 0:
            strategy_rows = strategy_rows[:max_strategies_per_sample]
        for strategy_index, strategy in enumerate(strategy_rows, start=1):
            candidate_strategy = str(strategy.get("strategy") or "legacy")
            prompt_path = prompt_dir / f"{index:02d}_{sample_id}_{candidate_strategy}.txt"
            packet = build_moe_prompt_packet(
                rank=index,
                route=route,
                contract=contract,
                reference_text_packet=(reference_text_bank or {}).get(sample_id),
                provider_prompt_path=prompt_path,
                distribution_route=distribution_route,
                candidate_strategy=candidate_strategy,
            )
            prompt_path.write_text(packet["provider_prompt"].rstrip() + "\n", encoding="utf-8")
            packets.append(packet)
```

Add:

```python
def _strategies_for_distribution_route(distribution_route: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not distribution_route:
        return [{"strategy": "legacy"}]
    rows = [dict(row) for row in distribution_route.get("candidate_strategies", []) if isinstance(row, dict)]
    return rows or [{"strategy": "aas_safe"}, {"strategy": "reference_style"}, {"strategy": "fid_diverse"}]
```

In `build_moe_prompt_packet`, after the aspect section, append:

```python
    strategy_section = render_distribution_strategy_section(distribution_route or {}, candidate_strategy=candidate_strategy)
    if strategy_section:
        provider_prompt += "\n\n" + strategy_section
    provider_prompt = soften_prompt_for_strategy(provider_prompt, candidate_strategy=candidate_strategy)
```

- [ ] **Step 6: Update the CLI**

In `scripts/track1_moe_prompt_packets.py`, import `load_distribution_routes` after it exists locally:

```python
from affectiveart.track1_moe_prompt_packets import (
    build_moe_prompt_packets,
    load_contracts,
    load_distribution_routes,
    load_reference_text_packets,
    load_routes,
    write_moe_prompt_packet_reports,
)
```

Add parser args:

```python
    parser.add_argument("--distribution-routes-json", type=Path)
    parser.add_argument("--max-strategies-per-sample", type=int, default=0)
```

Pass them:

```python
        distribution_routes=load_distribution_routes(args.distribution_routes_json),
        max_strategies_per_sample=args.max_strategies_per_sample or None,
```

Add loader to `affectiveart/track1_moe_prompt_packets.py`:

```python
def load_distribution_routes(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    return {str(row["sample_id"]): dict(row) for row in rows if isinstance(row, dict) and row.get("sample_id")}
```

- [ ] **Step 7: Run tests**

Run:

```bash
python3 -m pytest tests/test_track1_moe_prompt_packets.py tests/test_track1_prompt_lint.py tests/test_track1_distribution_router.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Run prompt lint on the old Top28 as a red check**

Run:

```bash
python3 scripts/track1_prompt_lint.py \
  --packets-json experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_clean_20260604/track1_moe_prompt_packets.json \
  --out-json experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_clean_20260604/prompt_lint_old_top28.json \
  --out-md experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_clean_20260604/prompt_lint_old_top28.md \
  --threshold 0.6 \
  --allow-fail
```

Expected: command exits `0` because `--allow-fail` is set, and report status is `fail`.

- [ ] **Step 9: Commit Task 4**

Run:

```bash
git add affectiveart/track1_moe_prompt_packets.py scripts/track1_moe_prompt_packets.py tests/test_track1_moe_prompt_packets.py
git commit -m "feat: add distribution-aware track1 prompt strategies"
```

---

### Task 5: Distribution Audit

**Files:**
- Create: `affectiveart/track1_distribution_audit.py`
- Create: `scripts/track1_distribution_audit.py`
- Create: `tests/test_track1_distribution_audit.py`

- [ ] **Step 1: Write failing audit tests**

Create `tests/test_track1_distribution_audit.py`:

```python
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from affectiveart.track1_distribution_audit import audit_candidate_distribution, write_distribution_audit_reports


class Track1DistributionAuditTest(unittest.TestCase):
    def test_audit_reports_aspect_and_prompt_phrase_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (768, 1024), (120, 80, 60)).save(image_dir / "track1_0001_c01.png")
            Image.new("RGB", (1600, 900), (60, 80, 120)).save(image_dir / "track1_0002_c01.png")
            rows = [
                {"sample_id": "track1_0001", "candidate_strategy": "aas_safe", "image_path": str(image_dir / "track1_0001_c01.png"), "provider_prompt": "front-facing portrait poster canvas"},
                {"sample_id": "track1_0002", "candidate_strategy": "fid_diverse", "image_path": str(image_dir / "track1_0002_c01.png"), "provider_prompt": "wide oil painting landscape"},
            ]

            report = audit_candidate_distribution(rows)

        self.assertEqual(report["summary"]["total"], 2)
        self.assertEqual(report["summary"]["aspect_labels"]["portrait"], 1)
        self.assertEqual(report["summary"]["aspect_labels"]["landscape"], 1)
        self.assertEqual(report["summary"]["strategy_counts"]["aas_safe"], 1)
        self.assertEqual(report["prompt_lint"]["phrases"]["front-facing"]["count"], 1)

    def test_reports_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "track1_0001_c01.png"
            Image.new("RGB", (768, 1024), (120, 80, 60)).save(image)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"rows": [{"sample_id": "track1_0001", "candidate_strategy": "aas_safe", "image_path": str(image), "provider_prompt": "front-facing"}]}), encoding="utf-8")
            out_json = root / "audit.json"
            out_md = root / "audit.md"
            report = audit_candidate_distribution(json.loads(manifest.read_text())["rows"])
            write_distribution_audit_reports(report, json_path=out_json, md_path=out_md)

            script = Path(__file__).resolve().parents[1] / "scripts" / "track1_distribution_audit.py"
            spec = importlib.util.spec_from_file_location("track1_distribution_audit_cli", script)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            code = module.main(["--manifest-json", str(manifest), "--out-json", str(out_json), "--out-md", str(out_md)])

        self.assertEqual(code, 0)
        self.assertIn("aspect_labels", out_json.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_track1_distribution_audit.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'affectiveart.track1_distribution_audit'`.

- [ ] **Step 3: Implement distribution audit module**

Create `affectiveart/track1_distribution_audit.py`:

```python
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageStat

from affectiveart.track1_prompt_lint import lint_prompt_batch
from affectiveart.track1_reference_family_bank import classify_aspect


def audit_candidate_distribution(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = [dict(row) for row in rows]
    audited: list[dict[str, Any]] = []
    for row in items:
        image_stats = _image_stats(str(row.get("image_path") or ""))
        aspect = classify_aspect(image_stats["width"], image_stats["height"]) if image_stats["exists"] else {"label": "missing", "ratio": 0}
        audited.append({**row, "image_stats": image_stats, "aspect": aspect})
    return {
        "summary": {
            "total": len(audited),
            "existing": sum(1 for row in audited if row["image_stats"]["exists"]),
            "aspect_labels": dict(Counter(str(row["aspect"]["label"]) for row in audited)),
            "strategy_counts": dict(Counter(str(row.get("candidate_strategy") or "unknown") for row in audited)),
        },
        "prompt_lint": lint_prompt_batch(audited, threshold=0.6),
        "rows": audited,
    }


def write_distribution_audit_reports(report: dict[str, Any], *, json_path: str | Path, md_path: str | Path) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    return [dict(row) for row in payload.get("rows", payload.get("candidates", []))]


def _image_stats(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    if not path_value or not path.exists():
        return {"exists": False, "width": 0, "height": 0, "mean_luma": 0.0, "luma_stddev": 0.0}
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        gray = rgb.convert("L")
        stat = ImageStat.Stat(gray)
        return {"exists": True, "width": width, "height": height, "mean_luma": round(stat.mean[0], 4), "luma_stddev": round(stat.stddev[0], 4)}


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Track1 Distribution Audit",
        "",
        f"- Total: {report['summary']['total']}",
        f"- Existing: {report['summary']['existing']}",
        f"- Aspect labels: {report['summary']['aspect_labels']}",
        f"- Strategy counts: {report['summary']['strategy_counts']}",
        f"- Prompt lint status: {report['prompt_lint']['summary']['status']}",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Implement CLI wrapper**

Create `scripts/track1_distribution_audit.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectiveart.track1_distribution_audit import audit_candidate_distribution, load_manifest_rows, write_distribution_audit_reports  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Track1 candidate distribution and prompt-template risk.")
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_candidate_distribution(load_manifest_rows(args.manifest_json))
    write_distribution_audit_reports(report, json_path=args.out_json, md_path=args.out_md)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m pytest tests/test_track1_distribution_audit.py tests/test_track1_prompt_lint.py -q
```

Expected: `5 passed`.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add affectiveart/track1_distribution_audit.py scripts/track1_distribution_audit.py tests/test_track1_distribution_audit.py
git commit -m "feat: add track1 distribution audit"
```

---

### Task 6: Top28 Dry-Run Pipeline And Review Packet

**Files:**
- Create: `scripts/track1_render_strategy_review_html.py`
- Create: `tests/test_track1_strategy_review_html.py`
- No changes to formal submission files.

- [ ] **Step 1: Write failing HTML smoke test**

Create `tests/test_track1_strategy_review_html.py`:

```python
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class Track1StrategyReviewHtmlTest(unittest.TestCase):
    def test_cli_writes_strategy_review_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "candidate.png"
            current = root / "current.jpg"
            reference = root / "reference.jpg"
            Image.new("RGB", (768, 1024), (100, 50, 30)).save(image)
            Image.new("RGB", (768, 1024), (80, 80, 80)).save(current)
            Image.new("RGB", (1600, 900), (30, 80, 100)).save(reference)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "sample_id": "track1_0803",
                                "caption": "Kremlin tower with searchlights.",
                                "candidate_strategy": "reference_style",
                                "image_path": str(image),
                                "current_image_path": str(current),
                                "reference_paths": [str(reference)],
                                "family_id": "kremlin_red_square",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out_html = root / "review.html"
            script = Path(__file__).resolve().parents[1] / "scripts" / "track1_render_strategy_review_html.py"
            spec = importlib.util.spec_from_file_location("track1_render_strategy_review_html_cli", script)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            code = module.main(["--manifest-json", str(manifest), "--out-html", str(out_html)])

        self.assertEqual(code, 0)
        html = out_html.read_text(encoding="utf-8")
        self.assertIn("track1_0803", html)
        self.assertIn("reference_style", html)
        self.assertIn("kremlin_red_square", html)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python3 -m pytest tests/test_track1_strategy_review_html.py -q
```

Expected: fail with `FileNotFoundError` or module import error for the missing script.

- [ ] **Step 3: Implement review HTML script**

Create `scripts/track1_render_strategy_review_html.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Track1 strategy review HTML.")
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--out-html", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(render_html([dict(row) for row in rows]), encoding="utf-8")
    print(args.out_html)
    return 0


def render_html(rows: list[dict[str, Any]]) -> str:
    cards = "\n".join(_render_card(row) for row in rows)
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Track1 Strategy Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f6f6f3; color: #1f2933; }}
    .sample {{ margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid #c9c9c0; }}
    .meta {{ font-size: 14px; line-height: 1.5; max-width: 1200px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 12px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #d8d8d0; padding: 8px; }}
    img {{ width: 100%; height: 360px; object-fit: contain; background: #ecece6; }}
    figcaption {{ font-size: 13px; margin-top: 6px; color: #334155; }}
  </style>
</head>
<body>
  <h1>Track1 三策略候选 Review</h1>
  {cards}
</body>
</html>
"""


def _render_card(row: dict[str, Any]) -> str:
    sample_id = html.escape(str(row.get("sample_id") or ""))
    caption = html.escape(str(row.get("caption") or ""))
    strategy = html.escape(str(row.get("candidate_strategy") or ""))
    family = html.escape(str(row.get("family_id") or ""))
    figures = []
    for label, path in [
        ("current", row.get("current_image_path")),
        (strategy or "candidate", row.get("image_path")),
    ]:
        if path:
            figures.append(_figure(label, str(path)))
    for index, ref_path in enumerate(row.get("reference_paths") or [], start=1):
        figures.append(_figure(f"reference {index}", str(ref_path)))
    return f"""
<section class="sample">
  <div class="meta">
    <strong>{sample_id}</strong> | strategy={strategy} | family={family}<br>
    {caption}
  </div>
  <div class="grid">{''.join(figures)}</div>
</section>
"""


def _figure(label: str, path: str) -> str:
    return f"<figure><img src='{html.escape(path)}'><figcaption>{html.escape(label)}</figcaption></figure>"


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test**

Run:

```bash
python3 -m pytest tests/test_track1_strategy_review_html.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Run Top28 no-generation dry path**

Run these commands after Tasks 1-5 exist:

```bash
python3 scripts/track1_distribution_router.py \
  --contract-json experiments/track1_reference_conditioned_pilot_20260603/final_contract_20260603/track1_final_gate_contracts.json \
  --out-json experiments/track1_reference_conditioned_pilot_20260603/distribution_router_20260604/track1_distribution_routes.json \
  --out-csv experiments/track1_reference_conditioned_pilot_20260603/distribution_router_20260604/track1_distribution_routes.csv \
  --out-md experiments/track1_reference_conditioned_pilot_20260603/distribution_router_20260604/track1_distribution_routes.md
```

```bash
python3 scripts/track1_moe_prompt_packets.py \
  --routes-json experiments/track1_reference_conditioned_pilot_20260603/expert_router_20260604/track1_expert_routes.json \
  --contract-json experiments/track1_reference_conditioned_pilot_20260603/final_contract_20260603/track1_final_gate_contracts.json \
  --distribution-routes-json experiments/track1_reference_conditioned_pilot_20260603/distribution_router_20260604/track1_distribution_routes.json \
  --reference-text-bank-jsonl experiments/track1_reference_conditioned_pilot_20260603/reference_text_bank_full_v2.jsonl \
  --out-dir experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260604 \
  --limit 28 \
  --max-strategies-per-sample 3
```

```bash
python3 scripts/track1_prompt_lint.py \
  --packets-json experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260604/track1_moe_prompt_packets.json \
  --out-json experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260604/prompt_lint.json \
  --out-md experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260604/prompt_lint.md \
  --threshold 0.6
```

Expected:

- Distribution routes JSON exists.
- MoE packets include three strategies for generated rows.
- Prompt lint status is `pass` or fails only for caption-required support phrases with documented reason. If it fails on `front-facing`, `flat printed poster`, or `portrait poster canvas`, do not generate images.

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add scripts/track1_render_strategy_review_html.py tests/test_track1_strategy_review_html.py
git commit -m "feat: add track1 strategy review html"
```

---

### Task 7: Verification And Safety Gate

**Files:**
- No new files required unless a previous task produced reviewed reports.

- [ ] **Step 1: Run all new tests**

Run:

```bash
python3 -m pytest \
  tests/test_track1_reference_family_bank.py \
  tests/test_track1_distribution_router.py \
  tests/test_track1_prompt_lint.py \
  tests/test_track1_moe_prompt_packets.py \
  tests/test_track1_distribution_audit.py \
  tests/test_track1_strategy_review_html.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 2: Run repository whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Confirm formal champion package remains unchanged**

Run:

```bash
git diff -- submissions/track1_submission.json submissions/track1_submission.zip submissions/track1/images | wc -l
```

Expected:

```text
0
```

- [ ] **Step 4: Ask Gemini-agent for plan/patch review before any generation**

Use `mcp__gemini_agent.gemini_diff_review` with this input:

```text
Review the Track1 reference-distribution anti-template implementation diff. Check whether it preserves champion package immutability, keeps metadata out of provider prompts, avoids excessive template phrases, and supports AAS-safe/reference-style/FID-diverse strategy separation.
```

Expected: no blocking issues. If Gemini-agent flags a blocker, fix it before generation.

- [ ] **Step 5: Commit final verification report if generated**

If Tasks 1-6 generated only source and tests, no additional commit is needed after Task 6. If a final verification report is created under `experiments/track1_reference_conditioned_pilot_20260603/`, commit it separately:

```bash
git add experiments/track1_reference_conditioned_pilot_20260603/<verified-report-file>
git commit -m "docs: record track1 anti-template verification"
```

---

## First Generation Runs After This Plan

After the implementation plan is complete and verified, the first image-generation run should be a ten-sample smoke experiment. Use samples that cover several risk families: Kremlin, naval flags, train window, tank/soldiers, scroll support, surrender document, aviation, agriculture, cavalry relation, and a generic painting.

```bash
GEMINI_API_KEY="$(security find-generic-password -s affectiveart-gemini-api-key -a gemini -w)" \
/Users/yhryzy/.cache/emoart-track1-gemini-venv/bin/python scripts/track1_moe_generate_candidates.py \
  --packets-jsonl experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260604/track1_moe_prompt_packets.jsonl \
  --out-dir experiments/track1_reference_conditioned_pilot_20260603/moe_antitemplate_smoke10_20260604 \
  --sample-id track1_0803 \
  --sample-id track1_0077 \
  --sample-id track1_0091 \
  --sample-id track1_0128 \
  --sample-id track1_0063 \
  --sample-id track1_0721 \
  --sample-id track1_0081 \
  --sample-id track1_0491 \
  --sample-id track1_0747 \
  --sample-id track1_0370 \
  --max-candidates-per-sample 1
```

Then audit:

```bash
python3 scripts/track1_distribution_audit.py \
  --manifest-json experiments/track1_reference_conditioned_pilot_20260603/moe_antitemplate_smoke10_20260604/candidate_manifest.json \
  --out-json experiments/track1_reference_conditioned_pilot_20260603/moe_antitemplate_smoke10_20260604/distribution_audit.json \
  --out-md experiments/track1_reference_conditioned_pilot_20260603/moe_antitemplate_smoke10_20260604/distribution_audit.md
```

Render a smoke review sheet:

```bash
python3 scripts/track1_render_strategy_review_html.py \
  --manifest-json experiments/track1_reference_conditioned_pilot_20260603/moe_antitemplate_smoke10_20260604/candidate_manifest.json \
  --out-html experiments/track1_reference_conditioned_pilot_20260603/moe_antitemplate_smoke10_20260604/track1_antitemplate_smoke10_review_zh.html
```

Proceed to Top28 only if the smoke review shows that `reference_style` and `fid_diverse` candidates increase visual variety without losing mandatory content, text, support surfaces, or relation logic.

Do not build a replacement package until human review confirms clear wins and `scripts/affectiveart_challenge.py validate-track1` passes for the candidate package.

## Self-Review

- Spec coverage: the plan implements reference-family bank, distribution router, three candidate strategies, prompt lint, distribution audit, review HTML, and safety verification.
- Marker scan: no unresolved draft markers remain in this plan.
- Type consistency: `sample_id`, `family_id`, `candidate_strategy`, `hard_constraints`, `style_freedom`, `provider_prompt`, `image_path`, and `reference_paths` are used consistently across tasks.
- Submission safety: all generated outputs go under `experiments/track1_reference_conditioned_pilot_20260603/`; formal Track1 champion files remain read-only for this plan.
