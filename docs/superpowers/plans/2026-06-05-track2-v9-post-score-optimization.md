# Track2 v9 Post-Score Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After Codabench submission `781601` finishes, calibrate the local Track2 scorer and prepare one optimized v9-derived candidate for the next submission window.

**Architecture:** Treat `781601` as the decisive calibration point for the aggressive v9/v3 path. Keep classification scoring, VULCA entailment scoring, and final submission packaging as separate layers so text improvements cannot mask unsafe label changes. Produce only side-path JSON/ZIP candidates until the final submit decision is explicit.

**Tech Stack:** Python unittest, `affectiveart.track2_local_shadow_evaluator`, `affectiveart.track2_fused_shadow_evaluator`, `affectiveart.track2_vulca_entailment_gate`, Codabench Track2 validator.

---

### Task 1: Capture Official Score And Calibrate The Local Ledger

**Files:**
- Read: Codabench submission `781601`
- Modify: `experiments/track2_local_shadow_evaluator_20260605/calibration_ledger.jsonl`
- Output: `experiments/track2_local_shadow_evaluator_20260605/post_781601_calibration/`

- [ ] **Step 1: Record the visible official score**

Open Codabench submission `781601` and copy the visible Track2 metrics:

```text
overall
classification
description
emotion_accuracy
emotion_macro_f1
valence_accuracy
valence_macro_f1
arousal_accuracy
arousal_macro_f1
```

- [ ] **Step 2: Append calibration entry**

Run this command and enter the official Codabench values when prompted:

```bash
python3 - <<'PY'
import subprocess

overall = input("official overall for 781601: ").strip()
classification = input("official classification for 781601: ").strip()
description = input("official description for 781601: ").strip()

subprocess.run(
    [
        "python3",
        "scripts/track2_local_shadow_evaluator.py",
        "ledger-add",
        "--ledger",
        "experiments/track2_local_shadow_evaluator_20260605/calibration_ledger.jsonl",
        "--submission-id",
        "781601",
        "--file-name",
        "track2_submission_v3_mid_gemini35_desc_192_candidate.zip",
        "--shadow-overall-expected",
        "0.865584",
        "--shadow-classification-expected",
        "0.778150",
        "--shadow-description-expected",
        "0.953017",
        "--official-overall",
        overall,
        "--official-classification",
        classification,
        "--official-description",
        description,
        "--notes",
        "v3/v9 aggressive same-quadrant classification calibration",
    ],
    check=True,
)
PY
```

Expected: command exits `0` and prints a calibration summary with `entry_count >= 1`.

- [ ] **Step 3: Re-run fused comparison**

```bash
python3 scripts/track2_fused_shadow_evaluator.py score \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate moe_v2_anchor=submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate v3_mid_gemini35_desc_192=submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json \
  --candidate v9_vulca_entailment=submissions/track2_submission_v9_vulca_entailment_candidate.json \
  --candidate public_style_review11_vulca=submissions/track2_submission_v10_public_style_review11_vulca_fused_candidate.json \
  --out-dir experiments/track2_local_shadow_evaluator_20260605/post_781601_calibration/fused_compare \
  --expected-row-count 1000
```

Expected: report is written to `experiments/track2_local_shadow_evaluator_20260605/post_781601_calibration/fused_compare/fused_shadow_score_report.md`.

### Task 2: Decide Whether v9 Is A Valid Base

**Files:**
- Read: `experiments/track2_local_shadow_evaluator_20260605/post_781601_calibration/fused_compare/fused_shadow_score_report.json`
- Read: `submissions/track2_submission_v9_vulca_entailment_candidate.json`
- Output: `experiments/track2_v9_post_score_optimization_20260606/v9_decision.md`

- [ ] **Step 1: Apply the v9 base decision rule**

Use these thresholds:

```text
Use v9 as base if:
- official overall for 781601 >= 0.845
- official classification for 781601 >= 0.745
- official description for 781601 >= 0.940

Do not use v9 as base if:
- official overall for 781601 < 0.840
- official classification for 781601 < 0.735
- official description for 781601 < 0.930
```

- [ ] **Step 2: Write the decision note**

Create `experiments/track2_v9_post_score_optimization_20260606/v9_decision.md` with this exact structure:

```markdown
# Track2 v9 Post-Score Decision

- Submission checked: `781601`
- Official overall:
- Official classification:
- Official description:
- Decision:
- Reason:
- Next candidate base:
```

Expected: decision is one of `use_v9_base`, `hold_v9_use_stable_v10`, or `needs_manual_review`.

### Task 3: Build v9-Optimized Candidate

**Files:**
- Read: `submissions/track2_submission_v9_vulca_entailment_candidate.json`
- Read: `submissions/track2_submission_v10_public_style_review11_vulca_fused_candidate.json`
- Create: `submissions/track2_submission_v11_v9_optimized_candidate.json`
- Create: `submissions/track2_submission_v11_v9_optimized_candidate.zip`
- Output: `experiments/track2_v9_post_score_optimization_20260606/v11/`

- [ ] **Step 1: Select candidate base**

If `v9_decision.md` says `use_v9_base`, use:

```text
submissions/track2_submission_v9_vulca_entailment_candidate.json
```

If it says `hold_v9_use_stable_v10`, use:

```text
submissions/track2_submission_v10_public_style_review11_vulca_fused_candidate.json
```

- [ ] **Step 2: Apply VULCA entailment gate to the selected base**

```bash
SELECTED_BASE_JSON=$(python3 - <<'PY'
from pathlib import Path

decision_path = Path("experiments/track2_v9_post_score_optimization_20260606/v9_decision.md")
text = decision_path.read_text(encoding="utf-8")
if "use_v9_base" in text:
    print("submissions/track2_submission_v9_vulca_entailment_candidate.json")
elif "hold_v9_use_stable_v10" in text:
    print("submissions/track2_submission_v10_public_style_review11_vulca_fused_candidate.json")
else:
    raise SystemExit("v9_decision.md must contain use_v9_base or hold_v9_use_stable_v10 before building v11")
PY
)

python3 scripts/track2_vulca_entailment_gate.py \
  --source-json "$SELECTED_BASE_JSON" \
  --out-json submissions/track2_submission_v11_v9_optimized_candidate.json \
  --out-zip submissions/track2_submission_v11_v9_optimized_candidate.zip \
  --report-json experiments/track2_v9_post_score_optimization_20260606/v11/vulca_entailment_report.json \
  --report-md experiments/track2_v9_post_score_optimization_20260606/v11/vulca_entailment_report.md \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --shadow-out-dir experiments/track2_v9_post_score_optimization_20260606/v11/base_shadow_eval \
  --expected-row-count 1000
```

Expected: `classification_label_changes=0` in the VULCA report for this step.

- [ ] **Step 3: Score v11 against current candidates**

```bash
python3 scripts/track2_fused_shadow_evaluator.py score \
  --baseline-json submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate moe_v2_anchor=submissions/track2_submission_moe_v2_accept5_candidate.json \
  --candidate v9_vulca_entailment=submissions/track2_submission_v9_vulca_entailment_candidate.json \
  --candidate public_style_review11_vulca=submissions/track2_submission_v10_public_style_review11_vulca_fused_candidate.json \
  --candidate v11_v9_optimized=submissions/track2_submission_v11_v9_optimized_candidate.json \
  --out-dir experiments/track2_v9_post_score_optimization_20260606/v11/fused_compare \
  --expected-row-count 1000
```

Expected: `v11_v9_optimized` appears in the ranking and `decision=recommend_submit`.

### Task 4: Validate And Final Gate

**Files:**
- Read: `submissions/track2_submission_v11_v9_optimized_candidate.json`
- Read: `experiments/track2_v9_post_score_optimization_20260606/v11/fused_compare/fused_shadow_score_report.json`

- [ ] **Step 1: Run Track2 validator**

```bash
python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v11_v9_optimized_candidate.json
```

Expected: `OK`.

- [ ] **Step 2: Run Track2 tests**

```bash
python3 -m unittest discover -s tests -p 'test_track2*.py' -v
```

Expected: all Track2 tests pass.

- [ ] **Step 3: Run whitespace check**

```bash
git diff --check
```

Expected: no output and exit code `0`.

- [ ] **Step 4: Apply scarce-submission gate**

Submit only if all conditions hold:

```text
- validator OK
- Track2 tests pass
- fused decision = recommend_submit
- safety_issue_codes is empty
- cross_quadrant_changes = 0
- official feedback from 781601 does not contradict the selected base route
```

If any condition fails, do not submit and write a hold note at:

```text
experiments/track2_v9_post_score_optimization_20260606/v11/hold_reason.md
```

### Task 5: Submission Handoff

**Files:**
- Read: `submissions/track2_submission_v11_v9_optimized_candidate.zip`
- Output: user-facing final instruction

- [ ] **Step 1: Present the exact ZIP path**

Use this exact wording:

```text
Upload this file only if you confirm we are spending one Codabench submission attempt:
/Users/yhryzy/dev/emoart-130k/submissions/track2_submission_v11_v9_optimized_candidate.zip
```

- [ ] **Step 2: Do not submit automatically**

Do not click upload, do not replace formal submission files, and do not submit another ZIP unless the user explicitly confirms the exact file path.
