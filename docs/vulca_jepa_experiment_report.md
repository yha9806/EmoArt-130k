# Vulca JEPA Experiment Report

## Summary

- Row count: 1000
- Model/current disagreements: 441
- Content-to-calm candidates: 181
- Model/KNN disagreements: 330

## Smoke Runs

These runs use a tiny Track2 slice (`Abstract Art`, 12 train examples, 3 test examples) to check local compatibility and qualitative behavior. They are not leaderboard estimates.

| Run | Wall seconds | Test distribution | Classifier/KNN agreement |
| --- | ---: | --- | ---: |
| I-JEPA ViT-H/16 | 23.3 | `content`: 1, `sad`: 2 | 0.0 |
| DINOv2 base | 13.5 | `calm`: 1, `content`: 1, `tired`: 1 | 0.3333 |
| SigLIP2 base patch16 224 | 12.2 | `bored`: 2, `tired`: 1 | 0.0 |

## Full I-JEPA Gate

Full I-JEPA Track2 was not run in this local pass. The smoke run completed without encoder failures and was under the 3x DINOv2 smoke ratio, but extrapolating the I-JEPA smoke throughput to the full 132,885-train plus 1,000-test Track2 run gives an optimistic estimate of about 58 hours on this machine. I-JEPA remains a research baseline for structure auditing, not a challenge submission backbone.

## Track1 Generation Risk

The Track1 audit integration now scores generated-image rows for high-style, low-content risk. A row is marked high risk when multiple evidence signals fire, including strong style score with weak caption fidelity, weak structure preservation, and low caption fidelity. This is intended to catch Vulca over-stylization cases where cultural style is strong but the literal caption or subject structure is under-preserved.

## Final Verification

- Focused tests passed: JEPA registry, Vulca JEPA audit, and Track2 embedding script guard.
- Full unit suite passed: 86 tests.
- Track2 submission JSON validation passed.
- Track1 submission ZIP validation passed after extracting the ZIP and checking `submission.json` plus packaged image files.
- Track1 and Track2 ZIP layouts include `submission.json`.

## Review Samples

- `track2_0998` priority=1.45 content -> excited
- `track2_0996` priority=1.45 content -> calm
- `track2_0994` priority=1.45 content -> calm
- `track2_0982` priority=1.45 aroused -> excited
- `track2_0975` priority=1.45 content -> calm
- `track2_0966` priority=1.45 content -> calm
- `track2_0956` priority=1.45 content -> calm
- `track2_0942` priority=1.45 content -> calm
- `track2_0941` priority=1.45 content -> calm
- `track2_0937` priority=1.45 content -> calm
- `track2_0928` priority=1.45 content -> calm
- `track2_0914` priority=1.45 content -> calm
- `track2_0908` priority=1.45 content -> calm
- `track2_0904` priority=1.45 content -> calm
- `track2_0899` priority=1.45 content -> calm
- `track2_0880` priority=1.45 content -> calm
- `track2_0871` priority=1.45 content -> calm
- `track2_0869` priority=1.45 content -> calm
- `track2_0866` priority=1.45 content -> calm
- `track2_0864` priority=1.45 content -> calm
- `track2_0842` priority=1.45 annoyed -> calm
- `track2_0841` priority=1.45 content -> calm
- `track2_0832` priority=1.45 content -> calm
- `track2_0830` priority=1.45 content -> calm
- `track2_0809` priority=1.45 content -> calm
- `track2_0807` priority=1.45 content -> calm
- `track2_0806` priority=1.45 annoyed -> calm
- `track2_0800` priority=1.45 content -> calm
- `track2_0798` priority=1.45 content -> calm
- `track2_0795` priority=1.45 content -> calm
- `track2_0794` priority=1.45 content -> calm
- `track2_0791` priority=1.45 content -> calm
- `track2_0781` priority=1.45 content -> calm
- `track2_0779` priority=1.45 content -> excited
- `track2_0773` priority=1.45 content -> calm
- `track2_0765` priority=1.45 content -> calm
- `track2_0761` priority=1.45 content -> calm
- `track2_0751` priority=1.45 content -> calm
- `track2_0748` priority=1.45 content -> calm
- `track2_0746` priority=1.45 content -> calm

## Backbone Registry

- `clip-vit-b-32`: ViT-B/32 (image, clip) - Fast baseline; useful for leakage audit and retrieval sanity checks.
- `siglip2-base-patch16-224`: google/siglip2-base-patch16-224 (image, siglip) - Current strongest Track2 single backbone in local holdout.
- `dinov2-base`: facebook/dinov2-base (image, dino) - Good structural vision baseline; weaker than SigLIP2 on Track2 emotion labels.
- `ijepa-vith16-1k`: facebook/ijepa_vith16_1k (image, jepa) - ViT-H 448px I-JEPA; smoke works but local full Track2 is too slow on MPS.
- `ijepa-vith14-1k`: facebook/ijepa_vith14_1k (image, jepa) - ViT-H 224px I-JEPA; lower resolution, not lower parameter count.
- `vjepa2-vitl`: facebook/vjepa2-vitl-fpc64-256 (video, jepa) - Video model; repeated still frames are not a meaningful Track2 signal.
