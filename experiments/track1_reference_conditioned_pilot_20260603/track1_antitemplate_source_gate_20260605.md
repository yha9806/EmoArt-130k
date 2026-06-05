# Track1 Anti-Template Source Gate - 2026-06-05

## Conclusion

Source-only gate passed. The anti-template planning layer is ready for a small image-generation smoke run, but it has not modified the current champion package and has not generated replacement images.

Do not build a replacement package yet.

## Verified Outputs

- Distribution routes: `experiments/track1_reference_conditioned_pilot_20260603/distribution_router_antitemplate_20260605/track1_distribution_routes.json`
- Prompt packets: `experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/track1_moe_prompt_packets.jsonl`
- Prompt lint: `experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/prompt_lint.json`
- Prompt review HTML: `experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/track1_strategy_prompt_review_zh.html`

Dry-run summary:

- Routes: 1000 samples.
- Candidate prompt budget: 28 packets.
- Covered dry-run samples: 10.
- Strategy counts: `aas_safe=10`, `reference_style=9`, `fid_diverse=9`.
- Prompt lint status: `pass`.
- Failed template phrases: none.
- Generated image files in dry-run output dirs: 0.

Important interpretation: this is a 28-packet dry budget, not 28 samples x 3 strategies. For a full Top28 three-strategy generation packet, use an 84-packet budget or adjust packet selection semantics.

## Verification Commands

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

Result: `69 passed, 33 subtests passed`.

```bash
git diff --check
```

Result: passed.

```bash
git diff -- submissions/track1_submission.json submissions/track1_submission.zip submissions/track1/images | wc -l
```

Result: `0`.

```bash
rg -n "sample_id|review_metadata|candidate_strategy|recommended_model|fallback_model|queue_score|provider_prompt_path|track1_|submissions/|\\{\\\"|\\\"sample_id\\\"|\\\"family_id\\\"" \
  experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top
```

Result: no matches.

## Gemini-Agent Review

Gemini-agent returned `verdict=caution`.

Potential blockers it raised:

- metadata might leak into provider prompts;
- champion package immutability might be violated through shared package mutation;
- FID-diverse should not be claimed as empirically proven before image generation.

Local adjudication:

- Metadata leakage was checked directly against the 28 provider prompt files; no routing metadata, sample IDs, model names, submission paths, or raw JSON keys were found.
- Champion package immutability was verified by protected diff result `0`; this source-only pipeline writes only under `experiments/`.
- The diversity claim remains a hypothesis only. The next step is a small generation smoke, followed by distribution audit and human visual review.

## Next Gate

Run a 10-sample smoke using the 20260605 prompt packets, then audit and review visually. Proceed to larger Top28 or 84-packet generation only if the smoke improves variety without losing required content, text, support surfaces, or relation logic.
