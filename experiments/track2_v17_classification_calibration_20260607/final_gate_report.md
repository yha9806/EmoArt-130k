# Track2 v17 Final Gate

- Decision: `hold_v17_keep_v15`
- Candidate: `v15_desc_expand300`
- Submission policy: `no_auto_submit`
- Caveat: This is a local/fused shadow gate, not official Codabench scoring; no auto-submit.
- Baseline: `v15_desc_expand300` overall_lower=0.835854
- Best v17: `v17_safe` overall_lower=0.777300
- Reason: Best v17 candidate v17_safe overall_lower 0.777300 does not strictly exceed baseline v15_desc_expand300 overall_lower 0.835854.

## Fused Ranking

| Candidate | Overall lower | Overall expected | Classification lower | Description lower | Shadow decision |
| --- | ---: | ---: | ---: | ---: | --- |
| v15_desc_expand300 | 0.835854 | 0.839902 | 0.723150 | 0.948558 | recommend_submit |
| official_782683_v12_stable_probe | 0.834595 | 0.838104 | 0.723150 | 0.946040 | recommend_submit |
| official_779605_moe_v2_anchor | 0.833408 | 0.836408 | 0.723150 | 0.943667 | recommend_submit |
| v17_safe | 0.777300 | 0.819682 | 0.609150 | 0.945450 | recommend_hold |
| v17_balanced | 0.729175 | 0.810182 | 0.512900 | 0.945450 | recommend_hold |
| v17_aggressive_probe | 0.532637 | 0.807962 | 0.123650 | 0.941625 | recommend_hold |
