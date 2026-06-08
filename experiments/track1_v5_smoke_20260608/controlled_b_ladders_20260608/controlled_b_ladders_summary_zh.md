# Track1 Controlled-B Ladder Summary

- 基底：`submissions/track1_candidate_v3_gate7_20260606/`
- 策略：1000 张完整提交包，仅替换 selected 样本；排除与旧 current 完全相同的候选图。
- Candidate pool: `experiments/track1_v5_smoke_20260608/controlled_b_ladders_20260608/controlled_b_candidate_pool.json`

| package | replacements | same_as_current | same_as_v3 | same_as_full1000 | zip MB | sha256 |
|---|---:|---:|---:|---:|---:|---|
| `submissions/track1_candidate_v5_controlled_b80_on_v3_20260608.zip` | 80 | 0 | 920 | 920 | 423.05 | `d8832362083449bfcd38e0ab5ba35f96a2d33470d4a0bead663a34870e2dfb30` |
| `submissions/track1_candidate_v5_controlled_b120_on_v3_20260608.zip` | 120 | 0 | 880 | 880 | 423.16 | `50aca7fc2f8f0602334ee9788e606f32543a6ae7912ff5958e9b5f8654cb0435` |
| `submissions/track1_candidate_v5_controlled_b160_on_v3_20260608.zip` | 160 | 0 | 840 | 840 | 423.16 | `60183237fc17929d79be8da50ade44ae3f0f49f7c592ef54cd68d6a59ffe44a1` |

## Source Counts

### B80
- `pilot100_human24_package_diff`: `8`
- `safe_subset_package_diff`: `24`
- `v5_human_accept48`: `48`

### B120
- `human_v25_wave2_package_diff`: `2`
- `pilot100_human24_package_diff`: `32`
- `safe_subset_package_diff`: `24`
- `v4_distribution_review_queue`: `14`
- `v5_human_accept48`: `48`

### B160
- `human_v25_wave2_package_diff`: `2`
- `pilot100_human24_package_diff`: `32`
- `safe_subset_package_diff`: `24`
- `v4_distribution_review_queue`: `54`
- `v5_human_accept48`: `48`
