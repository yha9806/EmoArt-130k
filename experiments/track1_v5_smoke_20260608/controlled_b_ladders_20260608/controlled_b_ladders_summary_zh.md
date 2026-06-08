# Track1 Controlled-B Ladder Summary

- 基底：`submissions/track1_candidate_v3_gate7_20260606/`
- 策略：1000 张完整提交包，仅替换 selected 样本；过滤 provider/JSON 占位图。
- current 回流策略：默认排除与旧 current 完全相同的候选图；仅 `track1_0735`、`track1_0740` 允许作为 v3 占位图救援。
- Candidate pool: `experiments/track1_v5_smoke_20260608/controlled_b_ladders_20260608/controlled_b_candidate_pool.json`

| package | replacements | same_as_current | same_as_v3 | same_as_full1000 | zip MB | sha256 |
|---|---:|---:|---:|---:|---:|---|
| `submissions/track1_candidate_v5_controlled_b80_on_v3_20260608.zip` | 80 | 0 | 920 | 918 | 424.31 | `6073f71db4d5c02829b8d40230d8db25b045bdf4743c6c0f2a3e2a2ba4c41f6f` |
| `submissions/track1_candidate_v5_controlled_b120_on_v3_20260608.zip` | 120 | 0 | 880 | 878 | 424.42 | `4b031f48e5b364063cd9f5d40dc8ac2b733c0d643fa8f151fedc07f63ab9df5d` |
| `submissions/track1_candidate_v5_controlled_b160_on_v3_20260608.zip` | 160 | 0 | 840 | 838 | 424.42 | `286499a7837ff5f195c109d7d42fd90269fd1e1c411562f81b3cd6a54cdadf6c` |

## Source Counts

### B80
- `pilot100_human24_package_diff`: `10`
- `safe_subset_package_diff`: `22`
- `v5_human_accept48`: `48`

### B120
- `human_v25_wave2_package_diff`: `2`
- `pilot100_human24_package_diff`: `34`
- `safe_subset_package_diff`: `22`
- `v4_distribution_review_queue`: `14`
- `v5_human_accept48`: `48`

### B160
- `human_v25_wave2_package_diff`: `2`
- `pilot100_human24_package_diff`: `34`
- `safe_subset_package_diff`: `22`
- `v4_distribution_review_queue`: `54`
- `v5_human_accept48`: `48`
