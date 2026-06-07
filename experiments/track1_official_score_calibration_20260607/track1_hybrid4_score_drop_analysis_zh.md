# Track1 Hybrid4 Official Score Drop Analysis

## 结论优先

这次 `hybrid_probe_redteam_fid_pass4.zip` 官方分数变低，根因不是 AAS 失败，而是 FID 失败。

| package | submission id | overall | FID | FID Score | AAS | content | style | attribute |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `track1_submit_v3_gate7_20260606.zip` | `782831` | `0.77` | `80.92` | `0.55` | `0.98` | `0.98` | `0.98` | `0.98` |
| `hybrid_probe_redteam_fid_pass4.zip` | `784403` | `0.74` | `105.66` | `0.49` | `0.99` | `0.99` | `0.99` | `0.99` |

Delta:

- Overall: `-0.03`
- FID: `+24.74`，显著变差
- FID Score: `-0.06`
- AAS: `+0.01`

所以这次结果说明：官方 AAS judge 认可这一类图的 prompt alignment，但官方 FID 不认可旧 current 基底的分布。

## 直接原因

`hybrid_probe_redteam_fid_pass4` 不是基于官方已知高分的 `v3_gate7` 包做 4 张替换，而是基于旧 `submissions/track1/images` current 包做 4 张替换。

真实图像哈希谱系：

| package | lineage |
| --- | --- |
| `hybrid_probe_redteam_fid_pass4` | `996` 张等同旧 current，`4` 张替换 |
| `v3_gate7` | `993` 张等同 `full1000_no_fallback`，`7` 张 gate 替换 |
| `safe_subset` | `966` 张 `v3/full1000` 相同，`7` 张 full，`27` 张新替换 |
| `strict_subset` | `970` 张 `v3/full1000` 相同，`7` 张 full，`23` 张新替换 |

这意味着：这次官方分数主要测到的是旧 current 基底的 FID 风险，不是 4 张替换本身。

## 校准器修正

已把 `784403` 加入 `track1_official_score_anchors.csv` 并重跑校准。

新报告：

- `track1_shadow_score_calibration_v3_with_hybrid4_official.json`
- `track1_shadow_score_calibration_v3_with_hybrid4_official_zh.md`

关键诊断：

- own official anchors: `2`
- local proxy direction: `anti_correlated`
- 旧 local Inception FID-like 在 `current` vs `v3/full1000` 之间方向错了

因此，旧报告里“current local fid_like 最好”不能再作为提交依据。后续必须把 `v3_gate7/full1000` 族作为已知官方高分基底。

## 后续提交策略

当前已知最稳回滚点：

- `submissions/track1_submit_v3_gate7_20260606.zip`
- 官方：overall `0.77`, FID `80.92`, AAS `0.98`

下一次 probe 不应该再从旧 `submissions/track1/images` 出发。候选应只在 `v3_gate7/full1000` 族上做小规模、可解释替换。

优先级：

1. `track1_submit_v2_safe_subset.zip`
   - 基底接近 `v3/full1000`
   - 27 张新增替换
   - 新校准预计官方 FID 接近或略优于 `v3_gate7`
   - 仍需承认模型只有 2 个 own anchors，置信度是 `low_medium`

2. `track1_submit_v2_strict_subset.zip`
   - 比 safe 少 4 张替换
   - 更保守，预计略低于 safe，但 AAS 风险也更低

3. 不再提交 `hybrid_redteam5`、`greedy_top80_raw20` 或任何以旧 current 为 996 张基底的包。

## 新硬规则

后续 Track1 每个提交包必须先生成 package lineage audit：

- 与 `v3_gate7` 相同多少张
- 与 `full1000_no_fallback` 相同多少张
- 与旧 current 相同多少张
- 有多少张是新替换

如果一个候选包大量回到旧 current 基底，必须默认 FID 高风险，除非官方 probe 证明相反。
