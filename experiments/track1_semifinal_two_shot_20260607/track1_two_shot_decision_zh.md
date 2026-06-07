# Track1 最后两次提交决策

## 结论

第一发建议提交：

`submissions/track1_submit_v2_safe_subset.zip`

原因不是它一定能第一，而是它是当前唯一“工程上干净、血缘正确、风险可解释”的半决赛包。它从 `v3_gate7/full1000` 高分家族出发，只替换 27 个经过 redteam 的 fallback 问题样本，不再重复最新 `hybrid_probe_redteam_fid_pass4` 的 old-current 血缘错误。

但必须明确：如果目标是第一名，`safe_subset` 大概率仍不够。我们和第一名的核心差距是 FID，不是 AAS。

## 为什么现在分数低

最新提交 `hybrid_probe_redteam_fid_pass4.zip` 的 AAS 很高，约 `0.99305`，但 FID 崩到了 `105.66`，Overall 只有 `0.73964`。

这说明：

- 图像逐张看起来更贴 prompt，不等于整包分布像官方参考艺术集。
- 最新失败包不是 v3/full1000 的微调，而是 `996` 张旧 current 加 `4` 张替换。
- 这个提交主要测到了旧 current 分布风险，不应该用来否定 v3/full1000 路线。

## 当前包血缘

| 包 | 与 v3 相同 | 与 full1000 相同 | 判断 |
| --- | ---: | ---: | --- |
| `v3_gate7` | 1000 | 993 | 已知官方锚点 |
| `full1000_no_fallback` | 993 | 1000 | v3 同家族 |
| `safe_subset` | 966 | 973 | 27 个高精 fallback 修复 |
| `strict_subset` | 970 | 977 | 23 个更保守 fallback 修复 |
| `hybrid_probe_redteam_fid_pass4` | 不属此家族 | 不属此家族 | 已证实风险线 |

## 第一名门槛

当前第一名：

- Overall `0.7969073906`
- FID `66.3621921022`
- FID Score `0.6010981145`
- AAS `0.9927166667`

若 AAS 是 `0.993`，要超过第一名，FID 约需 `<= 63.70`。

若 AAS 满分 `1.000`，FID 也仍需约 `<= 66.37`。

我们的 `v3_gate7` FID 约 `80.92`。所以只修几个 prompt alignment 或坏图，不足以第一。第一名需要整包分布进入 `66` 左右的 FID 区间。

## 本地校准投影

本地校准只有两个我们自己的官方锚点，置信度是 low/medium，只能用于风险控制，不能当官方分数。

| 包 | local fid_like | 预测/观测 FID | 预测/观测 Overall | 说明 |
| --- | ---: | ---: | ---: | --- |
| `safe_subset` | 63.308289 | 79.339175 | 0.769892 | 预测 |
| `strict_subset` | 63.287051 | 79.451891 | 0.769745 | 预测 |
| `full1000_no_fallback` | 63.098811 | 80.450932 | 0.768433 | 预测 |
| `v3_gate7` | 63.010429 | 80.920000 | 0.765000 | 官方锚点，四舍五入 |
| `hybrid_probe_redteam_fid_pass4` | 58.348904 | 105.660000 | 0.740000 | 官方锚点 |

所以 `safe_subset` 的意义是稳健测量，不是直接保证第一。

## 最后两发

### 第 4 次提交

提交：

`submissions/track1_submit_v2_safe_subset.zip`

用途：

- 测 v3-family + 27 个 fallback 修复的真实官方组件。
- 判断小范围 AAS 修复是否会保持或改善 FID。
- 为最后一发提供组件反馈。

### 第 5 次提交

不要提前锁死。根据第 4 次分数组件决定：

- 如果 `safe_subset` FID 到 `<= 75` 且 AAS `>= 0.99`，最后一发可以尝试更强但同家族的包。
- 如果 `safe_subset` FID 仍在 `79-81` 附近，最后一发不应该再做小修小补；必须先构建 FID-breakthrough 全量包。
- 如果 `safe_subset` FID 变差或 AAS 掉到 `< 0.985`，最后一发回到 `strict_subset` 或 v3-family 保守包。

## 接下来真正冲第一要做什么

继续离线做一个 FID-first 全量包：

1. 全 1000 样本使用官方 130k/reference 风格族检索，不是只找几个 reference。
2. 让 prompt adapter 保留自由度，避免所有海报变同一模板。
3. aspect ratio 由 caption 决定，不强制 1:1。
4. poster/scroll/album 只在 caption 要求时保留真实载体。
5. AAS gate 保留，但不让它把图像压成 rigid layout。
6. 做整包 style-family/FID proxy dashboard，而不是只看少数 contact sheet。

## 验证状态

已验证：

- `safe_subset` submission 文件 OK
- `strict_subset` submission 文件 OK
- `v3_gate7` submission 文件 OK
- `full1000_no_fallback` submission 文件 OK
- 四个 ZIP 均为 1000 JPG + `submission.json`
- ZIP 和对应目录哈希一致
- champion immutability 检查为 `0` 行 diff
