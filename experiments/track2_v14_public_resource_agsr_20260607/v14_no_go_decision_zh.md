# Track2 v14/v14b No-Go Decision

## 结论

当前 `v14_public_resource_agsr_max`、`v14_description_max_safe` 以及 v14b 阈值扫描候选都不建议作为下一次 Codabench 提交。

原因很直接：它们没有在本地 fused shadow evaluator 中超过当前最强稳定锚点 `official_782683_v12_stable_probe`，也没有产生足够大的分类变化来解释从 `0.84` 冲到 `0.89` 的路径。

## 当前最强锚点

`official_782683_v12_stable_probe` 仍是本地 fused ranking 第一：

- overall expected: `0.838104`
- overall lower: `0.834595`
- classification expected: `0.723150`
- description expected: `0.953057`
- changed labels: `0`

## v14 on 779605

`v14_public_resource_agsr_max` 从 `779605` 起步，只接受 3 个 near-same-work `content -> calm`：

- `track2_0210`
- `track2_0555`
- `track2_0869`

本地 fused score：

- overall expected: `0.836483`
- overall lower: `0.832359`
- classification expected: `0.723300`
- description expected: `0.949667`
- changed labels: `3`

它没有超过 `779605` 的 lower bound，也没有超过 v12。

## v14 on v12

从 v12 文本底座起步后，`v14_on_v12_public_resource_agsr_max` 仍然只改同样 3 个标签：

- overall expected: `0.838178`
- overall lower: `0.833545`
- classification expected: `0.723300`
- description expected: `0.953057`
- changed labels: `3`

它 expected 略高于 v12，但 lower bound 低于 v12。因此不值得消耗提交次数。

## v14b 阈值扫描

阈值扫描结果：

| candidate | changed labels | main transitions | overall expected | overall lower |
| --- | ---: | --- | ---: | ---: |
| `v14b_sameq_clip095` | 6 | `content->calm:5`, `frustrated->alarmed:1` | `0.836534` | `0.831284` |
| `v14b_sameq_clip094` | 23 | `content->calm:16`, `calm->content:6`, `frustrated->alarmed:1` | `0.836509` | `0.824209` |
| `v14b_sameq_clip093` | 48 | `content->calm:40`, `calm->content:6`, `frustrated->alarmed:1`, `glad->calm:1` | `0.837069` | `0.809769` |

放宽阈值会带来更多同象限变化，但本地 lower bound 快速下降。这和此前 `781601` 的线上失败模式一致：同象限变化不等于官方 12-way emotion 加分。

## 本地评分器状态

本地 scorer 的历史校验通过：

- `official_best=779605`
- `shadow_must_not_prefer=781601`
- `historical_alignment=pass`
- `spearman_proxy=1.0`

这不代表本地 scorer 等于官方 scorer，只说明它已经避免了之前“错误偏好 781601”的问题。

## 决策

当前推荐：

- `hold_all_v14_candidates`
- 不提交 v14/v14b。
- 下一步切换到更大收益路径：
  - `v15_description_max`: 目标把 Description 从约 `0.95` 推近 `0.99-1.00`。
  - `v15_classification_calibration`: 目标不是少量 public exact transfer，而是重建 majority-class emotion accuracy 校准。

如果必须立刻提交，最合理的仍是当前 v12 稳态包，而不是 v14/v14b。
