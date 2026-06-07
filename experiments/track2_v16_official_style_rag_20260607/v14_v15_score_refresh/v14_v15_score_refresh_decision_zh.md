# Track2 v14/v15 Local Scorer Refresh Decision

## 结论

本轮重新跑分后，`v15_desc_expand300` 仍是当前本地最优候选；`v14_public_resource_agsr_max` 和 v14b 阈值候选不建议提交。

核心原因是：v14 系列只带来很小的分类 expected 提升，但 lower bound 下降。它没有提供足够证据说明可以从线上约 `0.84` 冲到 `0.89`。

## Scorer 使用方式

本地评分器应该继续基于已知官方反馈校准，但不能把它当成完全复刻的官方 hidden evaluator。

当前使用方式：

- 用官方锚点 `779605` 锁定基准：overall `0.836408`，classification `0.723150`，description `0.949667`。
- 用失败提交 `781601` 惩罚大批量 same-quadrant 改动，尤其 `calm/content` 边界批量互换。
- 用 fused scorer 给 VULCA/Judge++ description 风险改善一个小幅、封顶的加成，避免文本优化覆盖分类风险。

这意味着评分器适合作候选排序、风险过滤和提交前 gate；不适合宣称已经还原官方标准答案。

## Fresh Fused Ranking

| rank | candidate | overall expected | overall lower | class expected | desc expected | label changes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `v15_desc_expand300` | `0.839902` | `0.835854` | `0.723150` | `0.956655` | 0 |
| 2 | `v15_desc_reuse192` | `0.839496` | `0.835570` | `0.723150` | `0.955842` | 0 |
| 3 | `official_782683_v12_stable_probe` | `0.838104` | `0.834595` | `0.723150` | `0.953057` | 0 |
| 4 | `v14_description_max_safe` | `0.838104` | `0.834595` | `0.723150` | `0.953057` | 0 |
| 5 | `v14_public_resource_agsr_max` | `0.838178` | `0.833545` | `0.723300` | `0.953057` | 3 |
| 6 | `official_779605_moe_v2_anchor` | `0.836408` | `0.833408` | `0.723150` | `0.949667` | 0 |

## Fresh Classification-Only Shadow Ranking

纯分类/官方公式侧的 local shadow scorer 不计算 VULCA 文本加成，因此所有 text-only 候选和 anchor 持平。

| rank | candidate | overall expected | overall lower | class expected | label changes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `official_779605_moe_v2_anchor` | `0.836408` | `0.833408` | `0.723150` | 0 |
| 2 | `official_782683_v12_stable_probe` | `0.836408` | `0.833408` | `0.723150` | 0 |
| 3 | `v15_desc_expand300` | `0.836408` | `0.833408` | `0.723150` | 0 |
| 4 | `v14_public_resource_agsr_max` | `0.836483` | `0.832359` | `0.723300` | 3 |
| 5 | `v14b_sameq_clip095` | `0.836534` | `0.831284` | `0.723400` | 6 |
| 6 | `v14b_sameq_clip094` | `0.836508` | `0.824209` | `0.723350` | 23 |

## Decision

- 不提交 v14/v14b。
- 如果必须现在提交，优先 `v15_desc_expand300`，因为它是 description-only，分类风险最低。
- 真正冲 `0.89` 必须继续 v16：用官方失败批次校准后的 RAG teacher + 多 backbone 共识，只接受强证据分类改动。

## Fresh Output Paths

- Fused scorer report: `experiments/track2_v16_official_style_rag_20260607/v14_v15_score_refresh/fused_shadow_score_report.md`
- Classification-only scorer report: `experiments/track2_v16_official_style_rag_20260607/v14_v15_local_shadow_refresh/shadow_score_report.md`
- Current best local ZIP: `submissions/track2_submission_v15_desc_expand300_candidate.zip`
