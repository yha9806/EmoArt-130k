# Track2 v15 Description Expand300 本地裁决

## 结论

`v15_desc_expand300` 是当前本地 fused shadow evaluator 下分数最高的低风险文本候选，但仍不能解释或接近 `0.89` 冠军分数。

推荐状态：`best_description_candidate_but_not_champion_solution`

不推荐把 v14/v14b 分类 public-reference 改动作为下一次提交主线；推荐把下一阶段重心转向 `classification v16`。

## 与现有候选对比

| candidate | local overall expected | local overall lower | class expected | desc expected | label changes | VULCA issues |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v15_desc_expand300 | 0.839902 | 0.835854 | 0.723150 | 0.956655 | 0 | 39 |
| v15_desc_reuse192 | 0.839496 | 0.835570 | 0.723150 | 0.955842 | 0 | 42 |
| official_782683_v12_stable_probe | 0.838104 | 0.834595 | 0.723150 | 0.953057 | 0 | 50 |
| official_779605_moe_v2_anchor | 0.836408 | 0.833408 | 0.723150 | 0.949667 | 0 | 62 |

解释：

- v15 expanded 相对 reuse192 的本地 expected 只增加约 `+0.000406`。
- 它确实继续降低了 VULCA/Judge++ 文本风险：候选问题数从 42 降到 39。
- 它没有改变 emotion/valence/arousal，因此 classification 仍卡在 `0.723150`。
- 如果 classification 不提升，即使 Description Score 达到 1.0，overall 上限也只有约 `0.861575`，仍不足以达到 `0.89`。

## 本轮 Gemini 扩展结果

- Source JSON: `submissions/track2_submission_v12_stable_probe_candidate.json`
- Existing decisions: 192
- New Gemini 3.5 Flash decisions: 300
- Combined unique decisions: 492
- Accepted changed rows: 72
- Accepted changed fields: 405
- Classification label changes: 0
- Validator: OK

主要改动类型：

- 删除或弱化图像不可直接验证的作者、地点、系列、时代和风格断言。
- 修正画面一致性问题，例如把误称的主体状态改回可见图像证据。
- 增加具体视觉证据，例如书法面板、版画线条、构图方向、主体姿态。
- 压缩冗长描述，保证每个字段是一句自然、具体、可核验的英文。

## 风险判断

`v15_desc_expand300` 相对 v12/reuse192 是更好的本地文本候选，但不是无风险提交：

- 本地评分器不是官方 hidden evaluator，只能做 proxy 排序。
- 它的 lower bound 仍未明显超过当前最佳官方 exact anchor。
- 官方 Description Score 已经给我们约 `0.95`，继续文本优化边际收益变小。
- 下一次提交如果目标是“冲第一”，文本-only 版本很可能只是小幅波动，不会产生 0.05 级别跳升。

## 提交策略

如果短期必须提交一个最稳的文本版：

- 首选 ZIP: `submissions/track2_submission_v15_desc_expand300_candidate.zip`
- 备选 ZIP: `submissions/track2_submission_v15_desc_reuse192_candidate.zip`
- 不提交 v14/v14b public-resource classification patch。

如果目标是冲 `0.89`：

- 不应把下一次有限提交机会消耗在 v15 expanded 上。
- 应先做 `classification v16`，目标是把 emotion accuracy proxy 从当前 `0.57` 风格拉向 `0.70+`，同时保持 emotion macro-F1 不崩。
- v15 expanded 可以作为 v16 的文本基底，因为它不改变分类，且本地 description proxy 最强。

## 下一步 v16 Gate

v16 classification candidate 在提交前必须同时满足：

- 基于官方已知提交的 calibration，不再使用 same-quadrant 数量作为收益代理。
- 相对 v15 expanded，分类改动必须有逐样本证据来源：公开训练分布风格、重复/近重复证据、多 backbone 一致预测、或官方提交反事实约束。
- `validate-track2 == OK`
- `classification_label_changes > 0` 时必须输出 transition ablation。
- 本地 fused scorer 必须显示：
  - `overall expected > v15_desc_expand300`
  - `overall lower >= v15_desc_expand300` 或有明确提交探测价值
  - `cross_quadrant_changes` 不出现无证据大批量漂移

本地评分报告：

- `experiments/track2_v15_description_max_20260607/expand300/fused_shadow_compare/fused_shadow_score_report.md`
- `experiments/track2_v15_description_max_20260607/expand300/rewrite_report.md`
