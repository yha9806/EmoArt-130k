# Track2 v15 Description Reuse192 本地裁决

## 结论

`v15_desc_reuse192` 是当前本地评分器下最强的低风险候选，但它不是 0.89 冲刺答案。

推荐状态：`candidate_for_low_risk_submission_or_expansion_base`

原因：

- 不改变 `emotion`、`emotional_valence`、`emotional_arousal_level`，所以不会重演 v9/v13/v14 这类分类冒进导致掉分的风险。
- 复用了已经由 Gemini 3.5 Flash multimodal 审过的 192 条 description decisions。
- 实际只接受 39 个样本、226 个文本字段修改，主要删除无图像证据的艺术史归因、作者/系列名、过强 iconography 解释，并提升文本字段的视觉落地性。
- 本地 fused shadow evaluator 排名第一，超过 v12 stable probe 和 779605 官方锚点。

## 本地评分

评分器：`track2_fused_shadow_score_v1_formula_plus_vulca_entailment`

| candidate | overall expected | overall lower | class expected | desc expected | label changes | cross quadrant |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v15_desc_reuse192 | 0.839496 | 0.835570 | 0.723150 | 0.955842 | 0 | 0 |
| official_782683_v12_stable_probe | 0.838104 | 0.834595 | 0.723150 | 0.953057 | 0 | 0 |
| official_779605_moe_v2_anchor | 0.836408 | 0.833408 | 0.723150 | 0.949667 | 0 | 0 |

解释：

- v15 的提升来自 description risk delta，不来自分类。
- 这说明本地 scorer 认为文本质量仍可继续挖，但分类分数没有被改善。
- 和第一名 0.89 的差距主要仍在 emotion accuracy / description 满分化，不是 JSON 格式或小规模 label patch。

## 安全阈值

本版本的安全阈值是：

- `classification_label_changes == 0`
- `cross_quadrant_changes == 0`
- `validate-track2 == OK`
- 只接受 `confidence >= 0.74` 且三个 description 维度分数都 `>= 0.72` 的 rewrite
- 禁止 evaluator manipulation 文本
- 禁止新增无图像证据的作者、系列、地点、历史事实或 iconography 断言

## 为什么 v14 不是当前最优

v14 public-resource AGSR 做的是分类标签修正。它能找到一些 public reference 风格/近似证据，但本地评分器显示：

- 分类改动幅度太小，不能支撑冲 0.89。
- 放宽阈值后容易重复 781601 的失败模式：same-quadrant 改动看似安全，但官方 hidden gold 未必按 public reference label 风格打。
- 即使在 v12 上叠加 v14，`overall expected` 只有轻微提升，`overall lower` 反而弱于 v12。

所以 v14 结论是 `hold/no-go`，不能作为下一次有限提交机会的主文件。

## 下一步

优先继续做 v15 expansion：

1. 以 v12 stable probe 为文本基底。
2. 对未覆盖的高风险 description 样本继续跑 Gemini 3.5 Flash multimodal rewrite。
3. 每轮 apply 后重新跑：
   - `validate-track2`
   - fused shadow evaluator
   - VULCA/Judge++ description risk
4. 只有当新版本同时满足：
   - `overall expected > v15_desc_reuse192`
   - `overall lower >= v15_desc_reuse192`
   - `classification_label_changes == 0`
   - 文本风险未新增
   才能替代 v15 reuse192。

当前候选文件：

- JSON: `submissions/track2_submission_v15_desc_reuse192_candidate.json`
- ZIP: `submissions/track2_submission_v15_desc_reuse192_candidate.zip`
- 本地评分报告: `experiments/track2_v15_description_max_20260607/reuse_192/fused_shadow_compare/fused_shadow_score_report.md`
