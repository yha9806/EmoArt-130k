# Track2 `desc_192` 冲榜差距报告

## 结论优先

`v3_mid_gemini35_desc_192` 是目前最强的“文本增强 + 分类稳态”候选，但还不能在本地证明稳超第一名。

- 候选 JSON: `submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.json`
- 候选 ZIP: `submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.zip`
- Gemini 3.5 Flash 多模态审计样本: `192`
- guard 接受重写行: `39`
- 文本字段改动: `226`
- 分类标签改动: `0`
- Track2 validator: OK
- 本地 Description audit issue count: `0`
- 未覆盖正式 `submissions/track2_submission.json/.zip`

当前建议：**不要立刻提交**。它适合作为下一次线上 calibration 的候选，但如果目标是“本地先超过第一名再提交”，还需要 classification v5 或更可信的 Description proxy。

## 当前分数位置

最后一次线上可见基准：

| item | overall | classification | description |
| --- | ---: | ---: | ---: |
| 我们已提交 `779605` | `0.836408` | `0.723150` | `0.949667` |
| 榜首可见目标 | `0.89` | `0.78` | `1.00` |
| 本地 `v3_mid_gemini35_desc_192` proxy expected | `0.863908` | `0.778150` | `0.949667` |

注意：这里的 Description 是本地 proxy，不是官方 LLM-assisted multimodal evaluator。本地 proxy 只能确认“没有明显文本问题”，不能证明官方会给 1.0。

## 如果官方 Description 提升，会发生什么

以当前本地 classification expected `0.778150` 计算：

| assumed official description | implied overall |
| ---: | ---: |
| `0.949667` | `0.863908` |
| `0.970000` | `0.874075` |
| `0.980000` | `0.879075` |
| `0.990000` | `0.884075` |
| `1.000000` | `0.889075` |


即使官方 Description 达到满分，整体也约为 `0.889075`，刚好贴近 `0.89`，不是稳超。要稳超，至少需要以下组合之一：

| classification | description | overall |
| ---: | ---: | ---: |
| `0.780` | `1.000` | `0.890` |
| `0.790` | `0.990` | `0.890` |
| `0.800` | `0.980` | `0.890` |

所以现在的瓶颈不是 JSON 格式，而是两件事：

1. 官方 Description 是否会把 192 轮重写从 `0.949667` 拉到接近 `1.0`；
2. classification 是否能从本地 `0.778150` 再稳定提高到 `0.780+` 或 `0.790+`。

## 为什么 v4 同象限扩展没有继续提分

本地 shadow evaluator 对同象限改动设了保守上限：最多奖励 `+0.055` classification。`v3_mid` 的 85 个同象限改动已经达到这个上限，因此继续加入弱证据同象限改动，expected 不再上升，只会扩大不确定性。

这不是说同象限新增一定没用，而是说本地 proxy 已经无法证明它有收益。继续做同象限扩张，除非能拿到新的强证据，否则不应消耗线上提交。

## Gemini-agent 独立审阅意见

Gemini-agent 的计划审阅给出 `caution`：

- 最大盲区是 Description metric 没有被本地校准；
- 0 次线上校准会让我们无法知道官方 LLM 是否奖励这些文本重写；
- 它建议用 1 次提交做 `v3_mid_gemini35_desc_192` calibration，然后反推本地 Description proxy。

我对这个建议的处理：

- 不自动提交，因为你明确说还剩 4 次，要先本地冲到足够有把握；
- 把 `v3_mid_gemini35_desc_192` 标记为“可校准候选”，不是“稳超候选”；
- 继续本地做 classification v5 的高证据跨象限审查。

## 下一步门槛

下一轮不应该全量乱改，而是只处理 evidence matrix 里的高证据跨象限候选。

推荐 v5 放行阈值：

- `cross_quadrant_risk=True` 的候选默认 hold；
- 只有同时满足以下条件才可进入 v5 candidate：
  - `evidence_score >= 9`；
  - `supporting_family_count >= 5`；
  - `supporting_source_count >= 9`；
  - `gemini35_objection=False`；
  - 追加一轮 Gemini 3.5 Flash image-only/label-only 仲裁仍支持 proposed；
  - 不触发 public duplicate/reference 的反证。

按这个阈值，当前只有 `track2_0547 annoyed->calm` 进入第一优先审查；其他 score=7 的跨象限样本先留在 review queue，不自动放行。

## 当前推荐

- **保守线上候选**：`submissions/track2_submission_v3_safe_plus_candidate.zip`
- **冲榜/校准候选**：`submissions/track2_submission_v3_mid_gemini35_desc_192_candidate.zip`
- **当前不要提交**：除非我们决定用 1 次线上机会校准官方 Description 对 192 轮文本重写的真实奖励。
- **下一步本地工作**：生成并审查 v5 跨象限队列，优先检查 `track2_0547`，再决定是否构造 `v5_cross1` 候选。
