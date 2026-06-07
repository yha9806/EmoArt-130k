# Track2 v16 Top6 RAG Gate Decision

## 结论

`v16_rag_top6_conf085` 暂不建议提交。它是一个有效的分类 probe，但不是比 `v15_desc_expand300` 更稳的冲榜包。

## 本轮做了什么

- 从 v16 RAG queue 中只取最高证据的 6 个 `allow_strong_consensus` 样本。
- 使用 Gemini 3.5 Flash 做视觉仲裁。
- 用 `min-confidence=0.85` 生成旁路候选：
  - `submissions/track2_submission_v16_rag_top6_conf085_candidate.json`
  - `submissions/track2_submission_v16_rag_top6_conf085_candidate.zip`

## Gemini Top6 决策结果

- 接受 label changes: 2
- 被拒绝:
  - `kept_current`: 2
  - `low_confidence`: 2
- 接受变化:
  - `track2_0756`: `content -> calm`
  - `track2_0946`: `content -> calm`
- 跨象限变化: 0
- label consistency issues: 0
- validator: OK

## 本地评分结果

Fused scorer:

| rank | candidate | overall expected | overall lower | class expected | desc expected | label changes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `v15_desc_expand300` | `0.839902` | `0.835854` | `0.723150` | `0.956655` | 0 |
| 2 | `v16_rag_top6_conf085` | `0.839952` | `0.835154` | `0.723250` | `0.956655` | 2 |

Classification-only scorer:

| rank | candidate | overall expected | overall lower | class expected | label changes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `official_779605_moe_v2_anchor` | `0.836408` | `0.833408` | `0.723150` | 0 |
| 2 | `v15_desc_expand300` | `0.836408` | `0.833408` | `0.723150` | 0 |
| 3 | `v16_rag_top6_conf085` | `0.836458` | `0.832709` | `0.723250` | 2 |

## Decision

Hold `v16_rag_top6_conf085`.

Reason: expected score improves only `+0.00005` over v15, while lower bound drops by about `0.00070`. With only a few remaining Codabench submissions, this is not enough evidence to spend a submission.

## Next

To make v16 worth submitting, we need one of these:

- More accepted changes with stronger teacher confidence, without repeating the failed bulk calm/content pattern.
- Evidence from exact/near duplicate references that bypasses the same-quadrant batch penalty.
- A better classification scorer that identifies non-calm/content hard cases where official emotion accuracy is likely wrong in our current anchor.
