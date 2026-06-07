# Track2 v16 Top48 RAG Gate Decision

## 结论

`v16_rag_top48_conf085` 暂不建议提交。

它验证了 v16 RAG teacher/apply/scorer 闭环可以工作，但没有提供足够分类增益。当前最优本地候选仍是 `v15_desc_expand300`。

## 本轮结果

- RAG teacher 决策数: 48
- 接受 label changes: 3
- 跨象限变化: 0
- label consistency issues: 0
- validator: OK
- 主要接受变化:
  - `track2_0280`: `content -> calm`
  - `track2_0756`: `content -> calm`
  - `track2_0946`: `content -> calm`

拒绝原因:

- `kept_current`: 22
- `low_confidence`: 23

## 本地评分

Fused scorer:

| rank | candidate | overall expected | overall lower | class expected | desc expected | label changes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `v15_desc_expand300` | `0.839902` | `0.835854` | `0.723150` | `0.956655` | 0 |
| 2 | `v16_rag_top6_conf085` | `0.839952` | `0.835154` | `0.723250` | `0.956655` | 2 |
| 3 | `v16_rag_top48_conf085` | `0.839977` | `0.834804` | `0.723300` | `0.956655` | 3 |

Classification-only scorer:

| rank | candidate | overall expected | overall lower | class expected | label changes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `official_779605_moe_v2_anchor` | `0.836408` | `0.833408` | `0.723150` | 0 |
| 2 | `v15_desc_expand300` | `0.836408` | `0.833408` | `0.723150` | 0 |
| 3 | `v16_rag_top6_conf085` | `0.836458` | `0.832709` | `0.723250` | 2 |
| 4 | `v16_rag_top48_conf085` | `0.836483` | `0.832359` | `0.723300` | 3 |

## 为什么不提交

从 top6 扩展到 top48 后，只新增 1 个可接受变化。expected 微升，但 lower 继续下降。这说明当前队列过度集中在 `content -> calm`，而 Gemini teacher 对这些边界多数给出低置信或保持 current。

这条路径无法解释从线上约 `0.84` 到 `0.89` 的差距。

## 下一步

继续 v16，但需要重做 queue 策略：

- 降低 `content -> calm` 在 queue 中的优先级，避免重复线上失败批次。
- 主动挖少数类和高混淆边界：
  - `frustrated / alarmed / aroused / excited`
  - `glad / happy / content`
  - `bored / tired / sad`
- public reference 只用于 exact/near-duplicate 或明确同作品证据，不再用风格相似批量转移。
- scoring gate 继续以 lower bound 优先；没有超过 `v15_desc_expand300` lower 的分类候选不消耗提交机会。
