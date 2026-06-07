# Track2 v16 Non-Content Hard-Case Gate Decision

## 结论

`v16_rag_non_content_top24_conf085` 不产生可提交增益。

这轮专门避开了过度集中的 `content -> calm` 队列，抽取 24 个非 `content -> calm` hard cases，覆盖少数类和高唤醒边界。但 Gemini 3.5 Flash 视觉仲裁没有给出任何达到 `min-confidence=0.85` 的 label change。

## 本轮结果

- RAG teacher 决策数: 24
- 接受 label changes: 0
- 跨象限变化: 0
- label consistency issues: 0
- validator: OK

主要观察:

- `aroused -> excited`: 低置信，未接受。
- `content -> excited`: Gemini 多数保持 `content`。
- `tired/sad/annoyed/alarmed -> calm`: Gemini 多数保持 current 或低置信。
- `happy/excited/bored -> calm`: 低置信，未接受。

## 含义

当前的 Gemini-RAG 仲裁不是 0.89 的主要杠杆。

原因不是工具链不可用，而是候选证据不够强：

- 对 `content/calm`，teacher 大多低置信或保持 current。
- 对非 `content/calm` hard cases，teacher 也没有支持强改动。
- 因此继续按现有 queue 烧 Gemini 只会增加成本，不会显著提高 classification。

## 下一步建议

要继续冲第一，下一步不应再扩大当前 teacher 队列，而应换分类策略：

- 重新训练/校准一个 classification model 或 stacked head，目标不是 image caption reasoning，而是拟合官方 12-way 标注风格。
- 用 public/validation 风格分层做 per-emotion calibration，特别处理 `frustrated/alarmed/aroused/excited` 和 `glad/happy/content`。
- 用官方三次提交反馈更新 local scorer 的 transition priors，但避免 overfit 到同象限微调。
- Description Score 继续保留 `v15_desc_expand300` 作为文本底座；分类候选只有在 lower bound 超过 v15 时才考虑提交。
