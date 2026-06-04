# Track2 MoE Specialist Dry-Run Handoff Decision

日期：2026-06-04

## 结论

本轮 dry-run 不生成 Track2 candidate JSON/ZIP，也不覆盖任何正式提交包。

原因很直接：56 个 clean/inclusive disagreement 样本全部被 gate 判为 `hold`，没有任何 `accept_change`。这不是失败，而是高精度 gate 的预期行为。clean 和 inclusive 都来自 SigLIP2 logreg 同一模型家族，本轮已经显式标注为 `siglip2_logreg`，不能作为两个独立证据源互相背书。

当前默认安全提交包仍是：

- `submissions/final_track2_20260602_scaled_human_gate_gemini35_guarded_v1.json`
- `submissions/final_track2_20260602_scaled_human_gate_gemini35_guarded_v1.zip`

## 本轮结果

- 样本数：56
- `accept_change`：0
- `hold`：56
- `keep_current`：0
- 缺失 current sample：0
- 写正式提交包：否
- 写 candidate JSON：否
- 写 candidate ZIP：否

报告路径：

- JSON：`experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/track2_moe_specialist_dry_run_report.json`
- Markdown：`experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/track2_moe_specialist_dry_run_report.md`
- HTML：`experiments/track2_moe_specialist_ensemble_20260603/dry_run_v1/html_review/track2_moe_specialist_dry_run_review.html`

## 为什么全部 hold

本轮队列来自 clean/inclusive 的 56 个分歧样本。也就是说，输入本来就是“同一 SigLIP2 家族的 clean 版本和 inclusive 版本不同意”的样本。

gate 的安全阈值要求：

- 至少两个独立模型家族支持同一个 proposed label；或
- 一个高置信 specialist source 支持，且没有强反对证据。

本轮只有 `siglip2_logreg` 一个模型家族，因此没有任何 proposed label 达到自动放行门槛。

## 风险结构

主要转移模式：

- `content -> calm`：27
- `calm -> content`：8
- 其他转移多为单例，包括 `calm -> frustrated`、`calm -> annoyed`、`annoyed -> content`、`aroused -> calm` 等。

VA 风险：

- 35 个是 `Positive/Low -> Positive/Low`，主要是 calm/content 边界。
- 20 个会改变 valence 或 arousal，其中包含 `Positive/Low -> Negative/High`、`Negative/High -> Positive/Low` 等高风险翻转。

所以不能把这 56 个自动并入 final manifest。尤其 VA 翻转会同时影响 emotion、valence、arousal 三项分类分，错误成本很高。

## Final Allowlist 安全阈值

下一版可以放行的改标必须满足以下任一条件：

1. 两个独立模型家族同意同一个 proposed emotion，并且每个支持证据达到最低质量门槛：
   - confidence >= 0.50
   - margin >= 0.12
   - 没有高置信 current-label 或其他-label 反对
   - 没有 description contradiction
   - high-similarity public reference 样本必须有显式复核

2. 一个 specialist source 高置信支持：
   - role 属于 `boundary`、`tail`、`va`、`description`、`specialist`
   - confidence >= 0.86
   - margin >= 0.12
   - 没有 strong opposition

不能放行的情况：

- clean 和 inclusive 都来自 `siglip2_logreg`，即使同意也只能算一个模型家族。
- 单个 global model 的建议不能直接改 final label。
- VA 翻转必须有 VA specialist 或 VLM/文本一致性证据支持。
- high-similarity public reference 不能自动照搬 public 标签，必须保留显式复核门槛。

## 下一步

为了继续提分，下一轮不应该继续只比较 clean/inclusive。需要新增至少一个真正独立的 evidence source：

1. `boundary` specialist：专门处理 `calm/content/glad` 边界。
2. `va` specialist：专门处理 `Positive/Low`、`Positive/High`、`Negative/High`、`Negative/Low` 象限翻转。
3. `description` specialist：用 Gemini/Vulca/vulca-emnlp 做图像-文本一致性判断，重点检查 proposed label 是否和 caption/attribute 自洽。
4. 异构视觉模型：CLIP、DINOv2 或其他不同 backbone 的分类头，只作为独立支持源，不单独决定 final label。

推荐下一轮执行：

先生成一个 `boundary + va + description` 三源 expert JSON，再用本 dry-run gate 复跑 56 个 disagreement 样本。只有出现 `accept_change > 0` 且没有 VA/description 风险时，才考虑写旁路 candidate，例如 `submissions/track2_submission_moe_specialist_candidate_*.json/.zip`。

## 生成最终 Track2 Submission 还缺什么

- 至少一个独立 specialist/VLM/异构模型 evidence source。
- 对所有 `accept_change` 样本同步更新 emotion、valence、arousal。
- 对所有改标样本做 Description Score rewrite/audit，保证 caption 和 attribute 不和新 emotion 冲突。
- 运行 Track2 validator 和 focused tests。
- 只写旁路 candidate，不覆盖现有正式提交包。
