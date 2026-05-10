# Vulca JEPA 实验报告

## 结论摘要

这轮实验的目标不是让 JEPA 直接替代当前的 Vulca / Gemini / SigLIP 工作流，而是验证 JEPA 风格的视觉表征在 Vulca 里能不能提供额外的审计信号，尤其是结构保真、主体保留、风格过强导致内容丢失这几类问题。

当前结果比较明确：

- I-JEPA 可以在本机跑通，但在 Track2 情绪分类上不适合作为主干模型。
- I-JEPA 在 medium slice 上比 DINOv2 慢约 5.3 倍，且 holdout macro-F1 很低。
- SigLIP2 仍然是这批对照里更强的语义 / 情绪侧基线。
- DINOv2 更适合作为结构视觉基线，而不是情绪分类主模型。
- JEPA/DINO/SigLIP 的最大价值是产生分歧样本，帮助人工或后续 VLM 审计，而不是自动改 submission。

报告产物：

- `experiments/vulca_jepa_audit/report.json`
- `experiments/vulca_jepa_smoke/manifest.json`
- `experiments/vulca_jepa_medium/manifest.json`
- `experiments/vulca_jepa_medium/comparison.json`
- `experiments/vulca_jepa_medium/review_candidates.json`

## Track2 审计概览

基于已有的 Track2 ensemble strict predictions，Vulca JEPA audit CLI 生成了 1000 行审计摘要。

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| 总样本数 | 1000 | 当前 Track2 测试集审计覆盖量 |
| 模型预测与当前标签不一致 | 441 | 候选复核池，不等于模型一定正确 |
| `content -> calm` 候选 | 181 | 当前标签是 `content`，模型更倾向 `calm` 的样本 |
| 分类器与 KNN 不一致 | 330 | 说明模型分类头和邻居证据存在分歧 |

这部分的实用意义是：它给出了一个“先看哪里”的排序，而不是直接生成新提交包。尤其是 `content -> calm` 的候选很多，说明当前标签里可能存在一批“画面情绪很安静，但被标成 content”的样本，需要结合 caption 和图像人工判断。

## Tiny Smoke 实验

第一轮 smoke 只用非常小的 Track2 切片：

- style filter：`Abstract Art`
- train examples：12
- test examples：3
- device：MPS
- batch size：4

这个实验只回答两个问题：

1. 模型能不能在本机完整加载、编码、训练分类器、写出 predictions / metrics。
2. 三类模型在极小样本上是否会立刻报错或出现明显不兼容。

| 模型 | 运行时间 | 测试预测分布 | 分类器/KNN 一致率 |
| --- | ---: | --- | ---: |
| I-JEPA ViT-H/16 | 23.3s | `content`: 1, `sad`: 2 | 0.0 |
| DINOv2 base | 13.5s | `calm`: 1, `content`: 1, `tired`: 1 | 0.3333 |
| SigLIP2 base patch16 224 | 12.2s | `bored`: 2, `tired`: 1 | 0.0 |

smoke 结论：

- 三个模型都能跑通，没有 encoder failure。
- I-JEPA 明显更慢，但 tiny smoke 受模型加载开销影响很大，不能直接作为 full-run 判断。
- tiny smoke 太小，预测分布没有统计意义，只能作为兼容性检查。

## Medium 实验

第二轮 medium 实验扩大到：

- style filter：`Abstract Art`
- train examples：120
- test examples：30
- device：MPS
- batch size：4

这个规模仍然是 bounded diagnostic，不是 leaderboard estimate，但已经足够比较运行成本、holdout macro-F1、预测分布和模型间分歧。

| 模型 | 运行时间 | Holdout macro-F1 | 测试预测分布 | 与当前标签一致 |
| --- | ---: | ---: | --- | ---: |
| I-JEPA ViT-H/16 | 92.4s | 0.0364 | `calm`: 14, `bored`: 6, 其他: 10 | 8/30 |
| DINOv2 base | 17.4s | 0.1186 | `content`: 11, `bored`: 8, 其他: 11 | 9/30 |
| SigLIP2 base patch16 224 | 16.3s | 0.1753 | `annoyed`: 6, `alarmed`: 5, `aroused`: 5, `bored`: 5, `calm`: 5, 其他: 4 | 3/30 |

medium 结论：

- SigLIP2 在这个 bounded slice 上 holdout macro-F1 最高。
- DINOv2 比 I-JEPA 更快，macro-F1 也更好。
- I-JEPA 的预测明显偏向 `calm`，在 30 个测试样本里预测了 14 个 `calm`。
- I-JEPA 的 medium runtime 是 DINOv2 的约 5.3 倍，超过 full-run gate 的 3x 阈值。
- I-JEPA 当前更适合保留为结构审计 research baseline，不适合进入 Track2 full candidate。

## 模型分歧分析

在 30 个共同测试样本上，三个模型的标签一致情况如下：

| 一致情况 | 样本数 |
| --- | ---: |
| 三个模型全一致 | 1 |
| 两个模型一致 | 18 |
| 三个模型全不同 | 11 |

这个结果说明：不同视觉表征看到的是不同信号。对 Vulca 来说，这反而有用。我们不应该把某一个模型当成裁判，而应该把“多模型一致但不同于当前标签”的样本作为人工复核或 VLM 二次审计入口。

## Medium 复核候选

`experiments/vulca_jepa_medium/review_candidates.json` 中记录了 13 个候选样本。这些样本满足：

- I-JEPA、DINOv2、SigLIP2 中至少两个模型给出同一标签。
- 这个共识标签不同于当前 submission 标签。

最强候选是 `track2_0015`：当前标签是 `calm`，三个模型都预测 `bored`。

| 样本 | 当前标签 | 共识标签 | 支持数 | I-JEPA | DINOv2 | SigLIP2 |
| --- | --- | --- | ---: | --- | --- | --- |
| `track2_0015` | calm | bored | 3 | bored | bored | bored |
| `track2_0001` | sad | bored | 2 | annoyed | bored | bored |
| `track2_0003` | excited | content | 2 | content | content | bored |
| `track2_0005` | calm | bored | 2 | bored | calm | bored |
| `track2_0007` | excited | calm | 2 | frustrated | calm | calm |
| `track2_0009` | content | calm | 2 | calm | bored | calm |

这些候选不应该直接覆盖提交标签。建议下一步对这些样本做图像级人工检查，重点看：

- 当前标签是否来自 caption 语义，而模型是否只看视觉氛围。
- `bored` / `calm` / `content` 之间是否存在标签定义边界模糊。
- SigLIP2 与 DINOv2 同意时，是否比 I-JEPA 单独意见更可信。
- I-JEPA 与 DINOv2 同意时，是否代表结构或场景层面的相似性。

## Full I-JEPA Gate

本轮没有跑 full I-JEPA Track2。

原因不是 I-JEPA 不能运行，而是 medium 结果已经足够让 full-run gate 失败：

- I-JEPA medium：92.4s
- DINOv2 medium：17.4s
- I-JEPA / DINOv2 runtime ratio：约 5.3x
- gate 规则：只有 I-JEPA 小规模运行低于 DINOv2 的 3x，才考虑 full run

此外，I-JEPA medium holdout macro-F1 只有 0.0364。即使忽略速度，它也没有显示出作为 Track2 情绪分类主干的价值。

结论：

- 不在本机跑 full I-JEPA。
- 不把 I-JEPA 纳入 challenge submission candidate。
- 保留 I-JEPA 作为研究型结构审计信号。

## Track1 生成图风险

Track1 侧新增了生成图风险评分函数：`score_track1_generation_risk`。

它关注的是 Vulca 生成图里很常见的一类问题：文化风格很强，但用户请求的具体内容、主体结构或 caption fidelity 不够稳定。

风险信号包括：

- style score 很高，但 caption fidelity 低。
- structure score 低。
- caption fidelity 本身低。

如果多个风险信号同时出现，样本会被标成 high risk。这个逻辑适合用来找 Vulca over-stylization，也就是“画得很像某种传统风格，但没有准确完成用户要求”的情况。

## 对 Vulca 的产品判断

这轮实验对 Vulca 的意义可以拆成三层。

第一层：情绪分类主干。

- SigLIP2 仍然更适合做 Track2 情绪侧主干。
- DINOv2 可以作为结构视觉对照。
- I-JEPA 暂时不适合作为本地 full Track2 分类主干。

第二层：结构和内容审计。

- JEPA / DINO 类模型更适合看视觉结构、主体保持、构图相似性。
- 如果 Vulca 生成图风格很对，但结构或主体丢失，可以用这类模型做辅助审计。
- 当前实验还没有证明 I-JEPA 比 DINOv2 更好，因此实际产品优先级应是 DINOv2 > I-JEPA。

第三层：分歧驱动复核。

- 多模型一致但不同于当前标签的样本，适合作为人工复核候选。
- 模型之间全不同的样本，适合作为“标签边界模糊”或“caption/image 信号冲突”候选。
- 对 Vulca 来说，这比追求单模型分数更有产品价值。

## 建议下一步

建议下一轮不要扩大 I-JEPA full run，而是做更贴近 Vulca 的审计闭环：

1. 对 `experiments/vulca_jepa_medium/review_candidates.json` 的 13 个候选样本做人工图像检查。
2. 选出 5-10 个明显案例，写成“当前标签 vs 多模型共识 vs 人工判断”的小表。
3. 对 Track1 生成图补一批真实 `caption_fidelity_score`、`style_score`、`structure_score`，让 `score_track1_generation_risk` 不只停留在函数测试。
4. 如果要继续测试 JEPA，优先试更快或更小的 image JEPA 变体；不要重复 still image 到 V-JEPA video frames 来冒充视频实验。
5. 把 DINOv2 作为默认结构审计 baseline，I-JEPA 作为可选 research mode。

## 最终验证

本轮验证结果：

- JEPA registry 聚焦测试通过。
- Vulca JEPA audit 聚焦测试通过。
- Track2 embedding script guard 聚焦测试通过。
- 全量 unittest 通过：86 tests。
- Track2 submission JSON validation 通过。
- Track1 submission ZIP 解压后，`submission.json` 和 packaged image files 检查通过。
- Track1 / Track2 ZIP 中都包含 `submission.json`。

## Backbone Registry 摘要

| 名称 | 模型 ID | 类型 | 本轮判断 |
| --- | --- | --- | --- |
| `clip-vit-b-32` | `ViT-B/32` | image / CLIP | 快速 baseline，可用于 sanity check |
| `siglip2-base-patch16-224` | `google/siglip2-base-patch16-224` | image / SigLIP | 当前更适合 Track2 语义和情绪侧 |
| `dinov2-base` | `facebook/dinov2-base` | image / DINO | 适合结构视觉审计 |
| `ijepa-vith16-1k` | `facebook/ijepa_vith16_1k` | image / JEPA | 可跑，但本地慢，分类弱 |
| `ijepa-vith14-1k` | `facebook/ijepa_vith14_1k` | image / JEPA | 低输入分辨率，不是小参数模型 |
| `vjepa2-vitl` | `facebook/vjepa2-vitl-fpc64-256` | video / JEPA | 不应用 repeated still frames 评估 Track2 |
