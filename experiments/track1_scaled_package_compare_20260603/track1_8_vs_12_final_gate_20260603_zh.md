# Track1 8-sample vs 12-sample Final Gate

日期：2026-06-03

## 结论

推荐使用 `expanded_12` 作为当前 Track1 冲奖候选包：

- 包目录：`submissions/track1_candidate_scaled_expanded_20260603/`
- ZIP：`submissions/track1_candidate_scaled_expanded_20260603.zip`
- 角色：当前推荐提交候选

保守 fallback 是 `high_precision_8`：

- 包目录：`submissions/track1_candidate_scaled_high_precision_20260602/`
- 角色：如果最后一分钟极度担心额外 4 个替换的模型评分偏差，则回退到它。

原因：官方邮件明确 Track1 最终为 `FID 50% + AAS 50%`，没有 LPIPS/current-similarity 项。`expanded_12` 覆盖更多已知坏点，人工/专家 gate 接受 12 个替换；同时两轮本地 Inception FID-like sanity 都显示 `expanded_12` 没有 FID-like 退化，反而略优于 `high_precision_8` 和 base。

重要限制：本地 Inception FID-like sanity 使用 512 维随机投影和 torchvision 预处理，不是 clean-fid，也不是官方 hidden FID。它只能说明“没有观察到明显 FID 风险”，不能声明官方 FID 一定更好。最终推荐 `expanded_12` 的主依据是 AAS/人评收益覆盖更多问题样本，FID-like sanity 只是风险排除证据。

## 官方评分口径

官方邮件给出的 Track1 评分：

- FID：生成图像集合与 reference artistic image set 的分布级相似度，越低越好，最后会转成分数。
- AAS：每张图和 caption 的 prompt-level alignment。
- AAS 三项：Content Alignment、Style Alignment、Attribute Alignment。
- 最终分数：FID 50% + AAS 50%。

所以本轮 final gate 不能把“和 current 是否相似”当作主目标。current 只是我们的基准，不是官方 reference，也不是官方 AAS judge。

## 对比对象

基准包：

- `submissions/track1_candidate_human_v21_plus0816_20260514/`

候选 A：`high_precision_8`

- 替换 8 个样本：`0151`, `0193`, `0235`, `0534`, `0686`, `0694`, `0709`, `0802`

候选 B：`expanded_12`

- 包含 `high_precision_8` 的 8 个替换。
- 额外替换 4 个样本：`0708`, `0769`, `0816`, `0971`

用户 sign-off：

- 接受：`0708`, `0816`, `0971`, `0769`
- 保留 current：`0665`, `0803`, `0593`
- rerun/不进包：`0881`

## 代理评估结果

### 1. SigLIP + expert metric proxy

使用匹配基准包的 SigLIP embedding、130k reference embedding、expert manifest。

| package | changed | accepted by proxy | changed mean delta | changed min delta |
|---|---:|---:|---:|---:|
| `high_precision_8` | 8 | 8 | 0.0682 | 0.0647 |
| `expanded_12` | 12 | 12 | 0.0686 | 0.0647 |

解释：在加入 expert/human accept 信号后，两包全部通过；`expanded_12` 覆盖更多样本，平均 delta 略高，最差 delta 不变。

### 2. SigLIP score sweep 压力测试

`score_sweep_siglip2_base` 推荐 `high_precision_8`。

这个结果要保留为风险提示，但不能作为主判据，原因是：

- 它不使用 expert_proxy。
- 它加入了 perceptual/current-similarity 项。
- 官方邮件没有 LPIPS/current-similarity 项。
- 对任何“明显修坏点但与 current 不同”的替换，它都会天然更保守。

它提示的真实风险是：`0708`, `0769`, `0971` 在 SigLIP distribution 上是轻微负 delta，官方 LLM-AAS 若不认可人眼看到的修复，额外 4 个替换可能收益变小。

### 3. Inception FID-like sanity

新增本地 InceptionV3 降维 Fréchet sanity check：

- 参考集：`Socialist Realism` + `Social Realism`
- 特征：torchvision InceptionV3 pool feature，固定随机投影到 512 维后计算 Fréchet distance。
- 注意：这不是官方 hidden FID，也不是 clean-fid。512 维投影会改变原始 2048 维 Inception 空间；该结果只用于相对 sanity，不用于声称官方 FID 数值。

512 reference 结果：

| package | fid_like |
|---|---:|
| `expanded_12` | 82.361616 |
| `high_precision_8` | 82.379847 |
| `base` | 82.406180 |

1024 reference 复核：

| package | fid_like |
|---|---:|
| `expanded_12` | 69.052224 |
| `high_precision_8` | 69.061162 |
| `base` | 69.078378 |

解释：两轮方向一致，`expanded_12` 没有表现出 FID-like 退化。差距很小，不能夸大为“FID 一定更高分”，但足以解除“多换 4 张会明显伤 FID”的主要担忧。

## 新增 4 个样本的风险拆分

| sample | decision | SigLIP distribution delta | final gate |
|---|---|---:|---|
| `track1_0708` | accept | -0.0071 | 进 `expanded_12`，AAS/人评收益优先 |
| `track1_0769` | accept | -0.0156 | 进 `expanded_12`，但属于额外 4 张里较保守风险项 |
| `track1_0816` | accept | +0.0166 | 进 `expanded_12`，FID-like 与 AAS 都支持 |
| `track1_0971` | accept | -0.0131 | 进 `expanded_12`，但属于额外 4 张里较保守风险项 |

`0769` 和 `0971` 是最需要最后视觉复核的两个额外替换；但它们已被人工 sign-off 接受，且包级 Inception FID-like 没有变差。

## Final Gate 建议

采用双轨提交策略：

1. 主推冲奖包：`track1_candidate_scaled_expanded_20260603.zip`
2. 保守 fallback：`track1_candidate_scaled_high_precision_20260602.zip`

在没有新的视觉 veto 前，推荐提交主推冲奖包。理由是官方 AAS 占 50%，我们现在最确定的收益来自修复 caption alignment、构图/逻辑、文字/海报表面等问题；FID-like sanity 没有显示 12-sample 包有分布退化。

不建议继续把 `0665`, `0803`, `0593`, `0881` 临时塞进最终包：

- `0665`, `0803`, `0593` 已由人工保留 current。
- `0881` 被标记 rerun，不能进入 final package。
- 最后阶段扩大替换数量的边际收益不如风险控制。

## 还缺什么

提交前只剩机械 gate：

- 再跑一次 Track1 validate。
- 确认 `submissions/track1_submission.json`, `submissions/track1_submission.zip`, `submissions/track1/images/` 没被改。
- 记录最终 ZIP SHA256。
- 上传时使用 `track1_candidate_scaled_expanded_20260603.zip`，不要误传 fallback 包。
