# Track1-only 官方锚点与本地评分器一致性审计

日期：2026-06-08

结论先行：**当前 Track1 本地评分器不能完全复现我们提交包的官方分数**。它只能做到两件事：

1. 已知官方组件分数时，`Overall = (FID Score + AAS) / 2` 可以精确复现。
2. 已知官方 FID 时，`FID -> FID Score` 的线性近似很接近，但不是完全一致。

不能做到的是：只凭本地 `fid_like` / `proxy` 就稳定预测官方 FID。两个自有官方锚点已经证明本地 `fid_like` 和官方 FID 方向相反，不能直接当冲榜评分器。

## 官方规则锚点

Track1 官方页面/邮件给出的评分结构：

| 项目 | 含义 | 权重 |
|---|---|---:|
| FID | 生成图集合与 reference artistic image set 的分布相似度，越低越好 | 50% |
| AAS | multimodal LLM-assisted evaluation，聚合 content/style/attribute alignment | 50% |
| Content Alignment | 图像是否符合 caption 主体内容 | AAS 子项 |
| Style Alignment | 是否符合艺术风格 | AAS 子项 |
| Attribute Alignment | 是否符合构图、笔触、颜色、线条、光照等属性 | AAS 子项 |

官方总分复现公式：

```text
Track1 Overall = (FID Score + AAS) / 2
```

## 自有官方提交锚点

### Anchor A：`track1_submit_v3_gate7_20260606.zip`

这是当前较强的自有官方锚点。页面显示 rounded `0.77`，按组件精确计算为 `0.765`。

| 字段 | 数值 |
|---|---:|
| submission_id | `782831` |
| local_package | `v3_gate7` |
| 官方 Overall 页面值 | `0.77` |
| 官方 Overall 组件精确值 | `0.765` |
| 本地公式复现 Overall | `0.765` |
| 公式误差 | `0.0` |
| 官方 FID | `80.92` |
| 官方 FID Score | `0.55` |
| 官方 AAS | `0.98` |
| Content Alignment | `0.98` |
| Style Alignment | `0.98` |
| Attribute Alignment | `0.98` |
| 本地 `fid_like` | `63.010429` |

内联判断：这个包 AAS 很强，但 FID 距离第一梯队仍然明显。它不是“内容失败”的包，而是“分布/FID 不够像官方 reference set”的包。

### Anchor B：`hybrid_probe_redteam_fid_pass4.zip`

这是一次 FID 探针提交，理论上本地 `fid_like` 更好，但官方 FID 明显变差。

| 字段 | 数值 |
|---|---:|
| submission_id | `784403` |
| local_package | `hybrid_redteam_fid_pass4` |
| 官方 Overall | `0.7396402011` |
| 本地公式复现 Overall | `0.7396402011` |
| 公式误差 | `5e-11` |
| 官方 FID | `105.6638160234` |
| 官方 FID Score | `0.4862304023` |
| 官方 AAS | `0.99305` |
| Content Alignment | `0.992` |
| Style Alignment | `0.9939` |
| Attribute Alignment | `0.99325` |
| 本地 `fid_like` | `58.348904` |

内联判断：这个包证明了一个反直觉事实：**本地 `fid_like` 从 `63.010429` 降到 `58.348904`，官方 FID 却从 `80.92` 恶化到 `105.663816`**。所以本地 FID proxy 不是官方 FID 的可靠替代品。

## 与第一梯队差距

当前公开/抓取到的 Track1 第一梯队：

| participant | Overall | FID | FID Score | AAS |
|---|---:|---:|---:|---:|
| `emosuis` | `0.7969073906` | `66.3621921022` | `0.6010981145` | `0.9927166667` |
| `trybest-1` | `0.7964185295` | `66.8417413832` | `0.5993703924` | `0.9934666667` |

我们的差距：

| 我们的包 | Overall | 距第一 | FID | FID Score | AAS |
|---|---:|---:|---:|---:|---:|
| `v3_gate7` | `0.765` | `-0.031907` | `80.92` | `0.55` | `0.98` |
| `hybrid_redteam_fid_pass4` | `0.7396402011` | `-0.057267` | `105.6638160234` | `0.4862304023` | `0.99305` |

内联判断：第一名不是靠 AAS 极端高赢的。第一名 AAS 约 `0.993`，我们 hybrid 也到 `0.99305`，但 FID 崩了。因此 Track1 冲第一的关键不是继续只堆 prompt 内容对齐，而是把 FID 拉到 `66-70` 区间，同时保持 AAS 不低于约 `0.99`。

## 评分器复现状态

### 官方总分公式复现

| 项目 | 数值 |
|---|---:|
| 样本数 | `13` |
| 最大绝对误差 | `5e-11` |
| 平均绝对误差 | `1.9e-11` |

结论：**完全可用**。只要官方返回了 `FID Score` 和 `AAS`，我们就能精确复现 `Overall`。

### 官方 FID 到 FID Score 近似

| 项目 | 数值 |
|---|---:|
| 样本数 | `13` |
| 线性模型 | `fid_score = 0.76613901 - 0.00260548 * fid` |
| R2 | `0.992139` |
| RMSE | `0.00493274` |
| MAE | `0.00448162` |
| 最大残差 | `0.00791356` |

结论：**近似可用，但不是官方精确归一化器**。它适合估算差距，不适合决定最后两次提交。

### 本地 `fid_like` 到官方 FID

| 项目 | 数值 |
|---|---:|
| 自有锚点数 | `2` |
| readiness | `insufficient_own_anchors` |
| proxy_direction | `anti_correlated` |
| 线性拟合 | `official_fid = 415.38532256 - 5.30809467 * local_fid_like` |
| R2 | `1.0` |

结论：**不可用作真实评分器**。`R2=1.0` 是两个点强行拟合出来的假象，且方向已经反了。至少需要第三个自有官方提交锚点，并且要保留本地包特征和官方组件分数。

## 本地候选包状态

### v4/full1000 FID-oriented 包

| 包 | 本地状态 | 关键数值 | 结论 |
|---|---|---:|---|
| `partial757` | 757 张新图 + 243 current | local `fid_like=62.414763` | 官方未知；本地 proxy 不可信 |
| `full1000_no_fallback` | 1000 张新图 | local `fid_like=63.098811` | 官方未知；可能更接近 v3，但无法确认 |
| `hybrid_redteam_fid_pass4` | 4 张替换 | local `fid_like=58.348904` | 已官方验证失败，FID 恶化 |
| `greedy_top80_raw20` | 20 张替换 | local `fid_like=57.697893` | 高风险；同类 proxy 已被官方打脸 |

### v5 controlled-B / smoke 包

| 包 | placeholder/code-like 健康检查 | 官方分数状态 | 结论 |
|---|---:|---|---|
| `current` | `0` known placeholder | current baseline local only | 包体健康，但不是官方最高 |
| `v3` | `2` known placeholder | `782831` 官方约 `0.765` | 已知强 anchor，但有 placeholder 风险 |
| `b80` | `0` known placeholder | 未提交/无官方组件 | 当前最干净候选之一 |
| `b120` | `0` known placeholder | 未提交/无官方组件 | 候选，但风险比 b80 高 |
| `b160` | `0` known placeholder | 未提交/无官方组件 | 候选，但替换更多，AAS/FID 不确定性更大 |

内联判断：v5 controlled-B 的意义不是“评分器已经证明会涨分”，而是它解决了 v3 中 `0730/0735/0740/0772` 这类 code-like/placeholder 问题，并且包体健康检查更干净。它是“下一次探针”的合理候选，不是已证明冠军包。

## 为什么现在不能说评分器和官方完全一致

不能完全一致的原因不是公式错，而是我们缺少官方 hidden reference evaluator：

1. 官方 FID 使用它们的 hidden reference artistic image set 和固定实现；我们只能用本地 reference/proxy。
2. 官方 AAS 使用固定 multimodal LLM protocol；我们不知道完整 prompt、rubric、inference settings。
3. 本地 `fid_like` 曾经把 `hybrid_redteam_fid_pass4` 排得更好，但官方 FID 实际更差。
4. 目前 Track1 自有官方锚点只有两个，不足以校准一个可信预测器。

## 当前决策建议

最后两次提交不能继续用“本地 FID-like 最低”作为唯一目标。

推荐策略：

1. 下一次提交用 **v5 controlled-B 小到中等替换包**，优先 `b80_fixed` 或 `accept48/b80` 这种 placeholder 清零、AAS 风险低的包。
2. 等官方返回后，把该提交作为第三个自有 anchor，更新 Track1 scorer。
3. 最后一枪根据第三个 anchor 判断：如果 controlled-B 提高，扩大到 `b120/b160`；如果下降，回退到 v3-style 稳健包，只做最小 placeholder 修复。

硬约束：

1. 不再提交基于 `greedy_top80_raw20` 或纯本地 FID-like 贪心的包，除非额外通过人工和 reference 分布审计。
2. 不把 `p1_light_proxy` 当官方分数，只当包体健康/placeholder 检查。
3. 任何最终包必须重新跑 `validate-track1`，并确认 `submissions/track1_submission.json`、`submissions/track1_submission.zip`、`submissions/track1/images/` 未被修改。
