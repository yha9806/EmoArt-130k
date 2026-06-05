# Track2 本地冲榜差距报告

## 结论

本轮已完成 score-calibrated champion pass 的第一版本地闭环：

- 构造了 evidence matrix；
- 生成了 `v3_safe_plus`、`v3_mid`、`v3_push` 三档旁路候选；
- 生成了 JSON/ZIP/Markdown/HTML review；
- 跑过 Track2 validator；
- 跑过本地 shadow evaluator。

但是，当前本地分数还没有达到“稳超第一名”的阈值。

## 当前线上目标

Codabench 当前可见第一名：

- Overall: `0.89`
- Classification: `0.78`
- Description: `1.00`

我们的已提交官方分数：

- Overall: `0.836408`
- Classification: `0.723150`
- Description: `0.949667`

## 本轮 v3 候选结果

| candidate | changed rows | cross quadrant | shadow classification | shadow description | shadow overall expected | shadow lower | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `v3_safe_plus` | 45 | 0 | `0.777150` | `0.949667` | `0.863409` | `0.840159` | recommend_submit |
| `v3_mid` | 85 | 0 | `0.778150` | `0.949667` | `0.863909` | `0.822659` | recommend_submit |
| `v3_push` | 116 | 17 | `0.739950` | `0.949667` | `0.844809` | `0.687609` | recommend_hold |

本地第一选择是 `v3_safe_plus`，不是 `v3_mid`，因为 `v3_mid` 虽然 expected 略高，但 lower bound 明显更低。

## 为什么还不能提交

如果 Description Score 不变，`v3_safe_plus` 和 `v3_mid` 都只能到约 `0.864`，离第一名 `0.89` 仍有明显差距。

即使假设 Description Score 达到满分：

- `v3_safe_plus`: `0.5 * 0.777150 + 0.5 * 1.0 = 0.888575`
- `v3_mid`: `0.5 * 0.778150 + 0.5 * 1.0 = 0.889075`

这仍然只是贴近 `0.89`，不是稳超。要稳超第一，本地候选至少需要满足其中之一：

- Classification `>= 0.785` 且 Description `>= 1.00`；
- Classification `>= 0.790` 且 Description `>= 0.99`；
- Classification `>= 0.800` 且 Description `>= 0.98`。

## 本轮完成的技术工作

新增本地工具：

- `affectiveart/track2_score_calibrated_champion.py`
- `scripts/track2_score_calibrated_champion.py`
- `tests/test_track2_score_calibrated_champion.py`

核心能力：

- 多候选证据聚合；
- 同象限/跨象限风险识别；
- 支持来源和来源家族计数；
- public-style、rollback、human gate 证据接口；
- 三档候选 ladder；
- formal submission 路径保护；
- candidate JSON/ZIP 输出；
- evidence matrix CSV/JSON；
- HTML review。

## 下一步

不要提交当前 v3。下一步应继续本地冲分：

1. **Description 满分化审计**
   对 `v3_safe_plus` 和 `v3_mid` 的 changed rows 做 Gemini/Vulca 图文一致性审计，找出可能拖 Description 的 caption/attribute 字段。

2. **Classification 再提升**
   当前 shadow classification 已接近 `0.778`，但需要推到 `0.785+` 才能在 Description 满分时稳超 `0.89`。

3. **构造 v4**
   不是扩大到 cross-quadrant，而是在同象限中继续找少量高收益行，尤其：
   - `calm/content/glad` 边界；
   - `happy/excited/aroused` 边界；
   - `sad/tired/bored` 边界。

4. **提交门槛**
   只有当本地报告同时满足以下条件，才建议消耗下一次 Codabench 提交：
   - shadow overall expected `>= 0.890`;
   - shadow lower bound `>= 0.845`;
   - validator OK;
   - label consistency issues = 0;
   - missing emotions = none;
   - Description audit 没有系统性风险；
   - 不覆盖正式 `submissions/track2_submission.json/.zip`。
