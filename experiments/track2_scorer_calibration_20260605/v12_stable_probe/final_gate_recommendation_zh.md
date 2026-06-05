# Track2 v12 Stable Probe Final Gate

## 结论

- 推荐候选：`v12_stable_probe`
- JSON: `submissions/track2_submission_v12_stable_probe_candidate.json`
- ZIP: `submissions/track2_submission_v12_stable_probe_candidate.zip`
- 决策：`low_risk_description_probe`
- 不建议再提交：`v9_vulca_entailment` 或任何 85 行同象限大批量改标签版本。

## 为什么选 v12_stable_probe

`v12_stable_probe` 保留 779605 anchor 的全部分类标签，只从 `public_style_review11_vulca` 复制 12 行开放文本字段。

- 分类标签变化：`0`
- 文本变化：`12` 行、`12` 个字段
- validator: `OK`
- ZIP 内部结构：`submission.json`
- formal submission overwritten: `False`

本地 fused shadow evaluator 排名：

| rank | candidate | overall expected | overall lower | classification expected | description expected | label changes | VULCA issues |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `v12_stable_probe` | `0.838104` | `0.834595` | `0.723150` | `0.953057` | `0` | `50` |
| 2 | `moe_v2_anchor` | `0.836408` | `0.833408` | `0.723150` | `0.949667` | `0` | `62` |
| 3 | `public_style_review11_vulca` | `0.838129` | `0.828620` | `0.723200` | `0.953057` | `16` | `50` |

## Scarce Submission Gate

通过项：

- classification expected 不依赖 same-quadrant 数量加分。
- description expected >= `0.945`。
- cross-quadrant changes = `0`。
- label changes = `0`。
- safety issue codes 为空。
- VULCA risk 改善：issues `62 -> 50`，mean risk `0.02525 -> 0.02105`。

保留风险：

- overall lower = `0.834595`，低于 779605 官方 overall `0.836408`。
- 因此它不是“严格 lower-bound 已确认超过当前线上 anchor”的候选。
- 它适合作为低风险 description probe：如果官方 Description Score 对 VULCA 文本修复敏感，可能小幅超过 779605；如果官方 evaluator 不吃这 12 行文本修复，分数大概率接近 779605。

## v9 Ablation 结论

校准后的分类 scorer 不再奖励 85 个同象限批量改标签。

| candidate | label changes | local overall expected | local overall lower | 结论 |
| --- | ---: | ---: | ---: | --- |
| `v12_content_to_calm_only` | `20` | `0.836909` | `0.826408` | lower 太低，不适合直接提交 |
| `v12_public_duplicate_only` | `16` | `0.836434` | `0.827433` | 有诊断价值，但不如 description-only 稳 |
| `v12_human_high_confidence_only` | `1` | `0.836359` | `0.832983` | 变化太少，预期收益不足 |
| `v12_no_calm_to_content` | `42` | `0.836444` | `0.812744` | 批量风险高 |
| `v12_multisource_consensus_only` | `37` | `0.834734` | `0.814034` | 包含大量 `calm->content`，风险高 |
| `v9_vulca_entailment` | `85` | `0.836909` | `0.780357` | block / hold |

## 下一次提交建议

如果要花一次 Codabench 机会做低风险线上 probe，上传：

`/Users/yhryzy/dev/emoart-130k/submissions/track2_submission_v12_stable_probe_candidate.zip`

如果策略是“只提交本地 lower bound 明确超过 779605 的候选”，则暂不提交，继续做更强的 classification v13。

## 验证

- `python3 -m affectiveart.challenge validate-track2 submissions/track2_submission_v12_stable_probe_candidate.json`: `OK`
- 所有 v12 ablation JSON validate-track2: `OK`
- `python3 -m unittest discover -s tests -p 'test_track2*.py' -v`: `233 tests OK`
- `git diff --check`: pass
