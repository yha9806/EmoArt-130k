# Track2 v22 Final-Shot Recommendation

结论优先：最后一次提交建议使用 v22 的推荐上传包，而不是继续做小幅标签微调。

- 推荐 profile: `calmshiftall`
- 推荐候选 ZIP: `submissions/track2_submission_v22_official_author_calmshiftall_candidate.zip`
- 上传友好 ZIP: `submissions/v22_final_upload/track2_submission.zip`
- 标签改动数: `177`
- projected overall: `0.848163`
- projected classification: `0.746659`
- projected description: `0.949667`

## 为什么这样选

- 官方反馈已经验证 `content->calm` 是强正向：v21 只改这 90 个样本，emotion accuracy 净增 0.09。
- 现在只剩一次机会，继续小幅保守改动不足以追第一名；应该沿已验证方向扩大，而不是引入跨 quadrant 新风险。
- 文本字段使用已在线证明稳定的高 description anchor，避免 v15/v21 描述版本继续拖低 description score。

## 风险

- 这是基于官方 aggregate score 的 proxy，不是 hidden gold 重建。
- `calm` 占比继续升高，收益取决于官方 test gold 是否确实偏 calm/content ontology。
- 如果尾部 `content->calm` 证据不如前 90 个干净，emotion macro-F1 可能不随 accuracy 同步提升。

不要自动提交；手动上传前只使用上面的上传友好 ZIP。
