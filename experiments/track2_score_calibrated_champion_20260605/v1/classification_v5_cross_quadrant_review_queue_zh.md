# Track2 classification v5 跨象限候选审查队列

## 结论

跨象限改动是唯一可能继续提升 Classification 的高收益路径，但也是当前最大风险源。默认不放行，先做小队列审查。

## 自动放行阈值

只有 `evidence_score>=9`、`supporting_family_count>=5`、`supporting_source_count>=9`、`gemini35_objection=False` 且追加多模态仲裁支持 proposed 的样本，才允许进入 `v5_cross` candidate。

## 候选表

| priority | sample_id | transition | evidence | families | sources | gemini objection | current VA | proposed VA | recommendation |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | `track2_0547` | `annoyed->calm` | 11 | 5 | 10 | `False` | `Negative/High` | `Positive/Low` | `first_priority_review` |
| 2 | `track2_0088` | `calm->tired` | 7 | 4 | 7 | `False` | `Positive/Low` | `Negative/Low` | `hold_review_only` |
| 3 | `track2_0139` | `calm->tired` | 7 | 4 | 7 | `False` | `Positive/Low` | `Negative/Low` | `hold_review_only` |
| 4 | `track2_0319` | `calm->bored` | 7 | 4 | 7 | `False` | `Positive/Low` | `Negative/Low` | `hold_review_only` |
| 5 | `track2_0414` | `calm->bored` | 7 | 4 | 7 | `False` | `Positive/Low` | `Negative/Low` | `hold_review_only` |
| 6 | `track2_0457` | `calm->aroused` | 7 | 4 | 7 | `False` | `Positive/Low` | `Positive/High` | `hold_review_only` |
| 7 | `track2_0665` | `calm->aroused` | 7 | 4 | 7 | `False` | `Positive/Low` | `Positive/High` | `hold_review_only` |
| 8 | `track2_0728` | `alarmed->aroused` | 7 | 4 | 7 | `False` | `Negative/High` | `Positive/High` | `hold_review_only` |
| 9 | `track2_0925` | `calm->tired` | 7 | 4 | 7 | `False` | `Positive/Low` | `Negative/Low` | `hold_review_only` |
| 10 | `track2_0153` | `calm->sad` | 5 | 3 | 6 | `False` | `Positive/Low` | `Negative/Low` | `hold_review_only` |
| 11 | `track2_0286` | `calm->tired` | 5 | 3 | 6 | `False` | `Positive/Low` | `Negative/Low` | `hold_review_only` |
| 12 | `track2_0318` | `calm->tired` | 5 | 3 | 6 | `False` | `Positive/Low` | `Negative/Low` | `hold_review_only` |
| 13 | `track2_0796` | `calm->tired` | 5 | 3 | 6 | `False` | `Positive/Low` | `Negative/Low` | `hold_review_only` |
| 14 | `track2_0815` | `content->annoyed` | 5 | 3 | 6 | `False` | `Positive/Low` | `Negative/High` | `hold_review_only` |
| 15 | `track2_0829` | `calm->bored` | 5 | 3 | 6 | `False` | `Positive/Low` | `Negative/Low` | `hold_review_only` |
| 16 | `track2_0873` | `aroused->calm` | 3 | 3 | 4 | `False` | `Positive/High` | `Positive/Low` | `block_or_low_priority` |
| 17 | `track2_0299` | `calm->tired` | 3 | 2 | 5 | `False` | `Positive/Low` | `Negative/Low` | `block_or_low_priority` |
| 18 | `track2_0452` | `calm->tired` | 1 | 2 | 3 | `False` | `Positive/Low` | `Negative/Low` | `block_or_low_priority` |
| 19 | `track2_0605` | `calm->sad` | 1 | 2 | 3 | `False` | `Positive/Low` | `Negative/Low` | `block_or_low_priority` |
| 20 | `track2_0772` | `calm->tired` | 1 | 2 | 3 | `False` | `Positive/Low` | `Negative/Low` | `block_or_low_priority` |

## 操作建议

1. 先对 `track2_0547` 单点做 Gemini 3.5 Flash 多模态仲裁，并展示图像/当前标签/候选标签。
2. 如果 `track2_0547` 通过，再构造 `v5_cross1`，只加这一个跨象限改动，跑 validator、description audit、shadow evaluator。
3. score=7 的候选不直接放行；它们多为 `calm->tired/bored/aroused`，容易把低唤醒正向图误推成负类或高唤醒类。
4. `track2_0815` 继续 block：此前 public/reference 分析提示它可能是 Positive/Low，不能因模型投票改成 `annoyed/Negative/High`。
