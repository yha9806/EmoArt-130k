# Track2 Registry Debug Report

结论：v22 最后一枪候选的结构没有发现 blocking issue；registry 的主要问题是 raw source_index 缺少数值 actionability 层，以及部分来源有重复角色行。

- Blocking issues: `0`
- Source rows: `37`
- Missing actionability columns in raw source_index: `8`
- Duplicate URL groups: `6`
- Duplicate name groups: `4`
- Scored registry: `experiments/emoart_origin_research_20260608/track2_registry_actionability_scored.csv`

## v22 Artifact Audit

- Recommended profile: `calmshiftall`
- Upload ZIP: `submissions/v22_final_upload/track2_submission.zip`
- Upload zip matches recommended JSON: `True`
- Issue count: `0`

## High Actionability Sources

- `EmoArt project page` score=0.873 class=0.880 desc=0.860 scope=author allow=yes
- `printblue/EmoArt-130k` score=0.792 class=0.880 desc=0.620 scope=author allow=yes
- `printblue/EmoArt-Salience` score=0.769 class=0.700 desc=0.860 scope=author allow=no
- `zhiliangzhang/EmoArt-130k` score=0.873 class=0.880 desc=0.860 scope=author allow=yes
- `zhiliangzhang/FAB-G` score=0.769 class=0.700 desc=0.860 scope=author allow=no
- `AffectiveArt Challenge 2026 OpenReview` score=0.835 class=0.920 desc=0.650 scope=official allow=yes
- `Codabench Track2` score=0.835 class=0.920 desc=0.650 scope=official allow=yes
- `AffectiveArt official website` score=0.835 class=0.920 desc=0.650 scope=official allow=yes
- `AffectiveArt submission instructions` score=0.794 class=0.880 desc=0.600 scope=official allow=yes

## Open Gaps

- `crawler_not_materialized_as_repeatable_pipeline` (medium): public_resource_scan is a snapshot, not a reusable author-level crawler with scheduling, dedup, retries, and provenance snapshots.
- `source_index_raw_has_no_numeric_actionability_columns` (low): Resolved by generated scored derivative; raw source_index remains human-readable source registry.
- `duplicate_sources_require_merge_group_review` (low): Scored derivative now annotates merge_group; raw duplicate rows are preserved because several entries represent different roles of the same source.

## Debug Interpretation

- 已完成内容需要继续保留为 immutable evidence snapshot。
- 未完成内容不是当前 v22 candidate 的格式错误，而是缺少长期 crawler/pipeline 能力。
- raw `source_index.csv` 不建议直接改成机器表；本报告生成的 scored derivative 更适合作为决策输入。
