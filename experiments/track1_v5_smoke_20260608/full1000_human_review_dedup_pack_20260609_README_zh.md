# Track1 Full1000 Deduplicated Human Review Pack 20260609

这个包替代旧的 `full1000_human_review_pack_20260608.zip`。

旧包没有损坏，但候选列语义不清：`B80/B120/B160` 是 ladder subset，不是四套独立候选。因此旧 HTML 里会出现大量重复候选图。

## 重复统计

- `840 / 1000` 个样本：`v3/B80/B120/B160` 四列完全相同。
- `160 / 1000` 个样本：四列中只有 2 个唯一候选。
- `0 / 1000` 个样本：四列候选都不同。

## 新包修复

- 只显示唯一候选图。
- 每个候选明确标注来源列，例如 `候选 1: v3+B80+B120`、`候选 2: B160`。
- 附带 `track1_full1000_candidate_duplicate_report.csv`，可核查每个样本的候选分组。
- 保留 current 对照、P0/P1 风险、人工预标、CSV 导出。

## 已知预标问题

- `track1_0049`: 右侧枪托/武器接触逻辑问题。
- `track1_0080`: 脚的朝向/站姿逻辑问题。
- `track1_0910`: 多人 grappling 的肢体、抓握、木结构接触关系混乱。

## 使用方式

解压 `full1000_human_review_dedup_pack_20260609.zip`，打开 `track1_full1000_dedup_human_review_zh.html`。

这个包只用于人工审核，不会修改 current champion submission。
