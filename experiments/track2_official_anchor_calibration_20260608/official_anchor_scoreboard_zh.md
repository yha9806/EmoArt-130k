# Track2 v23 official-anchor calibrated scorer

## 结论

- 这个评分器先复现线上锚点，再对未提交候选做估计。
- `exact_official` 是精确线上分数锚点；`visible_official` 只能复现排行榜可见的两位小数；`estimated` 不是隐藏 gold 重建。
- 任何估计分超过 0.86 都会标为高外推风险，因为我们的真实锚点主要集中在 0.83-0.84。

## Ranking

| rank | candidate | kind | expected | visible | class | desc | label changes | text rows | warnings |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | v22_calmshiftall | estimated | 0.848163 | 0.85 | 0.746659 | 0.949667 | 177 | 0 | not_hidden_label_reconstruction |
| 2 | v22_calmshift150 | estimated | 0.847352 | 0.85 | 0.745038 | 0.949667 | 150 | 0 | not_hidden_label_reconstruction |
| 3 | v22_calmshift120 | estimated | 0.846335 | 0.85 | 0.743002 | 0.949667 | 120 | 0 | not_hidden_label_reconstruction |
| 4 | official_785979_v21_calmshift90 | exact_official | 0.842559 | 0.84 | 0.740034 | 0.945083 | 90 | 81 |  |
| 5 | official_779605_moe_v2_anchor | exact_official | 0.836408 | 0.84 | 0.723150 | 0.949667 | 0 | 0 |  |
| 6 | official_782683_v12_stable_probe | visible_official | 0.835833 | 0.84 | 0.721667 | 0.950000 | 0 | 12 | exact_unavailable_visible_leaderboard_only |
| 7 | official_781601_v3_mid | exact_official | 0.834027 | 0.83 | 0.719137 | 0.948917 | 85 | 39 |  |

## 0.89 requirement

- If Description=1.00, required Classification=0.780000.
- If Description=0.95, required Classification=0.830000.

## Anchor Residuals

| candidate | submission | kind | residual | blocking |
|---|---|---|---:|---|
| official_779605_moe_v2_anchor | 779605 | exact_official | 0.000000 | False |
| official_781601_v3_mid | 781601 | exact_official | 0.000000 | False |
| official_782683_v12_stable_probe | 782683 | visible_official | 0.000000 | False |
| official_785979_v21_calmshift90 | 785979 | exact_official | 0.000000 | False |
