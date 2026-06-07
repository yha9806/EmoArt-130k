# Track1 V4 FID Distribution Gate Conclusion

## 结论优先

当前最有提交价值的不是 `full1000_no_fallback`，也不是 raw greedy20，而是一个很小的 conservative probe：

- `hybrid_probe_redteam_fid_pass4`
- 只替换 4 张：`track1_0140`, `track1_0233`, `track1_0488`, `track1_0697`
- 本地 Inception FID-like：`58.348904`
- 当前冠军包本地 Inception FID-like：`58.526207`
- 本地改善：`0.177303`
- Gemini 3.5 Flash redteam：4 张都偏向 `full1000`
- submission 结构验证：`OK`

这仍然不是最终人工确认包，但它是目前第一版同时满足“包级 FID-like 改善”和“AAS/视觉 redteam 不反对”的候选。

## 不能提交的包

- `full1000_no_fallback`：本地 FID-like `63.098811`，明显差于 current。
- `partial757`：本地 FID-like `62.414763`，明显差于 current。
- `hybrid_redteam5`：5 张 high-delta redteam 支持候选，但包级 FID-like `58.602602`，仍差于 current。
- `greedy_top80_raw20`：本地 FID-like `57.697893` 最好，但 20 张里 Gemini redteam 只支持 4 张；其余 16 张存在 caption、文字、mockup、构图或物理逻辑风险，不能作为 AAS 安全包提交。

## 关键发现

1. SigLIP/metric proxy 的单样本正向 delta 不能直接用于替换。
   `top80` 中大量样本在 redteam 中被判 current 更好，甚至有系统错误图、metadata 图、mockup 边框、伪文字。

2. FID 必须用包级搜索，而不是逐图贪心视觉判断。
   high-delta redteam5 在视觉上有合理替换，但组合后本地 FID-like 变差。

3. AAS gate 必须压在 FID search 后面。
   raw greedy20 对 FID 很好，但 AAS/视觉风险太大；最终只能取 `FID search ∩ redteam pass` 的交集。

4. 当前真正的冲奖方向是小步正式测分。
   AAS 已接近天花板，下一次提交应该用 conservative hybrid4 测官方 FID 是否跟本地 Inception proxy 同向。

## 产物路径

- shortlist HTML：`experiments/track1_v4_fid_distribution_champion_20260607/distribution_shortlist_top80_zh.html`
- high-delta redteam：`experiments/track1_v4_fid_distribution_champion_20260607/redteam_high_delta_gemini35_flash/redteam_reviews.md`
- greedy20 redteam：`experiments/track1_v4_fid_distribution_champion_20260607/redteam_fid_greedy20_gemini35_flash/redteam_reviews.md`
- conservative hybrid4 review HTML：`experiments/track1_v4_fid_distribution_champion_20260607/fid_redteam_pass4_review_zh.html`
- conservative hybrid4 package：`experiments/track1_v4_fid_distribution_champion_20260607/hybrid_probe_redteam_fid_pass4.zip`
- conservative hybrid4 submission JSON：`experiments/track1_v4_fid_distribution_champion_20260607/hybrid_probe_redteam_fid_pass4/submission.json`
- local shadow calibration v2：`experiments/track1_official_score_calibration_20260607/track1_shadow_score_calibration_v2_with_hybrids_zh.md`

## 下一步建议

1. 人眼快速复核 4 张 conservative hybrid4。
2. 如果没有明显视觉问题，把 `hybrid_probe_redteam_fid_pass4.zip` 作为一次正式 Codabench 测分提交。
3. 回分后，把官方 FID/FID Score/AAS 加入 `track1_official_score_anchors.csv`，重新校准 shadow scorer。
4. 如果官方确认 hybrid4 同向改善，再扩大到 `FID search ∩ redteam pass ∩ human pass` 的 10-30 张包；如果官方不改善，则说明本地 Inception proxy 和官方 FID 有偏差，需要改 reference sampling 或特征维度。
