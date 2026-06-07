# Track1 V4 Broad Package FID Smoke Analysis

## 结论

不要直接提交 `full1000_no_fallback`、`partial757`、`safe_subset` 或 `strict_subset`。本地 all-style Inception FID-like smoke 显示，当前 `current` 包仍然是已有包里最接近 EmoArt-130K 公开参考分布的版本。

这不等于 current 已经足够冲奖；官方分数已经证明我们距离榜首还有 FID 差距。但它说明现有 Gemini broad-regeneration 包没有解决 FID，反而把图像分布推离了公开 EmoArt-130K 分布。

## 本地 smoke 结果

方法：`reference_limit=2048`，`reference_seed=20260607`，`fid_feature_dim=512`，all tar styles，device=`mps`。

| package | fid_like | 解释 |
| --- | ---: | --- |
| current | 58.526207 | 当前最好，本地代理下最贴公开参考分布 |
| partial757 | 62.414763 | 大量替换后分布变差 |
| v3_gate7 | 63.010429 | 不能证明局部替换提升 FID，且重打包/编码差异会影响代理 |
| full1000_no_fallback | 63.098811 | 全量新生成图不是冲奖方向 |
| strict_subset | 63.287051 | 不能直接提交 |
| safe_subset | 63.308289 | 不能直接提交 |

## 对策略的影响

1. v4 不能走“全量重生图覆盖 current”。
2. full1000 可以作为失败案例和候选池，但不是 final package 底座。
3. 下一步应保留 current 分布，只修 current 的高风险 outlier。
4. 替换数量要从 `50-100`、`120-220` 两档开始，而不是直接 400+。
5. 每个候选必须同时过三关：AAS/语义不差、视觉不出 artifact、本地 package FID 不退化。

## 下一步

- 先做 candidate-level metric proxy：current vs full1000、current vs partial757。
- 从正向样本中构建 `conservative80` 和 `balanced180` manifest。
- 如果干净替换数不足 120，再只针对 current outlier 生成新候选。
- 新候选的 prompt 应更像官方参考分布，减少 poster/template 化，不再用强硬统一版式。
