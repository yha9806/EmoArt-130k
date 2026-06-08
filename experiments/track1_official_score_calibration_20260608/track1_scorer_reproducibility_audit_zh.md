# Track1 Scorer Reproducibility Audit

这是本地评分器复现性审计，不是官方隐藏评测器。

## 结论

- 官方总分公式复现样本数：`13`
- 官方总分公式最大绝对误差：`5e-11`
- FID 到 FID Score 拟合 R2：`0.992139`
- FID 到 FID Score 拟合 RMSE：`0.00493274`
- 本地 local/official 锚点数：`2`
- 本地 proxy 方向：`anti_correlated`
- 本地 proxy 就绪度：`insufficient_own_anchors`
- 原因：至少需要 3 个自有提交锚点，并且每个锚点都要同时保留本地包特征和官方组件分数。

## 本地锚点

| package | submission | local fid_like | official FID | official FID Score | official AAS |
|---|---|---:|---:|---:|---:|
| `hybrid_redteam_fid_pass4` | `784403` | 58.348904 | 105.663816 | 0.4862304023 | 0.99305 |
| `v3_gate7` | `782831` | 63.010429 | 80.92 | 0.55 | 0.98 |

## FID Score 拟合最差残差

| participant | submission | FID | actual FID Score | predicted | residual |
|---|---|---:|---:|---:|---:|
| `zhbai` | `755551` | 130.772838 | 0.4333265599 | 0.425412996 | 0.0079135639 |
| `emosuis` | `778783` | 66.362192 | 0.6010981145 | 0.5932336457 | 0.0078644688 |
| `trybest-1` | `777611` | 66.841741 | 0.5993703924 | 0.5919841897 | 0.0073862027 |
| `vulcaart` | `782831` | 80.92 | 0.55 | 0.5553035684 | -0.0053035684 |
| `vulcaart` | `784403` | 105.663816 | 0.4862304023 | 0.4908340506 | -0.0046036483 |

## 解释

- 官方 `overall = 0.5 * FID Score + 0.5 * AAS` 可以精确复现。
- FID Score 对 FID 的公开映射可以高置信近似，但不是官方归一化函数源码。
- 本地 proxy 目前不能当作可复现官方评分器；它只能做提交前风险排序。
- 要让本地 scorer 真正可校准，至少还需要 3 个以上自有提交锚点，并且每个锚点要保留本地包特征。
