# Track1 P1-Light Proxy（本地轻量评分器）

这是本地 gate 报告，不是官方隐藏评测器的完整模拟。
它只用于在最后两次提交机会前排除明显坏包，并帮助选择下一轮进入视觉复核的候选包。

## 总结

- 比较包数量：`5`
- 当前推荐进入下一道 gate 的包：`b80_fixed`

## 包级结果

| 包 | 状态 | 已替换样本 | placeholder 命中 | 最高风格集中度 | P1-light 分数 |
|---|---|---:|---:|---:|---:|
| `b80_fixed` | `candidate_ok_for_next_gate` | 80 | 0 | 0.3 | 0.9875 |
| `b120_fixed` | `candidate_ok_for_next_gate` | 120 | 0 | 0.3 | 0.9867 |
| `b160_fixed` | `candidate_ok_for_next_gate` | 160 | 0 | 0.2938 | 0.9838 |
| `current` | `candidate_ok_for_next_gate` | 0 | 0 | 0.0 | 1.0 |
| `v3_gate7` | `reject_known_placeholder` | 7 | 2 | 0.2857 | 0.0 |
