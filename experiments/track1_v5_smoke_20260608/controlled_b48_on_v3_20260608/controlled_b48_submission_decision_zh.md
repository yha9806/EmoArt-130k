# Track1 Controlled-B48 on V3 Submission Decision

## 结论

`track1_candidate_v5_controlled_b48_on_v3_20260608.zip` 是比旧 `track1_candidate_v5_smoke_accept48_20260608.zip` 更安全的 controlled-B 候选，因为它建立在官方已验证较好的 `v3_gate7/full1000` 谱系上，而不是旧 current 谱系上。

这不是最终“最大化 FID”的终局包；它是一个可提交、可解释、风险受控的 B 方案第一版。

## 文件

- ZIP: `submissions/track1_candidate_v5_controlled_b48_on_v3_20260608.zip`
- submission.json: `submissions/track1_candidate_v5_controlled_b48_on_v3_20260608/submission.json`
- Replacement manifest: `experiments/track1_v5_smoke_20260608/human_accept48_20260608/track1_v5_smoke_human_accept48_replacement_manifest.json`
- Lineage audit: `experiments/track1_v5_smoke_20260608/controlled_b48_on_v3_20260608/lineage_audit.json`

## 验证

- validate-track1: `OK`
- submission rows: `1000`
- zip images: `1000`
- zip sha256: `10a3279e4da2264e9377e1d3bd20c619360d1e9b23eecc08e8287a1675a6415c`
- champion package diff lines: `0`

## Lineage

- same_as_current: `0`
- same_as_v3_gate7: `952`
- same_as_full1000_no_fallback: `948`
- changed_vs_v3_gate7: `48`

## 判断

可以把它作为倒数两次中的第一发 controlled-B 试探，前提是我们接受它的目标不是一次性冲到第一，而是用 v3 高分基底承载 v5 人评通过替换，避免旧 current FID 崩盘。

如果要更激进冲第一，下一步应在这个 v3 基底上继续扩展 80-180 个通过 AAS/FID/人评 gate 的替换，而不是提交旧 current 基底包。
