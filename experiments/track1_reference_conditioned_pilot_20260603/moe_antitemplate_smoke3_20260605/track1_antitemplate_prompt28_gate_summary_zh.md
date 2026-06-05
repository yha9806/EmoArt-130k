# Track1 anti-template prompt28 gate summary

日期：2026-06-05

## 结论

这轮已经不只是 dry-run。我们实际生成了 28 张候选图，覆盖 10 个高风险苏联宣传画/俄文文字类样本，并完成了本地分布审计、中文 contact sheet、gemini-agent contact-sheet 审查、Gemini 3.5 Flash 红队审查。

当前正式提交包保持不可变：`git diff -- submissions/track1_submission.json submissions/track1_submission.zip submissions/track1/images | wc -l` 输出为 `0`。

核心结论：

- anti-template / reference-text-bank 方向有效，尤其改善了 pseudo-Cyrillic、无意义大字、过度模板化海报构图。
- 但不能全自动替换。部分样本的 caption 结构、空间关系、历史符号比“文字更像俄文”更重要。
- 这轮最稳的高精度替换候选是 `0665 aas_safe`、`0803 aas_safe`、`0898 aas_safe`。
- `0077 aas_safe`、`0708 reference_style` 有明显潜力，但需要人工看 caption 结构和局部逻辑。
- `0370 reference_style` 是 AI 红队与先前人工偏好冲突样本，只能进入冲突复核，不能自动收。
- `0353`、`0476`、`0593`、`0747` 暂时保持 current 或 targeted rerun，不进自动替换。

## 已生成产物

- 候选 manifest：`candidate_manifest_review.json`
- 分布审计：`distribution_audit_review.md`
- 中文 HTML：`track1_antitemplate_prompt28_review_zh.html`
- 中文 contact sheet：
  - `track1_antitemplate_prompt28_contact_sheet_zh_01.jpg`
  - `track1_antitemplate_prompt28_contact_sheet_zh_02.jpg`
- Gemini 3.5 redteam：
  - `redteam_gemini35/redteam_reviews.json`
  - `redteam_gemini35/redteam_reviews.md`
- 本 gate 建议表：
  - `track1_antitemplate_prompt28_gate_recommendations.csv`

## 审计摘要

- planned/generated/cached 后最终候选数：28
- unique samples：10
- existing images：28
- missing images：0
- invalid/unreadable images：0
- prompt lint：pass
- aspect label：28 张均为 portrait

说明：全部 portrait 不是全量系统的最终分布，而是因为本次 prompt28 预算集中在高风险海报/俄文/苏联宣传画样本上。

## 分层建议

### high_precision_accept

这些可以进入最终人工确认队列，确认通过后进入 replacement manifest：

- `track1_0665` -> `aas_safe`
  - 修复 current 的横幅乱码，保留 medal/ribbon/Kremlin/crowd/fireworks。
  - 需要最终看 medal/ribbon 细节和底部是否有微小伪字。
- `track1_0803` -> `aas_safe`
  - 修复 current 的伪 Cyrillic，Kremlin、红星、苏英美旗帜更清楚。
  - 需要最终确认三面旗帜几何和 Kremlin 塔是否可信。
- `track1_0898` -> `aas_safe`
  - 修复 current 旗帜双 hammer-sickle 的符号错误，雪地冲锋和海报文字更稳。
  - 需要最终看枪、旗杆、身体接触关系。

### accept_but_visual_check

这些有潜力，但不能自动收：

- `track1_0077` -> `aas_safe`
  - 优点：俄文和海报感明显好。
  - 风险：caption 明确说 sailor hoisting red and white flags before panels of warships and sea battles；新图可能弱化了 panel 结构。
- `track1_0708` -> `reference_style`
  - 优点：当前可能有俄文拼写问题，新图士兵/战场/红旗更干净。
  - 风险：需要看主角武器、旗帜遮挡、文字是否截断。

### duplicate_candidate_resolution / conflict

- `track1_0370` -> redteam 选 `reference_style`，但这和先前人工“current 更好”的方向冲突。
  - AI 理由：current 左侧疑似 extra hand；reference_style 物理逻辑更干净。
  - 处理：必须全分辨率人工二选一。没有人工确认前不替换。

### keep_current_block

- `track1_0353`：红队选 current；候选有手、武器或边框 artifact。
- `track1_0476`：红队选 current；候选有船型时代混杂、绳子漂浮、炮弹卡通化等问题。
- `track1_0593`：红队选 current；唯一新候选有底部 microtext 和剑柄握持问题。

### rerun_or_hold

- `track1_0747`：保守 hold/targeted rerun。
  - 虽然红队选 current，但这个样本的关键不是“谁更漂亮”，而是 mounted soldiers 与 civilians 的关系是否读成保护/撤离，不能读成追逐/攻击。
  - 这类关系问题不能靠单个 caption-aware VLM gate 自动放行，需要 blind relation review 或更窄 prompt rerun。

## 为什么这轮比早期模板候选好

1. 文本 bank 起作用：模型不再随便造 pseudo-Cyrillic，而是倾向使用少数大号、清楚、风格合理的俄文短句。
2. reference family 起作用：Kremlin、海军、奖章、旗帜、战场构图开始像真实苏联宣传画，而不是泛化的红色海报模板。
3. MoE 策略分工起作用：
   - `aas_safe` 适合修复文字/主体保真。
   - `reference_style` 适合提高历史海报质感和构图。
   - `fid_diverse` 偶尔有更自由的构图，但也更容易出现裁切、微字、局部物理问题。
4. gate 终于能分清“更像海报”和“更适合提交”不是同一件事。

## 还缺什么才能进入最终 manifest

必须补：

1. 人工确认这 6 个可疑/候选样本：
   - `0665 aas_safe`
   - `0803 aas_safe`
   - `0898 aas_safe`
   - `0077 aas_safe`
   - `0708 reference_style`
   - `0370 reference_style` vs current
2. 对 `0747` 做 targeted relation rerun 或保留 current。
3. 把人工确认结果写成 accepted replacement manifest。
4. 用 candidate package builder 生成独立候选包，不触碰 champion package。
5. 跑 `validate-track1 --check-files`。

## 下一步建议

不要立刻全量 1000 x 3。下一步应该是：

1. 对这 10 个样本做人工 final gate，确认 anti-template 策略是不是可进入替换包。
2. 如果 high_precision_accept 至少 3/3 通过，再扩展到 100 个高风险样本。
3. 100 个样本通过后，再做全量 1000 的路由：
   - 文字/真实地标/旗帜/徽章/制服：reference-text-bank + factual anchors。
   - 非文字艺术图：降低 poster template，优先 style-family/FID diversity。
   - 关系/空间逻辑高风险：少量候选 + blind relation review。

这条路线比“一次性全量生成”慢一些，但更接近拿奖目标：把钱和 quota 花在 AAS 高方差、高收益、当前系统最容易犯错的地方，同时控制 FID 分布不要变成统一模板海报。
