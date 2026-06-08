# Track1 当前成果与 P1 优先级总结

日期：2026-06-08

目的：把当前 Track1 / EmoArt 公开资源检索、官方分数反馈、P0 本地资源使用情况、候选包审计结果和下一步 P1 方向固化下来，作为后续继续优化 Track1 的决策基线。

## 结论优先

1. P0 资源已经在本地，不需要重新下载。`EmoArt-130k` 的 `Annotation.json` 和 56 个 style tar 包都已经在 `/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k/`。
2. P0 已经用过，但还没有用到强队级别。我们已经做过 official reference bank v5/v7，也把 1000 个 Track1 sample 都路由到 official/public EmoArt references；但这些 reference 主要被当作 Gemini prompt 的参考图，还没有完全变成 FID optimizer、style-family scorer 和 final selection gate。
3. 当前 Track1 的主要短板不是 AAS，而是 FID/style distribution。我们的 v3 official anchor 是 Overall `0.77`、FID `80.92`、FID Score `0.55`、AAS `0.98`；公开榜首约 Overall `0.80`、FID `66.x`、AAS `0.99`。差距主要在 FID。
4. P1 现在比继续重复 P0 更重要。P1 的目标不是“多读论文”，而是把发起者和合作者的公开算法成果转成 Track1 的方法模块：reference distribution routing、salience-aware prompt compiler、anti-template generation router、本地 FID/AAS proxy、evidence-based package selector。
5. 当前候选包风险已经被识别：历史 v1/v2/full1000 等包里出现过 code/JSON/provider placeholder；B80/B120/B160 fixed 包已经修复 known placeholder 命中为 0。

## P0 本地资源现状

本地公开资源：

- `/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k/Annotation.json`
- `/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k/*.tar.gz`
- `/Users/yhryzy/dev/emoart-challenge/data/EmoArt-5k/annotation.json`
- `/Users/yhryzy/dev/emoart-challenge/data/EmoArt-5k/Images.tar.gz`
- `/Users/yhryzy/dev/emoart-challenge/data/emoart_annotation/Annotation.json`

本地 profile 统计：

- rows: `132895`
- top emotions:
  - `Calm`: `74350`
  - `Excited`: `20595`
  - `Contentment`: `20406`
  - `Sad`: `5827`
  - `Alarmed`: `5407`
  - `Frustrated`: `3313`
- valence:
  - `Positive`: `116861`
  - `Negative`: `16034`
- arousal:
  - `Low`: `101540`
  - `High`: `31355`
- top styles:
  - `Romanticism`: `15731`
  - `Realism`: `15307`
  - `Impressionism`: `11736`
  - `Expressionism`: `10065`
  - `Baroque`: `7995`
  - `Post-Impressionism`: `6146`
  - `Art Nouveau (Modern)`: `5899`
  - `Surrealism`: `4436`
  - `China_images`: `4157`
  - `Rococo`: `3733`

注意：本地 rows 统计和官方公开材料的 `132,664 artworks` 有轻微差异。后续写报告和外部引用以官方数字为准；本地统计用于工程 profile。

## P0 已经完成的工作

### Official Reference Bank

已有模块：

- `affectiveart.track1_official_reference_bank`
- `scripts/track1_build_official_reference_bank.py`

设计文件：

- `docs/superpowers/specs/2026-06-05-track1-official-reference-bank-v5-design.md`
- `docs/superpowers/plans/2026-06-05-track1-official-reference-bank-v5.md`

已观察结果：

- v5 使用 full `Annotation.json` 作为候选池，不是只从少量 curated reference 里选图。
- v5 把 unique reference assets 从旧 bank 的 `78` 扩到 `1680`，并让 `1000/1000` Track1 samples 都获得 official references。
- v7 reranker 降低 Socialist Realism 单一 victory poster 过度复用问题，最高 first-reference reuse 从 `58` 降到 `19`。

结论：P0 的“下载/接入/初步路由”已经做过；缺的是更深的分布优化和选择优化。

## Track1 官方分数与本地结论

当前重要官方/公开 anchors：

| source | package/team | overall | FID | FID Score | AAS | notes |
|---|---:|---:|---:|---:|---:|---|
| own | `track1_submit_v3_gate7_20260606.zip` | `0.77` | `80.92` | `0.55` | `0.98` | 当前可用 official anchor |
| own | `hybrid_probe_redteam_fid_pass4.zip` | `0.74` | `105.66` | `0.49` | `0.99` | AAS 更高但 FID 明显变差 |
| public top | `emosuis` visible row | `0.80` | `66.36` | `0.60` | `0.99` | 公开榜首上下文 |
| public top | `trybest-1` visible row | `0.80` | `66.84` | `0.60` | `0.99` | 公开榜首上下文 |

解释：

- 我们已经能把 AAS 做到 `0.98-0.99` 附近。
- 直接 micro-fix 不足以冲第一，因为 7 个或几十个替换无法显著改变整包 FID。
- 但盲目 full1000 或 broad replacement 也会有风险：我们已有一次 `hybrid_redteam_fid_pass4` 的官方反馈，说明 AAS 可以涨、FID 可以大跌。
- 因此下一步必须是 reference-distribution-aware package selector，而不是单纯“多生成几张好看的图”。

## Placeholder / Code-like 问题审计

已生成审计页面：

- `experiments/track1_v5_smoke_20260608/package_placeholder_audit_20260608/track1_package_review_index_zh.html`

known placeholder scan 结果：

| package | known placeholder hits | samples |
|---|---:|---|
| `current_champion` | `0` |  |
| `submit_v1` | `4` | `track1_0730`, `track1_0735`, `track1_0740`, `track1_0772` |
| `submit_v2` | `4` | `track1_0730`, `track1_0735`, `track1_0740`, `track1_0772` |
| `submit_v3_gate7` | `2` | `track1_0735`, `track1_0740` |
| `candidate_full1000_v7` | `4` | `track1_0730`, `track1_0735`, `track1_0740`, `track1_0772` |
| `candidate_b48` | `2` | `track1_0735`, `track1_0740` |
| `candidate_b80_fixed` | `0` |  |
| `candidate_b120_fixed` | `0` |  |
| `candidate_b160_fixed` | `0` |  |

结论：

- 之前一些包确实存在 provider/JSON/code-like placeholder 图像风险。
- 当前 `current_champion` 没有 known placeholder 命中。
- 新的 Controlled-B fixed 包已经消除已知 placeholder 命中。
- 后续所有包必须把 placeholder hash guard 作为 hard gate。

## V5 Smoke 与 Controlled-B 当前成果

### 人评确认 48 个替换

文件：

- `experiments/track1_v5_smoke_20260608/human_accept48_20260608/track1_v5_smoke_human_accept48_replacement_manifest_zh.md`

结果：

- accepted_count: `48`
- 用户在 review 页面确认“是的，都可以替换”。
- 策略包含：
  - `caption_faithful_guard`
  - `anti_template_diversifier`
  - `reference_family_primary`
- 这批是高质量候选池，不是最终提交包本身。

### Controlled-B 梯度包

文件：

- `experiments/track1_v5_smoke_20260608/controlled_b_ladders_20260608/controlled_b_ladders_summary_zh.md`

当前包：

| package | replacements | known placeholder hits | zip MB |
|---|---:|---:|---:|
| `track1_candidate_v5_controlled_b80_on_v3_20260608.zip` | `80` | `0` | `424.31` |
| `track1_candidate_v5_controlled_b120_on_v3_20260608.zip` | `120` | `0` | `424.42` |
| `track1_candidate_v5_controlled_b160_on_v3_20260608.zip` | `160` | `0` | `424.42` |

解释：

- 这三个包都以 `v3_gate7` 为基底，不修改 champion。
- 默认排除和旧 current 完全相同的候选图。
- `track1_0735`、`track1_0740` 被允许作为 v3 占位图救援。
- 这些包更像“可提交候选”，但仍需要结合 official submission 次数和风险选择，不应盲目连续提交。

## 为什么 P1 更重要

P0 给了我们数据，但 P1 给的是方法。现在的问题不是“有没有 reference”，而是：

- reference 应该如何影响生成，而不是把所有图变成模板？
- 哪些视觉属性是 AAS evaluator 真正在看的？
- FID 分布应该贴近哪种 style family，而不是全局平均？
- 哪些样本应该严格 caption lock，哪些样本应该放开 painterly freedom？
- 什么样的 local proxy 能区分“看起来更好”与“官方 FID 更好”？

P1 应优先检索和结构化这些人的公开成果：

| source group | 要找什么 | 对 Track1 的用法 |
|---|---|---|
| Hongxia Xie / Cheng Zhang | EmoArt、FAB-G、EmoVIT、MindPower、CookAnything、emotion instruction、attribute salience | AAS proxy、salience-aware prompt compiler、情绪/属性 gate |
| Wen-Huang Cheng / Hong-Han Shuai / Ling Lo | affective vision、robust feature、multimedia understanding、micro-expression/facial affect | judge 稳健性、细粒度情绪/视觉 cue 设计 |
| Sicheng Zhao | visual emotion analysis、affective gap、emotion distribution、domain adaptation | Track2 label prior，也能帮助 Track1 emotion-aware generation |
| Jianlong Fu | multimodal generation、PromptFix、MM-Diffusion、quality repair、vision-language systems | 生成修复、candidate repair、prompt/fix pipeline |
| Sanghoon Lee | perceptual quality assessment、image/video quality、generative AI media analysis | FID/quality proxy、visual artifact gate |
| Xing Huang / Lianxin Digital | psychological assessment、industrial affective computing | 组织者侧情绪分类/心理评估语义线索 |

## 下一步建议

### 1. 先做 P1 Method Intelligence Crawl

输出目标：

- `experiments/emoart_origin_research_20260608/p1_method_intelligence_crawl_zh.md`
- `experiments/emoart_origin_research_20260608/p1_method_cards.json`

每个 method card 至少包含：

- author / lab / paper / repo / dataset
- task
- public source URL
- core method
- transferable idea for Track1
- potential engineering module
- confidence
- risk / non-use reason

### 2. 把 P1 转成 Track1 模块

优先模块：

1. `style_family_distribution_retriever`
   - 不止 top4 reference，而是 sample-level family distribution。
2. `fid_style_proxy`
   - CLIP/DINO/Inception/color/aspect/text-density/template-similarity 多指标。
3. `salience_aware_prompt_compiler`
   - 根据 caption/style/emotion 选择 2-3 个关键 visual levers，避免五属性机械堆叠。
4. `anti_template_router`
   - 判断哪些样本需要 strict poster/scroll/document，哪些样本需要 painterly freedom。
5. `package_selection_gate`
   - 结合 human accept、placeholder guard、AAS risk、FID proxy 和 official anchor 选择最后提交包。

### 3. 再决定是否提交 Controlled-B 包

当前 B80/B120/B160 是可用候选，但提交前需要回答：

- 我们只剩几次提交？
- 是否要先提交 conservative B80/B120，还是等 P1 proxy 后选包？
- 当前 official scoring 对 v3 和 hybrid probe 的反馈显示，局部视觉质量不等于 FID 提升；因此建议不要在没有 P1 proxy 的情况下把最后提交机会花完。

## 当前硬约束

- 不修改：
  - `submissions/track1_submission.json`
  - `submissions/track1_submission.zip`
  - `submissions/track1/images/`
- 不把 challenge private test images 上传到公开服务。
- 不把 public EmoArt row 当 hidden test label。
- 不写 evaluator manipulation 文本。
- 所有新候选包必须过：
  - structure validation；
  - check-files；
  - known placeholder hash scan；
  - visual review / human gate；
  - local proxy sanity；
  - champion diff check。

## 当前阶段判断

P0 已经完成“下载 + 初步使用 + official reference routing”。下一阶段不应继续把重点放在 P0 爬取，而应把 P1 的公开算法、论文和 organizer 技术路线系统化，然后反向改造 Track1 pipeline。

一句话：**P0 是素材库，P1 是打法。我们现在缺的是打法。**
