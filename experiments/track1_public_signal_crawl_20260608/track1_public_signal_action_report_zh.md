# Track1 公开信号继续爬取报告

日期：2026-06-08

目标：继续以 AffectiveArt / EmoArt 作者与合作者公开输出为原点，提取能提升 Track1 分数的可执行信号。重点不是“破解官方评分器”，而是用公开数据、代码、论文和 leaderboard anchor 建一个更接近官方偏好的本地代理。

## 结论

公开爬取可以显著改善我们的 Track1 策略，但不能让本地 scorer 与官方 hidden evaluator 完全一致。

最有价值的新信号有三类：

1. **FID 分布信号**：EmoArt-130k 是合法、公开、最接近官方 reference artistic image set 的 distribution prior。它包含 56 个 style tar 包和 `Annotation.json`，应作为 style-family reference bank 的核心。
2. **AAS 属性信号**：EmoArt/FAB-G 明确把 Brushstroke、Composition、Color、Line、Light 作为核心视觉属性；FAB-G 进一步要求先判断 attribute salience，再做 emotion reasoning。
3. **Affective reasoning 信号**：EmoVIT 支持 emotion + reason 输出，说明组织者技术传统重视“图像中哪些视觉证据支撑情绪”，这可以变成我们的 VLM judge prompt 和 prompt compiler guard。

不能通过公开爬取得到的是：hidden FID reference set、完整 AAS judge prompt、官方 VLM 模型、官方 inference settings、test labels。

## 已验证公开资源

### 1. 官方 Track1 规则

来源：Codabench Track1。公开规则确认 Track1 使用 FID 和 AAS，最终 50% + 50%。AAS 分为 content/style/attribute alignment，并由 fixed prompts、predefined scoring rubrics、consistent inference settings 的 multimodal protocol 评估。

Track1 对我们的直接含义：

| 官方项 | 我们应建的本地代理 |
|---|---|
| FID | EmoArt style-family distribution distance，不再相信单一 `fid_like` |
| Content Alignment | caption object/support/relation gate |
| Style Alignment | style-family retrieval + anti-template detector |
| Attribute Alignment | FAB-G/EmoArt 五属性 salience gate |
| Invalid Content | text/watermark/evaluator-instruction/filename/sample-id detector |

### 2. EmoArt-130k dataset

来源：HuggingFace `printblue/EmoArt-130k`。

| 字段 | 抓取结果 |
|---|---|
| lastModified | `2025-05-31T06:06:36.000Z` |
| downloads | `1096` |
| likes | `8` |
| gated | `False` |
| sibling_count | `59` |
| 关键文件 | `Annotation.json` + 56 个 style `.tar.gz` |

抓取到的 style 文件包括：`Abstract Art`、`Abstract Expressionism`、`Impressionism`、`Realism`、`Romanticism`、`Expressionism`、`Baroque`、`China_images`、`Ink and wash painting`、`Socialist Realism`、`Ukiyo-e` 等。

Track1 含义：这不是“参考图随便找”的资源，而是我们应该做 full style-family retrieval 的核心库。当前 prompt/reference 只用了少量 reference 反复轮换，导致生成模板化；正确做法是每个 sample 检索一组同 style、同 medium、同 emotion/arousal 的 reference family。

### 3. EmoArt-130k code repo

来源：GitHub `zhiliangzhang/EmoArt-130k`。

| 字段 | 抓取结果 |
|---|---|
| pushed_at | `2026-05-12T10:17:00Z` |
| updated_at | `2026-05-24T08:42:15Z` |
| license | `Apache-2.0` |
| stars/forks | `6` / `1` |
| 关键文件 | `GUI.py, LICENSE, MTLD.py, README.md, Shannon entropy.py, TTR.py, artist.png, attributes_alignments.py, clip_score.py, display.jpg, index.html, overview.jpg` |

关键不是 star 数，而是 repo 里有 metric code：`attributes_alignments.py`、`clip_score.py`、`TTR.py`、`MTLD.py`、`Shannon entropy.py`。

Track1 含义：

| 文件/方法 | 可转成什么 |
|---|---|
| `attributes_alignments.py` | 本地 attribute alignment proxy |
| `clip_score.py` | prompt-image/style-caption CLIP proxy |
| `TTR.py` / `MTLD.py` / entropy | 文本/解释 anti-template 指标，可借给 AAS review |
| `GUI.py` | 可视化审查工具思路 |

### 4. FAB-G / EmoArt-Salience

来源：GitHub `zhiliangzhang/FAB-G` 和 HuggingFace `printblue/EmoArt-Salience`。

| 字段 | 抓取结果 |
|---|---|
| FAB-G pushed_at | `2026-05-12T10:06:28Z` |
| FAB-G top files | `README.md, fabg_sim, infer_fabg_qwen3vl.py, requirements.txt` |
| Salience lastModified | `2026-05-12T06:30:32.000Z` |
| Salience files | `.gitattributes, README.md, test.parquet, train.parquet` |

FAB-G 公开 README 明确：共享一个 Qwen3-VL-8B base，用六个 LoRA adapter：`color`、`composition`、`line`、`light`、`brushstroke`、`final`。前五个做 yes/no attribute salience，final 只用被选中的 cues 做 emotion/arousal/valence/explanation。

Track1 含义：我们不能再把五个属性机械全塞进 prompt。正确做法是：

1. 从 caption/style/emotion 推断 2-3 个最关键 attribute levers。
2. 生成 prompt 只强化这些 salient levers。
3. AAS gate 也只检查“该样本真正需要的属性”，避免过度约束导致模板化。

### 5. EmoVIT

来源：GitHub `aimmemotion/EmoVIT`。

| 字段 | 抓取结果 |
|---|---|
| description | `[CVPR 2024] EmoVIT: Revolutionizing Emotion Insights with Visual Instruction Tuning` |
| pushed_at | `2025-04-20T01:04:39Z` |
| updated_at | `2026-05-19T19:40:05Z` |
| stars/forks | `40` / `2` |
| top files | `FT.yaml, README.md, blip2_vicuna_instruct.py, emo, requirements_emo.txt, requirements_lavis.txt` |

公开 README 里有一个重要使用格式：`Predicted emotion: [emotion]. Reason: [explanation].` 这说明他们的技术传统不是只输出 emotion label，而是要求视觉证据解释。

Track1 含义：本地 AAS judge 应该让 VLM 输出：

| 子项 | 应问什么 |
|---|---|
| content | caption 主体、支持物、关系是否真的出现 |
| style | 是否符合 style family，而非 generic polished AI image |
| attribute | 哪些 brushstroke/composition/color/line/light 证据支撑 emotion |
| failure | 是否存在 sample-id、watermark、gallery/mockup、错误文字、物理逻辑失败 |

### 6. AffectiveArt2026 repo

来源：GitHub `lizihan363-ship-it/AffectiveArt2026`。

| 字段 | 抓取结果 |
|---|---|
| pushed_at | `2026-05-17T05:10:00Z` |
| updated_at | `2026-05-17T05:20:32Z` |
| top files | `README.md, Submission Instructions.md, assets` |

Track1 含义：它更像 submission instruction repo，不是 scorer repo。不要期待从这里拿到 hidden evaluator；主要用于确认格式和规则。

## 作者/合作者公开输出的可执行意义

### Cheng Zhang / Hongxia Xie / Wen-Huang Cheng 线

已验证输出：EmoArt、FAB-G/AGSR、EmoVIT。

对 Track1 的含义：

1. 官方 ontology 基本就是 EmoArt ontology：style、emotion、VA、五属性。
2. AAS evaluator 很可能重视 fine-grained attribute reasoning，而不是只看 caption noun matching。
3. 强队会用 EmoArt distribution 做训练/检索/筛选，不会只靠一键生成。

### Sicheng Zhao 线

公开信号：visual emotion / affective computing survey、emotion distribution learning、AICA 数据集和 bias/subjectivity 传统。

对 Track1 的含义：

1. Emotion 不应该被做成单一固定视觉模板。
2. 同一个 caption 的 emotion 表达可以是分布式的，因此 prompt 要保留 painterly freedom。
3. scorer 需要考虑 dataset bias 和 style/emotion prior。

### Jianlong Fu / Sanghoon Lee 线

公开信号更偏 multimodal generation、quality assessment、perceptual metrics。

对 Track1 的含义：

1. FID 之外可能有隐性质量审查风险，至少需要图像质量/清晰度/伪影 gate。
2. 生成图不能只是“像 reference”，还要避免 AI artifact、无效边框、乱码文字、空间逻辑错误。

## 对我们当前失败的解释

### 为什么 hybrid probe 掉分

`hybrid_probe_redteam_fid_pass4` 的本地 `fid_like` 更低，但官方 FID 更差。这说明我们本地 proxy 优化到了错误方向：可能更像某个局部 embedding/统计目标，却不像官方 reference artistic image set。

### 为什么 reference prompt 会模板化

之前 reference board 不够分布化：少数 reference 反复使用，且很多 poster prompt 被强行拉成统一 portrait poster、粗边框、中心人物、上下大字。这能提高局部 AAS，却会损害 FID distribution。

### 为什么继续爬作者有用

不是为了找 hidden answer，而是为了确认官方团队的研究偏好：

- 他们公开数据重视 56-style distribution；
- 他们公开算法重视 salience，不是属性堆叠；
- 他们公开模型重视 emotion reasoning，不是单标签；
- 他们公开规则明确惩罚 evaluator manipulation。

## 下一步工程任务

### P0：把公开信号落到 Track1 scorer

1. `style_family_retriever_v2`
   - 输入：sample caption/style/emotion。
   - 输出：20-50 张同 style / 同 medium / 同 emotion-neighborhood 的 EmoArt reference family。
   - 约束：不能只返回少数热门图；要覆盖 56-style distribution。

2. `attribute_salience_router`
   - 输入：caption + style + emotion。
   - 输出：2-3 个 salient attributes，如 color/composition/line/light/brushstroke。
   - 来源：FAB-G schema + EmoArt-Salience。

3. `track1_aas_judge_v2`
   - 让 VLM 输出 content/style/attribute 三分项。
   - attribute 只评 salient levers。
   - 加 failure flags：artifact、wrong support、wrong text、relation failure、physical logic failure。

4. `fid_distribution_ensemble_v2`
   - 不再用单一 `fid_like`。
   - 特征包括：style family CLIP/SigLIP distance、颜色统计、edge density、aspect/support、poster/text density、template similarity。

### P1：把公开作者资源转成本地工具

1. 拉取/审查 `attributes_alignments.py` 和 `clip_score.py` 的实现，决定是否直接复用或重写为我们的 scorer feature。
2. 检查 EmoArt-Salience parquet schema，做一个轻量 salience prior table。
3. 用 EmoVIT reasoning style 写 Track1 redteam prompt，不需要训练 EmoVIT。

### P2：用于最后两次提交的策略

1. 下一次提交仍应是 controlled-B / v5 clean package，而不是纯 FID-like greedy。
2. 等官方返回第三个 anchor 后，更新 scorer。
3. 最后一枪再决定扩大替换还是回退稳健包。

## 不能做/不该做

1. 不爬、不请求、不使用 hidden test labels。
2. 不把 private test set 或提交图公开上传。
3. 不把 evaluator instruction 写进图或 prompt。
4. 不把公开 EmoArt rows 当 hidden gold，只能当 distribution prior。
5. 不直接相信 Gemini-agent 或网络搜索返回的未验证仓库链接；本轮已发现 gemini-agent brief 里有疑似错误 repo，已排除。

## 源链接

- Codabench Track1：https://www.codabench.org/competitions/16299
- AffectiveArt OpenReview：https://openreview.net/forum?id=LbbHX8ofXZ
- EmoArt project：https://zhiliangzhang.github.io/EmoArt-130k/
- EmoArt-130k HF：https://huggingface.co/datasets/printblue/EmoArt-130k
- EmoArt-130k GitHub：https://github.com/zhiliangzhang/EmoArt-130k
- FAB-G GitHub：https://github.com/zhiliangzhang/FAB-G
- EmoArt-Salience HF：https://huggingface.co/datasets/printblue/EmoArt-Salience
- EmoVIT GitHub：https://github.com/aimmemotion/EmoVIT
- Affective Computing survey：https://arxiv.org/abs/1911.05609
