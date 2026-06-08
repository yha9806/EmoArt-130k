# P1 Method Intelligence Crawl

日期：2026-06-08

目标：在 P0 公开 EmoArt 数据已经本地化并初步使用的前提下，继续检索 AffectiveArt / EmoArt 发起者与合作者的公开论文、代码、数据集和算法成果，并把它们转成 Track1 可执行优化模块。

结构化 method cards 已写入：

- `experiments/emoart_origin_research_20260608/p1_method_cards.json`

## 结论优先

1. P1 确实更重要。P0 给我们素材，P1 决定怎么把素材转成分数。
2. 最值得立即落地的不是“再爬更多 reference”，而是四个方法模块：
   - `salience_aware_prompt_compiler`
   - `style_family_distribution_retriever`
   - `emotion_instruction_aas_proxy`
   - `candidate_repair_prompt_adapter`
3. 对 Track1 来说，P1 的核心不是训练一个新模型替代 Gemini。最终生成仍然可以用 Gemini；P1 的作用是让 Gemini 的输入、候选筛选、修复和最终包选择更像强队系统。
4. 第一名和我们之间的公开差距主要是 FID。AAS 现在已经很高，继续一味追求 caption lock 会让图像更模板、更不贴 EmoArt reference distribution。
5. 因此下一轮不应该盲交 B80/B120/B160。它们是可用候选，但应该先接入 P1 proxy，判断哪个包真正更可能降低官方 FID，而不是只看局部视觉质量。

## P1 直接落地卡

### 1. FAB-G / AGSR：属性显著性瓶颈

来源：

- 论文：https://arxiv.org/abs/2605.15755
- 代码：https://github.com/zhiliangzhang/FAB-G
- 数据：https://huggingface.co/datasets/printblue/EmoArt-Salience

公开方法：

- 五个属性 agent：`color`、`composition`、`line`、`light`、`brushstroke`。
- 先判断每个属性是否对 emotion 判断真正 salient。
- final agent 只基于保留下来的 cues 做 emotion / VA / explanation。

转成 Track1：

- 当前 prompt 的问题是容易把所有属性都塞进去，导致画面过度约束和模板化。
- Track1 应改成：每张图只选 2-3 个关键 visual levers。
- 对 AAS judge 也一样：不能让 judge 看到 caption 后自动“脑补通过”，要让它验证图像里是否真的有这些 salient visual cues。

落地模块：

- `salience_aware_prompt_compiler`
- `attribute_salience_aas_gate`

优先级：最高。

### 2. EmoVIT：情绪视觉指令调优

来源：

- 论文：https://arxiv.org/abs/2404.16670
- 代码：https://github.com/aimmemotion/EmoVIT
- CVPR 页面：https://openaccess.thecvf.com/content/CVPR2024/html/Xie_EmoVIT_Revolutionizing_Emotion_Insights_with_Visual_Instruction_Tuning_CVPR_2024_paper.html

公开方法：

- 用 GPT-assisted pipeline 生成 emotion visual instruction data。
- 训练 VLM 不只是描述物体，而是预测 emotion 并解释视觉证据。

转成 Track1：

- 我们不一定要训练 EmoVIT，但可以学习它的 prompt/judge 格式。
- 当前 AAS gate 容易被 caption anchor 牵引。EmoVIT 方向提醒我们：review prompt 应该先问“图像本身有哪些情绪证据”，再看 caption。

落地模块：

- `emotion_instruction_aas_proxy`
- `blind_affective_evidence_review`

优先级：高。

### 3. PromptFix：VLM 辅助 prompt adapter / repair

来源：

- 论文：https://arxiv.org/abs/2407.00640
- GitHub：https://github.com/microsoft/PromptFix
- GitHub mirror/implementation signal：https://github.com/yeates/PromptFix

公开方法：

- 用 VLM 理解输入图像状态，再通过 prompt adapter 生成修复 prompt。
- 核心不是从零生成，而是“识别缺陷 -> 定向修复”。

转成 Track1：

- 我们现在很多失败不是模型完全不会画，而是局部错误：手穿玻璃、旗帜物理错误、士兵追赶平民、画轴结构缺失、Kremlin reference 不准、文字过度/乱码。
- 与其无限重跑，不如把失败样本转成 repair prompt queue。

落地模块：

- `candidate_repair_prompt_adapter`
- `hardcase_repair_queue`

优先级：高。

### 4. AICA / Visual Emotion Analysis：affective gap 与分布学习

来源：

- 综述：https://arxiv.org/abs/2106.16125
- GitHub list：https://github.com/Priscilla-Lu/Affective-Image-Content-Analysis

公开方法：

- 视觉情绪分析不是确定性物体分类，而是存在 affective gap、主观性、domain adaptation、label distribution。
- 情绪和视觉风格之间有分布关系。

转成 Track1：

- Track1 不应把每个 caption 独立生成，而应维护 batch-level distribution。
- 我们要的是整包 FID + 单图 AAS；这天然是多目标优化。

落地模块：

- `emotion_style_distribution_router`
- `batch_level_fid_diversity_gate`

优先级：高。

### 5. Official proposal / FLUX-LoRA 风格适配信号

来源：

- OpenReview：https://openreview.net/forum?id=LbbHX8ofXZ
- PDF：https://openreview.net/pdf?id=LbbHX8ofXZ
- EmoArt project：https://zhiliangzhang.github.io/EmoArt-130k/

公开方法信号：

- 官方 proposal 明确 challenge 建在 EmoArt 132k / 56 style 上。
- proposal 里曾暴露 style adaptation / LoRA-like baseline 与 AAS attribute description 思路。
- 当前邮件/规则已经更新为 FID + AAS 各 50%，但“style distribution adaptation 很重要”这个信号没有变。

转成 Track1：

- 我们不能在 Gemini 内部训练 LoRA。
- 但可以在上游模拟 LoRA 的效果：style-family reference retrieval + route-specific prompt packet + package-level selection。

落地模块：

- `style_family_distribution_retriever`
- `style_conditioned_fid_proxy`

优先级：最高。

## P1 辅助落地卡

### 6. Micro-expression relation graph：关系语义 gate

来源：

- MER-GCN：https://arxiv.org/abs/2004.08915
- MER survey：https://arxiv.org/abs/2012.11307
- BASIC Lab：https://basiclab.lab.nycu.edu.tw/

公开方法：

- 微表情识别强调 subtle cues、region/graph relation、鲁棒性。

转成 Track1：

- 领域不同，但思想可迁移到 relation semantics。
- 之前 `track1_0747` 的问题不是对象缺失，而是关系读法错：从“保护撤离”变成“追赶平民”。

落地模块：

- `relation_semantics_graph_gate`

优先级：中。

### 7. EmoArt metric code：文本/属性/模板 proxy

来源：

- GitHub：https://github.com/ZHILIANGZHANG/EmoArt-130k

公开方法：

- 有 CLIP score、attribute alignment、TTR、MTLD、Shannon entropy 等评价代码。

转成 Track1：

- 用来做 prompt/template diversity proxy。
- 防止我们生成的图和 prompt 都变成统一模板。

落地模块：

- `aas_text_and_template_proxy`

优先级：高。

### 8. EmoVerse：body-action-scene 分解

来源：

- HF：https://huggingface.co/datasets/PiLabUSTC/EmoVerse-Dataset
- arXiv：https://arxiv.org/abs/2511.12554

公开方法：

- 用 body-action-scene triplets、情绪描述、grounding evidence 表达视觉情绪。

转成 Track1：

- 对叙事性 caption 做预分解：谁、做什么、在哪里、关系是什么、情绪如何表达。
- 适合战场、海报、火车、群众、人物交互类 hard cases。

落地模块：

- `body_action_scene_content_lock`

优先级：中。

## 观察队列

### Sanghoon Lee：perceptual quality / multimedia evaluation

来源：

- OpenReview organizer profile：https://openreview.net/forum?id=LbbHX8ofXZ
- Google Scholar profile：https://scholar.google.com/citations?user=Q1XQpC8AAAAJ

可迁移点：

- 方向上支持我们加入 perceptual quality 和 artifact gate。
- 但第一轮还没有抓到足够直接的 Track1 generation code，所以暂时作为 watch card。

落地建议：

- 先用现有 image metadata、placeholder hash、artifact review、style-conditioned visual gate。
- 不要马上追求复杂 IQA 模型。

### Hongxia Xie / generation-data lineage

来源：

- Research page：https://www.hongxiaxie.net/research

可迁移点：

- MindPower、CookAnything 等项目说明该团队偏好多阶段、多模态、数据驱动 generation pipeline。
- 这支持我们的系统方向：plan -> retrieve -> generate -> review -> repair -> package select。

落地建议：

- 作为架构信号即可。
- 当前优先级低于 FAB-G、EmoVIT、PromptFix、AICA。

## 对当前提交包的影响

### 不应马上做的事

- 不建议直接提交 B160 这种“看起来视觉更好”的包作为最后冲奖包。
- 不建议把 full1000/broad generation 当成最终包。
- 不建议继续只靠人工看小样本选包。

### 应该马上做的事

1. 用 `p1_method_cards.json` 建一个小型工程计划。
2. 先实现三个轻量 proxy，不训练大模型：
   - style-family distribution proxy；
   - salience-aware prompt/attribute proxy；
   - placeholder/artifact/template hard gate。
3. 用这些 proxy 重新比较：
   - v3 official anchor；
   - B80；
   - B120；
   - B160；
   - current champion；
   - any high-risk full1000 variants。
4. 再决定最后两次提交中的下一包。

## 推荐执行顺序

### Step 1: Method cards freeze

已完成：

- `p1_method_cards.json`
- 本报告

### Step 2: P1-light proxy implementation

优先实现：

- `track1_style_family_distribution_proxy`
- `track1_salience_prompt_proxy`
- `track1_package_hard_gate_report`

目标不是完美模拟官方，而是防止明显错误提交：placeholder、模板过强、reference 距离太远、过度海报化、AAS 属性堆叠。

### Step 3: Re-score existing packages

对以下包跑 proxy：

- current champion
- v3 gate7
- B80 fixed
- B120 fixed
- B160 fixed
- full1000 no fallback
- safe/strict subsets

输出：

- package-level ranking
- sample-level risk table
- 建议提交包

### Step 4: Human review only where proxy disagrees

人工不再全量看 1000 张；只看：

- proxy 认为收益高但 AAS 风险高的样本；
- proxy 认为 FID 好但人眼觉得差的样本；
- B80/B120/B160 的差异样本。

## 当前判断

P1 的公开方法充分支持我们继续优化 Track1，但它不会自动给出一个“稳赢”包。真正的收益来自把这些公开方法压成轻量工程模块，然后反向筛掉错误提交。

下一步最合理动作：**先实现 P1-light proxy，不马上花提交机会。**
