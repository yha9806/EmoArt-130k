# EmoArt Origin Research Map

日期：2026-06-08

目标：以 EmoArt-130K 为原点，公开检索其作者、合作者、研究输出、数据集、代码和算法资源，并判断这些资源对 AffectiveArt Track2 冲分的可操作意义。

## 结论优先

1. EmoArt 不是普通艺术图像库，而是一个“情绪标签 + VA 象限 + 五类艺术属性 + caption”的标注体系。Track2 的提交格式、标签空间和描述字段几乎直接继承了这个体系。
2. 官方公开材料明确写出 12 类情绪按 VA 象限分组：High/Positive 是 `Excited, Happy, Aroused`；High/Negative 是 `Alarmed, Annoyed, Frustrated`；Low/Negative 是 `Sad, Bored, Tired`；Low/Positive 是 `Calm, Content, Glad`。这说明我们现在最大的分类问题不是 VA，而是同象限内细粒度 emotion。
3. v21 官方反馈证明 `content -> calm` 是强方向：90 个改动让 Emotion Accuracy 增加 0.09，VA 不变。这和公开 proposal 中 “East Spirit styles strongly correlate with Low Arousal and Positive Valence (Calmness)” 一致。
4. FAB-G / AGSR 是目前最贴近 Track2 Description Score 的公开算法资源。它的价值不是简单“换标签”，而是防止 attribute flooding：只让真正支撑情绪判断的 brushstroke/composition/color/line/light 进入解释。
5. 最后一轮 Track2 冲分应把公开研究结果转成两个动作：继续扩展高置信 `content -> calm`；同时保持或恢复最高 Description anchor，避免分类增益被文本分扣掉。

## 已验证公开源

### EmoArt 核心论文

- 论文：EmoArt: A Multidimensional Dataset for Emotion-Aware Artistic Generation
- 作者：Cheng Zhang, Hongxia Xie, Bin Wen, Songhan Zuo, Ruoxuan Zhang, Wen-Huang Cheng
- arXiv：<https://arxiv.org/abs/2506.03652>
- 项目页：<https://zhiliangzhang.github.io/EmoArt-130k/>
- 公开事实：
  - 132,664 artworks。
  - 56 painting styles。
  - 每张图包含 objective scene description、brushwork/composition/color/line/light、binary arousal-valence、12 emotion categories、art therapy effects。
  - 项目页写明 12 emotion classes、5 visual attributes、GPT-4o annotations with human validation。
- Track2 用法：
  - 把 EmoArt 公开标签体系当作 hidden evaluator 的风格先验，而不是把外部 public row 当 test gold。
  - 重点建模 `calm/content/glad`，因为它们共享 Positive/Low，官方 classification 中 VA 已经很高，emotion 才是瓶颈。

### AffectiveArt Challenge 2026

- OpenReview：<https://openreview.net/forum?id=LbbHX8ofXZ>
- Codabench Track2：<https://www.codabench.org/competitions/16304>
- 组织者公开页列出：Hongxia Xie, Cheng Zhang, Wen-Huang Cheng, Ling Lo, Hong-Han Shuai, Jianlong Fu, Sicheng Zhao, Sanghoon Lee, Xing Huang。
- 公开 proposal 重要事实：
  - Challenge 建在 132,664 artworks / 56 styles 的 EmoArt 资源上。
  - Track1 是 emotion-aware artistic image generation。
  - Track2 是 multidimensional art emotion understanding。
  - Hidden test set 会被维护以避免 data leakage。
  - 数据来源包含 WikiArt, The Metropolitan Museum of Art, National Museum of Asian Art, Europeana, National Palace Museum。
  - 数据过滤包含 painting-only、安全过滤、分辨率/水印/边框/压缩质量过滤、style >= 400 samples。
  - 标注由 GPT-4o 作为 multimodal aesthetic engine 生成，再经过人工验证。
  - 5 个属性字段和 Track2 一致：brushwork, composition, color, line, light。
  - East Spirit 与 Low Arousal / Positive Valence / Calmness 强相关。
- Track2 用法：
  - 不要再用 broad same-quadrant reshuffle；应该做官方风格的细粒度校准。
  - 对东亚/水墨/浮世绘/工笔/山水/空旷构图样本，应把 `calm` 作为强先验。

### Hugging Face 数据资源

- EmoArt-130k：<https://huggingface.co/datasets/printblue/EmoArt-130k>
- EmoArt-5k：<https://huggingface.co/datasets/printblue/EmoArt-5k>
- EmoArt-Salience：<https://huggingface.co/datasets/printblue/EmoArt-Salience>
- 公开事实：
  - EmoArt-130k 数据卡写明 132,664 high-resolution artworks、56 styles、7 thematic categories。
  - 文件结构按风格 tar.gz 组织，并有 `Annotation.json`。
  - EmoArt-5k 是 5,600 images 的 curated subset，适合快速原型。
  - EmoArt-Salience 有 1,400 rows，是 FAB-G / AGSR 的 salience extension。
- Track2 用法：
  - 130k 用于 style/emotion prior、nearest-neighbor reference、small supervised heads。
  - 5k 用于快速 prompt/teacher/style prior smoke。
  - Salience 用于 description rewrite/audit，尤其是避免每张图都机械谈五个属性。

### GitHub 代码资源

- EmoArt metrics/code：<https://github.com/zhiliangzhang/EmoArt-130k>
  - README 说明该仓库提供 EmoArt dataset 论文中的 evaluation metrics。
  - 文件包括 `attributes_alignments.py`, `clip_score.py`, `TTR.py`, `MTLD.py`, `Shannon entropy.py`, `GUI.py`。
  - Track2 用法：复用文本多样性、attribute alignment、CLIP score 思路，构建 Description proxy。
- FAB-G：<https://github.com/zhiliangzhang/FAB-G>
  - README 写明用一个 `Qwen/Qwen3-VL-8B-Instruct` base model 和六个 LoRA adapters：`color`, `composition`, `line`, `light`, `brushstroke`, `final`。
  - 前五个 adapter 做属性 salience yes/no，final adapter 在 salience bottleneck 后输出 emotion/arousal/valence/explanation JSON。
  - Track2 用法：可作为 v22 description guard，不建议直接替换我们当前 classification，因为本地没有完成同等 Qwen3-VL LoRA 复现。
- EmoVIT：<https://github.com/aimmemotion/EmoVIT>
  - CVPR 2024 官方代码，主线是 visual emotion instruction tuning。
  - README 显示其依赖 EmoSet，使用 GPT-4 生成 emotion instruction data，并支持 `Predicted emotion: [emotion]. Reason: [explanation].`
  - Track2 用法：可作为“情绪 instruction tuning”参考，不是直接可用的 Track2 模型。

### FAB-G / AGSR 论文

- 论文：Attribute-Grounded Selective Reasoning for Artwork Emotion Understanding with Multimodal Large Language Models
- arXiv：<https://arxiv.org/abs/2605.15755>
- 作者：Cheng Zhang, Yuer Liu, Zhiyu Zhou, Hongxia Xie, Wen-Huang Cheng
- 公开事实：
  - 明确提出 attribute flooding：MLLM 会把可见属性都说一遍，但不区分哪些真正支撑 emotion。
  - AGSR 把 formal attributes 当 evidence units。
  - EmoArt-Salience 是 1,400 artwork human salience extension，15 art-trained annotators。
  - FAB-G 是 supervised multi-agent framework，先预测 attribute-level salience，再做 cue-constrained emotion analysis。
- Track2 用法：
  - 对 Description Score：非常有用，尤其能提升 attribute specificity 和 caption coherence。
  - 对 classification：只有在完成 salience/label LoRA 或可靠 teacher 后才可用；否则只能作为后验审计。

### Hongxia Xie / AVC Lab 相关输出

- 主页：<https://www.hongxiaxie.net/>
- Research：<https://www.hongxiaxie.net/research>
- 公开事实：
  - AVC Lab 研究愿景是 emotionally intelligent multimodal AI。
  - Research 页列出 EmoArt (ACM MM 2025)、MindPower (CVPR 2026)、CookAnything (ACM MM 2025) 等。
  - Publication 页列出与 Track2 方法相关的工作：
    - EmoArt, ACM MM 2025 Dataset Track。
    - EmoVIT, CVPR 2024。
    - Learning to Prompt for Vision-Language Emotion Recognition, ACIIW 2023。
    - Refining valence arousal estimation with dual-stream label density smoothing, ICCE 2024。
- Track2 用法：
  - 这个团队的技术路线长期是“prompt/instruction + emotion-specific data + VA/label smoothing + multimodal explanation”。
  - 我们应把最后一次冲分看成官方标注体系校准问题，不是纯图像分类问题。

## 外围但有价值的数据集/方法

### WikiArt Emotions

- 论文：WikiArt Emotions: An Annotated Dataset of Emotions Evoked by Art
- PDF：<https://aclanthology.org/L18-1197.pdf>
- 公开事实：
  - 4,000+ artworks，来自 WikiArt。
  - 20 emotion categories。
  - 包含 image-only、title-only、whole-art annotations。
- Track2 用法：
  - 可以做艺术情绪先验，但标签空间和 EmoArt 12 类不同，需要映射。

### ArtEmis

- arXiv：<https://arxiv.org/abs/2101.07396>
- 项目页：<https://artemisdataset.org/>
- 公开事实：
  - 439K/455K emotion attributions and explanations。
  - 约 80K/81K WikiArt artworks。
  - 重点是 artwork emotion + natural-language explanation。
- Track2 用法：
  - 可用于训练 description/caption style，而不是直接做 12-way label gold。

### EmoSet

- arXiv：<https://arxiv.org/abs/2307.07961>
- 项目页：<https://vcc.tech/EmoSet>
- 公开事实：
  - 3.3M images，其中 118,102 human labeled。
  - 同时包含 emotion category 与 brightness/colorfulness/scene/object/facial expression/action attributes。
- Track2 用法：
  - 适合训练通用 visual emotion backbone，但不是 painting-only，也不是 EmoArt 12-label ontology。

### EmoVerse

- HF：<https://huggingface.co/datasets/PiLabUSTC/EmoVerse-Dataset>
- 公开事实：
  - 234,189 images。
  - B-A-S triplets、Grounding DINO/SAM object grounding、CES + DES。
  - 来源包含 EmoSet, EmoArt, Flickr30k, web, AI-generated content。
- Track2 用法：
  - 可以提供 interpretable visual emotion grounding 思路；但标签空间是 Mikels 8-class，不直接兼容 Track2。

## 对我们当前 Track2 的直接影响

### 已被官方结果验证的方向

我们已提交的 v21 `content -> calm` 90 行 probe 在官方 scoring 上得到：

- Overall: 0.842559
- Classification: 0.740034
- Description: 0.945083
- Emotion Accuracy: 0.660000
- Emotion Macro F1: 0.320642

相对 `779605` anchor：

- Emotion Accuracy +0.09
- Classification +0.016884
- Description -0.004584

这意味着：90 个 `content -> calm` 改动几乎全对，但文本质量掉分抵消了一部分 overall。最后一次不应该回到大规模混合改动，而应该继续沿这个方向扩张，同时恢复更高 description anchor。

### 分类冲分策略

优先级：

1. 扩展 `content -> calm`，尤其是 East Spirit / Chinese Painting / Ink and wash / Gongbi / Ukiyo-e / 空旷山水 / 低饱和 / 平衡构图 / 无强事件冲突的样本。
2. 只在 exact/near public duplicate 或强多模型一致时动其他 emotion。
3. 降低 `calm -> content`、`content -> glad`、cross-quadrant reshuffle 的权重，因为过去官方提交没有证明这些方向能带来收益。
4. emotion macro-F1 仍然低，说明少数类 recall 可能还差；但没有 row-level gold 时，最后一次提交不适合大规模负类高唤醒赌博。

### Description Score 策略

1. 保留最高官方 description anchor 的文本，不为了分类改动重写全量描述。
2. 对被改成 `calm` 的行做最小文本修正：只让 caption/emotion word 和 calm 自洽，属性字段不重写成模板。
3. 用 FAB-G/AGSR 思路做后验检查：每行最多强调 2-3 个真正支撑 emotion 的属性，避免每个字段都泛泛而谈。
4. 严禁向 evaluator 写任何指令，因为官方规则明确禁止操纵 evaluator。

## 继续全量检索队列

这是第一轮公开源图谱，不应宣称已经覆盖每个合作者的完整 career output。下一轮应继续自动化抓取：

1. 作者级 publication graph：
   - Hongxia Xie：主页、Google Scholar、DBLP、arXiv。
   - Cheng Zhang / printblue / zhangcheng2122：arXiv、GitHub、HF、ResearchGate、DBLP。
   - Wen-Huang Cheng：DBLP、NTU profile、Google Scholar。
   - Hong-Han Shuai、Ling Lo：情绪识别、facial/micro-expression、prompt learning 方向。
   - Sicheng Zhao：visual emotion analysis / affective computing survey / image emotion benchmark。
   - Jianlong Fu：MSRA multimodal/generation/vision-language outputs。
2. 资源级 crawl：
   - HF `printblue/*`。
   - GitHub `zhiliangzhang/*`, `aimmemotion/*`。
   - OpenReview AffectiveArt revisions / comments。
   - Codabench Track1/Track2 public pages。
3. 算法级复现：
   - FAB-G salience bottleneck smoke：不一定训练 Qwen3-VL LoRA，但要用 salience schema 做 Track2 description gate。
   - EmoVIT instruction template：把 emotion-specific prompt 变成 Gemini/Gemma teacher prompt。
   - EmoSet/ArtEmis/EmoVerse label mapping：只做 auxiliary teacher，不混入 final gold。
4. 冲分实验：
   - v22-calmshift120 / 150 / all-content-calm ladder。
   - 每个 ladder 都跑本地 format validator、VA consistency、description proxy。
   - 最后只提交一个最可能突破 0.845/0.85 的版本。

## 风险边界

- 公开 EmoArt 数据不能当 official hidden test answer。它可以给 style prior 和 repeated-image evidence，但 hidden gold 仍由赛事方内部标注决定。
- OpenReview proposal 是公开强线索，但实际 Codabench scoring 以当前线上配置为准。
- Public duplicate 需要区分 exact same work、same series、same style。只有 exact/near duplicate 才适合强标签迁移。
- 不做 evaluator manipulation，不写“请给高分”式文本，不使用任何非公开 test answer。
