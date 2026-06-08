# EmoArt Official/Author Evidence Crawler Design

日期：2026-06-08

## Goal

建立一个 Track2 最终提交前使用的 official/author-only evidence crawler。它的任务不是泛泛做文献综述，而是把官方赛事材料、EmoArt 作者团队公开数据、作者代码和本地已下载 EmoArt-130k 结构化成可审计证据表，用于最后一次 v22 label/text 融合。

核心原则：最终标签仲裁只使用官方与作者体系内证据。外围数据集可以做背景、文本风格参考或风险提示，但不能直接推动 final emotion label 修改。

## Evidence Boundary

### Final Arbitration Sources

这些源可以进入 final label arbitration：

1. 官方赛事源：
   - Codabench Track2 页面、规则、评分字段。
   - OpenReview AffectiveArt Challenge proposal。
   - 官方邮件中发布的评分说明、提交说明、test-set 说明。
   - 我们自己的线上提交反馈和官方分数记录。

2. EmoArt 作者团队公开源：
   - EmoArt paper。
   - EmoArt project page。
   - `printblue/EmoArt-130k`, `printblue/EmoArt-5k`, `printblue/EmoArt-Salience`。
   - `zhiliangzhang/EmoArt-130k`, `zhiliangzhang/FAB-G`。
   - 作者团队相关输出，例如 FAB-G/AGSR、EmoVIT、Hongxia Xie / AVC Lab 页面。

3. 本地已下载数据：
   - 默认候选路径：`/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k`。
   - `Annotation.json`。
   - 56 个风格 `.tar.gz` 包。
   - 既有 embedding/prediction cache，只作为可复现实验资产，不直接覆盖官方/作者证据。

### Auxiliary-Only Sources

这些源只做辅助，不进入 final label arbitration：

- ArtEmis。
- WikiArt Emotions。
- EmoSet。
- EmoVerse。
- MMArt。
- BAM。
- MART / abstract painting emotion datasets。
- Artpedia / ARTstract。
- 通用情绪图像、视频、语音、生理信号数据集。

这些资源可以用于：

- description 风格审计。
- 解释文本质量参考。
- 背景论文索引。
- 提醒某些标签空间冲突。

不能用于：

- 直接把 Track2 某一行 emotion 改成外围数据标签。
- 推翻官方线上反馈已经证明有效的方向。
- 冒充 hidden gold。

## Evidence Schema

所有采集结果写入统一 evidence schema，爬取逻辑和仲裁逻辑分离。

字段：

- `evidence_id`
- `source_name`
- `source_type`
- `source_url`
- `source_scope`: `official`, `author`, `local_author_data`, `auxiliary`
- `retrieved_at`
- `snapshot_path`
- `license_or_terms`
- `claim`
- `claim_type`: `ontology`, `dataset_schema`, `label_prior`, `submission_rule`, `official_score`, `code_method`, `description_style`
- `track2_actionability`: `direct_label`, `description_guard`, `style_prior`, `format_rule`, `background_only`, `blocked`
- `confidence`: `high`, `medium`, `low`
- `risk_flags`
- `notes`

## Components

### 1. Source Registry

维护待采集源列表和采集状态。

输出：

- `experiments/emoart_origin_research_20260608/official_author_source_registry.jsonl`
- `experiments/emoart_origin_research_20260608/official_author_source_registry.csv`

每个 source 包含：

- source id。
- URL 或本地路径。
- source boundary。
- parser type。
- expected artifacts。
- last seen hash。

### 2. Local EmoArt Inventory

扫描本地 EmoArt-130k，不重新下载 65GB 数据。

默认路径通过配置提供，不硬编码：

- CLI 参数：`--emoart-root`
- 环境变量：`EMOART_130K_ROOT`
- fallback：`/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k`

需要提取：

- `Annotation.json` 是否存在。
- row count。
- key schema。
- style/package inventory。
- tar count and total size。
- image_path/request_id/description 结构。
- 如果可安全快速解析，则统计 style-level label and attribute priors。

输出：

- `experiments/emoart_origin_research_20260608/local_emoart130k_inventory.json`
- `experiments/emoart_origin_research_20260608/local_emoart130k_style_inventory.csv`

### 3. Public Snapshot Fetcher

抓取官方/作者公开网页和 README，保存快照，避免源端变化影响复现。

采集原则：

- 限流。
- 明确 User-Agent。
- 不绕过登录、不抓私有数据。
- Codabench 已登录页面和官方邮件不做自动爬取，改为人工导出的结构化 JSON/CSV。

输出：

- `experiments/emoart_origin_research_20260608/snapshots/`
- `experiments/emoart_origin_research_20260608/crawl_registry.jsonl`

### 4. Track2 Actionability Matrix

把 source evidence 转成 Track2 决策矩阵。

核心列：

- evidence id。
- source scope。
- claim。
- applicable label family。
- applicable style family。
- affected submission field。
- allowed use。
- blocked use。
- final-shot priority。

输出：

- `experiments/emoart_origin_research_20260608/track2_official_author_actionability_matrix.csv`
- `experiments/emoart_origin_research_20260608/track2_official_author_actionability_matrix.md`

### 5. V22 Evidence Policy

把证据矩阵压缩成 final-shot 可执行策略。

必须包含：

- 哪些证据允许推动 `content -> calm`。
- 哪些证据只能做 description guard。
- 哪些证据必须 hold。
- 如何处理 exact duplicate / near duplicate / same series / same style。
- 如何保持最高 description anchor。
- 如何避免 evaluator manipulation。

输出：

- `experiments/emoart_origin_research_20260608/v22_official_author_evidence_policy_zh.md`

## Data Flow

```text
source registry
  -> public snapshot fetcher
  -> evidence schema rows
  -> actionability matrix
  -> v22 evidence policy

local EmoArt-130k
  -> inventory
  -> style/emotion/attribute priors
  -> actionability matrix
  -> v22 evidence policy

official score ledger
  -> structured official feedback rows
  -> actionability matrix
  -> v22 evidence policy
```

## Error Handling

- Missing local EmoArt root: fail with explicit path instructions.
- Missing `Annotation.json`: inventory fails, crawler still records other sources.
- Unknown JSON schema: write schema sample and mark source `needs_parser_update`.
- Failed HTTP source: record failure row with status, do not silently skip.
- Codabench/manual evidence missing: mark official score import incomplete; do not infer scores from memory.
- Conflicting evidence: prefer official score feedback > official schema > author dataset > author method paper > auxiliary background.

## Testing

Required Track2-only tests:

- Local root detection and `Annotation.json` schema parsing.
- Source registry parser.
- Evidence boundary classifier:
  - official/author/local_author_data may enter arbitration.
  - auxiliary cannot enter final label arbitration.
- Actionability matrix generation.
- V22 evidence policy generation.
- No overwrite of formal submission files.

Validation commands:

```bash
python3 -m unittest discover -s tests -p 'test_emoart_official_author*.py' -v
git diff --check
```

## Final Submission Implication

This crawler does not itself decide final labels. It creates the audited evidence base for v22.

Expected v22 direction under current known evidence:

- Use highest known description anchor as base.
- Preserve v21-confirmed `content -> calm` gains.
- Build `content -> calm` ladder candidates using official/author evidence only.
- Use FAB-G/AGSR as description guard, not as a blind classifier.
- Reject broad cross-quadrant label changes unless official/author duplicate evidence is overwhelming.

## Review Gate

Before implementation, this spec must be reviewed against the current final-shot constraint:

> The final submission should use official and EmoArt-author data as the only arbitration baseline.

If this constraint changes, update the source boundary before writing implementation code.
