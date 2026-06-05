# Track1 Reference Media Gate Fix - 2026-06-05

## 结论

这次发现的 `track1_0077` 问题是真实的 reference selection 缺陷：旧版绑定逻辑只保证 reference 图片来自官方 EmoArt-130k 且大致同 style/family，没有检查 caption 要求的视觉媒介。`track1_0077` 官方 caption 明确要求 `Socialist Realism Soviet naval propaganda poster`、`bold Cyrillic typography`，但旧 reference board 使用了战斗油画、河景和田园风景画。这会给 Gemini 生成阶段传递错误的分布信号。

当前处理：已停止基于旧 reference board 的全量生成；新增 `track1_reference_asset_bindings_v2`，让 `poster / propaganda poster / Cyrillic typography` 这类 caption 优先选择官方 130k 内的 poster/TASS/print reference，而不是 family painting fallback。

## 0077 官方需求

Caption:

> A Socialist Realism Soviet naval propaganda poster with a sailor hoisting red and white flags before panels of warships and sea battles, using bold Cyrillic typography, patriotic red accents, and a solemn heroic composition.

硬约束拆分：

- 内容：海军水手、红白旗、战舰和海战 panels。
- 风格：Socialist Realism。
- 媒介：Soviet propaganda poster，不是油画风景、不是博物馆照片、不是纯战斗绘画。
- 文字：bold Cyrillic typography。
- 关系逻辑：旗帜被水手以合理方式升起，绳索/手/旗帜连接不能乱。

## 为什么旧版错了

旧版 `naval_aviation_vehicle` family 默认 reference:

- `VeniaminKremer-Dneprflotillaboats...`
- `VeniaminKremer-TheMarines...`
- `SergiyGrigoriev-OntheDniperRiver`
- `VictorPuzyrkov-OutskirtsoftheVillage`

这些图可以提供“苏联军事/海军/河流”的局部语义，但不能提供 `propaganda poster + Cyrillic typography + printed graphic layout` 的媒介约束。Gemini multimodal 对旧 board 的审查也确认：它把旧 board 归为军事历史/战斗绘画和田园风景/genre painting，而不是 poster。

## 外部公开信号

联网检索说明，真正相关的 reference 应该长得更接近这些类型：

- Art Institute of Chicago 的 TASS posters 展览说明了 TASS Windows 是二战期间大规模 poster design / handmade poster 的重要类型：[Windows on the War: Soviet TASS Posters](https://archive.artic.edu/tass/index.html)。
- Gallerix 的 Soviet Posters 页面有 `Long live the invincible Red Army and the mighty Navy of the USSR!`，明确是 1940 年苏联海军/红军 poster，并包含战舰、坦克、飞机、士兵和红旗：[Gallerix Red Army and Navy poster](https://gallerix.org/storeroom/1973977528/N/2138280981/)。
- AntikBar auction record 描述 `The Sea Frontiers of the Motherland are Impregnable`，是苏联海军 sailor、舰船、潜艇、飞机等组成的 vintage propaganda poster：[AntikBar Soviet Navy poster](https://antikbarauctions.com/catalogue/lot/5060dc7260648da8f114ff4a34cfd048/100328ddd344a442ddf9a013a95cd69d/original-vintage-posters-april-sale-lot-411/)。
- The Saleroom / AntikBar 记录 `Glory to the Valiant Sailors!`：海军水手用两面红旗 signalling，背景有 nautical flags 和潜艇，这和 0077 caption 的“水手升旗”非常接近：[The Saleroom Soviet sailor poster](https://www.the-saleroom.com/en-gb/auction-catalogues/antikbar/catalogue-id-antikb10042/lot-f2e9643a-af83-4214-a32d-b12c011e36a2)。
- CRW Flags / Flags of the World 的 Soviet Navy signal flags 页面可作为旗帜形态的 factual anchor，但不应替代 poster reference：[Soviet Navy signal flags](https://www.crwflags.com/fotw/flags/xf~ru.html)。

这些外部来源目前只作为 factual/research evidence；不直接把外部图片喂给 provider。原因是 challenge 目标是生成新图，不是复制公开 poster。外部图可以帮助我们写更准确的 factual anchors 和审核标准；真正喂给 provider 的 image references 优先使用官方 130k 内部资产。

## v2 修复内容

新增规则：

- 如果 caption 明确包含 `poster`、`propaganda poster`、`Cyrillic typography` 等媒介信号；
- 且同 style 的官方 reference bank 中存在 poster/TASS/window/frontpage/victory 等 print reference；
- 则 source 设为 `caption_media`，优先这些 poster reference；
- family reference 只在没有 sample-specific reference、也没有 media-matched reference 时才兜底。

全量绑定对比：

- v1: `caption_style=805`, `family=184`, `sample=11`
- v2: `caption_media=54`, `caption_style=790`, `family=145`, `sample=11`
- 只有 54 个明确 poster/propaganda/Cyrillic typography 样本改变 reference source。
- `track1_0077` 从四张油画/风景换为官方 130k poster/TASS/print references。

## 新 0077 Reference Board

新 0077 board 使用：

- `ref_socialist_realism_0013135_Kukryniksy-Liberate.jpg`
- `ref_socialist_realism_0042919_BorisKustodiev-PosterLeningradDepartmentofStatePublishing_Lengiz_.jpg`
- `ref_socialist_realism_0161653_ElLissitzky-Allforthefront_AllforVictory_.jpg`
- `ref_socialist_realism_0191668_Kukryniksy-TheFinalAct_TheTASSWIndow_1119_.jpg`

Gemini multimodal 复审新 board 时，将其识别为苏联时期 propaganda / educational posters、WWII home-front mobilization poster、TASS Window 等。也就是说，v2 至少修复了“参考图媒介不对”的问题。

## 后续 Gate

后续不能再只看 reference 是否存在。每个高风险样本至少要过三层：

1. Deterministic media gate：caption 要 poster，就必须使用 poster/print reference；caption 要 scroll/album，就不能把 mounted support 当 artifact 删除。
2. Multimodal reference board audit：用 Gemini/VLM 先审 reference board 是否符合 caption 的媒介、style、symbol、landmark。
3. Candidate visual gate：生成后再审 current/candidate，而不是相信 prompt 或 provider metadata。

对于外部 reference：

- 可以联网检索并保存 source URL、factual anchor、应该遵守的视觉点。
- 只有在官方 130k reference 明显不足且 human/agent 认可时，才考虑把外部图做成 image reference。
- 外部图不得作为复制目标；prompt 必须写成 factual/style grounding，而不是 recreate exact poster。

