# Track1 V4 Distribution Shortlist

这个报告只用于找高价值审图队列，不是 accepted replacement manifest。

## 结论

- 来源报告数：`2`
- 正向 candidate pair：`801`
- 去重后正向 sample：`460`
- 进入 review queue：`80`
- 高 delta 队列：`26`，阈值 `delta >= 0.05`
- watchlist：`366`，低于 review 阈值但 delta 为正

## 风险规则

- 这里没有候选 AAS/VLM 审计，不能自动替换。
- FID 是包级分布指标，单样本 delta 只能作为候选优先级。
- 下一步必须做：官方 caption 约束检查、视觉审图、Gemini redteam、hybrid 包级 FID-like sanity。

## 来源统计

- `full1000` positive=`460` changed=`460` mean_positive_delta=`0.01855` (>0.01: 285, >0.02: 155, >0.03: 94, >0.04: 53, >0.05: 26, >0.06: 10)
- `partial757` positive=`341` changed=`341` mean_positive_delta=`0.016248` (>0.01: 196, >0.02: 94, >0.03: 53, >0.04: 27, >0.05: 10, >0.06: 5)

## Review Queue

1. `track1_0499` source=`full1000` tier=`high_delta_review` delta=`0.1049` distribution_delta=`0.3159` perceptual_delta=`0.3833` duplicate_options=`2`
   caption: Abstract minimalist composition with a dark blue-black monochrome field, subtle tonal gradients, and a thin shadowed frame creating a calm, mysterious atmosphere.
2. `track1_0790` source=`full1000` tier=`high_delta_review` delta=`0.084` distribution_delta=`0.3139` perceptual_delta=`0.2457` duplicate_options=`1`
   caption: Ukiyo-e scene of travelers on a windy shoreline struggling with yellow umbrellas and scattered belongings, with waves, distant mountains, birds, fine linework, muted colors, and a vertical calligraphy panel.
3. `track1_0221` source=`full1000` tier=`high_delta_review` delta=`0.0755` distribution_delta=`0.2718` perceptual_delta=`0.2317` duplicate_options=`2`
   caption: A Baroque monochrome etching of a seated nude figure with loose expressive cross-hatching and stark light-shadow contrast on aged paper.
4. `track1_0725` source=`full1000` tier=`high_delta_review` delta=`0.0738` distribution_delta=`0.2349` perceptual_delta=`0.2574` duplicate_options=`1`
   caption: Ukiyo-e scene of three women in patterned kimonos beneath drooping willow branches, with muted colors, delicate linework, and a calm outdoor setting.
5. `track1_0609` source=`full1000` tier=`high_delta_review` delta=`0.0734` distribution_delta=`0.261` perceptual_delta=`0.2282` duplicate_options=`2`
   caption: Ukiyo-e scene of two kimono-clad women standing beneath a blossoming tree with distant mountains, delicate linework, muted earthy colors, and a calm atmosphere.
6. `track1_0957` source=`full1000` tier=`high_delta_review` delta=`0.0693` distribution_delta=`0.2757` perceptual_delta=`0.1859` duplicate_options=`1`
   caption: Ukiyo-e style print of a dramatic red-skinned warrior in ornate robes raising a fist above a crouching green creature, with bold outlines, flat colors, and a tense theatrical composition.
7. `track1_0049` source=`full1000` tier=`high_delta_review` delta=`0.0676` distribution_delta=`0.2376` perceptual_delta=`0.2135` duplicate_options=`1`
   caption: A Socialist Realism propaganda poster with bold flat colors and sharp graphic lines portrays a Soviet soldier with a rifle and bayonet confronting a swastika-marked fascist figure across a map of northern and southern seas, creating a tense wartime composition.
8. `track1_0630` source=`full1000` tier=`high_delta_review` delta=`0.0646` distribution_delta=`0.2229` perceptual_delta=`0.2076` duplicate_options=`2`
   caption: A whimsical Ukiyo-e scene with a large cat looming above lively koi fish in pale blue water, using flat colors, delicate outlines, and playful composition.
9. `track1_0911` source=`full1000` tier=`high_delta_review` delta=`0.061` distribution_delta=`0.2239` perceptual_delta=`0.1826` duplicate_options=`1`
   caption: Ukiyo-e print of fierce warriors in flowing robes, one gripping a sword beneath curling flames, rendered in a vertical composition with fine ink lines and muted tan, orange, and black tones.
10. `track1_0401` source=`full1000` tier=`high_delta_review` delta=`0.0609` distribution_delta=`0.2451` perceptual_delta=`0.1604` duplicate_options=`2`
   caption: Ukiyo-e scene of an elderly seated man and a standing woman in patterned robes inside a tatami room with shoji screens, calligraphy above, delicate linework, and muted earthy colors.
11. `track1_0686` source=`full1000` tier=`high_delta_review` delta=`0.059` distribution_delta=`0.2298` perceptual_delta=`0.1639` duplicate_options=`2`
   caption: A Socialist Realism Soviet propaganda poster with Cyrillic text, Baltic republic emblems, a large red Roman numeral V, golden ribbons, oak leaves, and a pale map background on aged paper.
12. `track1_0976` source=`full1000` tier=`high_delta_review` delta=`0.0583` distribution_delta=`0.2253` perceptual_delta=`0.1629` duplicate_options=`1`
   caption: Abstract watercolor and ink composition with layered brown mask-like forms, a red and blue zigzag band, green washes, and bold black organic shapes in a loose, textured arrangement.
13. `track1_0839` source=`full1000` tier=`high_delta_review` delta=`0.058` distribution_delta=`0.2323` perceptual_delta=`0.1544` duplicate_options=`1`
   caption: Abstract geometric composition of intersecting black lines forming rectangles and triangles, with yellow blocks, blue diagonal hatching, gray striped areas, and visible colored-pencil texture on an off-white background.
14. `track1_0483` source=`full1000` tier=`high_delta_review` delta=`0.0576` distribution_delta=`0.2074` perceptual_delta=`0.1763` duplicate_options=`2`
   caption: Ukiyo-e style illustration of two fish, a large pink-red fish beneath a smaller dark mottled fish, with delicate linework, flat muted colors, and calligraphic inscriptions on a pale background.
15. `track1_0899` source=`full1000` tier=`high_delta_review` delta=`0.0572` distribution_delta=`0.2477` perceptual_delta=`0.1336` duplicate_options=`1`
   caption: Ukiyo-e scene of samurai warriors carrying spears through a snowy mountain pass among snow-laden pine trees, with crisp linework, muted winter colors, and a calm yet solemn atmosphere.
16. `track1_0740` source=`full1000` tier=`high_delta_review` delta=`0.0568` distribution_delta=`0.1764` perceptual_delta=`0.2025` duplicate_options=`1`
   caption: A Gongbi painting of robed figures gathered in a dim interior, rendered with fine controlled linework, muted earth tones, and a solemn, aged atmosphere.
17. `track1_0844` source=`full1000` tier=`high_delta_review` delta=`0.0562` distribution_delta=`0.2115` perceptual_delta=`0.1632` duplicate_options=`1`
   caption: A Ukiyo-e scene of travelers crossing a small arched bridge over a stream near a hillside building, framed by autumn trees, delicate linework, and muted earthy colors.
18. `track1_0771` source=`full1000` tier=`high_delta_review` delta=`0.0559` distribution_delta=`0.2257` perceptual_delta=`0.1469` duplicate_options=`1`
   caption: A Ukiyo-e woodblock print of women gathered inside a traditional room opening onto a quiet garden with rocks, pine trees, sliding screens, and delicate linework in a muted color palette.
19. `track1_0030` source=`full1000` tier=`high_delta_review` delta=`0.0552` distribution_delta=`0.2012` perceptual_delta=`0.167` duplicate_options=`2`
   caption: Ukiyo-e scene of travelers in traditional robes walking through a hilly landscape with trees, distant figures on a path, and Mount Fuji in the background, rendered with delicate linework and muted earthy colors.
20. `track1_0881` source=`full1000` tier=`high_delta_review` delta=`0.055` distribution_delta=`0.2007` perceptual_delta=`0.1654` duplicate_options=`1`
   caption: A Socialist Realism propaganda poster of a decorated Soviet pilot holding a flight map before military airplanes in a blue sky, framed by laurel wreaths, red ribbons, bold Cyrillic lettering, and heroic patriotic composition.
21. `track1_0949` source=`full1000` tier=`high_delta_review` delta=`0.0529` distribution_delta=`0.2292` perceptual_delta=`0.1231` duplicate_options=`1`
   caption: Abstract ink sketch of a distorted standing figure with oversized arms and fragmented facial features, framed by rough linear borders and dense cross-hatched shadows on aged beige paper.
22. `track1_0659` source=`full1000` tier=`high_delta_review` delta=`0.0523` distribution_delta=`0.1838` perceptual_delta=`0.1645` duplicate_options=`2`
   caption: Ukiyo-e woodblock print of three elegantly dressed figures walking in diagonal rain, with a large patterned parasol, delicate linework, muted colors, and a calm street scene atmosphere.
23. `track1_0998` source=`full1000` tier=`high_delta_review` delta=`0.052` distribution_delta=`0.2129` perceptual_delta=`0.1337` duplicate_options=`1`
   caption: Abstract pencil sketch of a fragmented interior scene with angular geometric forms, overlapping objects, loose graphite lines, and a muted beige paper texture.
24. `track1_0897` source=`full1000` tier=`high_delta_review` delta=`0.0517` distribution_delta=`0.2101` perceptual_delta=`0.1346` duplicate_options=`1`
   caption: Ink and wash painting of a serene garden scene with a twisted pine tree, bamboo, rocks, a thatched pavilion where two scholars sit at a table, and a lone figure standing by a simple fence, rendered with delicate brushwork and soft muted washes.
25. `track1_0614` source=`full1000` tier=`high_delta_review` delta=`0.0507` distribution_delta=`0.2064` perceptual_delta=`0.1311` duplicate_options=`2`
   caption: Ukiyo-e scene of two women in patterned kimonos standing on a hillside above a quiet coastal landscape, one holding a closed parasol, with delicate linework and a muted earthy color palette.
26. `track1_0945` source=`full1000` tier=`high_delta_review` delta=`0.0503` distribution_delta=`0.1982` perceptual_delta=`0.1373` duplicate_options=`1`
   caption: Ukiyo-e scene of three elegantly dressed women in patterned kimonos standing beneath blooming cherry branches, with delicate linework, muted colors, and a calm spring atmosphere.
27. `track1_0785` source=`full1000` tier=`medium_delta_review` delta=`0.0497` distribution_delta=`0.1635` perceptual_delta=`0.1681` duplicate_options=`1`
   caption: Ink and wash painting of a serene mountainous landscape with clustered trees, small hillside houses, winding paths, and calligraphy, rendered with delicate brushwork and a muted earthy color palette.
28. `track1_0168` source=`full1000` tier=`medium_delta_review` delta=`0.0484` distribution_delta=`0.1509` perceptual_delta=`0.1719` duplicate_options=`2`
   caption: Ukiyo-e scene of elegant women in traditional robes gathered on a wooden veranda, with fine linework, muted colors, patterned textiles, and a calm domestic atmosphere.
29. `track1_0787` source=`full1000` tier=`medium_delta_review` delta=`0.0483` distribution_delta=`0.2083` perceptual_delta=`0.1141` duplicate_options=`1`
   caption: Ukiyo-e portrait of a standing Japanese actor or warrior in an ornate patterned kimono with a sword, surrounded by calligraphy on aged paper with bold outlines and muted red, gold, and black tones.
30. `track1_0746` source=`full1000` tier=`medium_delta_review` delta=`0.0473` distribution_delta=`0.2141` perceptual_delta=`0.1011` duplicate_options=`1`
   caption: Ukiyo-e scene of two kimono-clad figures walking under yellow paper umbrellas through diagonal rain, with delicate linework, muted colors, and a calligraphy cartouche in the upper corner.
31. `track1_0233` source=`full1000` tier=`medium_delta_review` delta=`0.0472` distribution_delta=`0.2143` perceptual_delta=`0.1006` duplicate_options=`2`
   caption: Ink and wash painting of a serene mountainous landscape with towering rocky peaks, dense pine trees, small riverside pavilions, and delicate monochrome brushwork on a vertical scroll.
32. `track1_0832` source=`full1000` tier=`medium_delta_review` delta=`0.0469` distribution_delta=`0.1803` perceptual_delta=`0.1326` duplicate_options=`1`
   caption: Ukiyo-e scene of two women in patterned kimonos, one standing and one kneeling with a fan beside tall flowering plants, rendered with flat warm colors and delicate linework.
33. `track1_0180` source=`full1000` tier=`medium_delta_review` delta=`0.0466` distribution_delta=`0.1683` perceptual_delta=`0.1426` duplicate_options=`2`
   caption: Ukiyo-e woodblock print of travelers crossing a high arched bridge over a blue river, with wooden supports, stone embankments, and a small village landscape in muted colors and crisp linework.
34. `track1_0890` source=`full1000` tier=`medium_delta_review` delta=`0.0462` distribution_delta=`0.1859` perceptual_delta=`0.1217` duplicate_options=`1`
   caption: A Ukiyo-e landscape of a winding blue river with small boats, grassy banks, pine and willow trees, and distant huts, rendered with bold outlines and a muted green and blue palette.
35. `track1_0872` source=`full1000` tier=`medium_delta_review` delta=`0.046` distribution_delta=`0.1673` perceptual_delta=`0.1394` duplicate_options=`1`
   caption: A serene Ukiyo-e landscape with a winding stream flowing through rocky banks, dense green hills, and trees silhouetted against a dark sky, rendered with bold outlines and textured wave patterns.
36. `track1_0080` source=`full1000` tier=`medium_delta_review` delta=`0.0449` distribution_delta=`0.1635` perceptual_delta=`0.1362` duplicate_options=`2`
   caption: Ukiyo-e style scene of two dramatic warriors grappling in richly patterned robes, with bold outlines, flat colors, and a vertical woodblock print composition.
37. `track1_0616` source=`full1000` tier=`medium_delta_review` delta=`0.044` distribution_delta=`0.1908` perceptual_delta=`0.1021` duplicate_options=`2`
   caption: Ink and wash painting of a gnarled pine tree rising from rocky ground beside grasses and a quiet shoreline, with delicate brushwork, muted sepia tones, calligraphy, and red seals creating a serene traditional landscape.
38. `track1_0914` source=`full1000` tier=`medium_delta_review` delta=`0.0433` distribution_delta=`0.1755` perceptual_delta=`0.1137` duplicate_options=`1`
   caption: Ukiyo-e print of whimsical yokai-like figures and tiny travelers parading with umbrellas, instruments, and strange sea-creature forms, rendered with fine outlines, muted colors, and a playful fantastical mood.
39. `track1_0008` source=`full1000` tier=`medium_delta_review` delta=`0.0432` distribution_delta=`0.1837` perceptual_delta=`0.1039` duplicate_options=`2`
   caption: Ukiyo-e rural landscape with travelers walking along a winding path beside fields and a narrow stream, tall trees rising in the foreground, muted colors, fine linework, and vertical composition.
40. `track1_0991` source=`full1000` tier=`medium_delta_review` delta=`0.0432` distribution_delta=`0.1664` perceptual_delta=`0.1216` duplicate_options=`1`
   caption: A Baroque engraved courtyard scene with grand arcaded architecture, a central tower, small figures gathered below, and winged mythological figures flying through a dramatic sky.
41. `track1_0488` source=`full1000` tier=`medium_delta_review` delta=`0.043` distribution_delta=`0.1876` perceptual_delta=`0.0996` duplicate_options=`2`
   caption: A Ukiyo-e landscape with a winding blue river, small boats, a wooden bridge, riverside buildings, green trees, and distant mountains beneath a softly graded sky.
42. `track1_0628` source=`full1000` tier=`medium_delta_review` delta=`0.0429` distribution_delta=`0.1537` perceptual_delta=`0.1325` duplicate_options=`2`
   caption: A Socialist Realism Soviet poster with bold Cyrillic typography contrasts wartime devastation behind barbed wire with liberated Ukrainian peasants harvesting golden wheat under a bright sky, using strong graphic composition and earthy colors.
43. `track1_0590` source=`full1000` tier=`medium_delta_review` delta=`0.0417` distribution_delta=`0.1587` perceptual_delta=`0.1192` duplicate_options=`2`
   caption: Ukiyo-e style scene of three figures in traditional Japanese robes walking past a wooden building and garden, rendered with delicate linework and a muted earthy color palette.
44. `track1_0314` source=`full1000` tier=`medium_delta_review` delta=`0.0416` distribution_delta=`0.1602` perceptual_delta=`0.1171` duplicate_options=`2`
   caption: A Socialist Realism Soviet propaganda poster with a bold Cyrillic headline, heroic soldiers, an industrial worker, anti-fascist caricature scenes, and stark wartime panels in muted reds, blues, and beige tones.
45. `track1_0915` source=`full1000` tier=`medium_delta_review` delta=`0.0415` distribution_delta=`0.153` perceptual_delta=`0.1233` duplicate_options=`1`
   caption: Ukiyo-e style scene of two women in patterned kimono walking under a large umbrella in falling snow, with delicate linework, muted earthy colors, and calligraphic text in the background.
46. `track1_0140` source=`full1000` tier=`medium_delta_review` delta=`0.0413` distribution_delta=`0.1621` perceptual_delta=`0.1134` duplicate_options=`2`
   caption: Ukiyo-e style portrait of a woman in a patterned kimono standing beside a small tea stand with a kettle, cup, wooden furnishings, vertical calligraphy, muted colors, and delicate ink linework on aged paper.
47. `track1_0523` source=`full1000` tier=`medium_delta_review` delta=`0.0411` distribution_delta=`0.1477` perceptual_delta=`0.1261` duplicate_options=`2`
   caption: Ukiyo-e style circular composition of a busy river harbor with boats, a large moored vessel in the foreground, distant waterfront buildings, muted colors, fine linework, and Japanese calligraphy on a dark background.
48. `track1_0697` source=`full1000` tier=`medium_delta_review` delta=`0.041` distribution_delta=`0.1492` perceptual_delta=`0.1243` duplicate_options=`2`
   caption: Abstract composition of geometric pyramids, spheres, and block-like forms on a faint grid, layered with energetic sketchy lines, splattered textures, and a vivid yellow, teal, orange, and red palette.
49. `track1_0223` source=`full1000` tier=`medium_delta_review` delta=`0.0409` distribution_delta=`0.1753` perceptual_delta=`0.0974` duplicate_options=`2`
   caption: An Early Renaissance monochrome illustration of a robed seated figure beside a walled garden and gate, with a small church, trees, hills, and delicate hatched linework in a muted sepia palette.
50. `track1_0952` source=`full1000` tier=`medium_delta_review` delta=`0.0409` distribution_delta=`0.1456` perceptual_delta=`0.1274` duplicate_options=`1`
   caption: A Ukiyo-e portrait of an elegant woman in a richly patterned blue kimono within a decorative interior, rendered with flat color blocks, delicate linework, and a muted vintage palette.
51. `track1_0611` source=`full1000` tier=`medium_delta_review` delta=`0.0408` distribution_delta=`0.1661` perceptual_delta=`0.1058` duplicate_options=`2`
   caption: Ink and wash painting of a serene mountainous riverside landscape with layered rocky peaks, misty water, trees, a small pavilion by the shore, a tiny boat, and delicate calligraphy rendered in monochrome brushwork.
52. `track1_0385` source=`full1000` tier=`medium_delta_review` delta=`0.0408` distribution_delta=`0.1458` perceptual_delta=`0.1264` duplicate_options=`2`
   caption: Ukiyo-e portrait of an elegantly dressed woman in a patterned kimono standing beneath leafy branches, holding a folding fan beside a quiet riverside landscape with delicate linework and muted colors.
53. `track1_0635` source=`full1000` tier=`medium_delta_review` delta=`0.0404` distribution_delta=`0.1264` perceptual_delta=`0.143` duplicate_options=`2`
   caption: Ukiyo-e scene of three elegantly robed figures standing near a garden doorway with flowering trees, rendered in delicate linework, muted colors, and an aged paper texture.
54. `track1_0884` source=`full1000` tier=`medium_delta_review` delta=`0.04` distribution_delta=`0.1499` perceptual_delta=`0.117` duplicate_options=`1`
   caption: A Socialist Realism poster-style scene of tanks rolling through a cheering city parade beneath sweeping red banners, red stars, and a monumental building, painted with bold colors and celebratory dramatic composition.
55. `track1_0838` source=`full1000` tier=`medium_delta_review` delta=`0.0396` distribution_delta=`0.1858` perceptual_delta=`0.0785` duplicate_options=`1`
   caption: Ukiyo-e style owl perched on a mossy twisting branch with red maple leaves, Japanese calligraphy, muted earthy colors, and delicate textured linework on aged paper.
56. `track1_0937` source=`full1000` tier=`medium_delta_review` delta=`0.0396` distribution_delta=`0.1715` perceptual_delta=`0.0923` duplicate_options=`1`
   caption: Ink and wash painting of a serene mountainous landscape with clustered trees, rocky slopes, small houses in a valley, calligraphy inscriptions, and red seal stamps rendered with delicate monochrome brushwork.
57. `track1_0815` source=`full1000` tier=`medium_delta_review` delta=`0.0396` distribution_delta=`0.1387` perceptual_delta=`0.1259` duplicate_options=`1`
   caption: Ukiyo-e scene of elegantly dressed Japanese figures gathered closely with folding fans and patterned robes, rendered in fine ink lines and a muted beige color palette.
58. `track1_0724` source=`full1000` tier=`medium_delta_review` delta=`0.039` distribution_delta=`0.1419` perceptual_delta=`0.1183` duplicate_options=`1`
   caption: Abstract pencil sketch of a sparse architectural landscape with faint angular building outlines, a curving path, delicate linear marks, and a muted beige paper texture.
59. `track1_0445` source=`full1000` tier=`medium_delta_review` delta=`0.0388` distribution_delta=`0.1536` perceptual_delta=`0.1052` duplicate_options=`2`
   caption: Ink and wash painting of a fan-shaped mountainous landscape with rocky peaks, sparse trees, misty valleys, a calligraphy inscription, warm sepia tones, and delicate brushwork.
60. `track1_0604` source=`full1000` tier=`medium_delta_review` delta=`0.0385` distribution_delta=`0.1817` perceptual_delta=`0.0747` duplicate_options=`2`
   caption: Ukiyo-e scene of two women in flowing robes sharing a large umbrella beside a quiet waterside with birds and reeds, rendered in delicate linework and a muted earthy palette.
61. `track1_0712` source=`full1000` tier=`medium_delta_review` delta=`0.0377` distribution_delta=`0.1149` perceptual_delta=`0.1361` duplicate_options=`2`
   caption: Ink and wash painting of a vertical scroll with a robed sage in a circular frame above swirling dragon-like forms and bold ornamental black brushstrokes on aged sepia paper.
62. `track1_0434` source=`full1000` tier=`medium_delta_review` delta=`0.037` distribution_delta=`0.171` perceptual_delta=`0.0754` duplicate_options=`2`
   caption: Ink and wash painting of a serene mountainous landscape with layered rocky ridges, dense trees, misty distant peaks, muted sepia tones, fine brushwork, and calligraphy inscription above.
63. `track1_0750` source=`full1000` tier=`medium_delta_review` delta=`0.037` distribution_delta=`0.1256` perceptual_delta=`0.1212` duplicate_options=`1`
   caption: A Socialist Realism wartime propaganda poster with comic-like panels of civilians resisting fascist forces through sabotage, explosions, burning vehicles, red flags, Cyrillic slogans, and bold red, black, and pale blue graphic linework.
64. `track1_0249` source=`full1000` tier=`medium_delta_review` delta=`0.0366` distribution_delta=`0.1533` perceptual_delta=`0.0906` duplicate_options=`2`
   caption: Abstract ink and watercolor composition of clustered skull-like heads and amorphous gray forms resting above a bold red horizontal band, with loose sketch lines on beige paper.
65. `track1_0812` source=`full1000` tier=`medium_delta_review` delta=`0.0366` distribution_delta=`0.1394` perceptual_delta=`0.1047` duplicate_options=`2`
   caption: Ukiyo-e scene of a woman holding a patterned parasol beside a child riding a small toy horse near a pink wooden building, with delicate linework, muted colors, and a calm spring atmosphere.
66. `track1_0665` source=`full1000` tier=`medium_delta_review` delta=`0.0365` distribution_delta=`0.1112` perceptual_delta=`0.1323` duplicate_options=`2`
   caption: A Socialist Realism Soviet victory poster with bold Russian typography, a large medal suspended from a black-and-orange ribbon above Kremlin towers, crowds, fireworks, and radiant blue searchlights in a solemn celebratory composition.
67. `track1_0818` source=`full1000` tier=`medium_delta_review` delta=`0.0363` distribution_delta=`0.1411` perceptual_delta=`0.1009` duplicate_options=`2`
   caption: A Socialist Realism wartime propaganda poster of Allied cargo ships, sailors, naval flags, and sea convoy scenes, rendered with bold graphic composition, flat colors, and a solemn patriotic atmosphere.
68. `track1_0930` source=`full1000` tier=`medium_delta_review` delta=`0.036` distribution_delta=`0.1423` perceptual_delta=`0.0981` duplicate_options=`1`
   caption: Ink and wash painting of a serene mountainous riverside landscape with layered rocky peaks, pine trees, a small pavilion by the water, tiny boats, and delicate calligraphy in a soft monochrome palette.
69. `track1_0699` source=`full1000` tier=`medium_delta_review` delta=`0.0357` distribution_delta=`0.1401` perceptual_delta=`0.0981` duplicate_options=`2`
   caption: Ink and wash painting of bold abstract black brushstroke forms on beige paper with Chinese calligraphy inscriptions, rough dry-brush textures, and a restrained monochrome palette.
70. `track1_0593` source=`full1000` tier=`medium_delta_review` delta=`0.0357` distribution_delta=`0.1261` perceptual_delta=`0.1122` duplicate_options=`2`
   caption: A Socialist Realism Soviet propaganda poster with bold Cyrillic text, a monumental sword-wielding historical figure above soldiers raising red flags amid a liberated battlefield town, rendered in dramatic colors and strong graphic brushwork.
71. `track1_0494` source=`full1000` tier=`medium_delta_review` delta=`0.0354` distribution_delta=`0.1314` perceptual_delta=`0.1042` duplicate_options=`2`
   caption: A Gongbi style double portrait of a solemn imperial couple in ornate embroidered robes and jeweled headdresses, arranged symmetrically on two warm-toned hanging scroll panels with fine linework and muted colors.
72. `track1_0157` source=`full1000` tier=`medium_delta_review` delta=`0.0351` distribution_delta=`0.1616` perceptual_delta=`0.0727` duplicate_options=`2`
   caption: An Early Renaissance triptych of the Nativity with Mary and Joseph kneeling before the infant Christ in a rustic manger, angels hovering above, donors in the side panels, ruined stone architecture, distant landscape, clear linear detail, and a solemn devotional mood.
73. `track1_0753` source=`full1000` tier=`medium_delta_review` delta=`0.0351` distribution_delta=`0.1064` perceptual_delta=`0.1278` duplicate_options=`1`
   caption: A Ukiyo-e portrait of a woman with elaborate black hair and patterned robes, surrounded by delicate decorative objects and muted aged paper tones.
74. `track1_0934` source=`full1000` tier=`medium_delta_review` delta=`0.0351` distribution_delta=`0.1037` perceptual_delta=`0.1303` duplicate_options=`1`
   caption: Abstract pop art composition of fragmented cow-like shapes and geometric panels with bold black outlines, diagonal hatching, flat yellow, blue, and cream colors, and vertical lettering along the right side.
75. `track1_0997` source=`full1000` tier=`medium_delta_review` delta=`0.0349` distribution_delta=`0.1546` perceptual_delta=`0.0784` duplicate_options=`1`
   caption: Ukiyo-e woodblock print of elegantly dressed Japanese figures gathered in an interior before sliding panels with a mountain landscape, rendered with fine linework, muted colors, and a calm theatrical composition.
76. `track1_0547` source=`full1000` tier=`medium_delta_review` delta=`0.0349` distribution_delta=`0.1455` perceptual_delta=`0.0873` duplicate_options=`2`
   caption: Abstract monochrome composition of blurred white light shapes and dark diagonal geometric shadows, creating a mysterious high-contrast play of light and texture.
77. `track1_0238` source=`full1000` tier=`medium_delta_review` delta=`0.0336` distribution_delta=`0.1434` perceptual_delta=`0.0802` duplicate_options=`2`
   caption: Ink and wash painting of a serene mountainous landscape with clustered trees, winding streams, small rustic structures, and delicate brushwork on aged muted paper.
78. `track1_0045` source=`full1000` tier=`medium_delta_review` delta=`0.0334` distribution_delta=`0.1353` perceptual_delta=`0.0868` duplicate_options=`2`
   caption: Ink and wash painting of a serene misty mountain landscape with layered hills, riverside trees, small huts, and a lone boat rendered in delicate brushwork and muted monochrome tones.
79. `track1_0882` source=`full1000` tier=`medium_delta_review` delta=`0.0333` distribution_delta=`0.1182` perceptual_delta=`0.104` duplicate_options=`1`
   caption: A Gongbi painting of delicate pink, white, and red poppy-like flowers with finely outlined green foliage, subtle shading, and calligraphy on a warm beige silk background.
80. `track1_0421` source=`full1000` tier=`medium_delta_review` delta=`0.0332` distribution_delta=`0.0947` perceptual_delta=`0.1268` duplicate_options=`2`
   caption: Ukiyo-e composition of an open folding fan on a beige paper background, decorated with delicate calligraphy, pale linework, and small painted motifs in a calm, minimal palette.
