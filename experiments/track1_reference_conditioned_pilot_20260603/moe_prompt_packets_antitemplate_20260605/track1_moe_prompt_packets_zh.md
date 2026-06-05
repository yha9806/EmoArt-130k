# Track1 Clean MoE Prompt Packets

这是一份干净 provider prompt 审核包。候选数、模型名、expert route、reference queries 只保存在 metadata，不进入 Gemini prompt。

## 摘要

- 样本数：28
- 候选图预算：28
- 模型：{'gemini-3-pro-image': 27, 'gemini-3.1-flash-image': 1}
- 专家：{'poster_expert': 28}

## 审核清单

### 01. `track1_0665`

- 官方 caption：A Socialist Realism Soviet victory poster with bold Russian typography, a large medal suspended from a black-and-orange ribbon above Kremlin towers, crowds, fireworks, and radiant blue searchlights in a solemn celebratory composition.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/01_track1_0665_aas_safe.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet victory poster with bold Russian typography, a large medal suspended from a black-and-orange ribbon above Kremlin towers, crowds, fireworks, and radiant blue searchlights in a solemn celebratory composition.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet victory poster with bold Russian typography, a large medal suspended from a black-and-orange ribbon above Kremlin towers, crowds, fireworks, and radiant blue searchlights in a solemn celebratory composition

Factual anchors:
- Kremlin tower silhouette with a red star crown, not a generic castle, cathedral, or Western clock tower.
- Use a Red Square/Kremlin poster cue: red brick tower, crenellated wall, and star-topped spire.
- Dramatic blue searchlight beams frame the tower.
- The black-and-orange ribbon remains a suspended victory ribbon above the Kremlin towers.
- The large medal is a Soviet victory emblem with star/medal geometry, not a random decorative badge.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- do not turn the ribbon into unrelated black drapery or random striped flags

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Kremlin/Red Square imagery should read as red brick towers, crenellated walls, and a red star when requested.

DISTRIBUTION-AWARE ART DIRECTION
- Keep the caption requirements and named symbols strict; use style variation only after the core scene is clear.
- Preserve: caption content; landmark reference; requested text policy; relation logic.
- Composition cues: red brick tower; searchlight; night scene.
- Medium choices: propaganda poster print.
- Allowed variation: aspect variation allowed; style variation allowed; crop variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 02. `track1_0665`

- 官方 caption：A Socialist Realism Soviet victory poster with bold Russian typography, a large medal suspended from a black-and-orange ribbon above Kremlin towers, crowds, fireworks, and radiant blue searchlights in a solemn celebratory composition.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/02_track1_0665_reference_style.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet victory poster with bold Russian typography, a large medal suspended from a black-and-orange ribbon above Kremlin towers, crowds, fireworks, and radiant blue searchlights in a solemn celebratory composition.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet victory poster with bold Russian typography, a large medal suspended from a black-and-orange ribbon above Kremlin towers, crowds, fireworks, and radiant blue searchlights in a solemn celebratory composition

Factual anchors:
- Kremlin tower silhouette with a red star crown, not a generic castle, cathedral, or Western clock tower.
- Use a Red Square/Kremlin poster cue: red brick tower, crenellated wall, and star-topped spire.
- Dramatic blue searchlight beams frame the tower.
- The black-and-orange ribbon remains a suspended victory ribbon above the Kremlin towers.
- The large medal is a Soviet victory emblem with star/medal geometry, not a random decorative badge.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- do not turn the ribbon into unrelated black drapery or random striped flags

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Kremlin/Red Square imagery should read as red brick towers, crenellated walls, and a red star when requested.

DISTRIBUTION-AWARE ART DIRECTION
- Match the reference family through medium, crop, density, color, brushwork, texture, and lighting while preserving caption logic.
- Composition cues: red brick tower; searchlight; night scene.
- Medium choices: propaganda poster print.
- Allowed variation: aspect variation allowed; style variation allowed; crop variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 03. `track1_0665`

- 官方 caption：A Socialist Realism Soviet victory poster with bold Russian typography, a large medal suspended from a black-and-orange ribbon above Kremlin towers, crowds, fireworks, and radiant blue searchlights in a solemn celebratory composition.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/03_track1_0665_fid_diverse.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet victory poster with bold Russian typography, a large medal suspended from a black-and-orange ribbon above Kremlin towers, crowds, fireworks, and radiant blue searchlights in a solemn celebratory composition.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet victory poster with bold Russian typography, a large medal suspended from a black-and-orange ribbon above Kremlin towers, crowds, fireworks, and radiant blue searchlights in a solemn celebratory composition

Factual anchors:
- Kremlin tower silhouette with a red star crown, not a generic castle, cathedral, or Western clock tower.
- Use a Red Square/Kremlin poster cue: red brick tower, crenellated wall, and star-topped spire.
- Dramatic blue searchlight beams frame the tower.
- The black-and-orange ribbon remains a suspended victory ribbon above the Kremlin towers.
- The large medal is a Soviet victory emblem with star/medal geometry, not a random decorative badge.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- do not turn the ribbon into unrelated black drapery or random striped flags

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Kremlin/Red Square imagery should read as red brick towers, crenellated walls, and a red star when requested.

DISTRIBUTION-AWARE ART DIRECTION
- Avoid repeating a generic centered portrait-poster template; choose a less common crop, depth, figure scale, or viewpoint when the caption permits.
- Keep required caption content, but let the artwork read as a natural historical artwork rather than a uniform generated layout.
- Composition cues: red brick tower; searchlight; night scene.
- Medium choices: propaganda poster print.
- Allowed variation: aspect variation allowed; style variation allowed; crop variation allowed.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 04. `track1_0803`

- 官方 caption：A Socialist Realism propaganda poster with the Soviet, American, and British flags flying above a Kremlin tower crowned by a red star, framed by dramatic blue searchlight beams, bold Russian text, and a solemn wartime color palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/04_track1_0803_aas_safe.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism propaganda poster with the Soviet, American, and British flags flying above a Kremlin tower crowned by a red star, framed by dramatic blue searchlight beams, bold Russian text, and a solemn wartime color palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism propaganda poster with the Soviet, American, and British flags flying above a Kremlin tower crowned by a red star, framed by dramatic blue searchlight beams, bold Russian text, and a solemn wartime color palette

Factual anchors:
- Kremlin tower silhouette with a red star crown, not a generic castle, cathedral, or Western clock tower.
- Use a Red Square/Kremlin poster cue: red brick tower, crenellated wall, and star-topped spire.
- Dramatic blue searchlight beams frame the tower.
- Soviet, American, and British flags fly above the Kremlin tower as the caption states.
- Soviet flag is largest and visually dominant, closest to the Kremlin red star.
- American and British flags are smaller allied flags flanking the Soviet flag.
- Do not place the American flag as the central or dominant flag.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.
- Use a full-bleed flat poster surface with no external presentation area.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- foreground soldiers, crowds, faces, hands, or human figures not requested by the caption
- pistols, rifles, or weapons not requested by the caption
- white presentation border, drop shadow, product mockup, or poster-on-page layout

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Kremlin/Red Square imagery should read as red brick towers, crenellated walls, and a red star when requested.
- Keep requested national flags distinct; do not invent unrelated flag hybrids.

DISTRIBUTION-AWARE ART DIRECTION
- Keep the caption requirements and named symbols strict; use style variation only after the core scene is clear.
- Preserve: caption content; landmark reference; requested text policy; relation logic.
- Composition cues: red brick tower; searchlight; night scene.
- Medium choices: propaganda poster print.
- Allowed variation: aspect variation allowed; style variation allowed; crop variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 05. `track1_0803`

- 官方 caption：A Socialist Realism propaganda poster with the Soviet, American, and British flags flying above a Kremlin tower crowned by a red star, framed by dramatic blue searchlight beams, bold Russian text, and a solemn wartime color palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/05_track1_0803_reference_style.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism propaganda poster with the Soviet, American, and British flags flying above a Kremlin tower crowned by a red star, framed by dramatic blue searchlight beams, bold Russian text, and a solemn wartime color palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism propaganda poster with the Soviet, American, and British flags flying above a Kremlin tower crowned by a red star, framed by dramatic blue searchlight beams, bold Russian text, and a solemn wartime color palette

Factual anchors:
- Kremlin tower silhouette with a red star crown, not a generic castle, cathedral, or Western clock tower.
- Use a Red Square/Kremlin poster cue: red brick tower, crenellated wall, and star-topped spire.
- Dramatic blue searchlight beams frame the tower.
- Soviet, American, and British flags fly above the Kremlin tower as the caption states.
- Soviet flag is largest and visually dominant, closest to the Kremlin red star.
- American and British flags are smaller allied flags flanking the Soviet flag.
- Do not place the American flag as the central or dominant flag.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.
- Use a full-bleed flat poster surface with no external presentation area.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- foreground soldiers, crowds, faces, hands, or human figures not requested by the caption
- pistols, rifles, or weapons not requested by the caption
- white presentation border, drop shadow, product mockup, or poster-on-page layout

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Kremlin/Red Square imagery should read as red brick towers, crenellated walls, and a red star when requested.
- Keep requested national flags distinct; do not invent unrelated flag hybrids.

DISTRIBUTION-AWARE ART DIRECTION
- Match the reference family through medium, crop, density, color, brushwork, texture, and lighting while preserving caption logic.
- Composition cues: red brick tower; searchlight; night scene.
- Medium choices: propaganda poster print.
- Allowed variation: aspect variation allowed; style variation allowed; crop variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 06. `track1_0803`

- 官方 caption：A Socialist Realism propaganda poster with the Soviet, American, and British flags flying above a Kremlin tower crowned by a red star, framed by dramatic blue searchlight beams, bold Russian text, and a solemn wartime color palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/06_track1_0803_fid_diverse.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism propaganda poster with the Soviet, American, and British flags flying above a Kremlin tower crowned by a red star, framed by dramatic blue searchlight beams, bold Russian text, and a solemn wartime color palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism propaganda poster with the Soviet, American, and British flags flying above a Kremlin tower crowned by a red star, framed by dramatic blue searchlight beams, bold Russian text, and a solemn wartime color palette

Factual anchors:
- Kremlin tower silhouette with a red star crown, not a generic castle, cathedral, or Western clock tower.
- Use a Red Square/Kremlin poster cue: red brick tower, crenellated wall, and star-topped spire.
- Dramatic blue searchlight beams frame the tower.
- Soviet, American, and British flags fly above the Kremlin tower as the caption states.
- Soviet flag is largest and visually dominant, closest to the Kremlin red star.
- American and British flags are smaller allied flags flanking the Soviet flag.
- Do not place the American flag as the central or dominant flag.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.
- Use a full-bleed flat poster surface with no external presentation area.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- foreground soldiers, crowds, faces, hands, or human figures not requested by the caption
- pistols, rifles, or weapons not requested by the caption
- white presentation border, drop shadow, product mockup, or poster-on-page layout

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Kremlin/Red Square imagery should read as red brick towers, crenellated walls, and a red star when requested.
- Keep requested national flags distinct; do not invent unrelated flag hybrids.

DISTRIBUTION-AWARE ART DIRECTION
- Avoid repeating a generic centered portrait-poster template; choose a less common crop, depth, figure scale, or viewpoint when the caption permits.
- Keep required caption content, but let the artwork read as a natural historical artwork rather than a uniform generated layout.
- Composition cues: red brick tower; searchlight; night scene.
- Medium choices: propaganda poster print.
- Allowed variation: aspect variation allowed; style variation allowed; crop variation allowed.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 07. `track1_0077`

- 官方 caption：A Socialist Realism Soviet naval propaganda poster with a sailor hoisting red and white flags before panels of warships and sea battles, using bold Cyrillic typography, patriotic red accents, and a solemn heroic composition.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/07_track1_0077_aas_safe.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet naval propaganda poster with a sailor hoisting red and white flags before panels of warships and sea battles, using bold Cyrillic typography, patriotic red accents, and a solemn heroic composition.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet naval propaganda poster with a sailor hoisting red and white flags before panels of warships and sea battles, using bold Cyrillic typography, patriotic red accents, and a solemn heroic composition
- sailor hoisting red and white signal flags
- two primary signal flags: one red and one white

Spatial logic:
- The sailor holds a believable hoisting line attached to the red and white signal flags; flag ropes stay simple and functional.
- Use one simple red flag and one simple white flag on a single clear halyard; hands and rope do not merge.
- The rope is visibly gripped by the sailor's hand with clear fingers around it, not passing through the palm.
- flag attachment points are visible and simple: each flag is tied to the halyard at a believable corner or short sleeve.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- dominant blue-white flag replacing the requested red and white flags
- no extra blue signal flag
- blue cross, St. Andrew's cross, or blue-white naval ensign replacing the requested white signal flag
- cropped text fragments, broken headline endings, or stray letters around the poster edge
- text on sailor hat, cap ribbon, uniform trim, or clothing; keep small clothing details plain
- plain sailor cap band only; no cap tally letters or microtext
- excess rope tangles that make the flag-hoisting action physically impossible

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Keep the requested red and white signal flags simple; do not replace them with Allied national flags.
- Vehicles share a coherent ground/water/track plane with nearby people and do not intersect bodies.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Keep the caption requirements and named symbols strict; use style variation only after the core scene is clear.
- Preserve: caption content; requested text policy; relation logic.
- Composition cues: uniformed sailor or pilot; ship or vehicle silhouette; transport machinery.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 08. `track1_0077`

- 官方 caption：A Socialist Realism Soviet naval propaganda poster with a sailor hoisting red and white flags before panels of warships and sea battles, using bold Cyrillic typography, patriotic red accents, and a solemn heroic composition.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/08_track1_0077_reference_style.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet naval propaganda poster with a sailor hoisting red and white flags before panels of warships and sea battles, using bold Cyrillic typography, patriotic red accents, and a solemn heroic composition.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet naval propaganda poster with a sailor hoisting red and white flags before panels of warships and sea battles, using bold Cyrillic typography, patriotic red accents, and a solemn heroic composition
- sailor hoisting red and white signal flags
- two primary signal flags: one red and one white

Spatial logic:
- The sailor holds a believable hoisting line attached to the red and white signal flags; flag ropes stay simple and functional.
- Use one simple red flag and one simple white flag on a single clear halyard; hands and rope do not merge.
- The rope is visibly gripped by the sailor's hand with clear fingers around it, not passing through the palm.
- flag attachment points are visible and simple: each flag is tied to the halyard at a believable corner or short sleeve.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- dominant blue-white flag replacing the requested red and white flags
- no extra blue signal flag
- blue cross, St. Andrew's cross, or blue-white naval ensign replacing the requested white signal flag
- cropped text fragments, broken headline endings, or stray letters around the poster edge
- text on sailor hat, cap ribbon, uniform trim, or clothing; keep small clothing details plain
- plain sailor cap band only; no cap tally letters or microtext
- excess rope tangles that make the flag-hoisting action physically impossible

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Keep the requested red and white signal flags simple; do not replace them with Allied national flags.
- Vehicles share a coherent ground/water/track plane with nearby people and do not intersect bodies.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Match the reference family through medium, crop, density, color, brushwork, texture, and lighting while preserving caption logic.
- Composition cues: uniformed sailor or pilot; ship or vehicle silhouette; transport machinery.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 09. `track1_0077`

- 官方 caption：A Socialist Realism Soviet naval propaganda poster with a sailor hoisting red and white flags before panels of warships and sea battles, using bold Cyrillic typography, patriotic red accents, and a solemn heroic composition.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/09_track1_0077_fid_diverse.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet naval propaganda poster with a sailor hoisting red and white flags before panels of warships and sea battles, using bold Cyrillic typography, patriotic red accents, and a solemn heroic composition.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet naval propaganda poster with a sailor hoisting red and white flags before panels of warships and sea battles, using bold Cyrillic typography, patriotic red accents, and a solemn heroic composition
- sailor hoisting red and white signal flags
- two primary signal flags: one red and one white

Spatial logic:
- The sailor holds a believable hoisting line attached to the red and white signal flags; flag ropes stay simple and functional.
- Use one simple red flag and one simple white flag on a single clear halyard; hands and rope do not merge.
- The rope is visibly gripped by the sailor's hand with clear fingers around it, not passing through the palm.
- flag attachment points are visible and simple: each flag is tied to the halyard at a believable corner or short sleeve.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- dominant blue-white flag replacing the requested red and white flags
- no extra blue signal flag
- blue cross, St. Andrew's cross, or blue-white naval ensign replacing the requested white signal flag
- cropped text fragments, broken headline endings, or stray letters around the poster edge
- text on sailor hat, cap ribbon, uniform trim, or clothing; keep small clothing details plain
- plain sailor cap band only; no cap tally letters or microtext
- excess rope tangles that make the flag-hoisting action physically impossible

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Keep the requested red and white signal flags simple; do not replace them with Allied national flags.
- Vehicles share a coherent ground/water/track plane with nearby people and do not intersect bodies.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Avoid repeating a generic centered portrait-poster template; choose a less common crop, depth, figure scale, or viewpoint when the caption permits.
- Keep required caption content, but let the artwork read as a natural historical artwork rather than a uniform generated layout.
- Composition cues: uniformed sailor or pilot; ship or vehicle silhouette; transport machinery.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 10. `track1_0370`

- 官方 caption：A Socialist Realism Soviet propaganda poster with a group of young people holding bouquets around a decorated soldier, backed by sweeping red flags, Cyrillic slogans, and a bright celebratory color palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/10_track1_0370_aas_safe.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet propaganda poster with a group of young people holding bouquets around a decorated soldier, backed by sweeping red flags, Cyrillic slogans, and a bright celebratory color palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet propaganda poster with a group of young people holding bouquets around a decorated soldier, backed by sweeping red flags, Cyrillic slogans, and a bright celebratory color palette

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- Use one short caption-appropriate Cyrillic headline.
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Keep requested national flags distinct; do not invent unrelated flag hybrids.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Keep the caption requirements and named symbols strict; use style variation only after the core scene is clear.
- Preserve: caption content; requested text policy; relation logic.
- Composition cues: caption-led subject; medium-appropriate composition; avoid batch template.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 11. `track1_0370`

- 官方 caption：A Socialist Realism Soviet propaganda poster with a group of young people holding bouquets around a decorated soldier, backed by sweeping red flags, Cyrillic slogans, and a bright celebratory color palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/11_track1_0370_reference_style.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet propaganda poster with a group of young people holding bouquets around a decorated soldier, backed by sweeping red flags, Cyrillic slogans, and a bright celebratory color palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet propaganda poster with a group of young people holding bouquets around a decorated soldier, backed by sweeping red flags, Cyrillic slogans, and a bright celebratory color palette

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- Use one short caption-appropriate Cyrillic headline.
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Keep requested national flags distinct; do not invent unrelated flag hybrids.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Match the reference family through medium, crop, density, color, brushwork, texture, and lighting while preserving caption logic.
- Composition cues: caption-led subject; medium-appropriate composition; avoid batch template.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 12. `track1_0370`

- 官方 caption：A Socialist Realism Soviet propaganda poster with a group of young people holding bouquets around a decorated soldier, backed by sweeping red flags, Cyrillic slogans, and a bright celebratory color palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/12_track1_0370_fid_diverse.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet propaganda poster with a group of young people holding bouquets around a decorated soldier, backed by sweeping red flags, Cyrillic slogans, and a bright celebratory color palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet propaganda poster with a group of young people holding bouquets around a decorated soldier, backed by sweeping red flags, Cyrillic slogans, and a bright celebratory color palette

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- Use one short caption-appropriate Cyrillic headline.
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Keep requested national flags distinct; do not invent unrelated flag hybrids.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Avoid repeating a generic centered portrait-poster template; choose a less common crop, depth, figure scale, or viewpoint when the caption permits.
- Keep required caption content, but let the artwork read as a natural historical artwork rather than a uniform generated layout.
- Composition cues: caption-led subject; medium-appropriate composition; avoid batch template.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 13. `track1_0747`

- 官方 caption：A Socialist Realism Soviet wartime propaganda poster with bold Cyrillic lettering, mounted soldiers, fleeing civilians, burning village ruins, aircraft overhead, and dramatic brushwork in a muted blue, red, and ochre palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/13_track1_0747_aas_safe.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet wartime propaganda poster with bold Cyrillic lettering, mounted soldiers, fleeing civilians, burning village ruins, aircraft overhead, and dramatic brushwork in a muted blue, red, and ochre palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet wartime propaganda poster with bold Cyrillic lettering, mounted soldiers, fleeing civilians, burning village ruins, aircraft overhead, and dramatic brushwork in a muted blue, red, and ochre palette

Spatial logic:
- The mounted soldiers read as protecting or escorting civilians away from the burning village, not attacking them.
- Place soldiers and civilians moving in a shared evacuation direction or with soldiers between danger and civilians.
- Use gesture, spacing, and eye-lines so the civilians are fleeing the village and aircraft, not fleeing from the soldiers.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- do not compose the horsemen as chasing or attacking civilians
- no cavalry weapons pointed at civilians

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Aircraft remain separate coherent background objects; flags and poles should not pierce fuselages or wings.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Keep the caption requirements and named symbols strict; use style variation only after the core scene is clear.
- Preserve: caption content; requested text policy; relation logic.
- Composition cues: battlefield depth; armored vehicle; mounted movement; infantry group.
- Medium choices: propaganda poster print; painting or brushwork surface.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 14. `track1_0747`

- 官方 caption：A Socialist Realism Soviet wartime propaganda poster with bold Cyrillic lettering, mounted soldiers, fleeing civilians, burning village ruins, aircraft overhead, and dramatic brushwork in a muted blue, red, and ochre palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/14_track1_0747_reference_style.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet wartime propaganda poster with bold Cyrillic lettering, mounted soldiers, fleeing civilians, burning village ruins, aircraft overhead, and dramatic brushwork in a muted blue, red, and ochre palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet wartime propaganda poster with bold Cyrillic lettering, mounted soldiers, fleeing civilians, burning village ruins, aircraft overhead, and dramatic brushwork in a muted blue, red, and ochre palette

Spatial logic:
- The mounted soldiers read as protecting or escorting civilians away from the burning village, not attacking them.
- Place soldiers and civilians moving in a shared evacuation direction or with soldiers between danger and civilians.
- Use gesture, spacing, and eye-lines so the civilians are fleeing the village and aircraft, not fleeing from the soldiers.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- do not compose the horsemen as chasing or attacking civilians
- no cavalry weapons pointed at civilians

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Aircraft remain separate coherent background objects; flags and poles should not pierce fuselages or wings.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Match the reference family through medium, crop, density, color, brushwork, texture, and lighting while preserving caption logic.
- Composition cues: battlefield depth; armored vehicle; mounted movement; infantry group.
- Medium choices: propaganda poster print; painting or brushwork surface.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 15. `track1_0747`

- 官方 caption：A Socialist Realism Soviet wartime propaganda poster with bold Cyrillic lettering, mounted soldiers, fleeing civilians, burning village ruins, aircraft overhead, and dramatic brushwork in a muted blue, red, and ochre palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/15_track1_0747_fid_diverse.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet wartime propaganda poster with bold Cyrillic lettering, mounted soldiers, fleeing civilians, burning village ruins, aircraft overhead, and dramatic brushwork in a muted blue, red, and ochre palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet wartime propaganda poster with bold Cyrillic lettering, mounted soldiers, fleeing civilians, burning village ruins, aircraft overhead, and dramatic brushwork in a muted blue, red, and ochre palette

Spatial logic:
- The mounted soldiers read as protecting or escorting civilians away from the burning village, not attacking them.
- Place soldiers and civilians moving in a shared evacuation direction or with soldiers between danger and civilians.
- Use gesture, spacing, and eye-lines so the civilians are fleeing the village and aircraft, not fleeing from the soldiers.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- do not compose the horsemen as chasing or attacking civilians
- no cavalry weapons pointed at civilians

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Aircraft remain separate coherent background objects; flags and poles should not pierce fuselages or wings.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Avoid repeating a generic centered portrait-poster template; choose a less common crop, depth, figure scale, or viewpoint when the caption permits.
- Keep required caption content, but let the artwork read as a natural historical artwork rather than a uniform generated layout.
- Composition cues: battlefield depth; armored vehicle; mounted movement; infantry group.
- Medium choices: propaganda poster print; painting or brushwork surface.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 16. `track1_0476`

- 官方 caption：A Socialist Realism propaganda poster with bold Cyrillic lettering, a naval medal, sailing ships, cannon, and uniformed sailors in a dramatic maritime battle scene rendered with graphic linework and muted watercolor tones.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/16_track1_0476_aas_safe.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism propaganda poster with bold Cyrillic lettering, a naval medal, sailing ships, cannon, and uniformed sailors in a dramatic maritime battle scene rendered with graphic linework and muted watercolor tones.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism propaganda poster with bold Cyrillic lettering, a naval medal, sailing ships, cannon, and uniformed sailors in a dramatic maritime battle scene rendered with graphic linework and muted watercolor tones

Spatial logic:
- Layer medal, ships, cannon, and sailors as a coherent maritime poster montage with believable scale hierarchy.
- Use the naval medal as an emblem or central motif; ships, cannon, and sailors should not physically intersect at impossible scales.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Artillery guns and large shells should remain coherent military equipment, not abstract pipes.
- Vehicles share a coherent ground/water/track plane with nearby people and do not intersect bodies.

DISTRIBUTION-AWARE ART DIRECTION
- Keep the caption requirements and named symbols strict; use style variation only after the core scene is clear.
- Preserve: caption content; requested text policy; relation logic.
- Composition cues: uniformed sailor or pilot; ship or vehicle silhouette; transport machinery.
- Medium choices: propaganda poster print; painting or brushwork surface.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 17. `track1_0476`

- 官方 caption：A Socialist Realism propaganda poster with bold Cyrillic lettering, a naval medal, sailing ships, cannon, and uniformed sailors in a dramatic maritime battle scene rendered with graphic linework and muted watercolor tones.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/17_track1_0476_reference_style.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism propaganda poster with bold Cyrillic lettering, a naval medal, sailing ships, cannon, and uniformed sailors in a dramatic maritime battle scene rendered with graphic linework and muted watercolor tones.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism propaganda poster with bold Cyrillic lettering, a naval medal, sailing ships, cannon, and uniformed sailors in a dramatic maritime battle scene rendered with graphic linework and muted watercolor tones

Spatial logic:
- Layer medal, ships, cannon, and sailors as a coherent maritime poster montage with believable scale hierarchy.
- Use the naval medal as an emblem or central motif; ships, cannon, and sailors should not physically intersect at impossible scales.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Artillery guns and large shells should remain coherent military equipment, not abstract pipes.
- Vehicles share a coherent ground/water/track plane with nearby people and do not intersect bodies.

DISTRIBUTION-AWARE ART DIRECTION
- Match the reference family through medium, crop, density, color, brushwork, texture, and lighting while preserving caption logic.
- Composition cues: uniformed sailor or pilot; ship or vehicle silhouette; transport machinery.
- Medium choices: propaganda poster print; painting or brushwork surface.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 18. `track1_0476`

- 官方 caption：A Socialist Realism propaganda poster with bold Cyrillic lettering, a naval medal, sailing ships, cannon, and uniformed sailors in a dramatic maritime battle scene rendered with graphic linework and muted watercolor tones.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/18_track1_0476_fid_diverse.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism propaganda poster with bold Cyrillic lettering, a naval medal, sailing ships, cannon, and uniformed sailors in a dramatic maritime battle scene rendered with graphic linework and muted watercolor tones.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism propaganda poster with bold Cyrillic lettering, a naval medal, sailing ships, cannon, and uniformed sailors in a dramatic maritime battle scene rendered with graphic linework and muted watercolor tones

Spatial logic:
- Layer medal, ships, cannon, and sailors as a coherent maritime poster montage with believable scale hierarchy.
- Use the naval medal as an emblem or central motif; ships, cannon, and sailors should not physically intersect at impossible scales.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Artillery guns and large shells should remain coherent military equipment, not abstract pipes.
- Vehicles share a coherent ground/water/track plane with nearby people and do not intersect bodies.

DISTRIBUTION-AWARE ART DIRECTION
- Avoid repeating a generic centered portrait-poster template; choose a less common crop, depth, figure scale, or viewpoint when the caption permits.
- Keep required caption content, but let the artwork read as a natural historical artwork rather than a uniform generated layout.
- Composition cues: uniformed sailor or pilot; ship or vehicle silhouette; transport machinery.
- Medium choices: propaganda poster print; painting or brushwork surface.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 19. `track1_0898`

- 官方 caption：Socialist Realism poster art of Soviet soldiers charging through a snowy battlefield with a raised rifle, a red flag, bold Cyrillic slogans, dramatic action, and a muted red-blue color palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/19_track1_0898_aas_safe.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
Socialist Realism poster art of Soviet soldiers charging through a snowy battlefield with a raised rifle, a red flag, bold Cyrillic slogans, dramatic action, and a muted red-blue color palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- Socialist Realism poster art of Soviet soldiers charging through a snowy battlefield with a raised rifle, a red flag, bold Cyrillic slogans, dramatic action, and a muted red-blue color palette

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Keep requested national flags distinct; do not invent unrelated flag hybrids.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Keep the caption requirements and named symbols strict; use style variation only after the core scene is clear.
- Preserve: caption content; requested text policy; relation logic.
- Composition cues: battlefield depth; armored vehicle; mounted movement; infantry group.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 20. `track1_0898`

- 官方 caption：Socialist Realism poster art of Soviet soldiers charging through a snowy battlefield with a raised rifle, a red flag, bold Cyrillic slogans, dramatic action, and a muted red-blue color palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/20_track1_0898_reference_style.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
Socialist Realism poster art of Soviet soldiers charging through a snowy battlefield with a raised rifle, a red flag, bold Cyrillic slogans, dramatic action, and a muted red-blue color palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- Socialist Realism poster art of Soviet soldiers charging through a snowy battlefield with a raised rifle, a red flag, bold Cyrillic slogans, dramatic action, and a muted red-blue color palette

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Keep requested national flags distinct; do not invent unrelated flag hybrids.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Match the reference family through medium, crop, density, color, brushwork, texture, and lighting while preserving caption logic.
- Composition cues: battlefield depth; armored vehicle; mounted movement; infantry group.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 21. `track1_0898`

- 官方 caption：Socialist Realism poster art of Soviet soldiers charging through a snowy battlefield with a raised rifle, a red flag, bold Cyrillic slogans, dramatic action, and a muted red-blue color palette.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/21_track1_0898_fid_diverse.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
Socialist Realism poster art of Soviet soldiers charging through a snowy battlefield with a raised rifle, a red flag, bold Cyrillic slogans, dramatic action, and a muted red-blue color palette.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- Socialist Realism poster art of Soviet soldiers charging through a snowy battlefield with a raised rifle, a red flag, bold Cyrillic slogans, dramatic action, and a muted red-blue color palette

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Keep requested national flags distinct; do not invent unrelated flag hybrids.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Avoid repeating a generic centered portrait-poster template; choose a less common crop, depth, figure scale, or viewpoint when the caption permits.
- Keep required caption content, but let the artwork read as a natural historical artwork rather than a uniform generated layout.
- Composition cues: battlefield depth; armored vehicle; mounted movement; infantry group.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 22. `track1_0708`

- 官方 caption：A Socialist Realism wartime poster of Soviet soldiers advancing through a rugged battlefield, with a front soldier holding a submachine gun and another raising a red banner, rendered in heroic composition, muted earth tones, and bold Cyrillic typography.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/22_track1_0708_aas_safe.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism wartime poster of Soviet soldiers advancing through a rugged battlefield, with a front soldier holding a submachine gun and another raising a red banner, rendered in heroic composition, muted earth tones, and bold Cyrillic typography.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism wartime poster of Soviet soldiers advancing through a rugged battlefield, with a front soldier holding a submachine gun and another raising a red banner, rendered in heroic composition, muted earth tones, and bold Cyrillic typography

Factual anchors:
- front soldier carries a PPSh-41-like Soviet submachine gun with a simple barrel and drum magazine cue.
- single correct Soviet red banner with a clear star or hammer-and-sickle cue, not competing flag hybrids.

Spatial logic:
- The front soldier holds the submachine gun with two clean contact points and a consistent advancing direction.
- The banner is raised by a separate visible pole with clear hand grip; it does not grow out of a weapon or sleeve.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.
- Use one large complete Cyrillic slogan if needed; avoid fragmented or misspelled dominant lettering.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- weapon hybrids, floating bayonets, malformed barrels, or missing drum magazine silhouette
- duplicate conflicting flag symbols, extra national flags, or banner poles fused with rifles

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- PPSh-41-like Soviet submachine gun cue: simple barrel and drum magazine silhouette.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Keep the caption requirements and named symbols strict; use style variation only after the core scene is clear.
- Preserve: caption content; requested text policy; relation logic.
- Composition cues: battlefield depth; armored vehicle; mounted movement; infantry group.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 23. `track1_0708`

- 官方 caption：A Socialist Realism wartime poster of Soviet soldiers advancing through a rugged battlefield, with a front soldier holding a submachine gun and another raising a red banner, rendered in heroic composition, muted earth tones, and bold Cyrillic typography.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/23_track1_0708_reference_style.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism wartime poster of Soviet soldiers advancing through a rugged battlefield, with a front soldier holding a submachine gun and another raising a red banner, rendered in heroic composition, muted earth tones, and bold Cyrillic typography.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism wartime poster of Soviet soldiers advancing through a rugged battlefield, with a front soldier holding a submachine gun and another raising a red banner, rendered in heroic composition, muted earth tones, and bold Cyrillic typography

Factual anchors:
- front soldier carries a PPSh-41-like Soviet submachine gun with a simple barrel and drum magazine cue.
- single correct Soviet red banner with a clear star or hammer-and-sickle cue, not competing flag hybrids.

Spatial logic:
- The front soldier holds the submachine gun with two clean contact points and a consistent advancing direction.
- The banner is raised by a separate visible pole with clear hand grip; it does not grow out of a weapon or sleeve.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.
- Use one large complete Cyrillic slogan if needed; avoid fragmented or misspelled dominant lettering.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- weapon hybrids, floating bayonets, malformed barrels, or missing drum magazine silhouette
- duplicate conflicting flag symbols, extra national flags, or banner poles fused with rifles

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- PPSh-41-like Soviet submachine gun cue: simple barrel and drum magazine silhouette.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Match the reference family through medium, crop, density, color, brushwork, texture, and lighting while preserving caption logic.
- Composition cues: battlefield depth; armored vehicle; mounted movement; infantry group.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 24. `track1_0708`

- 官方 caption：A Socialist Realism wartime poster of Soviet soldiers advancing through a rugged battlefield, with a front soldier holding a submachine gun and another raising a red banner, rendered in heroic composition, muted earth tones, and bold Cyrillic typography.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/24_track1_0708_fid_diverse.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism wartime poster of Soviet soldiers advancing through a rugged battlefield, with a front soldier holding a submachine gun and another raising a red banner, rendered in heroic composition, muted earth tones, and bold Cyrillic typography.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism wartime poster of Soviet soldiers advancing through a rugged battlefield, with a front soldier holding a submachine gun and another raising a red banner, rendered in heroic composition, muted earth tones, and bold Cyrillic typography

Factual anchors:
- front soldier carries a PPSh-41-like Soviet submachine gun with a simple barrel and drum magazine cue.
- single correct Soviet red banner with a clear star or hammer-and-sickle cue, not competing flag hybrids.

Spatial logic:
- The front soldier holds the submachine gun with two clean contact points and a consistent advancing direction.
- The banner is raised by a separate visible pole with clear hand grip; it does not grow out of a weapon or sleeve.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.
- Use one large complete Cyrillic slogan if needed; avoid fragmented or misspelled dominant lettering.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- weapon hybrids, floating bayonets, malformed barrels, or missing drum magazine silhouette
- duplicate conflicting flag symbols, extra national flags, or banner poles fused with rifles

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- PPSh-41-like Soviet submachine gun cue: simple barrel and drum magazine silhouette.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Avoid repeating a generic centered portrait-poster template; choose a less common crop, depth, figure scale, or viewpoint when the caption permits.
- Keep required caption content, but let the artwork read as a natural historical artwork rather than a uniform generated layout.
- Composition cues: battlefield depth; armored vehicle; mounted movement; infantry group.
- Medium choices: propaganda poster print.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 25. `track1_0353`

- 官方 caption：A Socialist Realism Soviet poster-style scene with a jubilant crowd of soldiers and civilians watching fireworks over the Kremlin and Red Square beneath bold Cyrillic lettering, rendered in a vintage muted palette with dramatic light and shadow.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/25_track1_0353_aas_safe.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet poster-style scene with a jubilant crowd of soldiers and civilians watching fireworks over the Kremlin and Red Square beneath bold Cyrillic lettering, rendered in a vintage muted palette with dramatic light and shadow.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet poster-style scene with a jubilant crowd of soldiers and civilians watching fireworks over the Kremlin and Red Square beneath bold Cyrillic lettering, rendered in a vintage muted palette with dramatic light and shadow
- jubilant crowd of soldiers and civilians watching fireworks

Factual anchors:
- Kremlin tower silhouette with a red star crown, not a generic castle, cathedral, or Western clock tower.
- Use a Red Square/Kremlin poster cue: red brick tower, crenellated wall, and star-topped spire.
- Fireworks happen over the Kremlin and Red Square, with the crowd below as the main scene.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- empty Kremlin-only night scene with no crowd

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- Use one short caption-appropriate Cyrillic headline.
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Kremlin/Red Square imagery should read as red brick towers, crenellated walls, and a red star when requested.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Keep the caption requirements and named symbols strict; use style variation only after the core scene is clear.
- Preserve: caption content; landmark reference; requested text policy.
- Composition cues: red brick tower; searchlight; night scene.
- Medium choices: propaganda poster print.
- Allowed variation: aspect variation allowed; style variation allowed; crop variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 26. `track1_0353`

- 官方 caption：A Socialist Realism Soviet poster-style scene with a jubilant crowd of soldiers and civilians watching fireworks over the Kremlin and Red Square beneath bold Cyrillic lettering, rendered in a vintage muted palette with dramatic light and shadow.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/26_track1_0353_reference_style.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet poster-style scene with a jubilant crowd of soldiers and civilians watching fireworks over the Kremlin and Red Square beneath bold Cyrillic lettering, rendered in a vintage muted palette with dramatic light and shadow.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet poster-style scene with a jubilant crowd of soldiers and civilians watching fireworks over the Kremlin and Red Square beneath bold Cyrillic lettering, rendered in a vintage muted palette with dramatic light and shadow
- jubilant crowd of soldiers and civilians watching fireworks

Factual anchors:
- Kremlin tower silhouette with a red star crown, not a generic castle, cathedral, or Western clock tower.
- Use a Red Square/Kremlin poster cue: red brick tower, crenellated wall, and star-topped spire.
- Fireworks happen over the Kremlin and Red Square, with the crowd below as the main scene.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- empty Kremlin-only night scene with no crowd

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- Use one short caption-appropriate Cyrillic headline.
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Kremlin/Red Square imagery should read as red brick towers, crenellated walls, and a red star when requested.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Match the reference family through medium, crop, density, color, brushwork, texture, and lighting while preserving caption logic.
- Composition cues: red brick tower; searchlight; night scene.
- Medium choices: propaganda poster print.
- Allowed variation: aspect variation allowed; style variation allowed; crop variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 27. `track1_0353`

- 官方 caption：A Socialist Realism Soviet poster-style scene with a jubilant crowd of soldiers and civilians watching fireworks over the Kremlin and Red Square beneath bold Cyrillic lettering, rendered in a vintage muted palette with dramatic light and shadow.
- 生成 metadata：model=`gemini-3-pro-image` candidates=`4` expert=`poster_expert` support=reference_expert, text_symbol_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/27_track1_0353_fid_diverse.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet poster-style scene with a jubilant crowd of soldiers and civilians watching fireworks over the Kremlin and Red Square beneath bold Cyrillic lettering, rendered in a vintage muted palette with dramatic light and shadow.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet poster-style scene with a jubilant crowd of soldiers and civilians watching fireworks over the Kremlin and Red Square beneath bold Cyrillic lettering, rendered in a vintage muted palette with dramatic light and shadow
- jubilant crowd of soldiers and civilians watching fireworks

Factual anchors:
- Kremlin tower silhouette with a red star crown, not a generic castle, cathedral, or Western clock tower.
- Use a Red Square/Kremlin poster cue: red brick tower, crenellated wall, and star-topped spire.
- Fireworks happen over the Kremlin and Red Square, with the crowd below as the main scene.

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster
- empty Kremlin-only night scene with no crowd

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- Use one short caption-appropriate Cyrillic headline.
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Kremlin/Red Square imagery should read as red brick towers, crenellated walls, and a red star when requested.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Avoid repeating a generic centered portrait-poster template; choose a less common crop, depth, figure scale, or viewpoint when the caption permits.
- Keep required caption content, but let the artwork read as a natural historical artwork rather than a uniform generated layout.
- Composition cues: red brick tower; searchlight; night scene.
- Medium choices: propaganda poster print.
- Allowed variation: aspect variation allowed; style variation allowed; crop variation allowed.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>

### 28. `track1_0593`

- 官方 caption：A Socialist Realism Soviet propaganda poster with bold Cyrillic text, a monumental sword-wielding historical figure above soldiers raising red flags amid a liberated battlefield town, rendered in dramatic colors and strong graphic brushwork.
- 生成 metadata：model=`gemini-3.1-flash-image` candidates=`3` expert=`poster_expert` support=reference_expert, text_symbol_expert, logic_expert, fid_guard_expert
- Provider prompt：`/Users/yhryzy/dev/emoart-130k/experiments/track1_reference_conditioned_pilot_20260603/moe_prompt_packets_antitemplate_20260605/provider_prompts_top/28_track1_0593_aas_safe.txt`

<details><summary>Clean provider prompt</summary>

```text
Create a single finished artwork image for an art-generation challenge.

CAPTION TO SATISFY
A Socialist Realism Soviet propaganda poster with bold Cyrillic text, a monumental sword-wielding historical figure above soldiers raising red flags amid a liberated battlefield town, rendered in dramatic colors and strong graphic brushwork.

OUTPUT BOUNDARY
- The output is the artwork itself, not a gallery/photo/mockup/catalog presentation.
- Make it clearly legible and fully filled by the artwork surface.
- Do not add sample IDs, filenames, watermarks, UI labels, captions, or explanation text.

REFERENCE-GUIDED CONTENT PLAN

Must include:
- A Socialist Realism Soviet propaganda poster with bold Cyrillic text, a monumental sword-wielding historical figure above soldiers raising red flags amid a liberated battlefield town, rendered in dramatic colors and strong graphic brushwork

Surface contract:
- Output must be the artwork surface itself, not a gallery photograph or photographed object.
- Poster means a flat clearly legible printed artwork; internal margins and printed border lines are allowed.

Logic-anchor variant:
- Resolve physical relations before decoration: figures, vehicles, flags, windows, and symbols must have coherent contact points and action direction.
- Treat factual anchors and spatial logic as hard constraints, not style suggestions.
- Keep only caption-requested symbols; do not substitute extra flags, emblems, walls, frames, or unrelated text blocks.

Flat-poster tight variant:
- Fill the output with the period printed artwork artwork; natural artwork edges are allowed, but no surrounding table, wall, drop shadow, or scanned-page border.
- Use only intentional headline/slogan typography requested by the caption; no tiny printer credits or footer microtext.
- Keep the poster clearly legible with no perspective tilt, curled edge, hanging clip, or photographed-paper presentation.

Defect-targeted variant:
- Prefer varied figure scale and clear silhouettes over close-up hands, ropes, weapons, or flag details.
- Use clear, selective, intentional text blocks; avoid tiny text, footer credits, hat text, document microtext, and edge labels.
- Simplify local mechanics so hands, ropes, rifles, bayonets, flags, documents, vehicles, and panels have clean contact points.

Text policy:
- Include short Cyrillic/Russian-looking headline blocks only where the caption requests typography.
- Prefer a few large legible-looking words over dense fake microtext.
- Do not add Latin sample IDs, filenames, watermarks, or unrelated labels.

Style policy:
- Use composed artwork with clear large shapes and controlled flat color areas.
- Use Socialist Realism poster language: heroic figures, idealized anatomy, bold diagonals, muted wartime palette.
- Keep symbolic objects historically coherent; do not invent unrelated insignia.

Must avoid:
- gallery wall, museum installation, product mockup, catalog page, drop shadow, watermark, filename label
- external photo frame, mat board, or wall-mounted display unless explicitly requested
- realistic wall scene, poster hanging from clips, curled paper edge, or photographed poster

GENERATION PRIORITY
1. Exact caption semantics and requested objects.
2. Spatial/physical coherence of relations and surfaces.
3. Requested artistic style, brushwork, palette, composition, and emotional atmosphere.
4. Attractive final image quality.

Produce only the image.

TEXT AND FACTUAL ANCHORS
- Text required: True; modes: cyrillic.
Allowed text cues:
- ПОБЕДА БУДЕТ ЗА НАМИ!
- ЗА РОДИНУ!
Text directives:
- Use a few large intentional text blocks; keep them integrated into the artwork surface.
- Prefer short, plausible text over dense fake microtext.
- Do not add sample IDs, filenames, UI labels, watermarks, or tiny footer credits.
Text must avoid:
- footer microtext
- sample ID or filename text
- watermark or UI label
- large malformed Cyrillic that changes the meaning of the main slogan
Factual anchors:
- Keep requested national flags distinct; do not invent unrelated flag hybrids.
- Military clothing should read as period poster uniforms, not modern generic outfits.

DISTRIBUTION-AWARE ART DIRECTION
- Keep the caption requirements and named symbols strict; use style variation only after the core scene is clear.
- Preserve: caption content; requested text policy; relation logic.
- Composition cues: battlefield depth; armored vehicle; mounted movement; infantry group.
- Medium choices: propaganda poster print; painting or brushwork surface.
- Allowed variation: free crop allowed; aspect variation allowed; style variation allowed.
- Poster support may remain visible when it helps the caption, but avoid a repeated stock layout.

ASPECT AND CANVAS PLAN
- Use a artwork surface; keep the full printed poster clearly legible and avoid left/right overflow.
- Aspect label: portrait_poster; target canvas: 768x1024.
```

</details>
