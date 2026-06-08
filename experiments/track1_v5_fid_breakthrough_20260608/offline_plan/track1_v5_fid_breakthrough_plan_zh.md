# Track1 v5 FID Breakthrough Offline Plan

这不是提交包，也不会调用 Gemini。它是最后两次提交前的全 1000 离线路由和 prompt 审核包。

## Summary

- total: `1000`
- candidate budget: `1000`
- route counts: `{'caption_faithful_guard': 475, 'reference_family_primary': 406, 'anti_template_diversifier': 119}`
- aspect counts: `{'square_artwork': 869, 'portrait_poster': 66, 'vertical_scroll': 36, 'album_spread': 23, 'horizontal_scroll': 4, 'panel_story': 2}`
- risk tags: `{'template_distribution_guard': 337, 'text_or_symbol_guard': 334, 'overused_reference_guard': 308, 'relation_logic_guard': 162, 'support_surface_guard': 76, 'real_world_reference_guard': 69}`

## First 40 Samples

- `track1_0001` route=`reference_family_primary` family=`ukiyoe_woodblock` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0002` route=`caption_faithful_guard` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=text_or_symbol_guard
- `track1_0003` route=`caption_faithful_guard` family=`ukiyoe_woodblock` v7_family=`generic_artwork` aspect=`square_artwork` risks=relation_logic_guard
- `track1_0004` route=`caption_faithful_guard` family=`ukiyoe_woodblock` v7_family=`generic_artwork` aspect=`square_artwork` risks=text_or_symbol_guard,relation_logic_guard
- `track1_0005` route=`reference_family_primary` family=`pastel_drawing` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0006` route=`reference_family_primary` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0007` route=`reference_family_primary` family=`generic_artwork` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0008` route=`reference_family_primary` family=`ukiyoe_woodblock` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0009` route=`anti_template_diversifier` family=`generic_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=template_distribution_guard,overused_reference_guard
- `track1_0010` route=`caption_faithful_guard` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=text_or_symbol_guard
- `track1_0011` route=`reference_family_primary` family=`generic_artwork` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0012` route=`reference_family_primary` family=`ukiyoe_woodblock` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0013` route=`reference_family_primary` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0014` route=`caption_faithful_guard` family=`ink_wash_painting` v7_family=`scroll_album_paper_support` aspect=`album_spread` risks=support_surface_guard
- `track1_0015` route=`reference_family_primary` family=`generic_artwork` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0016` route=`caption_faithful_guard` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=text_or_symbol_guard
- `track1_0017` route=`reference_family_primary` family=`ukiyoe_woodblock` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0018` route=`reference_family_primary` family=`generic_artwork` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0019` route=`caption_faithful_guard` family=`socialist_realism_poster` v7_family=`battle_tank_cavalry` aspect=`portrait_poster` risks=real_world_reference_guard,template_distribution_guard,overused_reference_guard
- `track1_0020` route=`anti_template_diversifier` family=`generic_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=template_distribution_guard,overused_reference_guard
- `track1_0021` route=`caption_faithful_guard` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=text_or_symbol_guard,template_distribution_guard,overused_reference_guard
- `track1_0022` route=`reference_family_primary` family=`ukiyoe_woodblock` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0023` route=`caption_faithful_guard` family=`album_leaf_ink` v7_family=`scroll_album_paper_support` aspect=`square_artwork` risks=text_or_symbol_guard,template_distribution_guard,overused_reference_guard
- `track1_0024` route=`reference_family_primary` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0025` route=`caption_faithful_guard` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=text_or_symbol_guard
- `track1_0026` route=`caption_faithful_guard` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=text_or_symbol_guard
- `track1_0027` route=`reference_family_primary` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0028` route=`caption_faithful_guard` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=text_or_symbol_guard
- `track1_0029` route=`caption_faithful_guard` family=`album_leaf_ink` v7_family=`scroll_album_paper_support` aspect=`square_artwork` risks=text_or_symbol_guard,template_distribution_guard,overused_reference_guard
- `track1_0030` route=`reference_family_primary` family=`ukiyoe_woodblock` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0031` route=`caption_faithful_guard` family=`ink_wash_painting` v7_family=`scroll_album_paper_support` aspect=`album_spread` risks=text_or_symbol_guard,support_surface_guard
- `track1_0032` route=`reference_family_primary` family=`ukiyoe_woodblock` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0033` route=`caption_faithful_guard` family=`ukiyoe_woodblock` v7_family=`generic_artwork` aspect=`square_artwork` risks=text_or_symbol_guard,relation_logic_guard
- `track1_0034` route=`reference_family_primary` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0035` route=`reference_family_primary` family=`ukiyoe_woodblock` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0036` route=`caption_faithful_guard` family=`generic_artwork` v7_family=`generic_artwork` aspect=`square_artwork` risks=text_or_symbol_guard,template_distribution_guard,overused_reference_guard
- `track1_0037` route=`anti_template_diversifier` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=template_distribution_guard,overused_reference_guard
- `track1_0038` route=`caption_faithful_guard` family=`socialist_realism_poster` v7_family=`battle_tank_cavalry` aspect=`portrait_poster` risks=text_or_symbol_guard,real_world_reference_guard,relation_logic_guard,template_distribution_guard,overused_reference_guard
- `track1_0039` route=`reference_family_primary` family=`generic_artwork` v7_family=`generic_artwork` aspect=`square_artwork` risks=none
- `track1_0040` route=`anti_template_diversifier` family=`ink_wash_painting` v7_family=`generic_artwork` aspect=`square_artwork` risks=template_distribution_guard,overused_reference_guard
