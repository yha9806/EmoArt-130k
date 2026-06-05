# Track1 Reference Role Gate

这份报告检查每个 reference board 是否覆盖官方 caption 需要的关键角色：媒介、主体、符号、真实锚点、动作关系和空间逻辑。

## 摘要

- Total: 1000
- Pass: 968
- Fail: 32
- Provider fallback: 1

## 缺失角色统计

- `medal_symbol`: 14
- `aircraft`: 8
- `cannon_artillery`: 6
- `allied_flags`: 4
- `mounted_civilian_relation`: 2
- `train_window`: 1
- `naval_battle`: 1

## Fail Rows

- `track1_0665`: missing=[medal_symbol] flags=[]
- `track1_0803`: missing=[allied_flags] flags=[]
- `track1_0747`: missing=[mounted_civilian_relation, aircraft] flags=[provider_reference_fallback]
- `track1_0476`: missing=[medal_symbol, cannon_artillery] flags=[]
- `track1_0091`: missing=[train_window] flags=[]
- `track1_0816`: missing=[mounted_civilian_relation] flags=[]
- `track1_0721`: missing=[allied_flags] flags=[]
- `track1_0802`: missing=[medal_symbol] flags=[]
- `track1_0534`: missing=[medal_symbol] flags=[]
- `track1_0327`: missing=[cannon_artillery] flags=[]
- `track1_0686`: missing=[medal_symbol] flags=[]
- `track1_0038`: missing=[medal_symbol] flags=[]
- `track1_0197`: missing=[cannon_artillery] flags=[]
- `track1_0643`: missing=[aircraft] flags=[]
- `track1_0973`: missing=[cannon_artillery] flags=[]
- `track1_0241`: missing=[allied_flags] flags=[]
- `track1_0661`: missing=[medal_symbol, cannon_artillery] flags=[]
- `track1_0108`: missing=[allied_flags] flags=[]
- `track1_0865`: missing=[aircraft] flags=[]
- `track1_0705`: missing=[medal_symbol] flags=[]
- `track1_0308`: missing=[cannon_artillery] flags=[]
- `track1_0386`: missing=[medal_symbol] flags=[]
- `track1_0005`: missing=[aircraft] flags=[]
- `track1_0273`: missing=[medal_symbol] flags=[]
- `track1_0770`: missing=[aircraft] flags=[]
- `track1_0663`: missing=[medal_symbol] flags=[]
- `track1_0692`: missing=[aircraft] flags=[]
- `track1_0717`: missing=[medal_symbol] flags=[]
- `track1_0881`: missing=[medal_symbol, aircraft] flags=[]
- `track1_0235`: missing=[aircraft] flags=[]
- `track1_0732`: missing=[naval_battle] flags=[]
- `track1_0251`: missing=[medal_symbol] flags=[]
