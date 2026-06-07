# Track1 Greedy Inception Hybrid FID Search

这是本地 Inception FID-like 子集搜索，不是官方评分器，也不是最终替换清单。

## Summary

- candidate pool: `4`
- selected replacements: `4`
- baseline fid_like: `58.526207`
- best fid_like: `58.348904`
- total improvement: `0.177303`

## Selected Replacements

1. `track1_0140` candidate=`full1000_no_fallback_redteam_pass` fid_like=`58.446207` improvement=`0.08`
2. `track1_0488` candidate=`full1000_no_fallback_redteam_pass` fid_like=`58.41008` improvement=`0.036127`
3. `track1_0233` candidate=`full1000_no_fallback_redteam_pass` fid_like=`58.378457` improvement=`0.031623`
4. `track1_0697` candidate=`full1000_no_fallback_redteam_pass` fid_like=`58.348904` improvement=`0.029553`

## Stop Conditions

- max replacements reached or no candidates were provided
