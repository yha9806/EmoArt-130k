# Track1 Local Shadow Score Calibration

> This is a local shadow calibration, not the official Codabench evaluator.

## Summary

- Calibration confidence: `low`
- Own local/official anchors: `1`
- Anchor package: `v3_gate7`
- Anchor official: FID `80.92`, FID Score `0.55`, AAS `0.98`
- Anchor local fid_like: `63.010429`
- Best public overall in ledger: `0.8`
- Gap to best public overall: `0.03`

## FID To FID Score Fit

- Rows: `7`
- slope: `-0.00261805`
- intercept: `0.76757502`
- r2: `0.989724`

## Package Ranking

| package | local fid_like | expected overall | lower | upper | expected FID Score | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `greedy_top80_raw20` | 57.697893 | 0.771338 | 0.769600 | 0.774815 | 0.562677 | local_proxy_better_than_anchor |
| `hybrid_redteam_fid_pass4` | 58.348904 | 0.770912 | 0.769386 | 0.773964 | 0.561824 | local_proxy_better_than_anchor |
| `current` | 58.526207 | 0.770796 | 0.769328 | 0.773731 | 0.561592 | local_proxy_better_than_anchor |
| `hybrid_redteam5` | 58.602602 | 0.770746 | 0.769303 | 0.773631 | 0.561492 | local_proxy_better_than_anchor |
| `partial757` | 62.414763 | 0.768251 | 0.768056 | 0.768641 | 0.556502 | local_proxy_better_than_anchor |
| `v3_gate7` | 63.010429 | 0.767861 | 0.767861 | 0.767861 | 0.555722 | local_proxy_similar_to_anchor |
| `full1000_no_fallback` | 63.098811 | 0.767803 | 0.767745 | 0.767832 | 0.555607 | local_proxy_similar_to_anchor |
| `strict_subset` | 63.287051 | 0.767680 | 0.767499 | 0.767770 | 0.555360 | local_proxy_similar_to_anchor |
| `safe_subset` | 63.308289 | 0.767667 | 0.767471 | 0.767764 | 0.555333 | local_proxy_similar_to_anchor |

## Interpretation

- Use this to rank candidate packages before spending Codabench submissions.
- Do not treat the expected score as an official score.
- Public leaderboard rows calibrate the FID-score scale only; they are not our hidden-test labels.
- More own submissions with component scores are required before this can become a medium-confidence scorer.
