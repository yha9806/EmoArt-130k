# Track1 Local Shadow Score Calibration

> This is a local shadow calibration, not the official Codabench evaluator.

## Summary

- Calibration confidence: `low_medium`
- Own local/official anchors: `2`
- Anchor package: `v3_gate7`
- Anchor official: FID `80.92`, FID Score `0.55`, AAS `0.98`
- Anchor local fid_like: `63.010429`
- Best public overall in ledger: `0.8`
- Gap to best public overall: `0.03`

## FID To FID Score Fit

- Rows: `8`
- slope: `-0.00262359`
- intercept: `0.76793769`
- r2: `0.991003`

## Local Proxy To Official FID

- model: `linear_local_fid_like_to_official_fid`
- own anchors: `2`
- proxy direction: `anti_correlated`
- slope: `-5.30727605`
- intercept: `415.33374088`
- r2: `1.0`

## Package Ranking

| package | local fid_like | expected official FID | expected overall | lower | upper | expected FID Score | method | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `safe_subset` | 63.308289 | 79.339175 | 0.769892 | 0.769892 | 0.769892 | 0.559784 | own_anchor_fit | projected_official_fid_better_than_anchor |
| `strict_subset` | 63.287051 | 79.451891 | 0.769745 | 0.769745 | 0.769745 | 0.559489 | own_anchor_fit | projected_official_fid_better_than_anchor |
| `full1000_no_fallback` | 63.098811 | 80.450932 | 0.768433 | 0.768433 | 0.768433 | 0.556867 | own_anchor_fit | projected_official_fid_similar_to_anchor |
| `v3_gate7` | 63.010429 | 80.920000 | 0.765000 | 0.765000 | 0.765000 | 0.550000 | observed_official | observed_official |
| `partial757` | 62.414763 | 84.081364 | 0.763672 | 0.763672 | 0.763672 | 0.547343 | own_anchor_fit | projected_official_fid_worse_than_anchor |
| `hybrid_redteam_fid_pass4` | 58.348904 | 105.660000 | 0.740000 | 0.740000 | 0.740000 | 0.490000 | observed_official | observed_official |
| `hybrid_redteam5` | 58.602602 | 104.313555 | 0.737131 | 0.737131 | 0.737131 | 0.494262 | own_anchor_fit | projected_official_fid_worse_than_anchor |
| `current` | 58.526207 | 104.719004 | 0.736599 | 0.736599 | 0.736599 | 0.493198 | own_anchor_fit | projected_official_fid_worse_than_anchor |
| `greedy_top80_raw20` | 57.697893 | 109.115095 | 0.730832 | 0.730832 | 0.730832 | 0.481664 | own_anchor_fit | projected_official_fid_worse_than_anchor |

## Interpretation

- Use this to rank candidate packages before spending Codabench submissions.
- Do not treat the expected score as an official score.
- Public leaderboard rows calibrate the FID-score scale only; they are not our hidden-test labels.
- More own submissions with component scores are required before this can become a medium-confidence scorer.
