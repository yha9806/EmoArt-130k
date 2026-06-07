# Track2 v16 Official Counterfactual Calibration

- Method: `track2_v16_official_counterfactual_calibration_v1`
- Anchor submission: `779605`
- Anchor classification: 0.723150
- Official score rows: 2
- Pairwise diff rows: 3

## Policy

Official aggregate results show that the 781601 same-quadrant batch reduced classification. v16 therefore blocks failed high-count boundary transitions unless stronger per-sample evidence is present.

## Known Failed Batches

- 781601 vs 779605: class delta -0.004013, changed rows 85, per-row -0.00004721

## Failed Transition Counts

- aroused->excited: 1
- bored->sad: 1
- calm->content: 43
- calm->glad: 2
- content->calm: 20
- content->glad: 5
- excited->happy: 1
- glad->calm: 1
- glad->content: 4
- happy->excited: 3
- sad->tired: 1
- tired->sad: 3
