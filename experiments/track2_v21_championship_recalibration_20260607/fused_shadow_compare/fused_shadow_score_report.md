# Track2 Fused Shadow Evaluator Report

> This is a local fused shadow score. It is not the official Codabench score and must not be treated as hidden-test ground truth.

- Method: `track2_fused_shadow_score_v1_formula_plus_vulca_entailment`
- Baseline JSON: `submissions/track2_submission_moe_v2_accept5_candidate.json`
- Formal submission overwritten: False

## Policy

Classification comes from the calibrated local shadow evaluator. Description is conservatively adjusted by VULCA/Judge++ entailment risk delta with a small capped effect, so text-risk improvements can break ties but cannot override classification hold decisions.

## Candidate Ranking

| rank | candidate | decision | overall lower | overall expected | class expected | desc expected | vulca delta | vulca issues | changes | cross quadrant |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | v15_desc_expand300 | recommend_submit | 0.835854 | 0.839902 | 0.723150 | 0.956655 | 0.006987 | 39 | 0 | 0 |
| 2 | v21_calmshift90 | recommend_hold | 0.780279 | 0.840827 | 0.725000 | 0.956655 | 0.006987 | 39 | 90 | 0 |
| 3 | v21_precision80 | recommend_hold | 0.701689 | 0.812402 | 0.668150 | 0.956655 | 0.006987 | 39 | 68 | 12 |
| 4 | v21_champion120 | recommend_hold | 0.648392 | 0.807962 | 0.668150 | 0.947775 | -0.001892 | 41 | 89 | 18 |
| 5 | v21_lastshot160 | recommend_hold | 0.636767 | 0.807962 | 0.668150 | 0.947775 | -0.001892 | 41 | 104 | 18 |
