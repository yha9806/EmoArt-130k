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
| 2 | v15_desc_reuse192 | recommend_submit | 0.835570 | 0.839496 | 0.723150 | 0.955842 | 0.006175 | 42 | 0 | 0 |
| 3 | official_782683_v12_stable_probe | recommend_submit | 0.834595 | 0.838104 | 0.723150 | 0.953057 | 0.003390 | 50 | 0 | 0 |
| 4 | v14_description_max_safe | recommend_submit | 0.834595 | 0.838104 | 0.723150 | 0.953057 | 0.003390 | 50 | 0 | 0 |
| 5 | v14_public_resource_agsr_max | recommend_submit | 0.833545 | 0.838178 | 0.723300 | 0.953057 | 0.003390 | 50 | 3 | 0 |
| 6 | official_779605_moe_v2_anchor | recommend_submit | 0.833408 | 0.836408 | 0.723150 | 0.949667 | 0.000000 | 62 | 0 | 0 |
| 7 | v14b_sameq_clip094_exactnear | recommend_submit | 0.832359 | 0.836483 | 0.723300 | 0.949667 | 0.000000 | 62 | 3 | 0 |
| 8 | v14b_sameq_clip095_exactnear | recommend_submit | 0.832359 | 0.836483 | 0.723300 | 0.949667 | 0.000000 | 62 | 3 | 0 |
| 9 | v14b_sameq_clip095 | recommend_submit | 0.831284 | 0.836534 | 0.723400 | 0.949667 | 0.000000 | 62 | 6 | 0 |
| 10 | v14b_sameq_clip094 | recommend_submit | 0.824209 | 0.836508 | 0.723350 | 0.949667 | 0.000000 | 62 | 23 | 0 |
| 11 | v13_precision | recommend_submit | 0.810048 | 0.837349 | 0.725030 | 0.949667 | 0.000000 | 62 | 48 | 0 |
| 12 | v13_balanced | recommend_hold | 0.779278 | 0.837228 | 0.724790 | 0.949667 | 0.000000 | 62 | 88 | 0 |
