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
| 2 | v16_rag_top6_conf085 | recommend_submit | 0.835154 | 0.839952 | 0.723250 | 0.956655 | 0.006987 | 39 | 2 | 0 |
| 3 | official_782683_v12_stable_probe | recommend_submit | 0.834595 | 0.838104 | 0.723150 | 0.953057 | 0.003390 | 50 | 0 | 0 |
| 4 | official_779605_moe_v2_anchor | recommend_submit | 0.833408 | 0.836408 | 0.723150 | 0.949667 | 0.000000 | 62 | 0 | 0 |
