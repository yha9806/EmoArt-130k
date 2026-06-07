# Track2 v19 Ablation Report

| ablation | candidate | changes | cross | class lower | overall lower | decision | repeats 781601 family |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| no_aggregate_calibration | v15_desc_expand300 | 0 | 0 | 0.723150 | 0.835854 | recommend_submit | False |
| transition_family_constrained_calibration | v19_precision | 16 | 4 | 0.639400 | 0.793979 | recommend_hold | False |
| balanced_transition_family_constrained_calibration | v19_balanced | 29 | 10 | 0.526400 | 0.737464 | recommend_hold | False |
| global_aggregate_probe | v19_probe | 39 | 20 | 0.398900 | 0.670292 | recommend_hold | True |

Decision: `hold_v19_keep_v15`

All v19 label-changing candidates have lower local classification and overall lower bounds than v15_desc_expand300.
