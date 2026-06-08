# Track2 v25 signal audit

## 结论

- decision: `needs_new_classification_signal`
- reasons: `best_candidate_below_target, top_emotion_collapse`
- best_existing_candidate: `track2_submission_v24_final_candidate`
- best_existing_overall: `0.853471`
- best_existing_classification: `0.756060`
- best_existing_description: `0.950882`
- candidates_above_0.86: `0`
- candidates_above_target_0.89: `0`
- three_model_exact_agreement_changes: `140`

## 模型信号

| model | rows | changes | high-conf changes | top emotion | top share |
|---|---:|---:|---:|---|---:|
| clip | 1000 | 425 | 2 | calm | 0.609 |
| dinov2 | 1000 | 458 | 3 | calm | 0.554 |
| ensemble | 1000 | 391 | 0 | calm | 0.614 |
| siglip2 | 1000 | 397 | 4 | calm | 0.589 |

## 三模型一致变化 Top Transitions

- `content->calm`: `81`
- `tired->sad`: `7`
- `tired->calm`: `6`
- `annoyed->calm`: `4`
- `content->excited`: `4`
- `sad->calm`: `4`
- `frustrated->annoyed`: `3`
- `sad->frustrated`: `3`
- `alarmed->content`: `2`
- `alarmed->sad`: `2`
- `aroused->calm`: `2`
- `calm->aroused`: `2`
- `calm->content`: `2`
- `happy->calm`: `2`
- `tired->bored`: `2`
- `tired->content`: `2`
- `alarmed->aroused`: `1`
- `alarmed->calm`: `1`
- `alarmed->frustrated`: `1`
- `annoyed->content`: `1`
- `aroused->annoyed`: `1`
- `aroused->excited`: `1`
- `bored->content`: `1`
- `calm->bored`: `1`
- `calm->tired`: `1`
- `excited->calm`: `1`
- `excited->happy`: `1`
- `happy->excited`: `1`

## 判断

- 现有候选池没有达到本地 0.89 gate；继续组合旧候选没有冲第一依据。
- 三模型一致变化主要用于发现新分类候选，但它本身受 public calm prior 影响，不能无门槛全量套用。
- 下一步若继续冲分，应训练/校准一个不塌缩的 classification v26/v27，然后重新跑 v23/v25 gate。
