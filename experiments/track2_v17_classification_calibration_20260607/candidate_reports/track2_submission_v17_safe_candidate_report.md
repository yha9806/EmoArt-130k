# Track2 v17 Classification Calibration Candidate

- Method: `track2_v17_classification_calibration_candidate_v1`
- Profile: `safe`
- Base JSON: `submissions/track2_submission_v15_desc_expand300_candidate.json`
- Candidate JSON: `submissions/track2_submission_v17_safe_candidate.json`
- Candidate ZIP: `submissions/track2_submission_v17_safe_candidate.zip`
- Accepted label changes: 8
- Label consistency issues: 0
- Missing emotions: none
- Top emotion: calm (49.1%)
- Formal submission overwritten: False

## Transition Counts

- aroused->calm: 1
- calm->tired: 1
- content->excited: 2
- tired->calm: 1
- tired->content: 1
- tired->sad: 2

## Distribution

- alarmed: 54
- annoyed: 22
- aroused: 28
- bored: 20
- calm: 491
- content: 237
- excited: 17
- frustrated: 46
- glad: 5
- happy: 8
- sad: 35
- tired: 37

## Accepted Changes

- track2_0063: tired->sad; support=2.96; confidence=0.79; sources=clip,dinov2,public_clean,public_inclusive,siglip2
- track2_0196: tired->sad; support=3.13; confidence=0.82; sources=clip,dinov2,public_clean,public_inclusive,siglip2
- track2_0215: tired->content; support=1.96; confidence=0.81; sources=dinov2,public_clean,public_inclusive,siglip2
- track2_0382: content->excited; support=3.84; confidence=0.95; sources=clip,dinov2,public_clean,public_inclusive,public_near_public_duplicate,siglip2
- track2_0505: tired->calm; support=3.54; confidence=0.81; sources=clip,dinov2,public_clean,public_inclusive,siglip2
- track2_0779: content->excited; support=2.92; confidence=0.87; sources=clip,dinov2,public_clean,public_inclusive,siglip2
- track2_0873: aroused->calm; support=1.68; confidence=0.95; sources=clip,public_clean,public_inclusive,public_near_public_duplicate,siglip2
- track2_0925: calm->tired; support=2.81; confidence=0.81; sources=clip,dinov2,public_clean,public_inclusive,siglip2
