# Track2 v17 Classification Calibration Candidate

- Method: `track2_v17_classification_calibration_candidate_v1`
- Profile: `balanced`
- Base JSON: `submissions/track2_submission_v15_desc_expand300_candidate.json`
- Candidate JSON: `submissions/track2_submission_v17_balanced_candidate.json`
- Candidate ZIP: `submissions/track2_submission_v17_balanced_candidate.zip`
- Accepted label changes: 15
- Label consistency issues: 0
- Missing emotions: none
- Top emotion: calm (48.7%)
- Formal submission overwritten: False

## Transition Counts

- alarmed->sad: 1
- aroused->calm: 1
- calm->aroused: 1
- calm->bored: 1
- calm->excited: 1
- calm->sad: 1
- calm->tired: 1
- content->excited: 2
- content->happy: 1
- tired->calm: 1
- tired->content: 1
- tired->sad: 3

## Distribution

- alarmed: 53
- annoyed: 22
- aroused: 29
- bored: 21
- calm: 487
- content: 236
- excited: 18
- frustrated: 46
- glad: 5
- happy: 9
- sad: 38
- tired: 36

## Accepted Changes

- track2_0030: alarmed->sad; support=2.47; confidence=0.73; sources=clip,dinov2,public_clean,public_inclusive,siglip2
- track2_0063: tired->sad; support=2.96; confidence=0.79; sources=clip,dinov2,public_clean,public_inclusive,siglip2
- track2_0090: calm->excited; support=2.21; confidence=0.73; sources=dinov2,public_clean,public_inclusive,siglip2
- track2_0196: tired->sad; support=3.13; confidence=0.82; sources=clip,dinov2,public_clean,public_inclusive,siglip2
- track2_0215: tired->content; support=1.96; confidence=0.81; sources=dinov2,public_clean,public_inclusive,siglip2
- track2_0319: calm->bored; support=2.51; confidence=0.73; sources=clip,dinov2,public_clean,public_inclusive,siglip2
- track2_0382: content->excited; support=3.84; confidence=0.95; sources=clip,dinov2,public_clean,public_inclusive,public_near_public_duplicate,siglip2
- track2_0457: calm->aroused; support=1.93; confidence=0.75; sources=clip,dinov2,public_clean,public_inclusive,siglip2
- track2_0505: tired->calm; support=3.54; confidence=0.81; sources=clip,dinov2,public_clean,public_inclusive,siglip2
- track2_0567: tired->sad; support=1.78; confidence=0.76; sources=clip,public_clean,public_inclusive,siglip2
- track2_0591: calm->sad; support=1.74; confidence=0.73; sources=clip,public_clean,public_inclusive,siglip2
- track2_0770: content->happy; support=1.36; confidence=0.75; sources=clip,dinov2
- track2_0779: content->excited; support=2.92; confidence=0.87; sources=clip,dinov2,public_clean,public_inclusive,siglip2
- track2_0873: aroused->calm; support=1.68; confidence=0.95; sources=clip,public_clean,public_inclusive,public_near_public_duplicate,siglip2
- track2_0925: calm->tired; support=2.81; confidence=0.81; sources=clip,dinov2,public_clean,public_inclusive,siglip2
