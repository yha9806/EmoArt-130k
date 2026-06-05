# Track2 v5 cross score>=7 Gemini arbitration summary

## Summary

- Reviewed candidates: `9`
- Strict candidate accepts: `1`
- Score normalization: raw Gemini scores outside 0-1 are normalized with the same scale guard used in the Track2 description pipeline.

## Table

| sample_id | transition | evidence | preferred | confidence | current fit | proposed fit | margin | strict enter |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `track2_0547` | `annoyed->calm` | 11 | `proposed` -> calm / Positive / Low | 0.80 | 0.35 | 0.68 | 0.33 | `True` |
| `track2_0088` | `calm->tired` | 7 | `current` -> calm / Positive / Low | 0.75 | 0.70 | 0.40 | -0.30 | `False` |
| `track2_0139` | `calm->tired` | 7 | `proposed` -> tired / Negative / Low | 0.70 | 0.40 | 0.60 | 0.20 | `False` |
| `track2_0319` | `calm->bored` | 7 | `current` -> calm / Positive / Low | 0.75 | 0.60 | 0.40 | -0.20 | `False` |
| `track2_0414` | `calm->bored` | 7 | `current` -> calm / Positive / Low | 0.70 | 0.60 | 0.40 | -0.20 | `False` |
| `track2_0457` | `calm->aroused` | 7 | `current` -> calm / Positive / Low | 0.70 | 0.60 | 0.40 | -0.20 | `False` |
| `track2_0665` | `calm->aroused` | 7 | `current` -> calm / Positive / Low | 0.75 | 0.68 | 0.52 | -0.16 | `False` |
| `track2_0728` | `alarmed->aroused` | 7 | `current` -> alarmed / Negative / High | 0.80 | 0.65 | 0.45 | -0.20 | `False` |
| `track2_0925` | `calm->tired` | 7 | `current` -> calm / Positive / Low | 0.70 | 0.60 | 0.40 | -0.20 | `False` |

## Decision

The strict algorithmic review finds candidates worth testing locally:

- `track2_0547` `annoyed->calm`: preferred `proposed` with normalized confidence `0.80` and fit margin `0.33`.

The eight score=7 cross-quadrant rows do not pass the strict gate. Most are calm-to-negative or calm-to-high-arousal proposals where the image evidence is not strong enough to justify taking quadrant risk.

## Notes

For score=7 rows, a proposed label is not enough; it must be clearly better than the current label. Borderline abstract or calm/content cases remain hold.
