# Track1 Prompt Template Lint

- Status: `fail`
- Total prompts: 28
- Threshold: 0.6
- Failed phrases: ['front-facing', 'medium-distance figures', 'fewer, larger']
- Warned phrases: ['portrait poster canvas', 'flat printed poster', 'internal poster margins', 'graphic poster composition']
- Justified phrases: ['portrait poster canvas', 'flat printed poster', 'internal poster margins', 'graphic poster composition']

## Phrase Counts

| Phrase | Status | Count | Ratio | Justified | Sample IDs |
| --- | --- | ---: | ---: | ---: | --- |
| `front-facing` | `fail` | 28 | 1.0 | 0 | track1_0665, track1_0803, track1_0077, track1_0370, track1_0747, track1_0476, track1_0898, track1_0708, track1_0353, track1_0593 |
| `portrait poster canvas` | `warn` | 27 | 0.964286 | 27 | track1_0665, track1_0803, track1_0077, track1_0370, track1_0747, track1_0476, track1_0898, track1_0708, track1_0353, track1_0593 |
| `flat printed poster` | `warn` | 27 | 0.964286 | 27 | track1_0665, track1_0803, track1_0077, track1_0370, track1_0747, track1_0476, track1_0898, track1_0708, track1_0353, track1_0593 |
| `medium-distance figures` | `fail` | 27 | 0.964286 | 0 | track1_0665, track1_0803, track1_0077, track1_0370, track1_0747, track1_0476, track1_0898, track1_0708, track1_0353, track1_0593 |
| `fewer, larger` | `fail` | 27 | 0.964286 | 0 | track1_0665, track1_0803, track1_0077, track1_0370, track1_0747, track1_0476, track1_0898, track1_0708, track1_0353, track1_0593 |
| `fewer, larger text blocks` | `pass` | 0 | 0.0 | 0 |  |
| `fewer/larger text blocks` | `pass` | 0 | 0.0 | 0 |  |
| `internal poster margins` | `warn` | 27 | 0.964286 | 27 | track1_0665, track1_0803, track1_0077, track1_0370, track1_0747, track1_0476, track1_0898, track1_0708, track1_0353, track1_0593 |
| `graphic poster composition` | `warn` | 27 | 0.964286 | 27 | track1_0665, track1_0803, track1_0077, track1_0370, track1_0747, track1_0476, track1_0898, track1_0708, track1_0353, track1_0593 |
