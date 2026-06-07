# Track1 Final Two-Shot Strategy

## Objective

Use the final two remaining Track1 submissions to maximize prize probability, with no mutation of the existing champion package.

## Current Evidence

- Official score is `0.5 * FID Score + 0.5 * AAS`.
- Visible first place is `0.7969073906`, with FID `66.3621921022`, FID Score `0.6010981145`, AAS `0.9927166667`.
- Our known `v3_gate7` anchor is roughly Overall `0.77`, FID `80.92`, FID Score `0.55`, AAS `0.98`.
- Our latest `hybrid_probe_redteam_fid_pass4` scored Overall `0.7396402011`, FID `105.6638160234`, FID Score `0.4862304023`, AAS `0.99305`.
- The latest failure was not a v3/full1000 derivative. It was old-current lineage: 996 old current images plus 4 replacements.

## Lineage Findings

| package | relation | implication |
| --- | --- | --- |
| `v3_gate7` | 1000/1000 v3 anchor | known official anchor |
| `full1000_no_fallback` | 993/1000 equal to v3 | same high-score family |
| `safe_subset` | 966/1000 equal to v3, 973/1000 equal to full1000 | conservative v3-family repair |
| `strict_subset` | 970/1000 equal to v3, 977/1000 equal to full1000 | stricter conservative repair |
| `hybrid_probe_redteam_fid_pass4` | old-current lineage | do not use for final package design |

## Score Math

To beat `0.7969073906`:

| assumed AAS | required FID Score | approximate required FID |
| ---: | ---: | ---: |
| 0.990 | 0.603815 | <= 62.56 |
| 0.993 | 0.600815 | <= 63.70 |
| 0.995 | 0.598815 | <= 64.46 |
| 1.000 | 0.593815 | <= 66.37 |

This means AAS repair alone cannot win. With FID near `80`, even perfect AAS stays around `0.78`. First place requires a full-package FID distribution improvement, not only localized artifact fixes.

## Local Projection

The local calibration is low/medium confidence because it only has two own official anchors. It is useful for risk control, not for claiming a winning score.

| package | local fid_like | projected/observed official FID | projected/observed overall | method |
| --- | ---: | ---: | ---: | --- |
| `safe_subset` | 63.308289 | 79.339175 | 0.769892 | projected |
| `strict_subset` | 63.287051 | 79.451891 | 0.769745 | projected |
| `full1000_no_fallback` | 63.098811 | 80.450932 | 0.768433 | projected |
| `v3_gate7` | 63.010429 | 80.920000 | 0.765000 | observed/rounded |
| `hybrid_probe_redteam_fid_pass4` | 58.348904 | 105.660000 | 0.740000 | observed |

So `safe_subset` is a conservative measurement package, not a realistic first-place package by itself.

## Submission Plan

### Submission 4: calibrated conservative measurement

Upload:

`submissions/track1_submit_v2_safe_subset.zip`

Why:

- It is v3/full1000 lineage, not old-current lineage.
- It applies 27 high-precision fallback fixes.
- It is already validated and ZIP contents match its package directory.
- It should tell us whether v3-family AAS repairs improve components without collapsing FID.

Do not upload:

- `hybrid_probe_redteam_fid_pass4.zip`
- any package derived mainly from old current images
- any package built by local fid_like greedy search, because that local proxy is anti-correlated with our official anchors

### Submission 5: choose after Submission 4 components

Use exact Codabench components from Submission 4:

- If FID is `<= 75` and AAS is `>= 0.99`, a stronger same-family package is allowed only if its visual/style-family audit is clean.
- If FID is still `79-81`, do not spend the last upload on another small AAS repair. Build a true FID-breakthrough full package first.
- If FID worsens or AAS drops below `0.985`, use `strict_subset` or rollback to v3-family conservative packaging.

Do not submit the fifth package without rerunning package validation and champion immutability checks.

## FID-Breakthrough Workstream

The remaining prize gap is FID, so the next offline work must focus on distribution:

1. Use official 130k/reference style clusters across all 1000 prompts.
2. Prevent one-template poster collapse; preserve support/surface only when caption asks for it.
3. Use aspect routing freely from caption, not forced square.
4. Keep AAS gates, but do not overfit prompts into rigid layouts.
5. Build package-level style-family dashboards before any final upload.
6. Treat Gemini/VLM reviews as evidence, not final truth; use human spot checks for relation and physical logic.

## Verified Files

- `submissions/track1_submit_v2_safe_subset.zip`
- `submissions/track1_submit_v2_strict_subset.zip`
- `submissions/track1_submit_v3_gate7_20260606.zip`
- `submissions/track1_candidate_full1000_aas_safe_official_v7_no_fallback_20260605.zip`

All four ZIPs have:

- `submission.json`
- 1000 JPG images
- no ZIP corruption
- no mismatch against their corresponding package directory

Champion immutability check:

- `git diff -- submissions/track1_submission.json submissions/track1_submission.zip submissions/track1/images | wc -l` returned `0`
