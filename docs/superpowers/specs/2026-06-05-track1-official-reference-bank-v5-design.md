# Track1 Official Reference Bank V5 Design

## Goal

Use the official/public EmoArt-130k reference corpus as the retrieval source for Track1 reference-grounded generation, replacing the previous 86-image curated mini-bank.

## Source Of Truth

The official/public local source is:

- `/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k/Annotation.json`
- `/Users/yhryzy/dev/emoart-challenge/data/EmoArt-130k/*.tar.gz`

The directory contains 132k annotated artworks across 56 styles. The images are stored inside style archives, so `find` sees zero direct image files until selected assets are extracted.

## Approach

V5 uses all annotation rows from `Annotation.json` as the candidate retrieval pool. For each Track1 test route, it:

1. Infers the official style archive from the caption, using explicit aliases plus automatic matching over the official 56 style names.
2. Scores every official artwork in that style using caption-token overlap, filename text, annotation text, and poster-specific boosts.
3. Applies a small usage penalty so repeated top references do not dominate.
4. Extracts only the top candidates per route into an experiment reference asset directory.
5. Writes a compatible `route_references` index consumed by `track1_attach_reference_assets.py`.

This is not a blind full extraction of 65GB. The full corpus is used for retrieval, while only selected references are materialized for Gemini prompt packets and visual review.

## Interface

New module:

- `affectiveart.track1_official_reference_bank`

New CLI:

- `scripts/track1_build_official_reference_bank.py`

Output directory:

- `experiments/track1_reference_conditioned_pilot_20260603/official_reference_bank_v5_20260605/`

The generated index has:

```json
{
  "route_references": {
    "track1_0001": [
      {
        "file": "official_Abstract_Art_0006080_YuriZlotnikov_Signalseries_1a2b3c4d5e.jpg",
        "note": "official EmoArt-130k retrieval; style=Abstract Art; score=..."
      }
    ]
  }
}
```

`route_references` has source `official_retrieval` and takes priority over legacy `references`, so old sample-specific mini-bank risks do not override official retrieval.

Extracted filenames include a stable hash suffix derived from the official archive member path, so two official files that normalize to the same safe basename cannot overwrite each other.

## Safety

- Do not mutate the champion package.
- Do not commit extracted generated bank outputs by default.
- Do not claim the official-bank output is automatically accepted for submission; it must go through reference board review, generation, redteam, and human gate.

## Observed Outcome

The v5 route/audit report should show many more unique reference assets than v4. In the 2026-06-05 run, unique references increased from 78 to 1680 across 4000 slots, with 1000/1000 samples receiving official references.

`diversity_limited_routes` remains 1000 in the top4 run because each route materializes exactly four official candidates and the downstream selector also requests four slots. That is not evidence that the official pool is small; the summary records the full official candidate pool by style.
