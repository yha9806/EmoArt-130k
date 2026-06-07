# Track1 Inception FID-Like Sanity Check

This is not the official hidden evaluator. It uses local torchvision InceptionV3 features, torchvision-style resize/normalization, sampled local EmoArt reference images, and optional deterministic Gaussian projection.

## Method

- Device: `mps`
- Reference styles: `all_tar_styles`
- Reference count: `2048`
- Reference seed: `20260607`
- FID feature dim: `512`

## Packages

- `current` fid_like=`58.526207` images=`1000` best
- `full1000_no_fallback` fid_like=`63.098811` images=`1000`

Wall seconds: `309.03`
