# Track1 Inception FID-Like Sanity Check

This is not the official hidden evaluator. It uses local torchvision InceptionV3 features, torchvision-style resize/normalization, sampled local EmoArt reference images, and optional deterministic Gaussian projection.

## Method

- Device: `mps`
- Reference styles: `['Socialist Realism', 'Social Realism']`
- Reference count: `1024`
- Reference seed: `20260603`
- FID feature dim: `512`

## Packages

- `expanded_12` fid_like=`69.052224` images=`1000` best
- `high_precision_8` fid_like=`69.061162` images=`1000`
- `base` fid_like=`69.078378` images=`1000`

Wall seconds: `97.49`
