# Preprocessing contract (leaf model)

This is the **binding contract** for every platform that runs the leaf model. Each platform implements exactly these steps starting from the **JPEG file bytes**, and its implementation is proven by the golden-vector parity tests (`build/golden/`, `tests/test_prototype.py`).

## Steps, in order

1. **Decode** the JPEG bytes to pixels (any spec-compliant JPEG decoder; slight decoder differences are inside the parity tolerances).
2. **EXIF orientation**: apply the orientation tag if present (phones emit rotated JPEGs). If no EXIF library is available on the platform, the image must be rejected with a clear message rather than silently mis-oriented.
3. **Color order**: convert to **RGB** (3 channels, uint8). No alpha, no grayscale.
4. **Resize**: direct resize to **224 × 224**, **bilinear** interpolation, **no crop**, no letterbox, no aspect-ratio preservation. (This matches the evaluation transform used during training and validation: `torchvision.transforms.Resize((224, 224))`.)
5. **Scale**: convert uint8 `[0, 255]` → float `[0, 1]` (divide by 255).
6. **Normalize**: subtract ImageNet mean, divide by ImageNet std, per channel:
   - mean = `[0.485, 0.456, 0.406]`
   - std = `[0.229, 0.224, 0.225]`
7. **Layout**: NCHW, float32, batch dimension present (`1 × 3 × 224 × 224`).

## Output

A single float32 tensor `1×3×224×224`. This tensor is the model input. The model output is 7 raw logits; the UI applies softmax only to display a **model score** (uncalibrated — temperature scaling made calibration worse in our validation analysis; see `build/model_card.json`).

## Not used

- No center crop, no random crop at inference, no aspect-ratio letterboxing.
- No test-time augmentation, no ensembling.
- No per-image histogram equalization or color correction.

## Provenance

- Training/evaluation transform source: `tomatoguard/research/models.py` (`eval_tf`) via `analysis/trackA_lib.py`.
- ImageNet mean/std values read from the same module (`IMAGENET_MEAN`, `IMAGENET_STD`); mirrored in `build/model_meta.json` (generated, not hand-typed).
- Golden vectors: 50 fixed S2-test JPEGs with expected PyTorch logits in `build/golden/manifest.json` (each entry carries the JPEG's SHA-256).
