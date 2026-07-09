# Inference + AU12 accuracy evaluation (June 2026)

Using `configs/inference_genphoto/expression.yaml` (updated with the 100 k checkpoint path and real local paths for backbones), three text prompts were evaluated with target intensities `[0.0, 0.25, 0.5, 0.75, 1.0]`. Generated GIFs are under `inference_output/expression_eval/`.

## Results

| Prompt | Detected AU12 trajectory | Pearson *r* |
|--------|--------------------------|-------------|
| Young woman | [0.098, 0.072, 0.157, 0.771, 0.949] | **0.9099** |
| Elderly woman | [0.453, 0.756, 0.725, 0.820, 0.865] | **0.8728** |
| Young man | [0.890, 0.828, 0.929, 0.960, 0.952] | **0.7475** |
| **Mean** | | **0.8434** |

## Interpretation

- **Young woman:** Excellent monotonic control; strong dynamic range from near-neutral to broad smile.
- **Elderly woman:** Good control with a minor non-monotonic dip at frame 3.
- **Young man:** High baseline smile even at intensity 0.0 (AU12 ≈ 0.89). The model compresses dynamic range for this identity, lowering correlation. This is a **side-effect / bias** likely stemming from the MEAD training distribution.

## Fixes applied during evaluation

1. **`comp_metrics/expression_au_accuracy.py`** — The installed `py-feat` version rejected numpy arrays passed to `detect_image()`. The script now writes temporary PNG files per frame before detection.
2. **Detector constructor compatibility** — Updated `emotion_model="resmasknet"` and `facepose_model="img2pose"` to match `py-feat`'s allowed model names.
3. **`configs/inference_genphoto/expression.yaml`** — `/path/to/...` placeholders replaced with real local paths (HF cache + final checkpoint).
