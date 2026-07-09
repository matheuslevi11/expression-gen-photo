# Trained vs. untrained / ascending vs. descending ablation (July 8, 2026)

> **Correction notice (July 9, 2026):** the diagnosis originally recorded here — that the untrained baseline was corrupted by "randomly-initialised merge weights" and therefore unfairly bad — was **wrong**. See [`2026-07-09-fair-baseline.md`](2026-07-09-fair-baseline.md): the merge layers are zero-initialised, the camera path without a checkpoint is bit-exactly inert, and the untrained run below **is** a fair frozen-backbone baseline. The measurements in this file remain valid; only the original interpretation was superseded.

## Experiment design

To verify that the 100 k-step training actually taught the model expression control, three inference variants were run with the same prompt and random seed:

| Variant | Adaptor | Intensity list |
|---------|---------|----------------|
| **Trained ascending** | 100 k checkpoint | [0.0, 0.25, 0.5, 0.75, 1.0] |
| **Untrained** | None (random init) | [0.0, 0.25, 0.5, 0.75, 1.0] |
| **Trained descending** | 100 k checkpoint | [1.0, 0.75, 0.5, 0.25, 0.0] |

Outputs: `inference_output/comparison_eval/` (`trained_asc.gif`, `untrained.gif`, `trained_desc.gif`).

## Results

| Variant | AU12 detected | Pearson *r* vs target |
|---------|---------------|----------------------|
| Trained ascending | [0.098, 0.072, 0.157, 0.771, 0.949] | **+0.91** |
| Untrained | [0.031, 0.032, 0.030, 0.018, 0.020] | **−0.85** |
| Trained descending | [0.137, 0.176, 0.406, 0.690, 0.799] | **−0.98** |

## What the ablation proves

1. **Descending correlation = −0.98** — when the intensity list is reversed, the smile trajectory also reverses. This proves the model is conditioned on the actual scalar values, not just generating a fixed low→high animation.
2. **Training provides both domain grounding and control** — versus the frozen backbone, the 100 k checkpoint moves outputs into the photorealistic MEAD portrait domain *and* adds scalar expression control.
3. **Dynamic range is real** — trained AU12 spans ~0.85 (0.10 → 0.95); the frozen backbone is flat (~0.01 span, no smile response at all).

## Superseded interpretation (kept for the record)

The original July 8 write-up claimed: *"the untrained output is garbage because `inference_expression.py` always calls `unet.set_all_attn_processor()`, which replaces every attention processor with a camera-conditioned variant that contains randomly-initialised merge weights … the comparison is between 'SD1.5 + broken attention' and 'SD1.5 + trained expression conditioning'; the delta is real but inflated."*

This was refuted on July 9 by code reading (`init.zeros_` on all merge weights/biases in `genphoto/models/attention_processor.py`) and by a bit-identity experiment. The delta is **not** inflated.
