# Smoke training run (May 16, 2026)

Config: `configs/train_genphoto/expression_smoke.yaml`

- **5 optimization steps** completed.
- Checkpoint: `/tmp/genphoto_smoke/expression_smoke-2026-05-16T23-25-02/checkpoints/checkpoint-step-5.ckpt` (~2.6 GB).
- Validation at step 3 produced paired GIFs under `samples/sample-3/` (e.g. `W009_happy_level_1_001_sample.gif` vs `_reference.gif`).

This confirmed: dataset load → 6-channel embedding → encoder + adaptor forward → MSE loss backward → checkpoint save → `GenPhotoPipeline` validation sampling.

## Pretrained weights verified on disk

Present under the pandaphd Hugging Face snapshot (paths used by the smoke config):

- SD1.5 + `unet_merged`
- `RealEstate10K_LoRA.ckpt`
- `v3_sd15_mm.ckpt`
