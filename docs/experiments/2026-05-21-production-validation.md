# Production-path validation + blocker fixes (May 21, 2026)

Two checks on the **full** annotations (`MEAD_processed/annotations/{train,validation}.json`, 4,028 / 176 clips) with **real** pretrained backbones — i.e. exactly the code path the production run takes, just with `max_train_steps: 3` and `train_batch_size: 1`.

## (a) Static dataset + model + forward pass — `scripts/_validate_dataset_and_model.py`

```
[1/6] config loaded: expression_validation.yaml
       pretrained_model_path exists: True
       lora_ckpt / motion_module_ckpt / train+val annotations: all True
[2/6] ExpressionMEAD (train) ok  (4028 clips, build 0.2s)
[3/6] one sample loaded (0.3s)
       pixel_values    : (5, 3, 256, 384) range [-0.933, 1.000]
       camera_embedding: (5, 6, 256, 384) phys_mean=0.842 ccl_mean=2.3e-05
       intensity_values: [0.640, 0.815, 0.869, 0.918, 0.970]
       text            : 'A portrait photograph of a man, frontal view, neutral background.'
       encoder cin check: 384 == 6 * 8^2 OK
[4/6] UNet3DCondCC + CameraCameraEncoder + LoRA + motion module: ok (12.8s)
       encoder params      : 199.26 M
       'merge' attn params : 18.96 M
[5/6] fp16 fwd+bwd on cuda:0, batch=1, 5 frames @ 256x384
       pred shape   : (1, 4, 5, 32, 48) matches noise
       loss         : 1.3726
       fwd+bwd time : 1.05s
       peak VRAM    : 18.00 GiB (batch=1)
[6/6] all checks passed.
```

## (b) Real `train_expression.py` run with `configs/train_genphoto/expression_validation.yaml`

- All checkpoints loaded cleanly (`Loading the image lora checkpoint … done`, `Loading the motion module checkpoint … done`).
- Trainable param accounting matches expectations: **182 params total = 142 encoder (199.26 M) + 40 attention `merge` (18.96 M) = 218.22 M**.
- Step-0 sanity GIF written to `sanity_check/A-portrait-photograph-of-a-woman,-frontal-view,-neutral-background..gif`.
- Validation pipeline at step 2 generated **100 paired sample/reference GIFs** in `samples/sample-1/` (run killed before iterating through all 176 val clips — see Finding #2).

Outputs are under `output/expression_validation/expression_validation-2026-05-21T01-36-27/`.

## Validation findings & fixes applied

### Finding #1 — VRAM: production `train_batch_size: 4` would OOM on RTX 3090 — **FIXED**

| Measurement | Value |
|---|---|
| Peak VRAM, batch=1, 5 frames @ 256×384, fp16 autocast | **18.00 GiB** |
| Available RTX 3090 VRAM (per device) | 24.00 GiB |

**Why:** VAE, text encoder, full UNet3D, encoder, and attention-merge params all on one GPU; frozen modules stay fp32, only the forward region is autocast to fp16; activations kept for backward.

**Fix applied:** `train_expression.py` originally ignored `gradient_accumulation_steps` — `optimizer.step()` ran every micro-batch. The training loop was patched to:

- divide the per-micro-batch loss by `gradient_accumulation_steps` before `.backward()`,
- only call `optimizer.step()` / `lr_scheduler.step()` / `optimizer.zero_grad()` and increment `global_step` every `gradient_accumulation_steps` micro-batches,
- preserve grad-clip + GradScaler semantics around the real optimizer step.

`configs/train_genphoto/expression.yaml` now uses `train_batch_size: 1, gradient_accumulation_steps: 4` (effective batch 4). Re-validation measured **peak 18.98 GiB at effective batch=4** ✓.

(Optional further savings if needed later: `vae.half()`, `text_encoder.half()`, or `unet.enable_gradient_checkpointing()` — not needed to start.)

### Finding #2 — Validation loop iterates over all 176 val clips — **FIXED**

`train_expression.py` had no cap on the validation dataloader (~8.5 s/clip × 176 clips ≈ 25 min/event; ≈ 83 h over a 100 k-step run with `validation_steps: 500`).

**Fix applied:** added an optional `max_validation_samples: int = None` parameter to `train_expression.py` (the production config sets it to **8**). The val loop now logs `Reached max_validation_samples=N; skipping remaining M val clips.` and breaks. Verified during re-validation (output: `Reached max_validation_samples=2; skipping remaining 174 val clips.`).

Production config also bumps `validation_steps` from 500 → **5000** (20 events instead of 200) and sets `validation_steps_tuple: [200]` for one early regression-catch event.

### Finding #3 — Disk: `/` is 99 % full — **FIXED**

Checkpoints are ~2.6 GB each (100 ckpts × 2.6 GB ≈ 260 GB), so writing to `/` (13 GB free) would crash mid-run.

**Fix applied:** `output_dir: "/databases-4tb/levi-experiments/generative-photography/output/expression"` in `configs/train_genphoto/expression.yaml`. `/databases-4tb` has 1.7 TB free.

## Re-validation after fixes

Ran the patched `train_expression.py` against `configs/train_genphoto/expression_validation.yaml` (production code path, batch=1 + grad-accum=4 + max_validation_samples=2 + 3 optimizer steps):

```
Iter: 1/3, Loss: 0.0032, GPU memory: 17.33 G
Iter: 2/3, Loss: 0.0279, GPU memory: 18.98 G
Saved state to .../checkpoints (global_step: 3)
Iter: 3/3, Loss: 0.0360, GPU memory: 18.98 G
Reached max_validation_samples=2; skipping remaining 174 val clips.
```

- Wall time: **~80 s end-to-end** (vs. >25 min in the pre-fix run that had to be killed).
- Effective batch = 4 via accumulation: each `Iter` aggregates 4 micro-batches (per-micro-batch loss shown is the post-division value; varies widely with the random diffusion timestep, which is normal).
- Checkpoint at step 3: **2.5 GB on `/databases-4tb`**.
- 2 val sample/reference pairs generated, cap fired.
- No leaked GPU memory after exit.

## Non-findings (verified safe)

- All four required pretrained artifacts resolved on disk (SD1.5 base, `unet_merged`, RealEstate10K LoRA, AnimateDiff v3 motion module).
- `ExpressionMEAD` loads 4,028 train clips in **0.2 s** at process start and **0.3 s** per sample (CLIP encode is the bottleneck per sample — fine at `num_workers ≥ 2`).
- 6-channel embedding has expected statistics (physical channels mean ≈ 0.84 from the high-AU MEAD ramp; CCL channel mean ≈ 2e-5, i.e. small but non-zero text-embedding differences).
- `encoder cin == 6 × downscale²` invariant holds (384 = 6 × 64).
- Forward pred shape `(1, 4, 5, 32, 48)` matches noise — the adaptor returns the right tensor for the MSE loss.
- Initial MSE loss ~1.37, the expected order of magnitude for an untrained adaptor predicting noise.
- After killing the validation run, GPU 2 returned to idle (no leaked memory).
