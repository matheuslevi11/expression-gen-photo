# Expression adaptation — project status (May 2026)

This document summarizes the current state of the Generative Photography → **Generative Expressions** fork: what has been built, what has been verified on this machine, and what remains before a full training run can produce meaningful expression-synthesis results.

For design rationale, see [`expression-adaptation-plan.md`](expression-adaptation-plan.md). For usage commands, see the root [`README.md`](../README.md).

---

## Executive summary

| Layer | Status |
|-------|--------|
| Code (dataset, train, inference, metrics, preprocess) | **Complete** |
| Environment (`environment.yaml`, pinned deps) | **Complete** |
| Pretrained backbones (SD1.5 + LoRA + motion module) | **On disk** (Hugging Face cache) |
| MEAD preprocessing → training annotations | **Complete** (4,028 train / 176 val clips) |
| End-to-end training graph (forward, loss, validation GIF, checkpoint) | **Verified** (May 21, batch=1 + grad-accum=4 path) |
| Production `expression.yaml` (paths, batch, val cap, output_dir) | **Applied & verified** (May 21) |
| `train_expression.py` (real gradient accumulation + val-sample cap) | **Patched & verified** (May 21) |
| Trained expression adaptor + inference results | **Not done** (smoke checkpoint only; 5 steps) |

**How close are we?** Ready to launch. The production config has been validated on the actual code path — 3 optimizer steps at effective batch=4 finished in ~80 s with peak VRAM 18.98 GiB (fits 24 GiB), 2 paired validation GIFs generated, checkpoint saved on `/databases-4tb`. The only remaining task is to start the long-running training job.

---

## What changed vs. the original Generative Photography repo

The fork **replaces the camera axis with facial expression intensity** while keeping the 3D UNet, AnimateDiff motion module, 6-channel conditioning layout (`cin: 384`), and attention-injection mechanism unchanged.

| Area | Original | This fork |
|------|----------|-----------|
| Conditioning | bokeh / focal / shutter / color temp | scalar smile / AU12 intensity |
| Dataset | `genphoto/data/dataset.py` + BokehMe simulation | `genphoto/data/expression_dataset.py` (`ExpressionMEAD`) |
| Physical channels | blur kernel, crop mask, etc. | scalar broadcast (`create_intensity_embedding`) |
| CCL text | `<bokeh kernel size: …>` | `<smile intensity: …>` |
| Training / inference | `train_*.py` / `inference_*.py` (×4) | `train_expression.py` / `inference_expression.py` |
| Preprocessing | on-the-fly camera simulation | `scripts/preprocess_mead.py` (MEAD video → crops + AU12) |
| Eval (accuracy) | Laplacian / FOV correlation | `comp_metrics/expression_au_accuracy.py` (AU12 vs target Pearson *r*) |
| Removed | — | BokehMe, `depth_any`, Gradio `app.py`, all camera YAMLs and train/inference scripts |

**Unchanged (reused as-is):** `genphoto/models/*`, `genphoto/pipelines/pipeline_animation.py`, `GenPhotoPipeline`, `CameraAdaptor` / `CameraCameraEncoder` class names (semantically the expression encoder), LPIPS and CLIP metric scripts.

---

## New and modified artifacts (by area)

### Data & preprocessing

- **`scripts/preprocess_mead.py`** — Discovers MEAD `video.tar` layouts, extracts front-view happy clips, face-crops with MediaPipe, estimates per-frame AU12 via **py-feat** (or `--au-method timeline` for dependency-free smoke tests). Writes `frames/…` and `annotations/{train,validation}.json`.
- **`genphoto/data/expression_dataset.py`** — `ExpressionMEAD` builds 5-frame ramps (monotonic AU12 sort), 6-channel `camera_embedding` (3 physical + 3 CCL), and CLIP captions. Validation clips can carry a fixed `intensity_list` for eval while pixels still follow a visual low→high ramp in the clip.
- **`MEAD_processed/`** (gitignored) — On this machine, preprocessing **finished successfully** on 2026-05-17:
  - **4,028** train clips, **176** validation clips
  - **24** frames per clip, **~1.9 GB** of cropped JPEGs
  - Annotations include per-frame `au12` from py-feat; validation entries include `intensity_list` for fixed-target evaluation

### Training & inference

- **`train_expression.py`** — Port of `train_bokehK.py`: distributed training, trains expression encoder + attention `merge` weights only; frozen VAE, text encoder, UNet backbone, image LoRA, motion module. Removes accidental `pdb` breakpoint from upstream. Sanity-check GIFs on step 0; validation sample/reference GIFs on schedule.
- **`inference_expression.py`** — Text-to-expression GIF synthesis from a base scene prompt + 5 intensity values.
- **`configs/train_genphoto/expression.yaml`** — Full run template (`max_train_steps: 100000`, batch 4, etc.) with **`/path/to/...` placeholders**.
- **`configs/train_genphoto/expression_smoke.yaml`** — 5-step smoke config with **real local paths** (HF snapshot, `/tmp/mead_validation_out` tiny set).
- **`configs/inference_genphoto/expression.yaml`** + **`expression_smoke.yaml`** — Inference templates; smoke variant points at the smoke checkpoint path.

### Evaluation & docs

- **`comp_metrics/expression_au_accuracy.py`** — Pearson correlation between detected AU12 trajectory and target intensities on generated GIFs.
- **`docs/expression-adaptation-plan.md`** — Design plan (Options A/B/C for embeddings, risks, suggested first experiment).
- **`README.md`** — Rewritten for the expression fork (setup, preprocess, train, infer, eval).
- **`environment.yaml`** — Added `mediapipe`, `py-feat`, `lpips`; pins versions for torch 2.1 / numpy 1.x compatibility.

### Repo hygiene

- **`.gitignore`** — `MEAD`, `MEAD_processed` (large data stay out of git).
- Root **`sanity_grid.jpg`**, **`sanity_grid_v2.jpg`** — Visual sanity checks (not wired into CI).

---

## Verification already performed on this machine

### 1. Smoke training (May 16, 2026)

Config: `configs/train_genphoto/expression_smoke.yaml`

- **5 optimization steps** completed.
- Checkpoint: `/tmp/genphoto_smoke/expression_smoke-2026-05-16T23-25-02/checkpoints/checkpoint-step-5.ckpt` (~2.6 GB).
- Validation at step 3 produced paired GIFs under `samples/sample-3/` (e.g. `W009_happy_level_1_001_sample.gif` vs `_reference.gif`).

This confirms: dataset load → 6-channel embedding → encoder + adaptor forward → MSE loss backward → checkpoint save → `GenPhotoPipeline` validation sampling.

### 2. Pretrained weights

Present under the pandaphd Hugging Face snapshot (paths used by smoke config):

- SD1.5 + `unet_merged`
- `RealEstate10K_LoRA.ckpt`
- `v3_sd15_mm.ckpt`

### 3. Full MEAD preprocess (May 17, 2026)

Log ends with: `Successfully processed 4204 clips` → split into train/val JSON. Process is **not** currently running (`preprocess.pid` stale).

### 4. Production-path validation (May 21, 2026)

Two checks on the **full** annotations (`MEAD_processed/annotations/{train,validation}.json`, 4,028 / 176 clips) with **real** pretrained backbones — i.e. exactly the code path the production run will take, just with `max_train_steps: 3` and `train_batch_size: 1`.

**(a) Static dataset + model + forward pass** — `scripts/_validate_dataset_and_model.py`

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

**(b) Real `train_expression.py` run** with `configs/train_genphoto/expression_validation.yaml`:

- All checkpoints loaded cleanly (`Loading the image lora checkpoint … done`, `Loading the motion module checkpoint … done`).
- Trainable param accounting matches expectations: **182 params total = 142 encoder (199.26 M) + 40 attention `merge` (18.96 M) = 218.22 M**.
- Step-0 sanity GIF written to `sanity_check/A-portrait-photograph-of-a-woman,-frontal-view,-neutral-background..gif`.
- Validation pipeline at step 2 generated **100 paired sample/reference GIFs** in `samples/sample-1/` (run was killed before iterating through all 176 val clips — see Finding #2 below).

Outputs are under `output/expression_validation/expression_validation-2026-05-21T01-36-27/`.

---

## Validation findings & fixes applied

### Finding #1 — VRAM: production `train_batch_size: 4` would OOM on RTX 3090 — **FIXED**

| Measurement | Value |
|---|---|
| Peak VRAM, batch=1, 5 frames @ 256×384, fp16 autocast | **18.00 GiB** |
| Available RTX 3090 VRAM (per device) | 24.00 GiB |

**Why:** VAE, text encoder, full UNet3D, encoder, and attention-merge params all on one GPU; frozen modules stay fp32, only the forward region is autocast to fp16; activations kept for backward.

**Fix applied:** `train_expression.py` originally ignored `gradient_accumulation_steps` — `optimizer.step()` ran every micro-batch. The training loop has been patched to:

- divide the per-micro-batch loss by `gradient_accumulation_steps` before `.backward()`,
- only call `optimizer.step()` / `lr_scheduler.step()` / `optimizer.zero_grad()` and increment `global_step` every `gradient_accumulation_steps` micro-batches,
- preserve grad-clip + GradScaler semantics around the real optimizer step.

`configs/train_genphoto/expression.yaml` now uses `train_batch_size: 1, gradient_accumulation_steps: 4` (effective batch 4). Re-validation measured **peak 18.98 GiB at effective batch=4** ✓.

(Optional further savings if needed later: `vae.half()`, `text_encoder.half()`, or `unet.enable_gradient_checkpointing()` — not needed to start.)

### Finding #2 — Validation loop iterates over all 176 val clips — **FIXED**

`train_expression.py` had no cap on the validation dataloader (~8.5 s/clip × 176 clips ≈ 25 min/event; ≈ 83 h over a 100 k-step run with `validation_steps: 500`).

**Fix applied:** added an optional `max_validation_samples: int = None` parameter to `train_expression.py` (and the production config sets it to **8**). The val loop now logs `Reached max_validation_samples=N; skipping remaining M val clips.` and breaks. Verified during re-validation (output: `Reached max_validation_samples=2; skipping remaining 174 val clips.`).

Production config also bumps `validation_steps` from 500 → **5000** (20 events instead of 200) and sets `validation_steps_tuple: [200]` for one early regression-catch event.

### Finding #3 — Disk: `/` is 99 % full — **FIXED**

Checkpoints are ~2.6 GB each (100 ckpts × 2.6 GB ≈ 260 GB), so writing to `/` (13 GB free) would crash mid-run.

**Fix applied:** `output_dir: "/databases-4tb/levi-experiments/generative-photography/output/expression"` in `configs/train_genphoto/expression.yaml`. `/databases-4tb` has 1.7 TB free.

### Re-validation after fixes (May 21, 2026)

Ran the patched `train_expression.py` against `configs/train_genphoto/expression_validation.yaml` (production code path, batch=1 + grad-accum=4 + max_validation_samples=2 + 3 optimizer steps):

```
Iter: 1/3, Loss: 0.0032, GPU memory: 17.33 G
Iter: 2/3, Loss: 0.0279, GPU memory: 18.98 G
Saved state to .../checkpoints (global_step: 3)
Iter: 3/3, Loss: 0.0360, GPU memory: 18.98 G
Reached max_validation_samples=2; skipping remaining 174 val clips.
```

- Wall time: **~80 s end-to-end** (vs. >25 min in the pre-fix run that we had to kill).
- Effective batch = 4 via accumulation: each `Iter` aggregates 4 micro-batches (per-micro-batch loss shown is the post-division value; varies widely with the random diffusion timestep, which is normal).
- Checkpoint at step 3: **2.5 GB on `/databases-4tb`**.
- 2 val sample/reference pairs generated, cap fired.
- No leaked GPU memory after exit.

### Non-findings (verified safe)

- All four required pretrained artifacts resolved on disk (SD1.5 base, `unet_merged`, RealEstate10K LoRA, AnimateDiff v3 motion module).
- `ExpressionMEAD` loads 4,028 train clips in **0.2 s** at process start and **0.3 s** per sample (CLIP encode is the bottleneck per sample — fine at `num_workers ≥ 2`).
- 6-channel embedding has expected statistics (physical channels mean ≈ 0.84 from the high-AU MEAD ramp; CCL channel mean ≈ 2e-5, i.e. small but non-zero text-embedding differences).
- `encoder cin == 6 × downscale²` invariant holds (384 = 6 × 64).
- Forward pred shape `(1, 4, 5, 32, 48)` matches noise — the adaptor returns the right tensor for the MSE loss.
- Initial MSE loss ~1.37, which is the expected order of magnitude for an untrained adaptor predicting noise.
- After killing the validation run, GPU 2 returned to idle (no leaked memory).

---

## Gap analysis: path to real expression synthesis results

“Expression synthesis results” here means: a checkpoint trained long enough to generate coherent 5-frame smile ramps from a text prompt at inference time, measurable with AU accuracy / LPIPS / (optionally) ArcFace.

### Done — no further engineering required to *start* training

1. All entry points and dataset code exist and match the GenPhoto training pattern.
2. Training data is preprocessed and annotated at production scale (happy / front / py-feat AU12).
3. Smoke test proved the full graph on GPU.
4. Frozen backbones are downloaded.

### Remaining before meaningful results (ordered)

| Step | Effort | Notes |
|------|--------|-------|
| **1. Launch full training** | Days (GPU) | `expression.yaml` already wired up. Single 3090, effective batch 4, 100 k optimizer steps (~400 k micro-batches). At ~0.5 s/micro-batch this is roughly 2 days of compute. |
| **2. Monitor sanity / val GIFs** | Ongoing | Step-0 sanity GIF; capped val GIFs every 5 k optimizer steps + one at step 200. Early checkpoints will not look good — expect tens of thousands of steps before clear expression control. |
| **3. Inference** | Minutes | Set `expression_adaptor_ckpt` in `configs/inference_genphoto/expression.yaml`; run `inference_expression.py` with 5 intensities. |
| **4. Metrics** | Minutes–hours | `expression_au_accuracy.py`, existing LPIPS / CLIP scripts. ArcFace identity metric is **planned but not implemented**. |

### Launch command

```bash
conda activate genphoto
cd /databases-4tb/levi-experiments/generative-photography

# Pick a free GPU first (GPU 0 on this machine is often occupied by another job)
nvidia-smi --query-gpu=index,memory.free --format=csv

# Then launch on that index. `local_rank=0` inside the script maps to the only
# visible device, so the script always uses the GPU you list here.
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
  train_expression.py --config configs/train_genphoto/expression.yaml
```

**Why `CUDA_VISIBLE_DEVICES` is required:** without it, `torchrun` / `torch.distributed.launch` set `LOCAL_RANK=0`, and the script does `torch.cuda.set_device(local_rank)` → physical GPU 0. If anything else is on GPU 0 (common on shared machines) you get `CUDA OutOfMemoryError` during the `expression_adaptor.to(local_rank)` step. With `CUDA_VISIBLE_DEVICES=N`, only GPU N is visible and shows up as `cuda:0` inside the process.

(For a single GPU you can also skip `torchrun` entirely and set `MASTER_ADDR / MASTER_PORT / RANK / LOCAL_RANK / WORLD_SIZE` yourself plus `CUDA_VISIBLE_DEVICES`, as the re-validation runs did.)

### Known limitations (affect result quality, not startup)

- **MEAD happy clips** often have high AU12 even at lower labeled intensity levels; the dataset uses sorted-frame ramps and, for validation, decouples fixed `intensity_list` targets from which pixels are shown. The model must learn from real video dynamics, not perfect neutral→smile simulation like GenPhoto’s synthetic camera effects.
- **Option A embedding only** — scalar broadcast + CCL; no landmark flow or multi-AU control yet.
- **Single emotion filter** — default `happy` only; multi-emotion is a follow-up.
- **No identity regularizer** in the loss (ArcFace) — may matter for identity consistency across the 5 frames.
- **Camera adaptor checkpoints** from the original paper are **not** used; expression encoder trains from scratch.

---

## Suggested immediate next command

After editing `configs/train_genphoto/expression.yaml`:

```bash
conda activate genphoto
cd /databases-4tb/levi-experiments/generative-photography

python -m torch.distributed.launch --nproc_per_node=1 --use_env \
  train_expression.py --config configs/train_genphoto/expression.yaml
```

Example path substitutions (adjust if your layout differs):

```yaml
pretrained_model_path: "/home/levi/.cache/huggingface/hub/models--pandaphd--generative_photography/snapshots/92c29567186da6c7f8ada09eab8b5bfc7c998314/stable-diffusion-v1-5"
lora_ckpt: ".../weights/RealEstate10K_LoRA.ckpt"
motion_module_ckpt: ".../weights/v3_sd15_mm.ckpt"
train_data:
  root_path: "/databases-4tb/levi-experiments/generative-photography/MEAD_processed"
validation_data:
  root_path: "/databases-4tb/levi-experiments/generative-photography/MEAD_processed"
```

Optional shorter dry run before committing to 100k steps: temporarily set `max_train_steps: 500` and `checkpointing_steps: 100` in the same config.

---

## Readiness checklist

- [x] Expression dataset + 6-channel embeddings
- [x] Training script (distributed, checkpointing, validation sampling)
- [x] Real gradient accumulation in `train_expression.py`
- [x] Bounded validation loop (`max_validation_samples`) in `train_expression.py`
- [x] Inference script + configs
- [x] MEAD preprocess script
- [x] Full MEAD_processed annotations on disk
- [x] Pretrained GenPhoto backbones on disk
- [x] Smoke test: loss + checkpoint + val GIFs
- [x] Production-path validation on full annotations (`expression_validation.yaml`)
- [x] VRAM / disk / val-loop blocker analysis (Findings #1–#3)
- [x] Production `expression.yaml` paths + batch/val fixes applied & verified
- [ ] Full training run started / completed
- [ ] Inference on a converged checkpoint
- [ ] Quantitative eval (AU correlation, LPIPS, CLIP)
- [ ] (Optional) ArcFace identity metric + training regularizer

---

## File map (quick reference)

```
train_expression.py              # training entry
inference_expression.py          # inference entry
genphoto/data/expression_dataset.py
scripts/preprocess_mead.py
scripts/_validate_dataset_and_model.py            # static validator (May 21)
configs/train_genphoto/expression.yaml            # full train (placeholders)
configs/train_genphoto/expression_smoke.yaml      # verified 5-step smoke
configs/train_genphoto/expression_validation.yaml # verified production-path validation
configs/inference_genphoto/expression.yaml
comp_metrics/expression_au_accuracy.py
MEAD_processed/                                   # local data (gitignored)
output/expression_validation/                     # last validation run artifacts
docs/expression-adaptation-plan.md
```

---

*Last updated from repository state: May 21, 2026 (production-path validation).*

---

## Training completed & evaluated (May–June 2026)

### Run history

| Run directory | Steps | Note |
|---------------|-------|------|
| `expression-2026-05-21T21-40-07` | 50 | Initial test, stopped manually. |
| `expression-2026-05-21T22-07-08` | 23 250 | First real run, stopped manually. |
| `expression-2026-05-26T21-43-56` | 100 000 | **Resumed from step 23 000** and ran to completion. |

**Final checkpoint:**
`output/expression/expression-2026-05-26T21-43-56/checkpoints/checkpoint-step-100000.ckpt`

**Training stability:**
- Loss remained very low and stationary for most of the run (frequently < 0.01) with occasional spikes up to ~0.2. No divergence or OOMs.
- GPU memory stable at ~21.4 GiB (RTX 3090, 24 GiB).
- Validation GIFs emitted every 5 000 steps (capped at 8 clips).

### Inference + quantitative evaluation

Using `configs/inference_genphoto/expression.yaml` (updated with the 100 k checkpoint path and real local paths for backbones), three text prompts were evaluated with target intensities `[0.0, 0.25, 0.5, 0.75, 1.0]`.

| Prompt | Detected AU12 trajectory | Pearson *r* |
|--------|--------------------------|-------------|
| Young woman | [0.098, 0.072, 0.157, 0.771, 0.949] | **0.9099** |
| Elderly woman | [0.453, 0.756, 0.725, 0.820, 0.865] | **0.8728** |
| Young man | [0.890, 0.828, 0.929, 0.960, 0.952] | **0.7475** |
| **Mean** | | **0.8434** |

**Interpretation:**
- **Young woman:** Excellent monotonic control; strong dynamic range from near-neutral to broad smile.
- **Elderly woman:** Good control with a minor non-monotonic dip at frame 3.
- **Young man:** High baseline smile even at intensity 0.0 (AU12 ≈ 0.89). The model compresses dynamic range for this identity, lowering correlation. This is a **side-effect / bias** likely stemming from the MEAD training distribution.

### Fixes applied during evaluation

1. **`comp_metrics/expression_au_accuracy.py`** — The installed `py-feat` version rejected numpy arrays passed to `detect_image()`. The script now writes temporary PNG files per frame before detection.
2. **Detector constructor compatibility** — Updated `emotion_model="resmasknet"` and `facepose_model="img2pose"` to match `py-feat`’s allowed model names.
3. **`configs/inference_genphoto/expression.yaml`** — All `/path/to/...` placeholders replaced with real local paths (HF cache + final checkpoint).

### Readiness checklist (updated)

- [x] Expression dataset + 6-channel embeddings
- [x] Training script (distributed, checkpointing, validation sampling)
- [x] Real gradient accumulation in `train_expression.py`
- [x] Bounded validation loop (`max_validation_samples`) in `train_expression.py`
- [x] Inference script + configs
- [x] MEAD preprocess script
- [x] Full MEAD_processed annotations on disk
- [x] Pretrained GenPhoto backbones on disk
- [x] Smoke test: loss + checkpoint + val GIFs
- [x] Production-path validation on full annotations (`expression_validation.yaml`)
- [x] VRAM / disk / val-loop blocker analysis (Findings #1–#3)
- [x] Production `expression.yaml` paths + batch/val fixes applied & verified
- [x] **Full training run completed (100 k steps)**
- [x] **Inference on converged checkpoint**
- [x] **Quantitative eval (AU correlation)**
- [ ] LPIPS / CLIP metrics on inference outputs
- [ ] (Optional) ArcFace identity metric + training regularizer

### Suggested next steps

1. **Run LPIPS + CLIP metrics** on the generated GIFs in `inference_output/expression_eval/` to measure temporal consistency and prompt fidelity.
2. **Address male baseline bias** if cross-gender robustness is required — consider identity-conditioned training or ArcFace regularisation.
3. **Checkpoint housekeeping** — The final run produced ~77 checkpoints (~2.6 GB each, ~200 GB total). Safe to delete all but `checkpoint-step-100000.ckpt` and one mid-run fallback (e.g. `checkpoint-step-50000.ckpt`).