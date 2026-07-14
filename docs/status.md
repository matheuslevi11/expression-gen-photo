# Expression adaptation — project status

Living document: current state of the Generative Photography → **Generative Expressions** fork. Edited destructively — checked items get checked, wrong claims get deleted. History and full experiment write-ups live in [`experiments/`](experiments/) (append-only, date-stamped). Design rationale: [`plan.md`](plan.md). Usage commands: root [`README.md`](../README.md). Doc conventions: root [`CLAUDE.md`](../CLAUDE.md).

*Last updated: July 9, 2026.*

---

## Executive summary

The fork replaces GenPhoto's camera axis with **scalar facial-expression (smile / AU12) intensity**, keeping the 3D UNet, AnimateDiff motion module, 6-channel conditioning layout (`cin: 384`), and attention-injection mechanism unchanged. Training data is MEAD front-view happy clips (4,028 train / 176 val), preprocessed with MediaPipe crops + py-feat AU12 labels.

**Current state:** trained to 100 k steps on one RTX 3090; final checkpoint at
`output/expression/expression-2026-05-26T21-43-56/checkpoints/checkpoint-step-100000.ckpt`.
Batch-evaluated at scale (2026-07-14): ascending-ramp control **r = 0.77 ± 0.21** (40 prompts × 3 seeds), confirmed by a label-independent MediaPipe detector (**0.83**); frame-to-frame identity cosine **0.92** (baseline 0.85, and undetectable on 57% of baseline samples); descending reversal −0.98. **No absolute intensity calibration** (constant-list CLS = 0.07) — control is *relative/ordinal*, a consequence of ramp-only MEAD training; permuted lists partial (r = 0.38). Claim accordingly: identity-consistent, temporally coherent expression ramps with measured (and honestly reported) calibration limits.

| Area | Original GenPhoto | This fork |
|------|-------------------|-----------|
| Conditioning | bokeh / focal / shutter / color temp | scalar smile / AU12 intensity |
| Dataset | `genphoto/data/dataset.py` + BokehMe simulation | `genphoto/data/expression_dataset.py` (`ExpressionMEAD`) |
| Physical channels | blur kernel, crop mask, etc. | scalar broadcast (`create_intensity_embedding`) |
| CCL text | `<bokeh kernel size: …>` | `<smile intensity: …>` |
| Train / inference | `train_*.py` / `inference_*.py` (×4) | `train_expression.py` / `inference_expression.py` |
| Eval (accuracy) | Laplacian / FOV correlation | `comp_metrics/expression_au_accuracy.py` (AU12 Pearson *r*) |
| Removed | — | BokehMe, `depth_any`, Gradio `app.py`, camera YAMLs/scripts |

**Unchanged (reused as-is):** `genphoto/models/*`, `genphoto/pipelines/pipeline_animation.py`, `GenPhotoPipeline`, `CameraAdaptor` / `CameraCameraEncoder` class names (semantically the expression encoder), LPIPS and CLIP metric scripts.

---

## Experiment log

| Date | Write-up | One-line result |
|------|----------|-----------------|
| 2026-05-16 | [smoke test](experiments/2026-05-16-smoke-test.md) | 5-step training graph verified end-to-end on GPU. |
| 2026-05-17 | [MEAD preprocess](experiments/2026-05-17-mead-preprocess.md) | 4,028 train / 176 val clips, py-feat AU12 annotations. |
| 2026-05-21 | [production validation](experiments/2026-05-21-production-validation.md) | Production code path verified; VRAM / val-loop / disk blockers found and fixed. |
| 2026-05-26 | [training run 100k](experiments/2026-05-26-training-run-100k.md) | Stable full run; final checkpoint at step 100 000. |
| 2026-06 | [AU accuracy eval](experiments/2026-06-eval-au-accuracy.md) | Mean Pearson *r* = 0.84 over 3 prompts; male baseline-smile bias noted. |
| 2026-07-08 | [trained vs untrained](experiments/2026-07-08-trained-vs-untrained.md) | Reversal test −0.98 proves scalar conditioning. (Original "unfair baseline" diagnosis superseded.) |
| 2026-07-09 | [fair baseline](experiments/2026-07-09-fair-baseline.md) | Bypass ≡ zero-merge bit-exactly; the ablation baseline was already fair. |
| 2026-07-09 | [StyleGAN data note](experiments/2026-07-09-stylegan-data-note.md) | Design note (no run): StyleGAN as identity-paired ramp generator; calibration must be measured, not prescribed. |
| 2026-07-11 | [related-work study](experiments/2026-07-11-related-work-study.md) | Study note (no run): EmojiDiff / MagicFace / PixelSmile are single-image editing/transfer; FineFace (Jul 2024) **is prior art** for AU-intensity T2I generation — claim revised to identity-consistent temporal ramps + measured calibration; head-to-head baseline now mandatory. |
| 2026-07-14 | [Tier 1+2 batch eval](experiments/2026-07-14-tier12-batch-eval.md) | 540+120 samples: ramp control robust (r=0.77±0.21, n=120; MediaPipe confirms 0.83), identity 0.92 vs baseline; **no absolute calibration** (constant-list CLS=0.07, flat ~0.85) — control is relative/contextual; permuted lists partial (0.38). |

---

## Readiness checklist

- [x] Expression dataset + 6-channel embeddings
- [x] Training script (distributed, checkpointing, validation sampling; real grad accumulation; bounded val loop)
- [x] Inference script + configs
- [x] MEAD preprocess script + full `MEAD_processed` annotations on disk
- [x] Pretrained GenPhoto backbones on disk
- [x] Smoke test + production-path validation
- [x] Full training run completed (100 k steps)
- [x] Inference on converged checkpoint + quantitative eval (AU correlation)
- [x] Trained vs. untrained / ascending vs. descending ablation
- [x] Fair SD1.5 baseline (bypass verified bit-identical to zero-merge)
- [ ] LPIPS / CLIP metrics on inference outputs
- [x] Identity metric (facenet-VGGFace2 frame-to-frame cosine, 2026-07-14; add insightface/ArcFace for multi-model averaging before publication)
- [ ] (Optional) ArcFace identity *training regularizer*

## Evaluation Robustness checklist

What separates the current "does it work" evidence from a defensible paper. Re-prioritized 2026-07-11 after the FineFace finding: the revised claim is *identity-consistent, temporally coherent intensity ramps with measured dose–response calibration*, so identity + calibration measurements are now claim-critical, not optional. None require retraining.

### Tier 1 — claim-critical (the revised claim rests on these)

- [x] **Identity consistency** (2026-07-14) — facenet-VGGFace2 frame-to-frame cosine: trained **0.92 ± 0.05** vs baseline 0.85 (baseline faces undetectable on 57% of samples). Pre-publication: add insightface/ArcFace for multi-model averaging.
- [x] **Dose–response calibration curve** (2026-07-14) — **NEGATIVE: no absolute calibration.** Constant-list CLS = 0.07; detected AU12 flat at ~0.85 for every commanded level ≥ 0.1. Control is *relative/contextual* (same commanded 0.0 → 0.54 inside a ramp, 0.79 in a constant clip). Root cause: ramp-only MEAD training. Report as measured limitation; see fix directions in the experiment note.
- [ ] **FineFace head-to-head** — run public FineFace (`github.com/tvaranka/fineface`) AU12 sweeps as independent stills vs our ramps, same prompts: (1) ramp-following Pearson *r*, (2) cross-frame identity consistency, (3) **constant-intensity CLS** — static-trained FineFace may calibrate better absolutely while losing identity/trajectory coherence; measure both directions honestly.

### Tier 2 — minimum for a complete ablation

- [x] **Scaled evaluation** (2026-07-14) — 40 prompts × 3 seeds: r = **0.774 ± 0.212** (median 0.86, 87% > 0.5) vs baseline −0.05 ± 0.64. Gender gap quantified: female 0.83 vs male 0.71.
- [x] **Non-monotonic / permuted intensity lists** (2026-07-14) — r = **0.385 ± 0.505**: real per-frame conditioning, but much weaker than monotonic ramps — the temporal prior resists non-monotonic trajectories (no training support).
- [x] **Independent AU measurement** (2026-07-14) — MediaPipe mouth-corner proxy confirms ramp control at r = **0.829 ± 0.230** (n=120), breaking the py-feat label/eval circularity that all four studied papers share unflagged.

### Tier 3 — supporting evidence

- [ ] **Training-step dose–response** — evaluate *r* at intermediate checkpoints (e.g. 1 k / 5 k / 10 k / 25 k / 50 k / 100 k). Independent evidence that training drives the capability. ⚠️ **Do this before checkpoint housekeeping** — the ~100 intermediate checkpoints (~245 GB) slated for deletion are the raw material.
- [ ] **Prompt-engineering baseline figure** — 5 independent SD1.5 generations with graded smile prompts. Now doubly motivated: PixelSmile's zero-shot row (CLS-6 0.69 untrained) proves "just prompt it" is a real objection; expected result is coarse control with collapsed identity.
- [ ] **Out-of-range intensity probe** — intensities < 0 and > 1 (FineFace's Fig. 8 extrapolation demo); cheap disentanglement evidence.

---

## Known limitations (affect result quality, not correctness)

- **MEAD happy clips** often have high AU12 even at lower labeled intensity levels; the dataset uses sorted-frame ramps and, for validation, decouples fixed `intensity_list` targets from which pixels are shown.
- **No absolute intensity calibration** — commanded values are rendered *relative to the clip's trajectory*, not as absolute AU12 targets (constant-list CLS = 0.07, flat ~0.85 plateau). Consequence of ramp-only MEAD training; fix candidates: constant/permuted-list training augmentation, AU dropout + expression-CFG, label distribution smoothing (see 2026-07-14 note).
- **Male baseline-smile bias** — quantified 2026-07-14: ramp r = 0.71 (male prompts) vs 0.83 (female), Δ ≈ 0.11; a MEAD distribution artifact. Consider identity-conditioned training or ArcFace regularisation if cross-gender robustness is required.
- **Option A embedding only** — scalar broadcast + CCL; no landmark flow or multi-AU control yet.
- **Single emotion filter** — default `happy` only; multi-emotion is a follow-up.
- **No identity regularizer** in the loss (ArcFace).
- **Camera adaptor checkpoints** from the original paper are not used; the expression encoder trains from scratch.

---

## Commands

### Training (single GPU)

```bash
conda activate genphoto
cd /databases-4tb/levi-experiments/generative-photography

# Pick a free GPU first (GPU 0 is often occupied on this machine)
nvidia-smi --query-gpu=index,memory.free --format=csv

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
  train_expression.py --config configs/train_genphoto/expression.yaml
```

`CUDA_VISIBLE_DEVICES` is required: `torchrun` sets `LOCAL_RANK=0` and the script does `torch.cuda.set_device(local_rank)` → physical GPU 0 unless remapped.

### Inference + metrics

```bash
python inference_expression.py \
  --config configs/inference_genphoto/expression.yaml \
  --base_scene "A portrait photograph of a young woman, frontal view, neutral background." \
  --intensity_list "[0.0, 0.25, 0.5, 0.75, 1.0]" --seed 42

# Fair frozen-backbone baseline (no expression conditioning):
#   use configs/inference_genphoto/expression_baseline.yaml with the same --seed

python comp_metrics/expression_au_accuracy.py \
  --gifs-dir inference_output/expression \
  --intensity-list "[0.0, 0.25, 0.5, 0.75, 1.0]"
```

### Batch evaluation (Evaluation Robustness experiments)

```bash
# generate → score → summarize; each stage is resumable. Experiments:
# scaled | calibration | permuted | extrapolation. Repeat with the
# baseline config for the trained-vs-baseline comparison.
CUDA_VISIBLE_DEVICES=1 python scripts/batch_eval.py generate \
  --config <real-path inference yaml> --experiment calibration \
  --seeds 42,43,44 --out-dir inference_output/batch_eval/trained
CUDA_VISIBLE_DEVICES=1 python scripts/batch_eval.py score \
  --manifest inference_output/batch_eval/trained/manifest.jsonl
python scripts/batch_eval.py summarize \
  --results inference_output/batch_eval/trained/results.jsonl
```

---

## File map

```
train_expression.py              # training entry
inference_expression.py          # inference entry (--seed for reproducible ablations)
genphoto/data/expression_dataset.py
scripts/preprocess_mead.py
scripts/_validate_dataset_and_model.py            # static validator
scripts/export_results.sh                         # stage results tarball for scp/rsync pull
scripts/batch_eval.py                             # generate/score/summarize eval sweeps
configs/eval_prompts.txt                          # 40-prompt bank for batch eval
configs/train_genphoto/expression.yaml            # full train (placeholders)
configs/train_genphoto/expression_smoke.yaml      # verified 5-step smoke
configs/train_genphoto/expression_validation.yaml # verified production-path validation
configs/inference_genphoto/expression.yaml        # trained-adaptor inference
configs/inference_genphoto/expression_baseline.yaml # fair frozen-backbone baseline
comp_metrics/expression_au_accuracy.py
MEAD_processed/                                   # local data (gitignored)
output/expression/                                # training runs + checkpoints
inference_output/                                 # generated GIFs + ablation artifacts
docs/plan.md                                      # design rationale (frozen)
docs/experiments/                                 # append-only lab notebook
```
