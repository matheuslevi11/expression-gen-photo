# Expression adaptation — project status

Living document: current state of the Generative Photography → **Generative Expressions** fork. Edited destructively — checked items get checked, wrong claims get deleted. History and full experiment write-ups live in [`experiments/`](experiments/) (append-only, date-stamped). Design rationale: [`plan.md`](plan.md). Usage commands: root [`README.md`](../README.md). Doc conventions: root [`CLAUDE.md`](../CLAUDE.md).

*Last updated: July 9, 2026.*

---

## Executive summary

The fork replaces GenPhoto's camera axis with **scalar facial-expression (smile / AU12) intensity**, keeping the 3D UNet, AnimateDiff motion module, 6-channel conditioning layout (`cin: 384`), and attention-injection mechanism unchanged. Training data is MEAD front-view happy clips (4,028 train / 176 val), preprocessed with MediaPipe crops + py-feat AU12 labels.

**Current state:** trained to 100 k steps on one RTX 3090; final checkpoint at
`output/expression/expression-2026-05-26T21-43-56/checkpoints/checkpoint-step-100000.ckpt`.
Mean AU12-vs-target Pearson *r* = **0.84** over 3 prompts; ascending/descending reversal (+0.91 / −0.98) proves true scalar conditioning; a verified-fair frozen-backbone baseline shows zero expression response without training. Remaining work is evaluation robustness (below), not engineering.

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
- [ ] ArcFace identity metric (claim-critical since the [related-work study](experiments/2026-07-11-related-work-study.md) — see Evaluation Robustness)
- [ ] (Optional) ArcFace identity *training regularizer*

## Evaluation Robustness checklist

What separates the current "does it work" evidence from a defensible paper. Re-prioritized 2026-07-11 after the FineFace finding: the revised claim is *identity-consistent, temporally coherent intensity ramps with measured dose–response calibration*, so identity + calibration measurements are now claim-critical, not optional. None require retraining.

### Tier 1 — claim-critical (the revised claim rests on these)

- [ ] **ArcFace identity consistency** — frame-to-frame cosine similarity, trained vs baseline (EmojiDiff's Antelopev2-cosine implementation; optionally multi-model averaging à la PixelSmile). Verifies the "same face, different expression" half of the claim — the half FineFace cannot make (no identity metric, independent stills).
- [ ] **Dose–response calibration curve** — constant lists `[c,c,c,c,c]` for c ∈ {0.0, 0.1, …, 1.0}; plot detected AU12 vs c. The other half of the revised claim — FineFace shows sweeps only qualitatively and admits a nonlinear scale. Follow PixelSmile's CLS protocol (uniform commanded intensities → Pearson) for comparability, plus MagicFace-style AU MSE for absolute error.
- [ ] **FineFace head-to-head** — run public FineFace (`github.com/tvaranka/fineface`) AU12 sweeps as independent stills vs our ramps, same prompts: (1) AU12 dose–response Pearson *r* (must be measured — they may be competitive), (2) cross-frame ArcFace identity consistency (isolates our core contribution). Depends on the two items above being implemented.

### Tier 2 — minimum for a complete ablation

- [ ] **Scaled evaluation** — ~30–50 prompts × 3 seeds with the ascending intensity list; report mean ± std of Pearson *r* and the full distribution vs the frozen-backbone baseline. (Current evidence: 3 prompts × 1 seed — an anecdote, statistically. One model load, ~15 s/sample ≈ 1 h GPU.)
- [ ] **Non-monotonic / permuted intensity lists** — e.g. `[0.0, 1.0, 0.5, 0.25, 0.75]`; rules out "the model learned smooth ramps plus a direction bit" rather than per-frame scalar conditioning.
- [ ] **Independent AU measurement** — cross-check a subset with a detector that did **not** produce the training labels: **LibreFace** (used by MagicFace for exactly this signal) or **MediaPipe blendshapes** `mouthSmileLeft/Right` (already a dependency; EmojiDiff's Exp metric). Breaks the py-feat label/eval circularity — which all four studied papers share unflagged — and pre-empts EmojiDiff's detector-dependency critique.

### Tier 3 — supporting evidence

- [ ] **Training-step dose–response** — evaluate *r* at intermediate checkpoints (e.g. 1 k / 5 k / 10 k / 25 k / 50 k / 100 k). Independent evidence that training drives the capability. ⚠️ **Do this before checkpoint housekeeping** — the ~100 intermediate checkpoints (~245 GB) slated for deletion are the raw material.
- [ ] **Prompt-engineering baseline figure** — 5 independent SD1.5 generations with graded smile prompts. Now doubly motivated: PixelSmile's zero-shot row (CLS-6 0.69 untrained) proves "just prompt it" is a real objection; expected result is coarse control with collapsed identity.
- [ ] **Out-of-range intensity probe** — intensities < 0 and > 1 (FineFace's Fig. 8 extrapolation demo); cheap disentanglement evidence.

---

## Known limitations (affect result quality, not correctness)

- **MEAD happy clips** often have high AU12 even at lower labeled intensity levels; the dataset uses sorted-frame ramps and, for validation, decouples fixed `intensity_list` targets from which pixels are shown.
- **Male baseline-smile bias** — high AU12 at intensity 0.0 for the "young man" prompt; likely a MEAD distribution artifact. Consider identity-conditioned training or ArcFace regularisation if cross-gender robustness is required.
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

---

## File map

```
train_expression.py              # training entry
inference_expression.py          # inference entry (--seed for reproducible ablations)
genphoto/data/expression_dataset.py
scripts/preprocess_mead.py
scripts/_validate_dataset_and_model.py            # static validator
scripts/export_results.sh                         # stage results tarball for scp/rsync pull
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
