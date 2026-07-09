# Generative Photography → Generative Expressions (MEAD adaptation)

This is a **fork-style adaptation** of [Generative Photography (CVPR 2025)](https://arxiv.org/abs/2412.02168) that swaps the camera-parameter axis (bokeh / focal length / shutter speed / colour temperature) for a **facial expression intensity** axis. The architecture, attention injection mechanism, and motion-module pipeline are preserved; only the conditioning signal, dataset, and training/inference entry points are replaced.

The original paper, model weights, and Hugging Face resources are still relevant: the SD1.5 backbone, the merged UNet, the RealEstate10K image LoRA, and the AnimateDiff v3 motion module are all reused **as-is** and frozen during training.

The high-level design rationale is in [`docs/plan.md`](docs/plan.md). Current project state, checklists, and the experiment log are in [`docs/status.md`](docs/status.md); full experiment write-ups live in [`docs/experiments/`](docs/experiments/).

## What changed vs. the original repo

| Concern | Original | This fork |
|---|---|---|
| Conditioning axis | bokeh / focal / shutter / color temp | facial expression intensity (AU12 / smile by default) |
| Dataset | `genphoto/data/dataset.py` (camera classes) | `genphoto/data/expression_dataset.py` (`ExpressionMEAD`) |
| Physical embedding | bokeh kernel / crop mask / sensor sim / WB curve | scalar intensity broadcast (`create_intensity_embedding`) |
| CCL embedding | text diffs of `<bokeh kernel size: ...>` etc. | text diffs of `<smile intensity: ...>` |
| Training entry | 4× `train_*.py` | `train_expression.py` |
| Inference entry | 4× `inference_*.py` | `inference_expression.py` |
| Preprocessing | offline simulation from base images | `scripts/preprocess_mead.py` (raw .mp4 → cropped frames + AU12) |
| Eval (accuracy) | Laplacian / FOV correlation | `comp_metrics/expression_au_accuracy.py` (AU12 vs target Pearson r) |
| Removed | BokehMe, depth_any, Gradio app, camera configs | — |

The 6-channel layout (3 physical + 3 CCL) is preserved, so `cin: 384` in the encoder config is unchanged.

## 1. Environment

```bash
conda env create -f environment.yaml
conda activate genphoto
```

`mediapipe` and `py-feat` are required for preprocessing; `lpips` is required for the consistency metric.

## 2. Pretrained backbones to download

You still need the same upstream artifacts as the original paper:

- **Stable Diffusion v1.5** with the GenPhoto-style `unet_merged` subfolder (from [pandaphd/generative_photography](https://huggingface.co/pandaphd/generative_photography)).
- **RealEstate10K image LoRA** (`RealEstate10K_LoRA.ckpt`).
- **AnimateDiff v3 motion module** (`v3_sd15_mm.ckpt`).

You do **not** need the released camera adaptor checkpoints — the expression encoder + attention `merge` parameters are trained from scratch.

## 3. MEAD preprocessing

Download MEAD from the [official source](https://wywu.github.io/projects/MEAD/MEAD.html) and point the script at the root.

```bash
python scripts/preprocess_mead.py \
    --mead-root /data/MEAD \
    --out-root  /data/MEAD_processed \
    --emotion happy \
    --view front \
    --frames-per-clip 24 \
    --au-method pyfeat \
    --val-actors W009 W011    # held-out actors for validation
```

Output layout:

```
/data/MEAD_processed/
    frames/<actor>/<emotion>/level_<n>/<clip>/frame_*.jpg
    annotations/train.json
    annotations/validation.json
```

If you cannot install `py-feat`, use `--au-method timeline` for a triangular-ramp proxy that lets you smoke-test the pipeline before installing the real AU detector.

## 4. Training

Edit `configs/train_genphoto/expression.yaml` to point at your preprocessed dataset and the pretrained backbones, then:

```bash
# Always set CUDA_VISIBLE_DEVICES to a free GPU index. With a single visible
# device, local_rank=0 maps to it; otherwise the script will try cuda:0 and
# OOM if GPU 0 is already in use.
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
    train_expression.py --config configs/train_genphoto/expression.yaml
```

Pick the GPU index by running `nvidia-smi --query-gpu=index,memory.free --format=csv` first. `torch.distributed.launch` also works but is deprecated in favour of `torchrun`.

Trainable parameters: the expression encoder (the `CameraCameraEncoder` module — name preserved for reuse) + the attention `merge` weights. Image LoRA, motion module, VAE, text encoder, and the SD1.5 UNet are frozen.

Sanity-check GIFs of training batches land in `output/expression/<run>/sanity_check/`.

## 5. Inference

After training, set `expression_adaptor_ckpt` in `configs/inference_genphoto/expression.yaml` to a saved checkpoint, then:

```bash
python inference_expression.py \
    --config configs/inference_genphoto/expression.yaml \
    --base_scene "A portrait photograph of a young woman, frontal view, neutral background." \
    --intensity_list "[0.0, 0.25, 0.5, 0.75, 1.0]"
```

The list must have **5 values** (matches the released backbone configuration). Values are smile / AU12 intensity in roughly `[0, 1]`.

## 6. Evaluation

- **Expression accuracy** — Pearson correlation between detected AU12 trajectory and target intensities:

  ```bash
  python comp_metrics/expression_au_accuracy.py \
      --gifs-dir inference_output/expression/ \
      --intensity-list "[0.0, 0.25, 0.5, 0.75, 1.0]"
  ```

- **Frame-to-frame consistency** — reuse the original LPIPS metric (`comp_metrics/consistency_by_LPIPS/comp_LPIPS.py`).
- **Identity preservation** — recommended add-on: **ArcFace** cosine similarity between the first frame and every other frame in a sequence (not yet implemented; see the plan doc).
- **Prompt following** — reuse the original CLIP score (`comp_metrics/quality_prompt_following_by_CLIP/clip.py`).

## 7. Open follow-ups

These were intentionally left out of the first pass (see [`docs/plan.md`](docs/plan.md) for the rationale):

- Identity regularizer (ArcFace cosine) added to the training loss.
- Multi-emotion model (current default filters to `happy` clips).
- Landmark-displacement (Option C) physical channel as a richer alternative to the scalar broadcast.
- Gradio demo (the original `app.py` was deleted with the camera scripts; rebuild after a checkpoint exists).

## Citation

If you build on this fork, please cite the original paper:

```bibtex
@article{Yuan_2024_GenPhoto,
  title={Generative Photography: Scene-Consistent Camera Control for Realistic Text-to-Image Synthesis},
  author={Yuan, Yu and Wang, Xijun and Sheng, Yichen and Chennuri, Prateek and Zhang, Xingguang and Chan, Stanley},
  journal={arXiv preprint arXiv:2412.02168},
  year={2024}
}
```
