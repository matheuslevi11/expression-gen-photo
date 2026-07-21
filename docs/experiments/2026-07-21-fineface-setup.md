# FineFace setup for the head-to-head comparison (July 21, 2026)

Setup note (getting the closest prior art running; see [2026-07-11 study](2026-07-11-related-work-study.md)). FineFace source is vendored at `forks/fineface/` (upstream `github.com/tvaranka/fineface`, at merge commit `6e9cc8d`). It now runs; this records the environment, the one non-obvious blocker and its fix, and the working invocation.

## What FineFace is (as installed)

Frozen SD2.1-base UNet + custom AU cross-attention processors (`AUAttnProcessor` adds `au_to_k`/`au_to_v` per cross-attn layer) + a linear `AUEncoder` + a LoRA. Three small checkpoints pulled from the `Tvaranka/fineface` HF repo (`attn_processors.ckpt`, `pytorch_lora_weights.safetensors`, `au_encoder.ckpt`). Input: text prompt + AU dict (12 AUs, intensities on a **0–5** scale); output: a single 512×512 still. No temporal/sequence notion — an intensity sweep is N independent generations.

## Environment (isolated; not our `genphoto` env)

FineFace pins torch 2.2 / diffusers 0.27 / transformers 4.41 / peft 0.7 — incompatible with `genphoto`'s fragile torch-2.1/diffusers-0.24 pins (CLAUDE.md warns not to disturb them). So a separate conda env was created, kept off the shared `/` disk:

```
env:       /databases-4tb/levi-experiments/envs/fineface   (python 3.11, 5.6 GB)
HF cache:  /databases-4tb/levi-experiments/.hf-cache        (5.0 GB)
```

Installed exactly `forks/fineface/requirements.txt` (torch 2.2.0+cu121, diffusers 0.27.2, transformers 4.41.0, huggingface_hub 0.23.0, peft 0.7.0, numpy 1.26.3). CUDA verified on RTX 3090.

## Blocker: SD2.1-base was removed from Hugging Face

`FineFacePipeline` hardcodes `stabilityai/stable-diffusion-2-1-base` as the frozen backbone. **Stability AI pulled all SD2.x repos from HF (2025)** — the canonical id 404s even with a valid token (anonymous → 401; authenticated → 404; sibling ids `stable-diffusion-2-1`, `stable-diffusion-2-base` also 404). FineFace's own `Tvaranka/fineface` weights are still reachable (302→CDN).

**Fix — faithful mirror.** Four independent community mirrors all carry a **bit-identical** UNet (`unet/diffusion_pytorch_model.safetensors` sha256 `6dfae3e5f7d459b5…`, and `.bin` `a434f8fdbb51f080…`) in full diffusers layout (unet/vae/text_encoder/tokenizer/scheduler/feature_extractor). Identical hashes across unrelated re-uploads = the true original weights, so FineFace's trained attn/LoRA (which were trained on exactly SD2.1-base) apply correctly. Using `Manojb/stable-diffusion-2-1-base` (most-downloaded, ~82k).

The fork was edited (fork commit `8c51a5c`) so the backbone id reads from `FINEFACE_SD_REPO` (default unchanged), rather than hardcoding the mirror — keeps the diff minimal and documents the reason inline.

## Working invocation

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /databases-4tb/levi-experiments/envs/fineface
export HF_HOME=/databases-4tb/levi-experiments/.hf-cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token)          # mirror is public; token is for HF egress on this host
export PYTHONPATH=/databases-4tb/levi-experiments/generative-photography/forks/fineface
export FINEFACE_SD_REPO=Manojb/stable-diffusion-2-1-base
cd /databases-4tb/levi-experiments/generative-photography/forks/fineface
CUDA_VISIBLE_DEVICES=<free gpu> python <script>              # `from fineface import FineFacePipeline`
```

(`PYTHONPATH` is required: running a script from outside the repo root otherwise can't import the `fineface` package.)

## Verification result

AU12 sweep {0, 1.25, 2.5, 3.75, 5.0} at fixed seed on prompt "A portrait photograph of a young woman, frontal view, neutral background." (`scratchpad/fineface_test.py`): clean **monotonic smile ramp** neutral → broad open-mouth smile. As predicted for the head-to-head, **identity visibly drifts across the sweep** (face shape, hair, eye colour shift between the independent stills) — the failure mode our temporal-attention ramps avoid by construction, and exactly what the planned cross-frame identity metric will quantify.

## Next (the actual head-to-head, Tier 1 checklist item)

1. Generate FineFace AU12 sweeps over our `configs/eval_prompts.txt`, mapping our [0,.25,.5,.75,1.0] → [0,1.25,2.5,3.75,5.0]; write a manifest with `"frames": [png,…]` per sweep.
2. Score with our `scripts/batch_eval.py score` (manifest entries use `"frames": [png,…]` instead of `"gif"`, already supported by `load_frames`) — py-feat AU12 + MediaPipe + identity, the same pipeline used for our model, so numbers are directly comparable.
3. Compare: ramp-following Pearson *r*, **constant-intensity CLS** (FineFace may calibrate absolutely better — it trained on graded stills), and **cross-frame identity** (expected decisive win for us).
