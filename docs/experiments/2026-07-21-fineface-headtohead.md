# FineFace head-to-head: setup + comparison (July 21, 2026)

The Tier-1 external comparison against FineFace (arXiv 2407.20175), the closest prior art (AU-intensity-conditioned text-to-image generation; see [2026-07-11 study](2026-07-11-related-work-study.md)). Covers getting it running and the full quantitative result.

**Bottom line (honest):** on matched prompts/seeds, FineFace **ties us on ramp-following**, **decisively beats us on absolute intensity calibration** (CLS 0.50 vs our 0.07), and we **modestly but consistently beat it on cross-frame identity** (worst-case adjacent cosine +0.12, winning 68% of prompts). Our defensible contribution narrows to the **sequence-native temporal formulation** (one identity-consistent clip vs N independent stills), not superior expression control.

## Setup

Source vendored at `forks/fineface/` (upstream `github.com/tvaranka/fineface` @ `6e9cc8d`). Frozen SD2.1-base UNet + custom AU cross-attention (`au_to_k`/`au_to_v` per cross-attn layer) + linear `AUEncoder` + LoRA; three checkpoints from `Tvaranka/fineface`. Input: prompt + AU dict (12 AUs, 0–5 scale) → one 512×512 still. No temporal model.

**Isolated env** (FineFace pins torch 2.2 / diffusers 0.27, incompatible with `genphoto`'s fragile pins; both kept off the shared `/` disk):
```
env:      /databases-4tb/levi-experiments/envs/fineface   (python 3.11, 5.6 GB)
HF cache: /databases-4tb/levi-experiments/.hf-cache        (5.0 GB)
```
Installed `forks/fineface/requirements.txt` exactly. CUDA verified on RTX 3090.

**Blocker — SD2.1-base removed from HF.** `FineFacePipeline` hardcodes `stabilityai/stable-diffusion-2-1-base`; Stability pulled all SD2.x repos (2025), so it 404s even authenticated (anonymous → 401; siblings also 404). FineFace's own weights (`Tvaranka/fineface`) still resolve. **Fix:** four independent mirrors carry a bit-identical UNet (`unet/diffusion_pytorch_model.safetensors` sha256 `6dfae3e5f7d459b5…`) in full diffusers layout — identical hashes across unrelated re-uploads ⇒ the true original weights, so FineFace's trained attn/LoRA apply correctly. Using `Manojb/stable-diffusion-2-1-base`. The fork was made to read the backbone id from `FINEFACE_SD_REPO` (default unchanged; fork commit `8c51a5c`).

**Invocation** (fineface env): `HF_HOME`, `HF_TOKEN`, `PYTHONPATH=forks/fineface`, `FINEFACE_SD_REPO=Manojb/stable-diffusion-2-1-base`. Generation via `scripts/fineface_generate.py`; scoring via `scripts/batch_eval.py score` (genphoto env; manifest uses `"frames"` entries) — the **identical** pipeline our model went through, so numbers are directly comparable.

## Experimental design

Matched to our [2026-07-14 eval](2026-07-14-tier12-batch-eval.md), same prompt bank (`configs/eval_prompts.txt`) and seeds (42/43/44):

- **scaled** (ramp-following + identity): 40 prompts × 3 seeds; 5 independent stills at commanded [0,.25,.5,.75,1.0], FineFace AU12 = commanded × 5. **Seed held fixed within a sweep** (shared initial noise) — the *most charitable* identity setting for a per-still model. Targets = commanded values → r/identity comparable to our 5-frame ramps.
- **calibration** (CLS): 10 prompts × 3 seeds × 11 single stills at commanded {0,…,1.0}. FineFace's default 50 steps / guidance (no handicap).

## Results (paired, n=120 scaled; same prompts & seeds)

| Metric | Ours (2026-07-14) | FineFace | Δ (ours − ff) | ours wins |
|---|---|---|---|---|
| AU12 Pearson *r* (ramp-following) | 0.774 ± 0.212 | **0.786 ± 0.333** | −0.011 | 43% |
| MediaPipe smile *r* (independent) | 0.829 ± 0.230 | **0.880 ± 0.238** | −0.052 | 30% |
| AU12 MSE | 0.142 | 0.141 | ~0 | — |
| Identity, adjacent-frame cosine (mean) | **0.919 ± 0.054** | 0.871 ± 0.124 | +0.047 | 63% |
| Identity, adjacent-frame cosine (min) | **0.877 ± 0.086** | 0.756 ± 0.212 | +0.121 | 68% |

**Calibration (CLS, absolute intensity):**

| | Ours | FineFace |
|---|---|---|
| CLS (pooled Pearson commanded→detected) | 0.075 | **0.497** |
| Dose–response curve | flat ~0.85 all levels | monotonic 0.05 (c=0) → 0.57 (c=1) |

## Interpretation (honest)

1. **Ramp-following: a tie, FineFace marginally ahead.** AU12 r is statistically indistinguishable (Δ −0.011, we win <half the prompts); on the independent MediaPipe detector FineFace is clearly ahead (0.88 vs 0.83, we win 30%). We cannot claim better expression control than the closest prior art.
2. **Absolute calibration: FineFace wins decisively (0.50 vs 0.07).** Its dose–response rises monotonically; ours is flat at ~0.85. This confirms the [2026-07-14](2026-07-14-tier12-batch-eval.md) diagnosis — our ramp-only MEAD training yields *relative/ordinal* control, not an absolute dial — and shows a static-stills competitor calibrates far better. A real limitation of our approach, not a wash.
3. **Identity: our win, modest but consistent and conservative.** We lead on mean (+0.05) and especially worst-case adjacent transition (+0.12), with ~2.5× lower variance, winning 63–68% of prompts. Conservative because (a) FineFace got its best-case shared-seed setup, and (b) our frames are lower-res MEAD-domain video with real head motion and blinks (which *lower* cosine), yet still score higher. The temporal-attention binding is doing real work.
4. **The gender bias is ours specifically, not the task's.** FineFace ramp *r* is *higher* for male prompts (0.82) than female (0.77) — the opposite of our male deficit (0.71 M / 0.83 F). This localizes our male-baseline-smile bias to the MEAD-happy training distribution, not to AU-conditioned face generation in general.

**Caveat:** the identity metric compares across visual domains (our 384×256 video frames vs FineFace 512² portraits); facenet embeddings are generally *more* stable on cleaner images, so the domain gap biases against us — the identity edge is if anything understated, not inflated.

## Revised positioning (post-comparison)

Drop any "better/finer expression control" or "better-calibrated" framing. The honest, defensible claim:

> A **sequence-native** formulation of expression-intensity control: identity-consistent, temporally coherent smile ramps generated as a **single clip**, distinct from per-still AU-conditioned generation (FineFace). Ramp-following is competitive with prior art and cross-frame identity is better (esp. worst-case); the trade-off is weaker absolute-intensity calibration, a consequence of ramp-only training.

Calibration is now a **named limitation with a fix path** (constant/permuted-list training augmentation, AU dropout + CFG, label smoothing), not a selling point. A qualitative figure (`inference_output/batch_eval/fineface/identity_compare_p000_s42.jpg`) shows a FineFace sweep vs our ramp for the same prompt/seed.

Note: an earlier draft of this note (setup stage) said FineFace identity "visibly drifts" — that was from a non-re-seeded quick test. Under the fair shared-seed setup, FineFace identity is actually decent (0.87); our advantage is real but modest, as the numbers above show.

## Reproduce

```bash
# generate (fineface env): scripts/fineface_generate.py --experiment {scaled,calibration}
#   with HF_HOME / HF_TOKEN / PYTHONPATH=forks/fineface / FINEFACE_SD_REPO=Manojb/stable-diffusion-2-1-base
# score (genphoto env):  scripts/batch_eval.py score --manifest .../fineface/manifest.jsonl
# summarize:             scripts/batch_eval.py summarize --results .../fineface/results.jsonl
```
Artifacts under `inference_output/batch_eval/fineface/` (gitignored): 450 sweeps (600 scaled + 330 calibration stills), `manifest.jsonl`, `results.jsonl`, `summary.json`.
