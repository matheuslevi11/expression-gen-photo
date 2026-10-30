# Expression adaptation — project status

Living document: current state of the Generative Photography → **Generative Expressions** fork. Edited destructively — checked items get checked, wrong claims get deleted. History and full experiment write-ups live in [`experiments/`](experiments/) (append-only, date-stamped). Design rationale: [`plan.md`](plan.md). Usage commands: root [`README.md`](../README.md). Doc conventions: root [`CLAUDE.md`](../CLAUDE.md).

*Last updated: October 29, 2026.*

---

## Executive summary

The fork replaces GenPhoto's camera axis with **scalar facial-expression (smile / AU12) intensity**, keeping the 3D UNet, AnimateDiff motion module, 6-channel conditioning layout (`cin: 384`), and attention-injection mechanism unchanged. Training data is MEAD front-view clips, preprocessed with MediaPipe crops + py-feat AU12 labels: happy (4,028 train / 176 val) plus, since 2026-10-29, neutral (1,796 train) for the bottom of the scale.

**Main model (2026-10-29):** the low-end retrain's treatment arm, EMA weights at 25k:
`output/lowend/treat/expression_lowend_treat.local-2026-10-29T14-58-39/checkpoints/checkpoint-step-25000-ema.ckpt`
(happy + neutral clips, per-actor normalized labels, neutral → smile bridge ramps plus the constant/permuted mix, cosine LR + EMA, 25k steps on 4 H100). On held-out units: **CLS 0.62 py-feat / 0.48 MediaPipe, ramp r 0.81 / 0.84, permuted r 0.67 / 0.73, fail 7%**. It ties the original 25k on ramps and aug 15k on calibration and permuted following, beats each on the other's axis, and costs 0.01–0.02 worst-case identity. It replaces both in reporting.
Earlier lineage: the 100k run on one RTX 3090 (`output/expression/expression-2026-05-26T21-43-56/`; ramp control peaks at 25k–50k, 2026-08-11). Its batch eval (2026-07-14) established ramp control (**r = 0.77 ± 0.21**, MediaPipe **0.83**), frame-to-frame identity **0.92** (frozen baseline 0.85, undetectable on 57% of its samples) and descending reversal −0.98. **Constant commands must be dithered** (2026-10-07): exact `[c]×5` lists zero the CCL channels (an input never seen in training) and collapse calibration to CLS 0.07.

**FineFace head-to-head (2026-07-21, revised 2026-10-28 and 2026-10-29; closest prior art):** the main model **ties FineFace on MediaPipe calibration** (CLS 0.48 held-out vs 0.49, inside the CI) and on ramp AU12 r (0.81 vs 0.79), and **trails slightly on ramp MediaPipe r** (0.84 vs 0.88). These are unpaired: FineFace ran on the original units. The low end is detector-dependent: py-feat reads our c = 0 at 0.32, the level of real MEAD neutral faces, vs FineFace's 0.05, while MediaPipe ranks ours lower; visually our c = 0 frames are neutral. We **win cross-frame identity** modestly (worst-case +0.12 to +0.17 for earlier checkpoints; the main model is 0.01–0.02 lower). Net: the defensible contribution is the **sequence-native formulation** (one identity-consistent clip vs independent per-still generation) at parity, *not* better expression control. Claim/position accordingly.

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
| 2026-07-14 | [Tier 1+2 batch eval](experiments/2026-07-14-tier12-batch-eval.md) | 540+120 samples: ramp control robust (r=0.77±0.21, n=120; MediaPipe confirms 0.83), identity 0.92 vs baseline; **no absolute calibration** (constant-list CLS=0.07, flat ~0.85) — control is relative/contextual; permuted lists partial (0.38). ⚠️ Calibration reinterpreted 2026-10-07 (zero-CCL artifact). |
| 2026-07-21 | [FineFace head-to-head](experiments/2026-07-21-fineface-headtohead.md) | Tier-1 comparison (n=120 matched): ramp-following **tie** (r 0.79 ff / 0.77 ours); FineFace **wins calibration** (CLS 0.50 vs 0.07; ⚠️ halved but standing after dither, 2026-10-28); we **win identity** modestly (min-adjacent +0.12, 68% of prompts). Contribution narrows to the sequence-native temporal formulation. |
| 2026-08-11 | [AU-vector proposal](experiments/2026-08-11-au-vector-proposal.md) | Design note (no run): Option B (multi-AU, FineFace-style) is a **large, data-gated** direction, not a cheap experiment — happy-only MEAD gives no non-smile-AU variance (needs multi-emotion re-preprocess + retrain), the vector doesn't fit the 6-ch spatial map, and it walks onto FineFace's turf. Backlog it as **multi-AU/multi-emotion temporal *trajectories***, after the cheap claim-closing items. |
| 2026-08-11 | [training-step dose–response](experiments/2026-08-11-training-step-dose-response.md) | 6 checkpoints × 120 ramps: **training drives control** — baseline −0.05 → AU12 r 0.55 @1k → plateau 0.85 @25–50k (MediaPipe agrees, Spearman +0.60; std shrinks 0.43→0.13). **100k is mildly past-peak** (0.77 < 0.85 @50k, t≈3.4); control sweet-spot is 25k–50k. In-session 100k regen bit-matched July (determinism ✓). Checkpoint deletion now unblocked. |
| 2026-08-11 | [provenance rescue](experiments/2026-08-11-provenance-rescue.md) | Rescue note (no run), qualifying Step 0: (1) "100k" run is 3 segments — **AdamW moments reset at the step-23k resume** (optimizer_state_dict saved but never read); (2) `plan.md` stale — code trains encoder + attention `merge` (**LoRA re-frozen**), uses py-feat not OpenFace/FID, file is `expression_dataset.py`; (3) train/val is a **clean 2-actor holdout** (W009/W011, 0 overlap, female-only val); **no MEAD in the headline eval loop**. |
| 2026-08-25 | [AU co-activation](experiments/2026-08-25-au-coactivation.md) | Re-scored all 20 py-feat AUs (+ **LibreFace** independent + geometry) over the ramp: the AU12 axis drives the whole **MEAD-happy covariance** — smile cluster up (**AU06 Duchenne** 0.88, AU25 0.88 via LibreFace/geo, AU10), mouth-closing/frown **down** (AU23 −0.72), + **subtle brow entanglement** (localization only partial, detector-dependent). Co-activation **emerges with training** (AU06 0.70→0.94). Cross-check caught a py-feat **AU20 artifact** (LibreFace disagrees). Motivates multi-AU; feeds Cap05 generality. |
| 2026-09-24 | [prompt-decoupled operator](experiments/2026-09-24-prompt-decoupled-operator.md) | Design note (no run), **advisor proposal** post-qualification: take the prompt out of the core loop and isolate the **S-operator** — face embedding (from a prompt *or any source*) + per-frame S → temporal embeddings → VAE-decode ("give a neutral face, I make it smile 0.2/0.8"). Adds editing/animation the current model lacks. Instantiations: (A) pure latent operator (cheap, risk = VAE realism of large smiles), (B) reusable face-embedding conditioning (IP-Adapter-style, practical), (C) framing-only. **Requires absolute calibration** as a prerequisite. Two cheap no-retrain probes proposed first. |
| 2026-09-24 | [probe 1: latent smile direction](experiments/2026-09-24-probe-latent-smile.md) | Feasibility probe for path A (no retrain): a global **linear** SD-VAE smile direction pushes detected AU12 0.43→0.81 across α, but is **entangled with a global darkening/skin-tone shift** — reaching a strong smile (AU12≳0.7) forces identity to collapse (0.66→0.37); the "brightness shortcut" risk. VAE roundtrip faithful (ceiling 0.87). **Linear latent editing not viable**; a learned disentangling operator would be needed. Favors **path B** (diffusion editing) → run probe 2 next. |
| 2026-09-24 | [probe 2: diffusion editing (SDEdit)](experiments/2026-09-24-probe-diffusion-edit.md) | Editing probe for path B (no retrain), 6 inputs × 4 strengths: the trained S-operator **animates a supplied in-domain neutral face** into a graded, identity-preserving smile with no darkening — but only in a **narrow strength band** (S≈0.7: ramp r=0.85, id→input 0.68; by S≥0.9 identity collapses to 0.24 = pure regeneration) and it does **not transfer to out-of-domain faces** zero-shot (they get regenerated, not edited). **Path B confirmed as the direction, but zero-shot SDEdit is not the deliverable** — the production operator needs trained reusable-face-embedding conditioning + inversion + absolute calibration. |
| 2026-10-01 | [SDumont2nd smoke](experiments/2026-10-01-sdumont2nd-smoke.md) | First cluster runs: 6-step smoke passes on 1×2 and 2×2 H100 (cross-node NCCL over IB/GPUDirect). Fixed 3 launch bugs: `h100_dev` QOS caps 2 GPU/node, `set -u` vs conda hooks, inherited activated env → base python. |
| 2026-10-02 | [calibration augmentation](experiments/2026-10-02-calibration-augmentation.md) | Train-time ramp/constant/permuted mix (level drawn *uniformly*) vs fresh ramp-only control, 25k each on H100: **CLS 0.03 → 0.16 on both py-feat and MediaPipe** (Δ CIs exclude 0), ramp r and identity unchanged; **permuted r 0.16 → 0.49**. Partial: response is flat for c ≤ 0.6 (data-starved low end) and constant lists zero the CCL channels (train/eval mismatch). Compared at 25k, which turned out to be a degraded checkpoint (see 2026-10-05). |
| 2026-10-05 | [cluster control discrepancy](experiments/2026-10-05-cluster-control-discrepancy.md) | Ramp r 0.65 (cluster ctrl 25k) vs 0.85 (original 25k) explained: the cluster eval reproduces the workstation numbers exactly (original ckpt → 0.855), code is unchanged since May, and the cluster run **matches the original up to 10k (0.79 vs 0.78), then regresses**: a failure tail (24% of ramps r<0.5) where the neutral start is lost. Single-run late training is unstable; select checkpoints by eval. **At 10k the calibration gain shrinks to +0.08 CLS** (significant on MediaPipe only; ctrl itself has CLS 0.15 there); permuted-r gain (+0.25) holds; costs: ramp MP r −0.07, permuted identity −0.02. |
| 2026-10-07 | [CCL zero test](experiments/2026-10-07-ccl-zero-test.md) | Inference-only: exact `[c]×5` lists zero the CCL channels (CCL is a "prompt changed" flag, not a magnitude). Same models, same seeds, ≤0.02 per-frame dither: **CLS 0.08 → 0.50 py-feat / 0.07 → 0.30 MediaPipe on the original lineage** (= FineFace's 0.497 on py-feat only; ⚠️ see 2026-10-28), 0.16 → 0.63 / 0.49 for the augmented arm at 25k; all 5 models gain on both detectors. July's "no absolute calibration" was mostly a protocol artifact. Residual: c = 0 over-smiles. |
| 2026-10-28 | [FineFace on both detectors](experiments/2026-10-28-fineface-both-detectors.md) | FineFace's MediaPipe CLS is **0.494** (py-feat 0.497). Dithered ramp-only checkpoints trail it on MediaPipe: 25k 0.30 (Δ −0.20), and the thesis 100k (transferred, reproduces July) 0.30 / 0.26 (Δ −0.19 / −0.24 on both detectors). FineFace's calibration win **stands at half the July size**. Aug 25k ties it on MediaPipe (0.491), but its ramps regressed; no checkpoint matches FineFace on both axes yet. 25k dominates 100k. Our c = 0 still smiles (py-feat ≥ 0.4). |
| 2026-10-28 | [checkpoint selection + held-out confirmation](experiments/2026-10-28-checkpoint-selection-confirmation.md) | The aug arm's 15k looked balanced vs FineFace on the eval units, but **held-out units** (prompts 10–29, seeds 45–47) show partial selection inflation. Aug 15k − original 25k: MediaPipe CLS **+0.06** (CI just excludes 0), ramp MediaPipe r −0.05, **permuted r +0.28/+0.31** (the robust augmentation effect). Averaging the 15k and 20k weights is worse than the 15k alone. All checkpoints degrade by losing the ramp's neutral start (constant LR, no EMA). No standalone stabilisation retrain; fold LR decay + EMA into the low-end-data retrain. |
| 2026-10-29 | [MEAD neutral clips + label bias](experiments/2026-10-29-mead-neutral-clips-label-bias.md) | 1,876 front neutral clips (47 actors) preprocessed on the cluster and merged (`*_happy_neutral.json`); the cluster labeler matches May's (r 0.998). Low-level constant support rises from 41/200 to 555/1,390 clips at c = 0/0.1. **py-feat AU12 has an actor-level offset tied to skin tone** (Spearman vs face luma −0.64 neutral / −0.40 happy; MediaPipe n.s.): darker faces read higher at the same geometry. Labels and eval both inherit it. |
| 2026-10-29 | [low-end retrain design](experiments/2026-10-29-low-end-retrain-design.md) | Pre-registration (results: next row). Control (aug recipe, happy, raw labels) vs treatment (happy + neutral, **per-actor normalized labels**, **neutral → smile bridge ramps**). Both arms: 25k steps, cosine LR + EMA 0.999. Primary: EMA-25k, treatment − control on held-out units; low end judged on MediaPipe. |
| 2026-10-29 | [low-end retrain results](experiments/2026-10-29-low-end-retrain-results.md) | All 4 pre-registered readings hold (treatment − control, held-out): CLS MediaPipe **+0.15**, ramp MP r +0.08, permuted r +0.13, MediaPipe low end −0.004 (CI < 0; mostly removing a c = 0 bump), fail 16% → 7%, identity −0.015. Treatment EMA-25k becomes the **main model**: it ties the original 25k on ramps and aug 15k on calibration/permuted, and beats each on the other's axis. Cosine + EMA remove the late drift, but the cosine control lands below aug 15k. The treatment's c = 0 reads at the py-feat level of real MEAD neutral faces and looks neutral; its py-feat gap to FineFace looks like a detector domain offset (MediaPipe ranks the other way). |
| 2026-10-29 | [longer-training design](experiments/2026-10-29-longer-training-design.md) | Pre-registration (no result yet). Treatment recipe at 50k steps (cosine over 50k) vs the 25k main model, plus a seed-142 replicate of the 25k as the run-to-run yardstick; fresh units (prompts 10–39 / all 40 × seeds 48–50). Primary: CLS MediaPipe Δ with CI > 0 and beyond the replicate gap; guardrails on ramps, permuted following and identity. |

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
- [x] **Dose–response calibration curve** (2026-07-14, reinterpreted 2026-10-07) — with exact-constant lists CLS = 0.07 (flat ~0.85), but that protocol zeroes the CCL channels. With a ≤0.02 per-frame dither the original lineage gives **CLS 0.50 py-feat / 0.30 MediaPipe at 25k** (rising from ~0.44 to ~0.84) and **0.30 / 0.26 at 100k** (flat ~0.72 for c ≤ 0.5). Main model (2026-10-29): **0.62 / 0.48** on held-out units, monotone on py-feat over all 11 levels. Report both protocols.
- [x] **FineFace head-to-head** (2026-07-21; both detectors 2026-10-28) — ramp-following a **tie** (AU12 r 0.786 ff / 0.774 ours; FineFace ahead on MediaPipe 0.88/0.83). Calibration: FineFace **0.497 py-feat / 0.494 MediaPipe**. Ours with exact-constant lists: 0.075 / 0.015. Ours dithered: 0.30 / 0.26 at 100k and 0.50 / 0.30 at 25k, i.e. FineFace led on MediaPipe by Δ −0.20 to −0.24. The main model (2026-10-29) reaches **0.62 / 0.48** on held-out units, so it ties FineFace on MediaPipe (inside the CI, unpaired) and leads on py-feat; its ramp MediaPipe r is 0.84 vs 0.88. We **win cross-frame identity** modestly (adjacent-min cosine +0.12, 63–68% of prompts, ~2.5× lower variance). Our male-bias is data-specific (FineFace has the opposite gender skew). Contribution = sequence-native temporal formulation, not superior control.

### Tier 2 — minimum for a complete ablation

- [x] **Scaled evaluation** (2026-07-14) — 40 prompts × 3 seeds: r = **0.774 ± 0.212** (median 0.86, 87% > 0.5) vs baseline −0.05 ± 0.64. Gender gap quantified: female 0.83 vs male 0.71.
- [x] **Non-monotonic / permuted intensity lists** (2026-07-14) — r = **0.385 ± 0.505**: real per-frame conditioning, but much weaker than monotonic ramps — the temporal prior resists non-monotonic trajectories (no training support). With constant/permuted training augmentation: **0.68** on held-out units vs 0.40 for the original 25k (2026-10-28); main model 0.67 AU12 / 0.73 MediaPipe (2026-10-29).
- [x] **Independent AU measurement** (2026-07-14) — MediaPipe mouth-corner proxy confirms ramp control at r = **0.829 ± 0.230** (n=120), breaking the py-feat label/eval circularity that all four studied papers share unflagged.

### Tier 3 — supporting evidence

- [x] **Training-step dose–response** (2026-08-11) — 6 checkpoints × 120 ramps over the continuous 1k→100k run: baseline −0.05 → AU12 r **0.55 @1k → plateau 0.85 @25–50k** (Spearman +0.60; MediaPipe agrees; per-prompt std shrinks 0.43→0.13). **Independent evidence training drives the capability**; acquired early, saturates ~25k. **100k is mildly past-peak** (0.77 < 0.85 @50k, Welch t≈3.4) — control sweet-spot is 25k–50k; supports the calibration deficit being data/formulation, not undertraining. Checkpoint-housekeeping constraint now **satisfied** (in-session 100k regen bit-matched July → determinism ✓).
- [ ] **Prompt-engineering baseline figure** — 5 independent SD1.5 generations with graded smile prompts. Now doubly motivated: PixelSmile's zero-shot row (CLS-6 0.69 untrained) proves "just prompt it" is a real objection; expected result is coarse control with collapsed identity.
- [ ] **Out-of-range intensity probe** — intensities < 0 and > 1 (FineFace's Fig. 8 extrapolation demo); cheap disentanglement evidence.

---

## Known limitations (affect result quality, not correctness)

- **MEAD happy clips** often have high AU12 even at lower labeled intensity levels; the dataset uses sorted-frame ramps and, for validation, decouples fixed `intensity_list` targets from which pixels are shown.
- **Constant commands must be dithered** — exact-constant lists `[c]×5` zero the CCL channels (an out-of-distribution input) and collapse calibration to CLS 0.07; a ≤0.02 per-frame dither restores it (original 25k 0.50 py-feat / 0.30 MediaPipe, 100k 0.30 / 0.26; 2026-10-02 augmentation at 25k: 0.63 / 0.49). Principled fix: a magnitude-aware CCL (needs retraining). Late-training drift under constant LR (2026-10-05) moved CLS; cosine LR + EMA removes it (2026-10-29).
- **Absolute low-end levels are detector-dependent across generators** (2026-10-29). py-feat reads the main model's c = 0 at 0.32, the same as real MEAD neutral frames, vs FineFace's 0.05, while MediaPipe ranks ours lower and is at a floor below c ≈ 0.2. Our c = 0 frames often have open, speech-like mouths (MEAD clips are speech). An absolute low-end claim needs a third detector or human rating.
- **Ramp-following is on par with, not better than, prior art** — matched-prompt AU12 r ties FineFace (0.77 vs 0.79). The contribution is the sequence-native temporal formulation + modestly better cross-frame identity, not superior control.
- **py-feat AU12 reads higher on darker skin at the same mouth geometry** (2026-10-29; luma proxy, MediaPipe shows no trend). It affects (i) raw training labels, which tie the intensity scale to identity (addressed in the low-end retrain by per-actor normalization), and (ii) the eval, whose prompts 14–23 (0-based `prompt_idx`) name ethnicities. Judge absolute levels on MediaPipe, and split results by prompt group before any demographic claim.
- **Male baseline-smile bias (data-specific)** — quantified 2026-07-14: ramp r = 0.71 (male prompts) vs 0.83 (female). FineFace shows the *opposite* skew (male 0.82 > female 0.77), localizing the bias to the MEAD-happy training distribution rather than AU-conditioned generation generally. Consider identity-conditioned training or ArcFace regularisation if cross-gender robustness is required. Measured with py-feat; given its skin-tone offset (above), part of it may be detector-driven.
- **Option A embedding only** — scalar broadcast + CCL; no landmark flow or multi-AU control yet.
- **Emotions** — happy only, plus neutral clips for the low end (2026-10-29); other emotions / multi-AU are a follow-up.
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

### Training (multi-node, SDumont2nd)

```bash
sbatch --nodes=1 scripts/slurm/smoke_expression_sdumont2nd.srm       # smoke (h100_dev)
sbatch --nodes=<N> scripts/slurm/train_expression_sdumont2nd.srm     # real run (petrobr-h100)
```

Account `crono`; uses the existing `--launcher slurm` path, no code changes. Staged (2026-10-01): `genphoto` env (verified on H100), backbones under `/petrobr/crono/matheus.aidano/backbones/` (sha256-checked), filled configs in `/petrobr/crono/matheus.aidano/configs_local/`, `MEAD_processed/` (all 100,896 frames present and decodable). 1- and 2-node smoke pass ([2026-10-01](experiments/2026-10-01-sdumont2nd-smoke.md)); `h100_dev` is capped at 2 GPUs/node by QOS. See root [`README.md`](../README.md#4b-training-on-sdumont2nd-slurm-multi-node).

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
genphoto/data/expression_dataset.py   # sequence_mode_probs: ramp/constant/permuted/bridge train-time mix
tests/                               # sampling + label-normalization unit tests (python tests/<file>.py)
scripts/preprocess_mead.py
scripts/merge_mead_emotion.py                     # merge sharded preprocess output for one emotion into MEAD_processed
scripts/normalize_labels_per_actor.py             # AU12 labels relative to each actor's neutral median
scripts/_validate_dataset_and_model.py            # static validator
scripts/export_results.sh                         # stage results tarball for scp/rsync pull
scripts/batch_eval.py                             # generate/score/summarize eval sweeps (--prompt-offset: held-out prompts)
scripts/compare_eval.py                           # paired run-vs-run comparison, bootstrap CIs
scripts/average_checkpoints.py                    # uniform weight average of adaptor checkpoints
configs/eval_prompts.txt                          # 40-prompt bank for batch eval
configs/train_genphoto/expression.yaml            # full train (placeholders)
configs/train_genphoto/expression_smoke.yaml      # verified 5-step smoke
configs/train_genphoto/expression_validation.yaml # verified production-path validation
configs/train_genphoto/expression_sdumont2nd.yaml       # full train, SDumont2nd cluster (placeholders)
configs/train_genphoto/expression_sdumont2nd_smoke.yaml # 6-step multi-node smoke, SDumont2nd
configs/train_genphoto/expression_calib_{aug,ctrl}.yaml # 2026-10-02 calibration arms (placeholders)
configs/train_genphoto/expression_lowend_{ctrl,treat}.yaml # 2026-10-29 low-end retrain arms (placeholders)
configs/train_genphoto/expression_lowend_treat_{50k,s142}.yaml # 2026-10-29 longer-training test (placeholders)
scripts/slurm/                                    # sbatch launchers for SDumont2nd (train, smoke, batch_eval, preprocess_mead)
configs/inference_genphoto/expression.yaml        # trained-adaptor inference
configs/inference_genphoto/expression_baseline.yaml # fair frozen-backbone baseline
comp_metrics/expression_au_accuracy.py
MEAD_processed/                                   # local data (gitignored)
output/expression/                                # training runs + checkpoints
inference_output/                                 # generated GIFs + ablation artifacts
docs/plan.md                                      # design rationale (frozen)
docs/experiments/                                 # append-only lab notebook
```
