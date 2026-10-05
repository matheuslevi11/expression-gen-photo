# Absolute-calibration remedy: constant/permuted-sequence training augmentation (October 2, 2026)

First attempt at the fix direction named in [2026-07-14](2026-07-14-tier12-batch-eval.md) (point 4) and [2026-07-21](2026-07-21-fineface-headtohead.md): the model has relative/ordinal smile control but **no absolute calibration** (constant-list CLS 0.07 vs FineFace 0.50), attributed to ramp-only training. Calibration is on the critical path of the post-qualification direction ([2026-09-24 prompt-decoupled operator](2026-09-24-prompt-decoupled-operator.md)).

## Diagnosis (code + labels)

- `_select_intensity_progression` makes **every** training sample an ascending ramp: frames sorted by AU12, evenly spaced. The model never sees a constant or non-monotonic sequence — exactly the inputs the calibration (constant `[c]×5`) and permuted protocols use.
- The py-feat AU12 labels are **heavily skewed high** (96,672 train frames): median 0.88, 46% of frames in [0.9, 1.0], ~10% below 0.3. Mean AU12 by MEAD level: 0.71 / 0.81 / 0.84 — the ~0.85 plateau.
- Constant sequences are feasible almost everywhere (100% of clips have 5 frames within ±0.1 of *some* level) but **not at the low end**: only 41 / 200 / 366 / 382 clips (1% / 5% / 9% / 9%) can fill a constant clip at c = 0.0 / 0.1 / 0.2 / 0.3. Plain constant augmentation would mostly teach "constant high" ⇒ the level must be drawn **uniformly over levels**, not over clips.

## Design

`ExpressionMEAD(sequence_mode_probs=...)` (training only; `None` = previous behaviour, no extra RNG calls). Per sample:

| Mode | p | Frames | Conditioning |
|---|---|---|---|
| ramp | 0.5 | sorted-AU12 ramp of the sampler's clip (unchanged) | frame labels |
| constant | 0.3 | level c ~ U{0.0, 0.1, …, 1.0}; clip ~ U(clips with ≥5 frames within ±0.1 of c); 5 such frames, temporal order | frame labels (within ±0.1 of c) |
| permuted | 0.2 | the ramp's frames in a random non-identity order | frame labels |

Conditioning is always the *shown frames' own* AU12, so supervision stays honest; identity always comes from one clip. Effect on the conditioning distribution (1,500 draws, real labels): share < 0.3 rises **9.9% → 16.5%**, median 0.885 → 0.814, share in [0.9, 1.0] 47% → 39%. Constant clips' within-clip label spread: median 0.10, max 0.20.

Two arms, identical except `train_data.sequence_mode_probs` (configs `configs/train_genphoto/expression_calib_{aug,ctrl}.yaml`):

- 25k steps (the [2026-08-11](2026-08-11-training-step-dose-response.md) plateau), 1 node × 4 H100, batch 1/GPU, grad-accum 1 ⇒ **effective batch 4 = the RTX-3090 run**; lr 1e-4 constant, no warmup (schedule identical to the original), seed 42, checkpoints every 5k.
- The control arm is a fresh ramp-only run in the same setup rather than the original 25k checkpoint, which is confounded by different hardware and the AdamW-moment reset at the step-23k resume ([provenance rescue](2026-08-11-provenance-rescue.md)).

Jobs 667037 (aug) and 667038 (ctrl), `petrobr-h100`, ~0.2 s/step.

## Verification before/while training

- `tests/test_expression_sampling.py` (5 tests): mode-prob validation, pool tolerance/level dropping, **uniform level rebalancing** (one rare low clip vs 50 high clips → ~50/50), permuted = same frames in non-identity order.
- 20-step smoke of the aug config (job 666999) and a 60-step timing run (667007): data time 0.00 s, 0.21 s/step on 2 GPUs — the new sampling does not starve the GPUs.
- The two arms' logged losses were near-identical over the first 1,000 steps. Not a bug: both share the main-process seed, hence identical timesteps and noise per step, and with a frozen backbone the ε-MSE is dominated by (t, noise). A replica of the training `DataLoader` (forked workers, `DistributedSampler` rank 0/4, seed 42) confirms the aug arm delivers 39 sorted / 28 constant / 13 permuted sequences in 80 draws vs 80/0/0 for control.

## Evaluation plan (fixed before results)

Same protocols as July via `scripts/slurm/batch_eval_sdumont2nd.srm` on each arm's `checkpoint-step-25000`: calibration (10 prompts × 3 seeds × 11 constant levels), scaled (40 × 3 ascending ramps), permuted (10 × 3 × 3 lists).

- **Primary:** pooled constant-list CLS (py-feat), aug vs ctrl, with a paired bootstrap CI over the 30 (prompt, seed) units.
- **Circularity guard:** the aug arm is trained to reproduce *absolute* py-feat values, so py-feat CLS is more circular than in July. The same pooled CLS on the label-independent MediaPipe proxy (`cls_mp_pearson_pooled`, added to `batch_eval.py summarize` for this) must move in the same direction for a calibration claim.
- **Guardrails (no regression):** ascending-ramp AU12 r and adjacent-frame identity cosine within 0.05 / 0.02 of control.
- **Secondary:** permuted-list r (July: 0.39).
- Reading: success = CLS clearly above control on both detectors with guardrails held. If CLS stays near 0.1 despite rebalancing, the limit is the **data** (MEAD-happy has almost no low-AU12 frames — only 41 clips support c = 0.0) and the next step is adding MEAD neutral clips, not more sampling tricks.

## Results (25k, jobs 667094 / 667095)

Both arms trained cleanly (2h04 each, 4 ranks, effective batch 4; checkpoints at 5k–25k under `output/calibration/{aug,ctrl}/`). 540 samples per arm, all scored. `scripts/compare_eval.py`, paired by sample id, 95% CI from a 2,000-draw bootstrap over (prompt, seed) units:

| Protocol | Metric | ctrl | aug | aug − ctrl |
|---|---|---|---|---|
| calibration (n=330) | **CLS py-feat** | 0.034 | 0.158 | **+0.124 [+0.041, +0.215]** |
| | **CLS MediaPipe** | 0.003 | 0.160 | **+0.156 [+0.085, +0.237]** |
| scaled (n=120) | AU12 r | 0.646 | 0.631 | −0.011 [−0.097, +0.074] |
| | MediaPipe r | 0.680 | 0.797 | +0.116 [+0.043, +0.190] |
| | id mean-adj | 0.949 | 0.950 | +0.001 [−0.004, +0.006] |
| | id min-adj | 0.923 | 0.923 | −0.001 [−0.009, +0.008] |
| permuted (n=90) | **AU12 r** | 0.158 | 0.489 | **+0.331 [+0.184, +0.475]** |
| | MediaPipe r | 0.302 | 0.566 | +0.264 [+0.150, +0.384] |
| | id min-adj | 0.882 | 0.915 | +0.033 [+0.008, +0.061] |

Detected smile by commanded constant level (mean over 30 clips × 5 frames):

| c | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| py-feat ctrl | .863 | .867 | .866 | .870 | .874 | .882 | .887 | .885 | .881 | .870 | .868 |
| py-feat aug | .842 | .812 | .813 | .813 | .816 | .820 | .828 | .841 | .867 | .878 | .896 |
| MediaPipe ctrl (×100) | 2.40 | 1.92 | 1.94 | 2.00 | 2.06 | 2.15 | 2.20 | 2.18 | 2.11 | 2.06 | 2.12 |
| MediaPipe aug (×100) | 1.13 | 0.79 | 0.85 | 0.81 | 0.86 | 0.89 | 0.99 | 1.17 | 1.36 | 1.62 | 1.80 |

Against the pre-registered criteria: **CLS rises above control on both detectors (CIs exclude 0) and the guardrails hold** (ramp AU12 r −0.011, identity ±0.001). So the augmentation produces real, detector-independent absolute calibration — but **far from closing the gap**: CLS 0.16 vs FineFace 0.50 ([2026-07-21](2026-07-21-fineface-headtohead.md)). The largest effect is the secondary metric: **non-monotonic trajectory following triples** (permuted AU12 r 0.16 → 0.49, MediaPipe 0.30 → 0.57), with better worst-case identity.

### Reading

1. **The response is a hockey stick, not a dial.** Flat from c = 0.1 to 0.6, then rising to 1.0, on both detectors. The low end is the data-starved region (1–9% of clips support c ≤ 0.3); on MediaPipe the low-c smile is ~40–60% below control, so the model does render *less* smile there, just not graded within that band. py-feat compresses this (0.81 vs 0.87), consistent with that detector saturating near its top.
2. **c = 0.0 sits above c = 0.1 on both detectors.** Structural cause: the CCL channels are *differences* of CLIP states between consecutive frames' prompts (`build_ccl_embedding`), so for a constant list `[c]×5` they are **exactly zero**, and at c = 0.0 the whole 6-channel conditioning is zero. In training, "constant" sequences carry their real labels (spread up to 0.2), so the CCL is never zero: evaluation-time constant lists remain out of distribution in half the channels, even with this augmentation. Exploratory, not pre-registered: excluding c = 0.0, CLS is 0.191 (py-feat) / 0.195 (MediaPipe) for aug vs 0.021 / 0.036 for ctrl.
3. **Open discrepancy: this control is much weaker than the original run.** Same `scaled` protocol, both at 25k: ctrl AU12 r **0.646** (MediaPipe 0.680) vs the original's **0.850** (0.854) in [2026-08-11](2026-08-11-training-step-dose-response.md) — below even the original 5k (0.746). Both detectors drop together, so this is likely the generated videos rather than the detector. It is unresolved whether the cause is the cluster *training* (DDP sharding vs grad-accum, numerics, run-to-run seed variance) or the cluster *evaluation* (H100 numerics, freshly downloaded py-feat weights). The aug-vs-ctrl comparison is unaffected (same setup, paired), but **cluster numbers must not be compared with July/August/FineFace numbers** until this is resolved.

### Next steps (in order)

1. Score the original 25k checkpoint on the cluster with the same pipeline: ~0.85 ⇒ the gap is cluster training; ~0.65 ⇒ cluster evaluation. Needs one 2.6 GB file from the workstation.
2. Inference-only CCL test on the aug 25k model: jittered-constant lists (c ± small noise, like the training constant sequences) vs exact `[c]×5`. A CLS jump would point to fixing the train/eval CCL mismatch (e.g. exact-c conditioning in constant mode) before any data work.
3. Low-end support: add MEAD neutral clips (raw MEAD needed), since the flat 0.1–0.6 band matches where training data is scarce.

Artifacts (gitignored): `inference_output/batch_eval/calib_{aug,ctrl}_25k/{calibration,scaled,permuted}/{manifest,results}.jsonl`, `summary.json`, `inference_output/batch_eval/calib_compare_25k.json`.
