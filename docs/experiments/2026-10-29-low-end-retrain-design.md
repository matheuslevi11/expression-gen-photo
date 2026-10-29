# Low-end-data retrain: design and pre-registration (October 29, 2026)

Follows the decision of [2026-10-28 checkpoint selection](2026-10-28-checkpoint-selection-confirmation.md): every checkpoint lacks a neutral bottom of the scale, so LR decay + EMA are folded into a retrain with low-end data. The data is the MEAD neutral clips of [2026-10-29 neutral clips](2026-10-29-mead-neutral-clips-label-bias.md), whose open labelling decision is settled here. This entry fixes the design and readings before any result; results go in a new entry.

## Decisions (agreed 2026-10-29)

1. **Labels: per-actor baseline normalization** of every label, happy and neutral (option b). Each actor's neutral median `b` (over all its neutral frames; 47 actors, `b` from 0.05 to 0.73, median 0.24) becomes the bottom of the scale.
   - **Refinement of the agreed `clip((au12 − b)/(1 − b), 0, 1)`:** frames below `b` are squeezed monotonically into `[0, 0.02)`; they are not clipped to 0.
   - The curve is piecewise linear and continuous: `b → 0.02` and `1 → 1`.
   - Reason: clipping would make 50% of neutral frames exactly 0. Constant c = 0 sequences would then have identical `0.000` prompts, i.e. zero CCL, the regime the eval's `calibration_jitter` (≤ 0.02 dither) exists to avoid. 0.02 matches that dither, and no frame is exactly 0 after the mapping.
   - Raw values are kept as `au12_raw`. Implemented in `scripts/normalize_labels_per_actor.py`. Outputs:
     - `MEAD_processed/annotations/{train,validation}_happy_neutral_actnorm.json`;
     - `actor_baselines_actnorm.json`.
2. **Bridge sequences**, a new sequence mode (`bridge` in `genphoto/data/expression_dataset.py`). Sampling: uniform actor, then one neutral and one happy clip of that actor. The existing ramp selection runs over the union of their 48 frames. Result: a neutral → full-smile ramp of one identity, the only training sample shaped like the eval ramp at the low end.
3. **Two arms, identical except for the data.**

## Arms

| | control (`expression_lowend_ctrl.yaml`) | treatment (`expression_lowend_treat.yaml`) |
|---|---|---|
| clips | MEAD happy, 4,028 train | happy + neutral, 4,028 + 1,796 train (45 actors) |
| labels | raw py-feat AU12 | per-actor normalized |
| sequence modes | ramp 0.5, constant 0.3, permuted 0.2 (the 10-02 aug recipe) | ramp 0.25, **bridge 0.25**, constant 0.3, permuted 0.2 |
| ramp / permuted clips | happy | happy (`ramp_emotions`) |
| constant clips | happy | happy + neutral |
| validation file | `validation.json` | `validation_happy_neutral_actnorm.json`, happy only |

Shared settings:
- 25k steps on 1 node × 4 H100, effective batch 4, seed 42.
- **Cosine LR 1e-4 → 0, no warmup**, so the first steps match the 10-02 arms. **EMA 0.999.**
- Checkpoints every 2.5k: the main checkpoint plus `checkpoint-step-N-ema.ckpt`. That is 2.5k rather than the 1k floated on 10-28, to keep each arm at ~40 GB.

The scheduler now steps per optimizer step. EMA and resume were smoke-tested on 10-28 (jobs 747844/747845).

Training-sample composition (20,000 simulated draws, treatment):

| | share | mean label by frame slot | median within-sequence range |
|---|---|---|---|
| bridge | 0.25 | 0.01, 0.06, 0.42, 0.76, 0.93 | 0.95 |
| ramp (happy) | 0.25 | 0.26, 0.62, 0.77, 0.85, 0.93 | 0.75 |
| constant | 0.30 | ≈ 0.49 (uniform level) | 0.11 |
| permuted (happy) | 0.20 | ≈ 0.69 | 0.76 |

- In 95% of bridges the first frame comes from the neutral clip and the last from the happy clip.
- Constant-pool support (clips with 5 frames within ±0.1), per level:

| level | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 41 | 200 | 366 | 382 | 449 | 647 | 1,016 | 1,651 | 2,530 | 3,560 | 2,949 |
| treatment | 2,193 | 2,462 | 1,216 | 948 | 840 | 981 | 1,284 | 1,812 | 2,664 | 3,310 | 2,546 |

- No constant c ≈ 0 sequence has identical prompts (0 of 836).

Normalized label distribution:

| | median | < 0.1 | > 0.5 |
|---|---|---|---|
| happy | 0.82 | 6.2% | 78.4% |
| neutral | 0.02 | 65.3% | 6.0% |

## Verification before submission

- **Unit tests** (`tests/test_expression_sampling.py`, `tests/test_label_normalization.py`, 11 pass):
  - bridge pairs need both emotions per actor;
  - bridge actor choice is uniform and clips are same-actor;
  - bridge ramps are monotone from the low clip to the high clip;
  - normalization maps 0 → 0, b → floor and 1 → 1, and is continuous and strictly monotone;
  - baselines use neutral frames only.
- **Regression:** under the control config, 150 seeded `_load_clip` draws are identical (pixels, caption, labels, clip id) between the new dataset module and HEAD's.
- **Contact sheet** of 6 bridges and 3 constant c = 0 sequences on real data:
  - the bridges are visibly neutral → smile ramps of one person, with consistent background and clothing;
  - the c = 0 frames are visibly neutral even where raw py-feat read 0.59–0.73 (W029).
- **Training smoke** of the treatment config: 8 steps on 2 H100 (h100_dev, job 750629), passed.
  - The LR follows the cosine to exactly 0 at step 8.
  - The main and EMA checkpoints were written at steps 4 and 8.
  - Bridge sampling works inside the DataLoader workers.

## Submission

- Control: job **750647**. Treatment: job **750648**.
- Each runs on 1 node × 4 H100 (`train_expression_sdumont2nd.srm --nodes=1 --time=05:00:00`) with the local copies of the two configs.
- The code was submitted uncommitted and is identical to commit `0e41505` (only this doc changed after submission). HEAD, `git status` and the full diff are also saved in `output/lowend/code_snapshot_2026-10-29.diff`.

## Pre-registered evaluation

**Primary checkpoint:** the **EMA checkpoint at 25k** of each arm, fixed now; no checkpoint is selected by looking at results. With cosine decay to 0, the last checkpoint is the natural endpoint.

**Primary comparison:** treatment − control, paired over held-out units:
- `calibration_jitter` and `permuted`: prompts 10–29 × seeds 45–47;
- `scaled`: 40 prompts × seeds 45–47;
- both detectors, 2,000-draw bootstrap.

Readings:

1. **Low end (the purpose).** The detected level of the ramp's first frame on MediaPipe (geometric, label-independent) and the `calibration_jitter` detected level at c ≤ 0.1 on MediaPipe should be lower for the treatment, with a CI excluding 0.
   - py-feat frame 0 is reported but is not the criterion. Under the treatment, c = 0 means "the actor's own neutral", whose raw py-feat reading is ~0.24 on MEAD. Absolute py-feat levels are therefore not comparable across arms.
2. **Calibration.** MediaPipe CLS of the treatment should be ≥ the control's: Δ CI lower bound > −0.05.
3. **Ramps.** Ramp MediaPipe r: Δ no worse than −0.05, the 10-28 tolerance.
4. **Non-monotonic following.** Permuted AU12 / MediaPipe r: Δ no worse than −0.05.
5. **Failure tail.** Fail rate (ramp AU12 r < 0.5) is reported together with identity min-adjacent cosine.

Secondary (descriptive):

- **Prompt groups.** Every metric split into ethnicity-named prompts (`prompt_idx` 14–23) and the rest, on both detectors. The question: does the py-feat / MediaPipe disagreement concentrate in that group, and does the treatment change the group gap?
- **Recipe stabilization.** Control EMA-25k vs aug 15k on the same held-out units (paired). The question: do cosine + EMA alone remove the late drift that forced the 15k pick?
- **Trajectory.** EMA checkpoints at 10k / 15k / 20k / 25k of both arms on the original units (prompts 0–9 × seeds 42–44). The question: is the 10-28 degradation signature (frame 0 rising, fail tail growing after ~15k) gone?
- **FineFace.** Unpaired comparison as on 10-28: FineFace's original-unit estimate against the held-out CI.

Decision rules:

- If readings 1–4 hold, the treatment recipe becomes the main model and replaces both the original 25k (ramps) and aug 15k (calibration) in reporting.
- If reading 1 holds but 3 or 4 fails, the low end is fixed at a following cost; tune the bridge share before scaling.
- If reading 1 fails, low-end data support is not the bottleneck. Then look at the conditioning: whether c = 0 prompts are distinguishable from nearby levels in CLIP space.
