# Multi-AU E1 dose–response: design (November 21, 2026)

A follow-up to the [E1 results](2026-11-21-multi-au-e1-results.md), pre-registered before any sample is generated. No training: it evaluates checkpoints that E1 already wrote.

## Question

**Does T's smile cost against C shrink with training?**
- **Background:** in E1, T's smile lost amplitude at the top of the scale. At 25k its MediaPipe CLS was 0.331 against C's 0.395, and its smile at c = 1 was weaker on all three judges.
- **Two explanations:**
  - **slower learning:** T splits its sequences over three channels and learns the smile more slowly, so the gap would shrink with steps;
  - **structural:** the 3-key conditioning costs smile amplitude at this recipe, and the gap would not shrink.
- **What depends on it:** whether a longer T schedule is worth training, or whether the next change has to be to the mix or the conditioning.

## Why C is evaluated too

E1 used a cosine LR from 1e-4 to 0 at 25k, so the learning rate is already small near the end:

| step | 10k | 15k | 20k | 25k |
|---|---|---|---|---|
| LR | 6.5e-5 | 3.5e-5 | 9.5e-6 | 0 |

T's own curve therefore flattens near 25k whatever the cause, and "still rising at 25k" cannot separate the two explanations. The T − C gap can: both arms see the same frames under the same schedule and differ only in conditioning.

## Runs

- **Checkpoints:** EMA at 10k, 15k and 20k for T and C:
  - `output/multiau/{t,c}/expression_multiau_{t,c}.local-2026-11-21T14-31-57/checkpoints/checkpoint-step-{10000,15000,20000}-ema.ckpt`;
  - local configs `infer_multiau_{t,c}_ema{10000,15000,20000}.local.yaml`, identical to the 25k ones except the checkpoint path.

  The 25k samples are E1's own.
- **Protocol:** `calibration_jitter` only, on the E1 units: prompts 10–39 × seeds 48–50 (90 units), 11 levels each, the held brow channels at 0.02 ± 0.02 as in E1. The same units and initial noise are used at every checkpoint.
- **Jobs:** six `batch_eval_sdumont2nd.srm` jobs (one per arm × checkpoint, 1 H100 each, `EXPERIMENTS=calibration_jitter SUBSET_OFFSET=10 SUBSET_PROMPTS=30 SEEDS=48,49,50`, no `ALLAU`). Then one LibreFace job on the six manifests.
- **Expected cost:** about 2 h 25 per job (58 min generation, 85 min scoring, from E1's logs), about 15 GPU-hours in all.
- **Outputs:** `inference_output/batch_eval/e1/multiau_{t,c}_ema{10000,15000,20000}/`.

## Measurements

- **Smile judges:** MediaPipe mouth-corner proxy (primary, as in E1 reading 4), py-feat AU12 and LibreFace AU12.
- **Response slope:** for each unit, the OLS slope of the detected smile on the commanded level over its 55 frames.
- **Gap:** gap(s) = mean over units of slope_T(s) − slope_C(s).
- **CLS:** the pooled CLS of E1 reading 4.

The E1 analysis already confirms that the slope carries the E1 deficit: at 25k, T's MediaPipe slope is 0.0201 against C's 0.0254, a gap of −0.0053 [−0.0078, −0.0030].

**Statistics:** a 2,000-draw bootstrap over the 90 units. Every arm and checkpoint is resampled with the same draws, so all differences are paired.

## Reading and outcome

**Primary:** d_gap = gap(25k) − gap(10k), on MediaPipe.
- **closing:** CI lower bound > 0 (the gap shrinks with training);
- **widening:** CI upper bound < 0;
- **constant:** the CI includes 0.

**Check:** the same rule on the MediaPipe CLS gap. If the slope and CLS categories disagree, the outcome is **inconclusive**.

**Consequences:**
- **Closing** supports slower learning. A longer T schedule, or a larger smile share of the mix, is a candidate next run.
- **Constant or widening** says the cost does not come from training length at this recipe. A longer schedule is not expected to remove it; the candidates change the mix (smile share, 25% now) or the conditioning.
- **Inconclusive:** no recommendation from this experiment.

Any next training run gets its own pre-registration.

**Also reported (descriptive):**
- the slope, gap and CLS at every checkpoint, on all three judges;
- the per-level curves;
- the change from 20k to 25k for each arm.

## Analysis

- **Script:** `scripts/analyze_multiau_e1_dose.py`, committed with this entry before the jobs are submitted.
- **Checks:** it stops if any directory lacks the 90 × 11 units or if the units or commanded levels differ between directories.
- **Validation on E1's 25k data:**
  - the same directories at every step give d_gap 0 → constant;
  - pairing T against M at 10k and against itself at 25k → closing;
  - the reverse → widening.

```
python scripts/analyze_multiau_e1_dose.py \
  --t 10000=inference_output/batch_eval/e1/multiau_t_ema10000 15000=inference_output/batch_eval/e1/multiau_t_ema15000 \
      20000=inference_output/batch_eval/e1/multiau_t_ema20000 25000=inference_output/batch_eval/e1/multiau_t_ema25000 \
  --c 10000=inference_output/batch_eval/e1/multiau_c_ema10000 15000=inference_output/batch_eval/e1/multiau_c_ema15000 \
      20000=inference_output/batch_eval/e1/multiau_c_ema20000 25000=inference_output/batch_eval/e1/multiau_c_ema25000 \
  --out /petrobr/crono/matheus.aidano/e1_multiau/results/dose_report.json
```
