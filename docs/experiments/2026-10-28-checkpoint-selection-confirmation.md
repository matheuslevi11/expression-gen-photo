# Aug-arm checkpoint trajectory, held-out confirmation and weight averaging (October 28, 2026)

Follow-up to [2026-10-28 FineFace on both detectors](2026-10-28-fineface-both-detectors.md), next step 1: find an augmented checkpoint that matches FineFace on calibration and ramp-following on both detectors at once.

## Part 1 — trajectory on the original units

The aug arm's 15k and 20k checkpoints (jobs 747013/747014) on `calibration_jitter`, `scaled` and `permuted`, the protocols 10k and 25k already had. Units are the usual ones: prompts 0–9 (40 for scaled) × seeds 42–44. Δ vs FineFace is paired (2,000-draw bootstrap); "fail" = ramps with AU12 r < 0.5; frame 0 = py-feat AU12 of the ramp's first (c = 0) frame.

| checkpoint | CLS py-feat | CLS MediaPipe (Δ vs FineFace) | ramp AU12 r | ramp MediaPipe r (Δ vs FineFace) | permuted AU12 / MP r | id min-adj | fail | frame 0 |
|---|---|---|---|---|---|---|---|---|
| FineFace | 0.497 | 0.494 | 0.786 | 0.880 | — | 0.756 | 8.3% | 0.156 |
| original 25k | 0.501 | 0.295 (−0.200 [−0.32, −0.09]) | 0.855 | 0.856 (−0.024 [−0.08, +0.03]) | — | 0.907 | 2.5% | 0.301 |
| aug 10k | 0.496 | 0.326 (−0.168 [−0.32, −0.02]) | 0.789 | 0.763 (−0.116 [−0.17, −0.06]) | 0.491 / 0.553 | 0.920 | 9.2% | 0.330 |
| **aug 15k** | 0.567 | **0.400 (−0.094 [−0.25, +0.06])** | 0.801 | **0.838 (−0.042 [−0.09, +0.01])** | **0.726 / 0.787** | 0.893 | 7.5% | 0.343 |
| aug 20k | 0.558 | 0.348 (−0.146 [−0.29, −0.01]) | 0.795 | 0.838 (−0.043 [−0.09, +0.01]) | 0.572 / 0.675 | 0.905 | 7.5% | 0.439 |
| aug 25k | 0.625 | 0.491 (−0.003 [−0.13, +0.13]) | 0.631 | 0.797 (−0.084 [−0.14, −0.02]) | 0.489 / 0.566 | 0.923 | 24.2% | 0.532 |

Aug 15k paired against its neighbours (Δ = 15k − other):

| vs | CLS MediaPipe | ramp MediaPipe r | permuted AU12 r | id min-adj (scaled) |
|---|---|---|---|---|
| aug 10k | +0.074 [−0.006, +0.149] | +0.075 [+0.036, +0.115] | +0.235 [+0.152, +0.319] | −0.026 [−0.037, −0.015] |
| aug 20k | +0.052 [−0.019, +0.123] | +0.001 [−0.032, +0.033] | +0.154 [+0.070, +0.243] | −0.012 [−0.023, −0.001] |
| aug 25k | −0.091 [−0.168, −0.005] | +0.042 [−0.004, +0.088] | +0.237 [+0.131, +0.354] | −0.030 [−0.041, −0.019] |
| original 25k | +0.106 [+0.012, +0.208] | −0.018 [−0.052, +0.018] | — | −0.014 [−0.026, −0.003] |

Observations:

- **Aug 15k is the only checkpoint whose MediaPipe CIs vs FineFace include 0 on both calibration and ramps.** The CIs are wide (±0.15 on CLS), so this means "not distinguishable", not "equal".
- **Permuted following peaks sharply at 15k** (0.73, +0.15 to +0.24 over every neighbour). It is the only axis where 15k separates from 20k. Its cost is identity: worst-case adjacent cosine 0.01–0.03 lower on ramps and 0.05 lower on permuted lists, still far above FineFace.
- **Metrics swing between checkpoints 5k steps apart** (permuted 0.49 → 0.73 → 0.57 → 0.49; MediaPipe CLS 0.33 → 0.40 → 0.35 → 0.49), so a single run cannot tell a property of the recipe from luck.
- **Degradation has one signature in all three series:** the original lineage 25k → 100k (frame 0 0.30 → 0.54), the cluster ctrl after 10k (0.42 → 0.56, fail 6% → 23%) and the aug arm after 20k (0.34 → 0.53, fail 8% → 24%). The ramp's first frame stops being neutral. The recipe has constant LR 1e-4, no decay and no EMA, and the drift is the recipe's, not the augmentation's.

**Selection caveat.** 15k was picked by looking at 4 checkpoints × ~6 metrics on the same 30 units it would be reported on, which inflates its numbers. Hence Part 2.

## Part 2 — design (fixed before results)

Three models on **held-out units**:

- aug 15k;
- **aug avg(15k, 20k)**, a uniform weight average (`scripts/average_checkpoints.py`; encoder + attention merge, optimizer state dropped; verified = (a + b)/2 to float32 rounding, ‖15k − 20k‖ = 37.2 encoder / 20.3 merge);
- original 25k as the reference.

Units:

- `calibration_jitter` and `permuted` on bank prompts **10–29** × seeds **45–47**: 60 units, 660 and 180 samples. None of these units were used for any selection.
- `scaled` on all 40 prompts × seeds 45–47 (new noise, same prompts).

`batch_eval.py generate --prompt-offset` and the launcher's `SUBSET_OFFSET`/`SUBSET_PROMPTS` were added for this. Offset 0 reproduces every existing job id.

FineFace is not rerun: it needs the workstation env, and its July units are prompts 0–9 × seeds 42–44. Comparisons to it are therefore **unpaired**: FineFace's original-unit estimate against our held-out estimate with its bootstrap CI.

Readings, fixed now:

1. **Selection check.** Aug 15k's held-out MediaPipe CLS, ramp MediaPipe r and permuted r should fall within their original-unit CIs. A drop below them means the 15k pick was inflated.
2. **Calibration remedy holds.** Aug 15k vs original 25k (paired, held-out): MediaPipe CLS gain with CI excluding 0, and ramp MediaPipe r no worse than −0.05.
3. **Averaging.** avg vs aug 15k (paired, held-out). If avg is no worse on any main metric (no CI excluding a loss) and has a lower fail rate or frame 0, averaging is a free stabiliser.
4. **Decision on retraining** (aug recipe with LR decay + EMA, checkpoints every 1k):
   - Go if reading 2 holds: the recipe has a good region worth stabilising.
   - If reading 2 fails, the 15k "balance" was selection noise, and the next lever is low-end data (MEAD neutral clips) rather than optimisation.

## Part 2 — results (jobs 747343–747345, 960 samples per model, all GPU speed)

Held-out estimates (95% CI over units):

| model | CLS py-feat | CLS MediaPipe | ramp AU12 r | ramp MediaPipe r | permuted AU12 / MP r | id min-adj ramp / permuted | fail | frame 0 |
|---|---|---|---|---|---|---|---|---|
| original 25k | 0.560 | 0.383 [0.32, 0.46] | 0.797 | 0.853 [0.81, 0.89] | 0.399 / 0.410 | 0.897 / 0.919 | 9.2% | 0.243 |
| aug 15k | 0.626 | **0.447 [0.37, 0.53]** | 0.748 | 0.802 [0.75, 0.85] | **0.676 / 0.717** | 0.891 / 0.861 | 15.0% | 0.354 |
| aug avg(15k, 20k) | 0.594 | 0.408 [0.34, 0.49] | 0.669 | 0.780 [0.72, 0.83] | 0.540 / 0.661 | 0.897 / 0.889 | 20.8% | 0.407 |

Readings against the fixed criteria:

1. **Selection check: partial inflation, on ramps.** Aug 15k's calibration (MediaPipe CLS 0.40 → 0.45) and permuted AU12 r (0.73 → 0.68) stay within their original-unit CIs. Its ramp AU12 r (0.80 → 0.75), ramp MediaPipe r (0.84 → 0.80) and permuted MediaPipe r (0.79 → 0.72) fall below them. Part of the ramp drop is the units, not the pick: the original 25k also loses ramp AU12 r on these seeds (0.855 → 0.797, fail 2.5% → 9.2%). Its ramp MediaPipe r does not drop (0.856 → 0.853), so about 0.03 of aug 15k's MediaPipe drop is selection. Calibration is easier on prompts 10–29 for both models.
2. **Calibration remedy: met on one criterion by a hair, missed on the other by a hair.**
   - Aug 15k − original 25k, paired over 60 held-out units: MediaPipe CLS **+0.064 [+0.00001, +0.134]** (py-feat +0.066 [+0.001, +0.136]). The CI excludes 0, but this is about half the original-unit gain (+0.106).
   - Ramp MediaPipe r **−0.051 [−0.098, −0.007]**, just past the −0.05 tolerance. Ramp AU12 r −0.050 [−0.101, +0.003] and ramp identity −0.006 [−0.016, +0.004].
   - Read honestly, this is a real but small calibration gain bought with a small ramp cost: a trade-off, not a free improvement.
3. **Averaging does not stabilise; it inherits the 20k's drift.** avg − aug 15k: ramp AU12 r −0.079 [−0.136, −0.023], permuted AU12 r −0.136 [−0.201, −0.073], CLS −0.03/−0.04 (CIs include 0). Fail rate is 20.8% vs 15.0% and frame 0 is 0.41 vs 0.35. The only gain is identity on permuted lists (+0.028). With weights 5k steps apart under constant LR, the average sits between the two and takes on the later checkpoint's lost neutral start.

Beyond the criteria:

- **The robust effect of the augmentation is non-monotonic following.** Permuted AU12 r is +0.278 [+0.194, +0.361] and MediaPipe r +0.307 [+0.240, +0.373] over the original 25k on fresh units (0.68 vs 0.40), consistent with 2026-10-02 (+0.33 vs its ctrl). The cost is worst-case identity on those lists (−0.058 [−0.071, −0.045]).
- **Unpaired vs FineFace** (FineFace's prompts 0–9 estimate against our held-out CI):
  - Aug 15k's MediaPipe CLS CI contains FineFace's 0.494, but its ramp MediaPipe r CI (0.75–0.85) excludes FineFace's 0.880.
  - The original 25k is the mirror image: ramps contain 0.880, while calibration (0.32–0.46) excludes 0.494.
  - On held-out units, then, **neither checkpoint matches FineFace on both axes**. The 15k "balance" of Part 1 was partly selection.

## Decision

The fixed rule (retrain with LR decay + EMA only if reading 2 holds) is not met cleanly: reading 2 sits on its thresholds. The averaging result also weakens the stabilisation case, because the late drift is not undone by mixing weights. The bottleneck shared by every checkpoint is the **bottom of the scale** (frame 0 ≥ 0.24, constant c = 0 ≥ 0.37 py-feat vs FineFace's 0.05–0.16), which is data. Recommendation (to be agreed): do not run a standalone stabilisation retrain. Fold LR decay + EMA + 1k checkpoints into the next retrain that low-end data (MEAD neutral clips) will need anyway. For reporting, use the original 25k as the ramp model and aug 15k as the calibration/trajectory variant, with the held-out numbers above.

Artifacts (gitignored): `inference_output/batch_eval/calib_aug_{15000,20000}/`, `inference_output/batch_eval/heldout/{aug_15000,aug_avg15k20k,orig_25k}/{calibration_jitter,scaled,permuted}/`, `heldout/heldout_report.json`, `trajectory_vs_fineface.json`, `aug15k_vs_*.json`. Averaged checkpoint: `output/calibration/aug/<run>/checkpoints/avg-step-15000-20000.ckpt`. Inference configs `configs_local/infer_calib_aug_{15000,20000,avg15k20k}.local.yaml` (outside git).
