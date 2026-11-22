# Multi-AU E1 dose–response: results (November 22, 2026)

Results of the design pre-registered in [2026-11-21 E1 dose–response design](2026-11-21-multi-au-e1-dose-response-design.md).
- **Frozen beforehand:** the reading, the rule and `scripts/analyze_multiau_e1_dose.py` were committed in `af7b797` before the jobs were submitted, and nothing was changed after.
- **Follow-up:** one check, marked *not pre-registered*, was added after the results.

**Outcome: constant.**
- **No detectable change in the gap:** on MediaPipe, the T − C smile gap does not change between 10k and 25k on the response slope or on CLS.
- **The pre-registered consequence:** the smile cost is not a training-length effect at this recipe, so a longer T schedule is not expected to remove it. The candidates change the mix or the conditioning.

The trajectory adds two things the single contrast hides:
- **The gap is not steady:**
  - at 10k it comes from T's raised bottom of the scale;
  - at 15k–20k it nearly closes;
  - from 20k to 25k it reopens, because C keeps gaining at the top while T stops.
- **Every gap here comes from one training run per arm.** The main model's seed replicate shows that the evaluation-unit bootstrap does not cover run-to-run variation. E1's 25k gap is still well beyond that variation on CLS.

## Runs

| job | checkpoint (EMA) | node | wall time |
|---|---|---|---|
| 807762 / 807763 / 807764 | T 10k / 15k / 20k | sdumont2nd2045 | 2 h 17 / 2 h 18 / 2 h 10 |
| 807765 / 807766 / 807767 | C 10k / 15k / 20k | sdumont2nd2045 / 2046 / 2046 | 2 h 12 / 2 h 19 / 2 h 19 |
| 807768 | LibreFace on the six manifests | sdumont2nd2045 | 5 min |

- **Resources:** 1 H100 and 24 CPUs per evaluation job; `calibration_jitter` only, no all-AU scoring.
- **Coverage:** 990 of 990 samples scored by all three judges in every directory; the 25k samples are E1's.
- **Logs:** no errors in any job log.
- **Analysis files:** in `/petrobr/crono/matheus.aidano/e1_multiau/results/`: `dose_report.json` and `dose_report.log` (pre-registered); `dose_posthoc.log` from `../posthoc_dose.py` (the follow-up).

## Pre-registered reading

Response slope = per-unit OLS slope of the detected smile on the commanded level. CIs are 2,000-draw unit bootstraps (90 units), paired across arms and steps.

**MediaPipe (primary):**

| step | LR | slope T | slope C | gap T − C | CLS T | CLS C | CLS gap |
|---|---|---|---|---|---|---|---|
| 10k | 6.5e-5 | 0.0114 | 0.0141 | −0.0027 [−0.0048, −0.0008] | 0.190 | 0.237 | −0.047 [−0.086, −0.013] |
| 15k | 3.5e-5 | 0.0144 | 0.0162 | −0.0018 [−0.0038, +0.0002] | 0.250 | 0.275 | −0.025 [−0.061, +0.011] |
| 20k | 9.5e-6 | 0.0190 | 0.0204 | −0.0014 [−0.0038, +0.0008] | 0.319 | 0.334 | −0.015 [−0.059, +0.029] |
| 25k | 0 | 0.0201 | 0.0254 | −0.0053 [−0.0078, −0.0030] | 0.331 | 0.395 | −0.064 [−0.106, −0.023] |

**Primary contrast, gap(25k) − gap(10k):**
- slope: −0.0026 [−0.0054, +0.0003] → constant;
- CLS: −0.017 [−0.066, +0.035] → constant.

The two agree, so the outcome is **constant**.

## Also reported (pre-registered, descriptive)

**The other judges disagree with each other on the contrast,** so neither changes the primary reading:

| judge | slope gap at 10k / 15k / 20k / 25k | slope contrast 25k − 10k | CLS contrast 25k − 10k |
|---|---|---|---|
| py-feat AU12 | −0.113 / −0.027 / −0.036 / −0.053 | +0.060 [+0.007, +0.114] (closing) | +0.051 [−0.006, +0.111] (constant) |
| LibreFace AU12 | −0.033 / −0.009 / −0.068 / −0.253 | −0.221 [−0.400, −0.035] (widening) | −0.061 [−0.140, +0.026] (constant) |

**What changes at 10k is the bottom of the scale.** T's smile at c = 0 is higher than C's on py-feat and LibreFace, and the two curves converge by 15k:

| judge | arm | c = 0 at 10k / 15k / 20k / 25k |
|---|---|---|
| py-feat | T | 0.43 / 0.33 / 0.32 / 0.31 |
| py-feat | C | 0.33 / 0.33 / 0.31 / 0.32 |
| LibreFace | T | 0.58 / 0.40 / 0.40 / 0.43 |
| LibreFace | C | 0.42 / 0.42 / 0.36 / 0.34 |

Early in training, T's "no smile" is not yet neutral. py-feat's contrast reads "closing" mostly because this early floor gap disappears. At 10k, T's py-feat range is 0.14 against C's 0.29; the floor accounts for 0.10 of that difference and the top for 0.05. The gap at the top is a different, later effect.

**What changes from 20k to 25k is the top, and only for C** (slope at 25k minus slope at 20k):

| judge | T | C |
|---|---|---|
| MediaPipe | +0.0011 [−0.0002, +0.0024] | +0.0050 [+0.0038, +0.0062] |
| py-feat | +0.025 [−0.005, +0.051] | +0.041 [+0.015, +0.066] |
| LibreFace | +0.011 [−0.077, +0.097] | +0.195 [+0.130, +0.256] |

Smile at c = 1 (MediaPipe / py-feat / LibreFace):
- **C:** 0.024 / 0.774 / 1.32 at 20k, rising to 0.027 / 0.811 / 1.55 at 25k;
- **T:** 0.019 / 0.696 / 1.17 at 20k and 0.020 / 0.716 / 1.21 at 25k.

In the last 5k steps (LR below 1e-5), C keeps widening its smile range and T stops. E1's 25k gap opens mostly in this stretch. LibreFace's "widening" reflects the same thing.

## Follow-up (not pre-registered): run-to-run yardstick

The main model has a seed replicate (seed 142, [2026-11-14](2026-11-14-longer-training-results.md)) evaluated on the same `calibration_jitter` units. LibreFace did not score it.

| judge | seed 42 − seed 142: slope | CLS |
|---|---|---|
| MediaPipe | −0.0020 [−0.0038, −0.0002] | +0.0048 [−0.031, +0.038] |
| py-feat | −0.011 [−0.051, +0.031] | −0.0015 [−0.046, +0.044] |

**What it shows:**
- **The unit bootstrap misses run-to-run variation.** A pure seed difference of one recipe gives a MediaPipe slope CI that excludes 0. The CIs in this entry and in E1 cover evaluation units, not training runs.
- **E1's 25k gap is well beyond this one draw,** especially on CLS:

  | judge | slope gap at 25k vs replicate | CLS gap at 25k vs replicate |
  |---|---|---|
  | MediaPipe | −0.0053 vs −0.0020 (2.6×) | −0.064 vs 0.005 (13×) |
  | py-feat | −0.053 vs −0.011 | −0.078 vs −0.002 |
- **The 15k–20k gaps are about the replicate's size on the slope** (−0.0018, −0.0014).

The replicate is one draw of the run-to-run distribution, so these ratios are indicative.

## What this means

- **Pre-registered:** constant. Training T longer on this recipe is not expected to close its smile gap to C.
- **The trajectory points the same way.** T's smile range stops growing after 20k on all three judges, while C's keeps growing until the schedule ends. Within this schedule, T saturates earlier. A longer cosine reshapes the whole schedule, so this does not strictly predict a longer run. For the main model, a 50k schedule did not raise MediaPipe calibration either (2026-11-14).
- **Next candidates (not decided):**
  - **Raise the smile share of T's mix** (25% now, against 100% for the main model).
  - **Change how the three channels are conditioned.** No specific design exists yet; the CCL is already one map per channel.

  Either needs its own pre-registration. A second training seed per arm would be needed to make per-checkpoint T − C differences robust to run-to-run variation.
