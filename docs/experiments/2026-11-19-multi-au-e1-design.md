# Multi-AU E1: 3-channel training against an AU12-only arm (design, November 19, 2026)

Pre-registration. Follows [2026-11-18 multi-AU E0 results](2026-11-18-multi-au-e0-results.md), GO with 3 channels on both panels. This entry fixes the data, the arms, the recipe and the readings before any E1 output exists. Results go in a separate entry.

## Decisions (agreed 2026-11-19)

The E0 entry recommended the choices below; the user approved them on November 19.

- **Data:** all 47 MEAD actors, with W009 and W011 held out for validation as in May.
- **Emotions:** neutral, happy, surprised, fear, angry and sad. Contempt and disgusted are left out because py-feat's smile label fires on them (E0 follow-up 2).
- **Labels:** py-feat for all three channels, re-expressed per actor with the piecewise-linear map of `normalize_labels_per_actor.py` on each channel (the actor's neutral median → 0.02, 1 → 1).
  - smile = the `au12` field (the existing AU12 label);
  - brow raise = mean(AU01, AU02);
  - brow lower = AU04.
- **Judges:**
  - brow raise: LibreFace mean(AU01, AU02);
  - brow lower: LibreFace AU04 and FaceMesh geometry (−inner brow gap);
  - geometry's brow raise is reported only by prompt group, since it failed validity on dark faces in E0.

## Data

- **Root:** `/petrobr/crono/matheus.aidano/MEAD_multiau/`, built by `scripts/build_multiau_root.py --tag e1` from the two E0 preprocess roots. The frames are hard links, not copies, so the E0 inputs stay intact.
- **Training file:** `annotations/train_e1.json`, 45 actors, 21,779 clips, 522,680 frames.

  | neutral | happy | surprised | fear | angry | sad |
  |---|---|---|---|---|---|
  | 1,796 | 4,028 | 4,000 | 3,982 | 3,936 | 4,037 |

  - **Happy and neutral are the main model's clips:** normalized AU12 r 0.9985 with its labels over 139,774 shared frames (mean |Δ| 0.009). The E0 continuity check showed the same.
  - **The four new emotions have 3,936–4,037 clips each,** about happy's size.
- **Validation file:** `annotations/validation_e1.json`, W009 and W011, 962 clips. Training only logs 8 of its happy clips.
- **Dropped:** 16 frames without a face (py-feat `aus` null).
- **Neutral baselines** (raw, per actor): AU12 0.05–0.73 (median 0.24, as in 2026-10-29); brow raise 0.31–0.81 (0.54); brow lower 0.07–0.53 (0.20). All map to 0.02.
- **Normalized labels**, training frames (median / share < 0.1 / share > 0.5):

  | emotion | smile | brow raise | brow lower |
  |---|---|---|---|
  | neutral | 0.02 / 65% / 6% | 0.02 / 70% / 0% | 0.02 / 76% / 1% |
  | happy | 0.82 / 6% / 78% | 0.02 / 62% / 10% | 0.01 / 81% / 1% |
  | surprised | 0.02 / 66% / 8% | 0.61 / 4% / 68% | 0.02 / 84% / 0% |
  | fear | 0.05 / 57% / 15% | 0.54 / 12% / 55% | 0.02 / 68% / 3% |
  | angry | 0.11 / 49% / **22%** | 0.01 / 95% / 0% | 0.52 / 10% / 52% |
  | sad | 0.02 / 63% / 14% | 0.02 / 66% / 5% | 0.43 / 14% / 42% |

  Each brow channel is high where E0 placed it and near the neutral level elsewhere. The cost accepted with the emotion set is visible on angry: 22% of its frames carry a normalized smile above 0.5 (py-feat's residual smile, E0 follow-up 2), against 6% of neutral frames.

## Arms

| | T (`expression_multiau_t.yaml`) | C (`expression_multiau_c.yaml`) |
|---|---|---|
| conditioning keys | `au12`, `brow_raise`, `brow_lower` | `au12` |
| physical channels (3) | one key each | AU12 broadcast (the single-axis embedding) |
| CCL channels (3) | one per key, from its own prompt: `<smile intensity: …>`, `<brow raise intensity: …>`, `<brow lower intensity: …>` | the smile prompt's CCL, broadcast |

Everything else is shared, so the arms differ only in what the encoder is told:
- **Encoder:** `cin: 384` and the 6-channel layout are unchanged; no new parameters.
- **Recipe:** the main model's. Cosine LR 1e-4 → 0 without warmup, EMA 0.999, 25k steps on 1 node × 4 H100, effective batch 4, seed 42, checkpoints every 2.5k (main + EMA).
- **Frames:** sequence construction uses the emotion's channel (`ramp_keys`), not the conditioning keys, so with the same seed both arms see the same frames. A unit test checks this (`tests/test_multiau_conditioning.py`).
- **Sequence mix:** ramp 0.25, bridge 0.25, constant 0.3, permuted 0.2, as in the main model, extended to the new emotions:
  - **ramp / permuted:** a clip of one of the five non-neutral emotions, ordered by its emotion's channel (happy → smile; surprised and fear → brow raise; angry and sad → brow lower);
  - **bridge:** a neutral → X ramp of one actor, with X drawn uniformly from the five emotions and the frames ordered by X's channel;
  - **constant:** one channel drawn uniformly and held near a uniform level, from any clip.

  Conditioning is always the shown frames' own labels.

Simulated composition (20,000 draws from the training sampler; `/petrobr/crono/matheus.aidano/e1_multiau/sim_composition.py`):

| sequences (target channel) | share | target channel, mean label by frame slot | median range: smile / raise / lower |
|---|---|---|---|
| ramp, smile (happy) | 0.052 | 0.25, 0.63, 0.78, 0.85, 0.93 | 0.77 / 0.12 / 0.08 |
| ramp, brow raise (surprised, fear) | 0.102 | 0.32, 0.47, 0.53, 0.58, 0.67 | 0.29 / 0.32 / 0.11 |
| ramp, brow lower (angry, sad) | 0.098 | 0.21, 0.38, 0.47, 0.55, 0.72 | 0.36 / 0.01 / 0.51 |
| bridge, smile | 0.049 | 0.01, 0.06, 0.40, 0.76, 0.92 | 0.95 / 0.18 / 0.14 |
| bridge, brow raise | 0.100 | 0.02, 0.05, 0.37, 0.53, 0.67 | 0.33 / 0.70 / 0.16 |
| bridge, brow lower | 0.099 | 0.01, 0.04, 0.26, 0.47, 0.72 | 0.40 / 0.14 / 0.75 |
| constant, smile / raise / lower | 0.102 / 0.097 / 0.103 | ≈ 0.48 (uniform level) | held channel 0.11 / 0.08 / 0.11 |
| permuted, smile / raise / lower | 0.042 / 0.078 / 0.077 | ≈ 0.68 / 0.53 / 0.47 | as the ramps |

- **Share by channel:** smile-targeted sequences are 25% of the mix, against 100% for the main model. Brow raise and brow lower are ~38% each.
- **Bridges:** in 92% of them the first frame comes from the neutral clip and the last from the high clip.
- **Constant-pool support** (clips with 5 frames within ±0.1), per level:

  | level | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | smile | 16,111 | 17,200 | 7,072 | 5,237 | 4,220 | 3,816 | 3,711 | 3,828 | 4,357 | 4,488 | 3,013 |
  | brow raise | 12,894 | 14,159 | 6,131 | 5,216 | 4,853 | 5,017 | 5,411 | 5,123 | 3,419 | 1,122 | **21** |
  | brow lower | 14,834 | 16,368 | 8,286 | 6,066 | 5,248 | 4,870 | 4,527 | 3,984 | 3,070 | 1,634 | 407 |

  Brow raise at 1.0 rests on 21 clips, so the top of its scale is thin.

**Primary checkpoint:** the EMA checkpoint at 25k of each arm, fixed now.

**Reference model (M):** the main model (2026-10-29 low-end treatment, EMA 25k). It is trained on happy + neutral only, and it enters the smile readings as a third reference, not as an arm.

## Verification before submission

- **Regression against HEAD (`fea0aea`).** Under the main model's settings, the new dataset module gives the same results as HEAD's, with stub CLIP and synthetic clips:
  - 300 seeded training draws give identical frames, labels and 6-channel embeddings;
  - all 36 validation clips are identical, with and without fixed `intensity_list` targets.

  The check is `/petrobr/crono/matheus.aidano/e1_multiau/checks/check_equivalence.py`, against HEAD's module saved next to it.

  Every existing `batch_eval.py` experiment keeps its job names, command lists and fields.
- **Unit tests:**
  - `tests/test_multiau_conditioning.py`, 10 pass:
    - one key per physical channel;
    - one CCL map per template, and a single template reproduces the single-axis embedding;
    - ramps are ordered by the emotion's channel;
    - bridges cover every high emotion within one actor;
    - constants draw every channel within tolerance;
    - arms that differ only in `label_keys` see the same frames;
    - validation targets drive the clip's own channel;
    - mismatched keys and templates are rejected.
  - The 11 earlier tests still pass.
- **Analysis code on synthetic data.** `analyze_multiau_e1.py` returns:
  - "Full success" when T follows brow commands and C's brows follow the smile;
  - "Fail" when T behaves like C.
- **Training smoke, T.** 8 steps on 2 H100 (h100_dev):
  - The first attempt (804035) failed in validation: the validation records carry a fixed smile `intensity_list`, which the 3-key path rejected. The fix applies the list to the clip's own channel and keeps the frames' labels on the others.
  - The rerun (804241) passed: the LR follows the cosine to 0 at step 8, main + EMA checkpoints were written at steps 4 and 8, a validation sample was saved, and data time is ~0 after the first step.
- **Training smoke, C.** Job 804249 passed, with the same checks. Its losses match T's at step 1 (0.5186) and differ only in the third decimal after it (0.0247 vs 0.0251 at step 2). Same frames and noise, different conditioning, as designed.

## Submission (November 21, 2026)

- **T:** job **807386** (sdumont2nd2054). **C:** job **807387** (sdumont2nd2059).
- Each runs on 1 node × 4 H100: `train_expression_sdumont2nd.srm --nodes=1 --time=05:00:00`, excluding nodes 2047, 2049–2051 and 2056, with the local copies of the two configs.
- **Code at submission:** uncommitted, on top of HEAD `fea0aea`. `output/multiau/code_snapshot_2026-11-21/` holds:
  - HEAD and `git status`;
  - the diff of tracked files;
  - a tar of the new files;
  - the two local configs.
- **Delay:** the smoke tests passed on November 19, but training was submitted only on November 21. Two agent sessions had run on different login nodes at once, so work stopped until one remained.

## Pre-registered evaluation

**Units** are the "fresh" units of [2026-11-14](2026-11-14-longer-training-results.md), so that M's existing samples can be reused:
- `calibration_jitter`, `permuted` and the brow `*_calibration_jitter` protocols: prompts 10–39 × seeds 48–50 (90 units; 11 levels each for calibration);
- `scaled` and the brow `*_scaled` protocols: 40 prompts × seeds 48–50 (120 units).

**Commands** (`scripts/batch_eval.py`):
- **Varied channel:** each protocol varies one channel with its usual list:
  - smile protocols as before;
  - `raise_scaled` / `lower_scaled`: the ascending ramp on that brow channel;
  - `raise_calibration_jitter` / `lower_calibration_jitter`: the 11 jittered levels;
  - `raise_lower_scaled`: both brow channels ramp together.
- **Held channels:** every other channel is held at 0.02 ± 0.02 jitter. 0.02 is the label of the actor's neutral face after normalization, and the jitter keeps the CCL non-zero, as in training. The held lists depend only on the unit, so they are identical across models.
- **C on brow protocols:** C cannot receive brow commands. On those protocols it generates from the held smile list on the same units and noise, which makes it the matched null for T.

**Measurements** (per frame):

| | primary judge | also reported |
|---|---|---|
| smile | MediaPipe mouth-corner proxy (label-independent) | py-feat AU12 (the labeler), LibreFace AU12 |
| brow raise | **LibreFace mean(AU01, AU02)** | py-feat mean(AU01, AU02); geometry brow raise, by prompt group only |
| brow lower | **LibreFace AU04** | geometry −inner brow gap; py-feat AU04 |
| identity | facenet min-adjacent cosine | mean-adjacent |

The no-face rate of each detector is reported next to every metric (the lesson of 2026-11-14).

**Statistics:** Δ = T − C, paired over units, with a 2,000-draw bootstrap 95% CI. A ramp r is the per-unit Pearson r between the commanded list and the detected values. CLS is the Pearson r between the commanded level and the per-frame detected value, pooled over the units of a calibration protocol. Leakage is the per-unit Spearman ρ between the smile command and a brow measure along a smile ramp, as in [2026-08-25](2026-08-25-au-coactivation.md).

### Readings

1. **Brow-raise control.** On `raise_scaled`, LibreFace brow-raise ramp r. Passes if the Δ CI lower bound is > 0 and T's mean r is ≥ 0.5, the ramp-failure threshold of the smile evaluation.
2. **Brow-lower control.** The same on `lower_scaled` with LibreFace AU04.
3. **Brow calibration.** LibreFace CLS on `raise_calibration_jitter` and `lower_calibration_jitter`. Passes per channel if the Δ CI lower bound is > 0.
4. **Smile kept (non-inferiority).** Each of these must have a Δ CI lower bound > −0.05, the 2026-10-28 tolerance:
   - MediaPipe CLS on `calibration_jitter`;
   - MediaPipe ramp r on `scaled`;
   - MediaPipe r on `permuted`.
5. **Localization (the motivation).** On `scaled`, the leakage of the smile command onto LibreFace brow raise. Passes if Δ (T − C) has a CI upper bound < 0, i.e. T's brows move less with the smile. If C's own leakage CI includes 0, there is nothing to remove, and the reading is reported as uninformative rather than failed.

**Guardrails** (reported with every reading):
- **Identity:** min-adjacent cosine Δ on `scaled`, `raise_scaled` and `lower_scaled`. A drop larger than 0.03 with a CI excluding 0 is reported as a trade-off.
- **Fail rate:** the share of units with ramp r < 0.5, per channel.
- **No-face rate:** per detector.

**Secondary (descriptive):**
- **Cost of the data change:** C − M on the smile protocols (same units). C trains on six emotions and sees py-feat's residual smile on angry, fear and sad.
- **What brow commands drag:** on `raise_scaled` / `lower_scaled`, Spearman of the brow command with MediaPipe smile, LibreFace AU12, AU25 and AU26 (mouth opening; brow raise comes from surprised and fear clips).
- **Leakage onto brow lower:** the leakage of reading 5 on LibreFace AU04, and on both brow measures for M.
- **Combined commands** (`raise_lower_scaled`): LibreFace brow-raise and brow-lower ramp r. Raise and lower co-occur in < 1% of training frames, so this is extrapolation.
- **The labeler's view:** py-feat versions of readings 1–3 (circular: the training labels are py-feat).
- **Prompt groups:** every brow metric split into ethnicity-named prompts (`prompt_idx` 14–23) and the rest. Geometry's brow raise is reported only within these groups.
- **Trajectory:** none planned. Intermediate EMA checkpoints are kept, so a dose–response can be added later as a separate entry.

### Decision

- **Full success:** readings 1, 2, 4 and 5 pass. The 3-channel model becomes the multi-AU line, and the next step widens the AU set toward FineFace's.
- **Brow control without localization:** readings 1, 2 and 4 pass, but 5 fails. The brow channels add control but do not remove the smile's brow drag. The leakage measures then decide whether conditioning or data drives the drag.
- **Partial:** only one brow channel passes 1 or 2 (with 4 passing). That channel is kept; the other is examined (data share, label quality) before any retrain.
- **Fail:** neither brow channel passes, or reading 4 fails. Write up before any further training.

Reading 3 is reported with the decision but does not change it: ramp control can exist without absolute calibration, as the smile history showed (constant commands needed dithering, 2026-10-07).

## Evaluation runs (planned)

- **Generation and scoring:** `scripts/slurm/batch_eval_sdumont2nd.srm` with `ALLAU=1` (py-feat all AUs + geometry after the usual scoring). Per model, 8 experiments (`scaled`, `calibration_jitter`, `permuted`, `raise_scaled`, `raise_calibration_jitter`, `lower_scaled`, `lower_calibration_jitter`, `raise_lower_scaled`) as two 4-GPU jobs, `SUBSET_OFFSET=10 SUBSET_PROMPTS=30 SEEDS=48,49,50`.
- **LibreFace:** `scripts/slurm/rescore_libreface_sdumont2nd.srm` on every manifest, M's three included.
- **Analysis:** `scripts/analyze_multiau_e1.py`, written and committed before any E1 sample is scored.
