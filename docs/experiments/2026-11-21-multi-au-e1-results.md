# Multi-AU E1: results (November 21, 2026)

Results of the design pre-registered in [2026-11-19 multi-AU E1 design](2026-11-19-multi-au-e1-design.md). The readings, thresholds and decision rule were not changed after the results.
- **Analysis code frozen beforehand:** `scripts/analyze_multiau_e1.py` was committed in `71a236c` at 14:34. That was before any E1 sample existed: the training jobs had not written a checkpoint yet, and the evaluation jobs started at 16:20.
- **No code changes since:** tracked code is unchanged from `71a236c`.

**Outcome: Fail**, by the pre-registered rule, on two counts:
- **Reading 4 (smile non-inferiority) fails** on MediaPipe calibration: CLS Δ −0.064 [−0.108, −0.023]; the lower bound is past the −0.05 tolerance. T's smile ramps and permuted following are as good as C's or better.
- **Reading 2 (brow-lower control) fails:** T's mean ramp r is 0.20, below 0.5, although T beats C (Δ +0.195 [+0.071, +0.318]).

What passed:
- **brow-raise control** (reading 1: T 0.57 vs C 0.12);
- **brow calibration on both channels** (reading 3).

Reading 5 (localization) also fails: on LibreFace, T's brow raise follows the smile as much as C's. Follow-ups that are *not pre-registered* show this measure depends on the judge:
- py-feat and geometry do not see the brows rise with the smile, in T, C or M;
- LibreFace reads the same rise on real MEAD happy faces, where py-feat and geometry also see none.

The follow-ups also locate the smile loss: T's strongest smile is weaker than C's and M's on all three judges, while the order of its smiles is intact.

## Runs

| job | what | node / resources | wall time |
|---|---|---|---|
| 807386 | training, T | sdumont2nd2054, 4 H100 | 1 h 57 |
| 807387 | training, C | sdumont2nd2059, 4 H100 | 1 h 49 |
| 807389 | evaluation smoke: `raise_scaled` and `lower_calibration_jitter` end to end (generation, scoring, all-AU, LibreFace, analysis), from the smoke-training checkpoint | 2 H100 | 2 min |
| 807392 / 807394 | T / C: `calibration_jitter`, `permuted`, `raise_calibration_jitter`, `lower_calibration_jitter`, `ALLAU=1` | 4 H100 each | 3 h 49 / 3 h 50 |
| 807393 / 807395 | T / C: `scaled`, `raise_scaled`, `lower_scaled`, `raise_lower_scaled`, `ALLAU=1` | 4 H100 each | 31 min each |
| 807396 | M: all-AU re-scoring of its existing `scaled`, `calibration_jitter` and `permuted` samples. The batch script found all 1,380 samples and their scores by job name, so nothing was regenerated. | 3 H100 | 1 h 05 |
| 807397 | LibreFace on all 19 manifests (after the five jobs above) | 1 H100 + 96 CPUs | 10 min |

- **Checkpoints:** `output/multiau/{t,c}/expression_multiau_{t,c}.local-2026-11-21T14-31-57/checkpoints/checkpoint-step-25000-ema.ckpt`. All intermediate checkpoints (2.5k–25k, main and EMA) are kept.
- **Samples:** `inference_output/batch_eval/e1/multiau_{t,c}_ema25000/`.
- **Analysis:** in `/petrobr/crono/matheus.aidano/e1_multiau/results/`:
  - pre-registered: `e1_report.json` and `e1_report.log`;
  - the follow-ups below: `e1_posthoc.json` and `e1_posthoc.log`, from `/petrobr/crono/matheus.aidano/e1_multiau/posthoc_e1.py`.

## Data checks

- **Coverage:** every unit of every protocol has all four records (manifest, MediaPipe/py-feat/identity, all-AU py-feat + geometry, LibreFace): 120 per ramp protocol, 990 per calibration protocol and 270 for `permuted`, for T, C and M.
- **No-face rates:**
  - MediaPipe and LibreFace: 0 on every protocol.
  - facenet found fewer than 5 faces in at most 1.7% of T's units and at most 5.0% of C's (C `scaled`). The identity guardrail drops those units (n 118–119 of 120).
- **Logs:** no tracebacks or errors in any E1 job log.

## Pre-registered readings

Δ = T − C, paired over units, 2,000-draw cluster bootstrap 95% CI. Smile on MediaPipe, brows on LibreFace.

| # | reading | T | C | Δ [95% CI] | n | criterion | result |
|---|---|---|---|---|---|---|---|
| 1 | brow-raise ramp r (`raise_scaled`) | 0.566 | 0.119 | +0.447 [+0.330, +0.561] | 120 | lower > 0 and T ≥ 0.5 | **pass** |
| 2 | brow-lower ramp r (`lower_scaled`) | 0.201 | 0.006 | +0.195 [+0.071, +0.318] | 120 | lower > 0 and T ≥ 0.5 | **fail** (T < 0.5) |
| 3 | brow-raise CLS | 0.377 | −0.003 | +0.380 [+0.324, +0.436] | 90 | lower > 0 | pass |
| 3 | brow-lower CLS | 0.232 | −0.001 | +0.232 [+0.185, +0.286] | 90 | lower > 0 | pass |
| 4 | smile CLS (`calibration_jitter`) | 0.331 | 0.395 | −0.064 [−0.108, −0.023] | 90 | lower > −0.05 | **fail** |
| 4 | smile ramp r (`scaled`) | 0.780 | 0.704 | +0.075 [+0.028, +0.125] | 120 | lower > −0.05 | pass |
| 4 | smile permuted r | 0.650 | 0.626 | +0.025 [−0.031, +0.078] | 90 | lower > −0.05 | pass |
| 5 | smile → brow-raise leakage, Spearman (`scaled`) | 0.460 | 0.441 | +0.019 [−0.076, +0.111] | 120 | upper < 0 | **fail** |

On reading 5, C's own leakage is 0.441 [0.341, 0.534]. Its CI excludes 0, so the reading counts as failed, not uninformative.

## Decision

**Fail.** The rule reads: "neither brow channel passes, or reading 4 fails." Reading 4 fails, which settles it.
- Had reading 4 passed, the outcome would have been Partial, since only brow raise passes readings 1–2.
- The rule's consequence applies: write up before any further training.
- The main model (2026-10-29) stays the model of record.

## Guardrails

- **Identity** (facenet min-adjacent cosine):

  | protocol | T | C | Δ [95% CI] | flag |
  |---|---|---|---|---|
  | `scaled` | 0.898 | 0.872 | +0.026 [+0.017, +0.036] | — |
  | `raise_scaled` | 0.860 | 0.901 | −0.041 [−0.057, −0.026] | trade-off |
  | `lower_scaled` | 0.870 | 0.901 | −0.031 [−0.046, −0.018] | trade-off |

  The brow protocols compare unlike things. There, C receives only the held smile list, so every channel sits near neutral and its faces barely move, while T's brows move. Any motion lowers min-adjacent similarity. On the shared smile ramp, T's identity is better than C's.
- **Fail rate** (share of units with ramp r < 0.5): `scaled` T 0.11 / C 0.18; `raise_scaled` T 0.28 / C 0.70; `lower_scaled` T 0.60 / C 0.76. No unit had an undefined r.

## Secondary (pre-registered, descriptive)

- **Cost of the data change (C − M, smile, MediaPipe).** These costs come before any conditioning change. C trains on six emotions, and only 25% of its sequences target the smile.

  | metric | C − M | 95% CI |
  |---|---|---|
  | ramp r | −0.104 | [−0.161, −0.053] |
  | permuted r | −0.120 | [−0.167, −0.074] |
  | CLS | −0.097 | [−0.140, −0.050] |

  Relative to that C, T loses a further 0.064 in CLS and gains 0.075 in ramp r.
- **What brow commands drag** (Spearman with the command; Δ is T − C):

  | command | measure | T | Δ [95% CI] |
  |---|---|---|---|
  | brow raise | LibreFace AU12 | +0.215 | +0.226 [+0.109, +0.336] |
  | brow raise | MediaPipe smile | +0.065 | +0.136 [+0.027, +0.247] |
  | brow raise | AU26 (jaw drop) | +0.103 | +0.129 [+0.014, +0.246] |
  | brow lower | MediaPipe smile | +0.053 | +0.124 [+0.017, +0.234] |

  The other drags have CIs that include 0.
- **Leakage onto brow lower** (LibreFace AU04, `scaled`): T 0.199, C 0.080, Δ +0.120 [+0.006, +0.228]; M 0.231 (C − M −0.152 [−0.288, −0.013]).
- **Combined commands** (`raise_lower_scaled`, extrapolation):
  - T brow-raise ramp r 0.671 (C 0.119);
  - T brow-lower ramp r 0.313 (C 0.006).

  Both are higher than on the single-channel ramps.
- **The labeler's view** (py-feat; circular, since the training labels come from it):

  | metric | channel | T | C | Δ [95% CI] |
  |---|---|---|---|---|
  | CLS | brow raise | 0.485 | 0.007 | +0.478 [+0.423, +0.531] |
  | CLS | brow lower | 0.280 | −0.014 | +0.294 [+0.228, +0.364] |
  | ramp r | brow raise | 0.366 | 0.095 | +0.271 [+0.133, +0.411] |
  | ramp r | brow lower | 0.063 | 0.004 | +0.059 [−0.074, +0.181] |
- **Prompt groups** (prompts 14–23, which name an ethnicity, vs the rest; T's ramp r, Δ over C in brackets):

  | measure | ethnicity-named | rest |
  |---|---|---|
  | LibreFace brow raise | 0.589 (+0.512) | 0.558 (+0.425) |
  | geometry brow raise | 0.515 (+0.359) | 0.551 (+0.341) |
  | LibreFace brow lower | 0.244 (+0.283) | 0.186 (+0.166) |
  | geometry brow lower | 0.026 (+0.210, CI includes 0) | −0.007 (+0.276) |

  Brow raise is confirmed by geometry in both groups.

## Follow-ups (not pre-registered)

`posthoc_e1.py` runs these after the pre-registered analysis. None of them changes the decision. Its bootstrap uses its own draws, so a CI it shares with the pre-registered report can differ in the third decimal.

### 1. Smile → brow leakage depends on the judge

Per-unit Spearman of the smile command with each brow measure, on `scaled` (n 120, mean [95% CI]):

| measure | T | C | M | T − C |
|---|---|---|---|---|
| LibreFace brow raise (reading 5) | +0.460 [+0.370, +0.545] | +0.441 [+0.345, +0.530] | +0.432 [+0.338, +0.523] | +0.019 [−0.083, +0.107] |
| py-feat brow raise | +0.089 [−0.001, +0.186] | +0.178 [+0.079, +0.276] | +0.228 [+0.126, +0.325] | −0.089 [−0.217, +0.042] |
| geometry brow raise | −0.118 [−0.215, −0.016] | −0.086 [−0.191, +0.019] | −0.198 [−0.300, −0.093] | −0.032 [−0.147, +0.077] |
| LibreFace brow lower | +0.199 | +0.080 | +0.231 | +0.120 [+0.012, +0.227] |
| py-feat brow lower | −0.167 | −0.202 | −0.098 | +0.035 [−0.103, +0.172] |
| geometry brow lower | +0.164 | +0.152 | +0.322 | +0.012 [−0.096, +0.118] |

The three judges disagree about what the brows do during a smile:
- LibreFace reads them rising;
- geometry reads them falling (strongly on ethnicity-named prompts: T −0.46, M −0.44);
- py-feat reads little either way.

[2026-08-25](2026-08-25-au-coactivation.md) found the same disagreement on the 100k model.

On py-feat, which motivated the localization question (our AU01 leakage vs FineFace's), T's leakage is the lowest of the three models. Its Δ over C still includes 0, and py-feat is the labeler.

### 2. The judges disagree on real MEAD faces too

These are the training clips that T and C saw. All 139,774 frames of the happy and neutral clips have LibreFace scores from E0, and all but 2 have geometry scores. The table gives the within-clip Spearman of the py-feat smile label with each measure, averaged over the 4,028 happy clips:

| measure | mean ρ [95% CI] |
|---|---|
| brow-raise label (py-feat; what T was told) | −0.139 [−0.148, −0.130] |
| LibreFace brow raise | **+0.174** [+0.163, +0.185] |
| geometry brow raise | −0.100 [−0.110, −0.090] |
| brow-lower label (py-feat) | −0.129 [−0.138, −0.119] |
| LibreFace AU04 | +0.069 [+0.057, +0.081] |
| geometry brow lower | +0.432 [+0.424, +0.440] |
| LibreFace AU12 (reference) | +0.653 [+0.646, +0.661] |

- **LibreFace reads brows rising as real actors smile; py-feat and geometry read the opposite.** A model that reproduced real smiles would therefore show LibreFace "leakage". Reading 5 as specified cannot tell brow movement from LibreFace's response to smiling faces.
- **The eval condition is in T's training distribution.** Of the happy frames with smile label > 0.7 (62,092), 56% have a brow-raise label ≤ 0.05 (median 0.020, the neutral level). So holding the brow channels at 0.02 under a strong smile is a common case in T's data, not an extrapolation.

### 3. The smile loss is amplitude at the top of the scale

Mean detected smile per commanded level on `calibration_jitter` (90 units per level):

| judge | model | 0.0 | 0.5 | 0.8 | 0.9 | 1.0 | CLS |
|---|---|---|---|---|---|---|---|
| MediaPipe | T | −0.001 | 0.007 | 0.014 | 0.016 | 0.020 | 0.331 |
| | C | 0.000 | 0.008 | 0.015 | 0.020 | 0.027 | 0.395 |
| | M | −0.002 | 0.006 | 0.017 | 0.022 | 0.029 | 0.492 |
| py-feat AU12 | T | 0.310 | 0.475 | 0.621 | 0.672 | 0.716 | 0.464 |
| | C | 0.321 | 0.517 | 0.658 | 0.740 | 0.811 | 0.541 |
| | M | 0.333 | 0.498 | 0.683 | 0.797 | 0.874 | 0.606 |
| LibreFace AU12 | T | 0.425 | 0.733 | 0.922 | 1.062 | 1.213 | 0.320 |
| | C | 0.339 | 0.654 | 0.913 | 1.111 | 1.545 | 0.422 |
| | M | 0.419 | 0.713 | 1.208 | 1.507 | 1.811 | 0.508 |

- **The bottom and middle of the scale are close;** T's top end is compressed. At 0.9 and 1.0 the order is M > C > T on all three judges.
- **The spread across units at a fixed level is the same** for T and C (SD of the sample-mean MediaPipe smile 0.015–0.019). So CLS falls because the range shrinks, not because responses get noisier.
- **This is why ramp r passes while CLS fails.** Ramp and permuted r measure order within a clip and ignore scale.
- **It is a two-step loss:** the data change costs part of the amplitude (C < M), and the 3-key conditioning costs more (T < C).

### 4. Brow-channel response curves

Mean per commanded level (levels 0.0, 0.5, 0.8, 0.9, 1.0) on the brow calibration protocols:

| channel | judge | T | C |
|---|---|---|---|
| raise | LibreFace | 0.138, 0.266, 0.571, 0.738, 0.784 | 0.24–0.27, flat |
| raise | py-feat | 0.293, 0.331, 0.404, 0.421, 0.434 | ≈ 0.30, flat |
| raise | geometry | 0.112, 0.126, 0.140, 0.143, 0.144 | 0.117, flat |
| lower | LibreFace AU04 | 0.807, 1.427, 1.640, 1.650, 1.725 | ≈ 1.0, flat |
| lower | py-feat AU04 | 0.427, 0.494, 0.544, 0.563, 0.569 | ≈ 0.44, flat |
| lower | geometry inner-brow gap | 0.144, 0.140, 0.137, 0.136, 0.135 | 0.142, flat |

- **Both channels move the brows on every judge,** monotonically, including geometry. So brow control is not a LibreFace artifact.
- **Brow raise rises slowly, then steeply:** on LibreFace, 0.14 → 0.27 from 0 to 0.5, then 0.34 → 0.74 from 0.6 to 0.9.
- **Brow lower rises from the start and flattens above 0.8.** But within a 5-frame ramp the change is small next to the variation within and between videos: ramp r 0.20, fail rate 0.60.

### 5. C fills "no smile" with other emotions' brows

- **C's low smile is not neutral.** On `calibration_jitter`, C's LibreFace brow raise falls from 0.254 at smile 0 to 0.175 at smile 1, while T stays at 0.14–0.16 and M at 0.15–0.19.
- **C's neutral is not neutral either.** On the brow protocols, where every channel C receives sits near neutral, its faces read brow raise ≈ 0.25 and AU04 ≈ 1.0. At its zero brow commands, T reads 0.14 and 0.81.
- **Why:** a single smile channel trained on six emotions has nothing to tell "no smile" from "surprised, fear, angry or sad", all of which carry low smile labels. T's brow channels hold those faces at neutral.
- **Consistent with this,** C does worse than T on the smile ramp: fail rate 0.18 vs 0.11, identity 0.872 vs 0.898.

## What this leaves open

The rule asks for a write-up before further training. These are the questions a next step would have to answer; none is decided here.
- **Smile amplitude in T.** Is the top-end compression still shrinking at 25k, given that smile sequences are 25% of T's mix against 100% for the main model? The intermediate EMA checkpoints are kept, so a dose–response on `calibration_jitter` can answer this without training. If it is not shrinking, the levers are the smile share of the mix or the training length.
- **Brow lower within a clip.** Brow lower has a monotonic response across videos but weak ramps.
  - Its training ramps come from angry and sad clips, whose median label range is 0.51 against 0.77 for smile ramps (design, composition table).
  - Label quality per emotion has not been checked.
- **A judge for localization.** LibreFace's brow raise rises with the smile on real MEAD faces, so it cannot test whether a model adds brow movement to a smile. Any later localization claim needs a judge that is validated on that co-activation, for example by manual inspection of a frame sample, before the reading is set.
