# FineFace head-to-head on both detectors, with dithered constant commands (October 28, 2026)

Closes the open half of the [2026-10-07](2026-10-07-ccl-zero-test.md) reversal: our dithered calibration matched FineFace on py-feat (0.501 vs 0.497), but FineFace's MediaPipe CLS was never computed, and py-feat is circular for us (our training labels come from it). Also reruns the July protocols on the **100k checkpoint the thesis cites**, now with dither.

## Design

- **FineFace:** the July results ([2026-07-21](2026-07-21-fineface-headtohead.md)), copied from the workstation (`inference_output/batch_eval/fineface/results.jsonl`, split per protocol). Calibration = 10 prompts × 3 seeds × 11 levels, one still per (prompt, seed, level) at exact AU12 = 5c (no CCL, so nothing to dither); scaled = 40 × 3 five-still sweeps. Per-frame `au12` and `mp_smile` were already scored by the same `batch_eval.py score` pipeline.
- **Ours:** `calibration_jitter` (≤0.02 per-frame dither) and `scaled` results already on the cluster for original 25k, aug 10k and aug 25k; new run for the July 100k checkpoint (job 746699; sha256 `587b01fd…74be5b` on the cluster), also with exact `calibration` to tie it to July on the same hardware.
- **Pairing:** `compare_eval.py --exp-b calibration_jitter`, paired by (prompt, seed, level) — the same prompt bank indices, seeds and levels on both sides; 2,000-draw bootstrap over the 30 (prompt, seed) units (120 for scaled). CLS is pooled Pearson of (commanded level, detected per frame): FineFace contributes 1 frame per level, we contribute 5. `compare_eval.py` now prints the by-level curve for MediaPipe as well as py-feat.
- **Reading rule:** MediaPipe is the claim-bearing detector for calibration (label-independent); py-feat is reported but circular, more so for the aug arm, which is trained to reproduce absolute py-feat values.

## Results

Calibration (CLS), ours − FineFace:

| model | CLS py-feat | Δ (95% CI) | CLS MediaPipe | Δ (95% CI) |
|---|---|---|---|---|
| **FineFace** | 0.497 | — | **0.494** | — |
| July 100k, exact (July protocol, workstation) | 0.075 | −0.422 [−0.547, −0.294] | 0.015 | −0.479 [−0.600, −0.375] |
| **July 100k, dithered** | 0.303 | **−0.194 [−0.285, −0.085]** | 0.257 | **−0.237 [−0.357, −0.139]** |
| original 25k, dithered | 0.501 | +0.004 [−0.115, +0.127] | 0.295 | **−0.200 [−0.316, −0.093]** |
| cluster aug 10k, dithered | 0.496 | −0.001 [−0.112, +0.113] | 0.326 | −0.168 [−0.317, −0.022] |
| cluster aug 25k, dithered | 0.625 | +0.128 [+0.009, +0.239] | **0.491** | −0.003 [−0.134, +0.131] |

Ramp-following and identity (`scaled`, n = 120), ours − FineFace (FineFace: AU12 r 0.786, MediaPipe r 0.880, id min-adjacent 0.756):

| model | AU12 r | Δ | MediaPipe r | Δ | id min-adj Δ |
|---|---|---|---|---|---|
| July 100k (workstation) | 0.774 | −0.011 [−0.080, +0.065] | 0.829 | −0.052 [−0.110, +0.010] | +0.121 [+0.083, +0.160] |
| original 25k | 0.855 | +0.070 [+0.013, +0.135] | 0.856 | −0.024 [−0.078, +0.033] | +0.152 [+0.118, +0.192] |
| cluster aug 10k | 0.789 | +0.004 [−0.057, +0.069] | 0.763 | −0.116 [−0.168, −0.056] | +0.161 [+0.127, +0.201] |
| cluster aug 25k | 0.631 | −0.155 [−0.241, −0.070] | 0.797 | −0.084 [−0.143, −0.023] | +0.167 [+0.134, +0.206] |

Detected smile by commanded level (FineFace exact stills; ours dithered clips):

| c | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| py-feat FineFace | .054 | .067 | .085 | .142 | .159 | .224 | .268 | .326 | .390 | .468 | .573 |
| py-feat July 100k | .717 | .723 | .733 | .703 | .711 | .733 | .782 | .784 | .830 | .876 | .888 |
| py-feat original 25k | .629 | .439 | .537 | .542 | .690 | .738 | .799 | .805 | .816 | .840 | .844 |
| py-feat aug 25k | .427 | .405 | .556 | .572 | .606 | .665 | .750 | .762 | .819 | .863 | .884 |
| MediaPipe FineFace (×100) | 0.05 | 0.20 | 0.33 | 0.63 | 0.82 | 1.21 | 1.47 | 1.74 | 1.89 | 2.25 | 2.64 |
| MediaPipe July 100k (×100) | 0.78 | 0.84 | 1.13 | 1.15 | 1.04 | 1.12 | 1.39 | 1.25 | 1.53 | 1.86 | 2.22 |
| MediaPipe original 25k (×100) | 0.59 | −0.30 | 0.06 | 0.23 | 0.65 | 0.86 | 1.27 | 1.30 | 1.36 | 1.68 | 1.79 |
| MediaPipe aug 25k (×100) | −0.69 | −1.00 | −0.40 | −0.33 | −0.14 | 0.04 | 0.62 | 0.81 | 1.03 | 1.51 | 1.77 |

**100k on the cluster (job 746699, 780 samples, GPU speed).** The run reproduces July on the same checkpoint: `scaled` AU12 r 0.766 vs 0.774 (Δ −0.008 [−0.026, +0.009]) with identity equal to 3 decimals, and exact-constant CLS 0.073 / 0.014 vs 0.075 / 0.015. That is the same H100-vs-3090 numerical drift seen on the 25k ([2026-10-05](2026-10-05-cluster-control-discrepancy.md)), so the transferred checkpoint is intact. Dither lifts its CLS from 0.073 to 0.303 on py-feat (+0.230 [+0.143, +0.314]) and from 0.014 to 0.257 on MediaPipe (+0.243 [+0.151, +0.337]).

Original 25k vs 100k, paired: dithered CLS −0.198 [−0.288, −0.105] (py-feat) and −0.037 [−0.147, +0.072] (MediaPipe); ramp AU12 r −0.090 [−0.132, −0.048]; worst-case identity −0.029 [−0.048, −0.013].

## Reading

1. **On the label-independent detector, FineFace still calibrates better than every ramp-only checkpoint.** MediaPipe CLS is 0.49 for FineFace vs 0.26 (100k) and 0.30 (25k), with CIs excluding 0. The 2026-10-07 "on par with FineFace" holds only on py-feat (our training-label detector) and only for the 25k. On the 100k the thesis cites, FineFace wins on both detectors even with dither (−0.19 / −0.24). On that same checkpoint, dither halves the July MediaPipe gap (−0.48 → −0.24). Net: July's verdict "FineFace wins calibration" stands at about half the size, while July's "no absolute calibration" does not (the dithered 100k rises monotonically above c ≈ 0.5).
2. **25k dominates 100k.** Under dither the two are equivalent on MediaPipe CLS, but the 100k loses 0.20 py-feat CLS: its py-feat response is flat at ~0.72 for c ≤ 0.5 and only then rises, where the 25k climbs from ~0.44. The 100k is also worse on ramp-following (−0.09) and worst-case identity (−0.03). This extends the [2026-08-11](2026-08-11-training-step-dose-response.md) past-peak finding from ramps to calibration. Reported numbers should come from the 25k, or show both.
3. **Only the augmentation reached FineFace on MediaPipe** (aug 25k 0.491 vs 0.494), but at that checkpoint ramp-following has fallen below FineFace (AU12 r −0.16, MediaPipe r −0.08). That is the cluster late-training tail of [2026-10-05](2026-10-05-cluster-control-discrepancy.md). The aug 10k keeps ramp AU12 r on par but is no better than the original lineage on MediaPipe CLS (0.33) and trails FineFace on ramp MediaPipe r by 0.12. **No checkpoint yet matches FineFace on calibration and ramp-following on both detectors at once.**
4. **CLS hides the missing bottom of the scale.** CLS is a correlation and ignores offset. FineFace's c = 0 is neutral (py-feat 0.05), while ours never goes below ~0.4 (25k models) or ~0.7 (100k) on py-feat: our "zero" still smiles. This is the c = 0 residual of 2026-10-07 seen across the whole lower half, and it matches the data limit diagnosed in [2026-10-02](2026-10-02-calibration-augmentation.md): only 1–9% of MEAD-happy clips support c ≤ 0.3. MediaPipe's absolute values are not compared across models, because they depend on the image domain (512² SD2.1 portraits vs 384×256 MEAD-domain frames).
5. **Identity and ramp verdicts unchanged.** Every model wins worst-case identity by +0.12 to +0.17 (CIs exclude 0). Ramp-following ties FineFace for the original lineage: MediaPipe r −0.02 to −0.06 (CIs include 0); AU12 r −0.02 for the 100k, and +0.07 for the 25k on the circular py-feat only.

## Implications

- The thesis claim on calibration: **dithered constant-command CLS 0.26–0.30 (MediaPipe) vs FineFace 0.49, up from 0.01–0.07 with exact lists.** It remains a named limitation, now about half the July size, and the cause is split between protocol (CCL zero, fixed by dither) and model/data (no neutral bottom).
- The augmentation is the evidenced lever for closing the MediaPipe gap. What it still needs is a checkpoint (or a stable run) that keeps ramp-following.

## Next steps

1. Evaluate the aug arm's 15k and 20k checkpoints (already on disk) under `calibration_jitter` + `scaled`, looking for one that matches FineFace on calibration and ramp-following on both detectors.
2. Low end: add MEAD neutral clips (raw MEAD needed). This is the only route to a neutral c = 0.
3. Make dithering the pipeline's behaviour for repeated consecutive values (inference-only, documented).

Artifacts (gitignored): `inference_output/batch_eval/{fineface,trained}/<protocol>/results.jsonl` (July results split per protocol), `orig_100k/{calibration,calibration_jitter,scaled}/`, `fineface_vs_*_{jitter,scaled}.json`, `fineface_vs_trained_exact.json`, `ccl_test_orig_100k.json`, `orig_25k_vs_100k.json`. Inference config `configs_local/infer_orig_100k.local.yaml` (outside git).
