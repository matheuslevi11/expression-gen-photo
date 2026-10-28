# Is the zero CCL of exact-constant lists hurting calibration? (October 7, 2026)

Inference-only test of the structural suspect from [2026-10-02](2026-10-02-calibration-augmentation.md) (Reading §2) and [2026-10-05](2026-10-05-cluster-control-discrepancy.md): the calibration protocol commands exact-constant lists `[c]×5`, which make the 3 CCL channels exactly zero (at c = 0.0 the whole 6-channel conditioning is zero). Training never produces that: even "constant" augmented sequences carry distinct real labels.

## The CCL is a "prompt changed" flag, not a magnitude

`build_ccl_embedding` takes differences of CLIP states between consecutive frames' prompts `"<smile intensity: {value:.3f}>"`. Measured mean |CCL| (clip-0 CLIP text encoder):

| list | mean \|CCL\| |
|---|---|
| exact constant `[0.5]×5` | **0** |
| 0.5 ± 0.02 jitter | 0.086 |
| 0.5 ± 0.10 jitter | 0.110 |
| real "constant" training labels (~0.5) | 0.088 |
| ramp `[0, .25, .5, .75, 1]` | 0.120 |
| real training ramp (0.64 → 0.97) | 0.112 |

Any change in the 3-decimal prompt text gives a full-size CCL, whatever the numeric step. So the encoder only ever sees CCL ≈ 0.1 in training, and the exact-constant evaluation feeds it 0.

## Design

New `batch_eval.py` protocol `calibration_jitter`: same 10 prompts × 3 seeds × 11 levels as `calibration`, but each frame commands a value within ±0.02 of c, five values **distinct at 3 decimals** (so no two frames share a prompt and no frame's CCL is zero), reflected at 0/1, drawn from a fixed per-(prompt, seed, level) RNG so every model gets identical lists. Mean commanded value: 0.011 at c = 0, 0.499 at c = 0.5, 0.991 at c = 1.

Run on four models: aug and ctrl × 10k and 25k. Each is compared **with itself** on the exact-constant `calibration` results already on disk: same prompts, seeds, levels and initial noise; only the CCL (and ≤ 0.02 of scalar) differs. `compare_eval.py --exp-b` pairs by (prompt, seed, level).

## Prediction and reading (fixed before results)

- If zero CCL is out-of-distribution in a way that matters, **CLS(jitter) > CLS(exact)** with paired CIs excluding 0 on both py-feat and MediaPipe, most clearly for the aug arm (trained on near-constant sequences, so jittered lists are in-distribution for it), and the c = 0.0 bump (detected smile at c = 0 above c = 0.1) should disappear.
- If CLS does not move, the CCL is not the bottleneck and the flat low end is a data/label problem → MEAD neutral clips next.
- If it moves, the fix belongs in the conditioning, not the data: e.g. make the CCL encode magnitude (value-scaled differences) or never emit exact-zero CCL at inference.

## Results

Five models, each compared with itself (exact `[c]×5` vs ±0.02 near-constant lists; paired by prompt, seed and level; 2,000-draw bootstrap over the 30 (prompt, seed) units). All calibration evaluations ran at GPU speed (one job hit a node whose allocated GPU already held 5.8 GB of a foreign process; the new launcher guard aborted it and it was rerun elsewhere).

| model | CLS py-feat exact → jitter | Δ (95% CI) | CLS MediaPipe exact → jitter | Δ (95% CI) |
|---|---|---|---|---|
| **original 25k** (RTX-3090 lineage) | 0.080 → **0.501** | +0.421 [+0.345, +0.500] | 0.068 → **0.295** | +0.226 [+0.168, +0.304] |
| cluster ctrl 10k | 0.146 → 0.347 | +0.201 [+0.135, +0.267] | 0.109 → 0.239 | +0.131 [+0.064, +0.218] |
| cluster ctrl 25k | 0.034 → 0.411 | +0.377 [+0.304, +0.465] | 0.003 → 0.255 | +0.252 [+0.196, +0.331] |
| cluster aug 10k | 0.221 → 0.496 | +0.275 [+0.215, +0.340] | 0.185 → 0.326 | +0.141 [+0.078, +0.209] |
| cluster aug 25k | 0.158 → **0.625** | +0.467 [+0.383, +0.544] | 0.160 → **0.491** | +0.331 [+0.280, +0.390] |

Detected py-feat AU12 by level, exact → jitter:

| c | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original 25k exact | .830 | .813 | .824 | .850 | .834 | .830 | .841 | .844 | .854 | .849 | .857 |
| original 25k jitter | .629 | .439 | .537 | .542 | .690 | .738 | .799 | .805 | .816 | .840 | .844 |
| aug 25k exact | .842 | .812 | .813 | .813 | .816 | .820 | .828 | .841 | .867 | .878 | .896 |
| aug 25k jitter | .427 | .405 | .556 | .572 | .606 | .665 | .750 | .762 | .819 | .863 | .884 |

Aug vs ctrl, both under jitter (paired across arms): 10k +0.149 [+0.065, +0.227] py-feat / +0.087 [−0.014, +0.167] MediaPipe; 25k **+0.214 [+0.111, +0.314] / +0.235 [+0.157, +0.313]**.

## Reading

1. **The CCL is the bottleneck the prediction named.** Every model gains on both detectors with CIs well clear of 0; the exact-constant curves are flat at ~0.82–0.87 and the jittered ones climb monotonically from ~0.4 to ~0.85. The "no absolute calibration" result of [2026-07-14](2026-07-14-tier12-batch-eval.md) is reproduced under its own protocol (original lineage, exact lists: CLS 0.08 here vs 0.075 then) but is **mostly an artifact of commanding exactly equal values**, which zero the CCL channels — an input the encoder never saw in training. It is not primarily a property of ramp-only training: the ramp-only original reaches CLS 0.50 py-feat once the CCL is training-like.
2. **This reverses the calibration half of the FineFace comparison** ([2026-07-21](2026-07-21-fineface-headtohead.md): FineFace 0.497 vs ours 0.075). With ≤0.02 dither the original lineage matches FineFace on py-feat CLS (0.501). Not yet established: the MediaPipe side (FineFace's per-frame `mp_smile` is in its July `results.jsonl` on the workstation; its MediaPipe CLS was never computed) and the 100k checkpoint July actually used.
3. **The augmentation and the CCL fix compound.** Under jitter the augmented arm beats its control by +0.21/+0.24 CLS at 25k (both detectors) and reaches 0.63 py-feat / 0.49 MediaPipe, the best calibration measured in this project. The 2026-10-02/05 conclusion that augmentation is "a modest calibration lever" was itself depressed by the exact-constant protocol.
4. **Residual low-end defect.** At c = 0.0 (commanded values in [0, 0.02], scalar channel ≈ 0) several models still smile more than at c = 0.1 (original 0.63 vs 0.44, ctrl 25k 0.78 vs 0.59); aug 25k is nearly clean (0.43 vs 0.41). Near-zero scalar input is rare in training (5% of frames have AU12 < 0.25).
5. **Scope.** Only the calibration protocol is affected: `scaled` ramps and the `permuted` lists command distinct values, so their CCL was always training-like and those results stand.

## Implications

- **Dithering is a legitimate, zero-cost inference procedure** for a constant command: it changes the commanded values by ≤0.02 while keeping the conditioning in-distribution. Any reported constant-intensity result must say whether it was used.
- The principled fix is in the conditioning design: a CCL that encodes the *size* of the change (or no CCL for repeated values) so that "no change" is not an out-of-distribution input. That needs retraining; dithering does not.

## Next steps

1. Compute FineFace's MediaPipe CLS from its July results (workstation) to complete the head-to-head on both detectors.
2. Make dithering the pipeline's behaviour for repeated consecutive values (inference-only, documented), and rerun the July calibration protocol on the 100k checkpoint if the thesis keeps citing it.
3. Investigate the c = 0 residual (near-zero scalar) before claiming the bottom of the scale.
