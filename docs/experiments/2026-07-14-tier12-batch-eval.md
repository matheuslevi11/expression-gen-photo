# Tier 1 + Tier 2 batch evaluation (July 13–14, 2026)

First statistically grounded evaluation of the 100 k checkpoint, run with `scripts/batch_eval.py` (commit `ab81908`). 540 trained samples + 120 frozen-backbone baseline samples, three measurements per sample (py-feat AU12, MediaPipe mouth-corner smile proxy, facenet-VGGFace2 identity cosine). Outputs under `inference_output/batch_eval/{trained,baseline}/` (manifests, per-sample results, summaries).

**Headline: the model has strong *relative* (trajectory-following) expression control and high identity consistency, but NO absolute intensity calibration — constant-intensity clips produce a near-constant broad smile regardless of the commanded level (pooled CLS = 0.07).** The July smoke-test anomaly is confirmed at scale.

## Setup

| Experiment | Variant | Design | n |
|---|---|---|---|
| scaled | trained + baseline | 40 prompts × 3 seeds × ascending [0,.25,.5,.75,1] | 120 + 120 |
| calibration | trained | 10 prompts × 3 seeds × constant [c]×5, c ∈ {0.0,…,1.0} | 330 |
| permuted | trained | 10 prompts × 3 seeds × 3 non-monotonic lists | 90 |

Baseline ran `scaled` only: conditioning is bit-exactly inert without the adaptor (2026-07-09 entry), so other baseline experiments would duplicate GIFs.

## Results

### Scaled evaluation (Tier 2) — strong, robust ramp control

| Metric | Trained (n=120) | Baseline (n=120) |
|---|---|---|
| AU12 Pearson *r* vs target | **0.774 ± 0.212** (median 0.86; 87% > 0.5; 65% > 0.8) | −0.05 ± 0.64 (only 63 valid) |
| MediaPipe smile-proxy *r* | **0.829 ± 0.230** | −0.02 ± 0.66 (44 valid) |
| AU12 MSE | 0.142 ± 0.092 | 0.326 ± 0.081 |
| Identity cosine, adjacent frames (mean) | **0.919 ± 0.054** | 0.853 ± 0.128 |
| Identity cosine, adjacent frames (min) | 0.877 ± 0.086 | 0.794 ± 0.152 |
| Samples with detectable faces (identity) | 120/120 | **52/120** |

- The earlier 3-prompt anecdote (r = 0.84) survives scaling: r = 0.77 ± 0.21 over 40 diverse prompts × 3 seeds, with an independent-detector cross-check (MediaPipe, trained on nothing of ours) agreeing at r = 0.83. **The py-feat circularity confound is broken** (Tier 2 independent-AU item).
- Identity consistency is high (0.92 adjacent-frame cosine) and *exceeds* the baseline even where the baseline is measurable; the baseline fails face detection entirely on 57% of samples.
- **Gender bias quantified** (known limitation, now with numbers): female-prompt r = 0.825 ± 0.178 (n=60) vs male-prompt r = 0.712 ± 0.228 (n=57) — a Δ ≈ 0.11 gap consistent with the MEAD male baseline-smile artifact from the 2026-06 eval.

### Dose–response calibration (Tier 1) — NEGATIVE: no absolute calibration

Constant-intensity clips `[c]×5`:

| Commanded c | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Detected AU12 | 0.79 | 0.87 | 0.85 | 0.85 | 0.85 | 0.85 | 0.86 | 0.85 | 0.86 | 0.86 | 0.86 |

Pooled CLS (Pearson over all (commanded, detected) pairs) = **0.075**. The curve is flat at ~0.85 for every level ≥ 0.1 — commanded absolute intensity carries almost no signal in a constant-intensity clip.

**The control is contextual, not absolute.** Direct evidence: frame 1 of an ascending ramp (commanded 0.0) detects AU12 = 0.54 ± 0.29, but a whole clip at constant 0.0 detects 0.79 ± 0.18. The same commanded value produces a different expression depending on the rest of the clip — the model renders "the low end of *this clip's* trajectory," not "AU12 = 0."

**Why (training-data structure):** every MEAD training clip is a sorted monotonic ramp, so the model never saw a constant-intensity sequence; and MEAD happy clips carry high AU12 even at low labeled intensities (known limitation), which sets the ~0.85 plateau. Constant lists are out-of-distribution in exactly the dimension the calibration test probes.

### Permuted lists (Tier 2) — partial per-frame conditioning

r = **0.385 ± 0.505** (AU12; MediaPipe agrees at 0.46 ± 0.45). Per-list: [0,1,.5,.25,.75] → 0.48; [1,0,.75,.25,.5] → 0.23; [.5,0,1,.75,.25] → 0.44. Clearly above the baseline's zero, clearly below monotonic ramps (0.77 ascending / −0.98 descending from 2026-07-08). The temporal prior (motion module + ramp-only training) resists non-monotonic trajectories; per-frame conditioning exists but is weakest exactly where the training distribution has no support.

## Interpretation

1. **What training produced is *ordinal trajectory control*:** the model reliably follows monotonic intensity ramps in either direction with high identity consistency — verified by two independent detectors at n=120. It does not implement an absolute AU12 dial.
2. **The revised claim must say "relative":** *identity-consistent, temporally coherent expression-intensity ramps* stands (and is now measured, not asserted); "measured dose–response calibration" is delivered as a **measurement with a negative result** — which is still an evaluation contribution no related work provides (FineFace shows sweeps qualitatively and admits a nonlinear scale; we quantify exactly how uncalibrated ramp-trained control is).
3. **For the FineFace head-to-head** this sharpens the design: compare ramp-following r *and* identity, and run their model through the same constant-intensity CLS protocol — static-image-trained FineFace may actually calibrate better absolutely while losing identity/trajectory coherence. That would make an honest, interesting comparison figure either way.
4. **Fix directions for absolute calibration** (future run, not eval): train-time intensity-list augmentation (constant and permuted lists with matching frame supervision — requires frame retrieval by AU12 value rather than sorted ramps), MagicFace-style AU dropout + CFG, and/or FineFace-style label distribution smoothing.

## Caveats of this evaluation

- Identity metric is facenet-pytorch InceptionResnetV1 (VGGFace2), not ArcFace-proper; the checklist item is satisfied in substance (strong face-recognition embedding, frame-to-frame cosine) but the paper should either rename or add insightface/ArcFace for the multi-model average.
- Baseline identity/correlation stats are computed on the subset where faces were detectable (52/120 and 63/120) — survivorship in the baseline's favor; the trained-vs-baseline gaps are lower bounds.
- MediaPipe smile proxy is ordinal only (arbitrary units); it corroborates correlations, not MSE.
- Calibration used 10 prompts (the first 10 of the bank); the flatness is so extreme (CLS 0.07) that more prompts cannot rescue absolute calibration.
