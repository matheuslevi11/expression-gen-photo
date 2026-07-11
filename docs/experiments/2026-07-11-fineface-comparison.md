# FineFace study: the closest prior art (July 11, 2026)

Study note (no run). Follow-up to [2026-07-11-related-work-comparison.md](2026-07-11-related-work-comparison.md), which flagged FineFace as the closest unexamined competitor. Full read confirms it: **FineFace is prior art for AU-intensity-conditioned text-to-image generation**, and our positioning must change accordingly.

## The paper

**"Towards Localized Fine-Grained Control for Facial Expression Generation"** (FineFace), Varanka, Khor, Li, Wei, Kung, Sebe, Zhao — arXiv:2407.20175v1, 25 Jul 2024. Univ. of Oulu + Trento; **same Oulu group as MagicFace** (FineFace = their generation paper, MagicFace = their editing sibling). Code + dataset public: `github.com/tvaranka/fineface`.

**Method:** frozen SD 2-1-base; conditioning is a continuous 12-dim AU vector y ∈ [0,5]¹² (AU1,2,4,5,6,9,12,15,17,20,25,26) through an AU encoder + IP-Adapter-style decoupled cross-attention (`Z = Attn(Q,K_text,V_text) + λ_AU·Attn(Q,K_AU,V_AU)`), plus a rank-32 LoRA; CFG on the AU condition (null = all-zero vector). Global token-level injection — no spatial map. Training data: DISFA (~90k lab frames, manual AU labels) + AffectNet (~90k stills, LibreFace auto-labels with ad-hoc manual distribution corrections; intensities < 1 zeroed as unreliable). Distribution smoothing (Gaussian label noise σ²=0.2 + 20% integer quantization) gives large consistency gains (their Table 3: CLIP-I 0.81→0.92).

**Evaluation:** 15 prompts × 12 AUs × 5 intensities + 50 combinations; AU MSE via a LibreFace classifier (4.71 individual / 7.54 combination, beating four self-constructed baselines) and CLIP-I vs the unconditioned sample as "character consistency" (self-admitted ambiguous). **Intensity control is shown qualitatively only** (sweep figures); they concede the intensity scale "is nonlinear"; there is **no commanded-vs-detected correlation metric anywhere** and **no face-identity metric at all**.

## The honest novelty ledger

**Already theirs (July 2024):** AU-intensity T2I generation with no source image; adapter on a frozen backbone; CFG on the expression condition; smooth qualitative intensity sweeps; out-of-range extrapolation (negative AU12 → frown); and *broader* control than ours (12 AUs + combinations vs our single AU12). They explicitly claim primacy ("no previous works on generating facial images with AU conditions"). Any "first AU-conditioned generation" phrasing on our side is dead.

**Still ours alone:**

1. **Identity-consistent intensity ramps within one generated sequence** — their sweep is N independent stills (shared seed, nothing binds identity architecturally); our temporal attention binds 5 frames to one identity/scene by construction, and the per-frame condition commands a *trajectory*. This is now the core defensible contribution.
2. **Measured dose–response calibration** — they never quantify commanded-vs-detected intensity; our CLS/Pearson protocol is an evaluation contribution over the closest prior art, and their admitted nonlinear scale is an exploitable weakness.
3. **CCL change-encoding** — direction/magnitude of change *between frames*; meaningless in a single-image setting.
4. **Video-derived ordered supervision** (real MEAD smile dynamics) vs static stills where low intensities had to be discarded as unreliable — precisely the regime where video ramps are richest.
5. **Unified multi-axis framing** — expression as one axis in GenPhoto's camera-parameter family with the identical mechanism.

**Revised claim:** *first identity-consistent, temporally coherent expression-intensity ramps as a first-class sequence-generation parameter, with measured dose–response calibration.*

## Mandatory new experiment: FineFace head-to-head

Their code is public. Run their AU12 sweep as independent stills vs our ramps, same prompts, and compare:

1. **AU12 dose–response Pearson r** — they may be competitive here; must be measured, not assumed.
2. **Cross-frame ArcFace identity consistency** — independent stills should lose decisively; this is the measurement that isolates our core contribution.

This is the single most important external comparison for the paper. Added to the Evaluation Robustness checklist.

## To borrow

1. **Distribution smoothing** of AU labels (σ²=0.2 noise + partial quantization) — one-line training change with evidenced gains; applicable to our AU12 labels in a future run.
2. **Negative / out-of-range intensity probe** (their Fig. 8) as a disentanglement demo — test intensities < 0 and > 1.
3. **CLIP-I(uncond vs cond)** as a cheap character-consistency proxy alongside ArcFace.
4. Their 12-AU multi-label + smoothing recipe is the ready-made blueprint for our Option B (AU-vector) extension.
5. Citation hygiene: cite FineFace as prior art, MagicFace as the same group's editing sibling.

## Their weaknesses we avoid

No identity mechanism or metric across intensity levels; no quantitative calibration (claims rest on figures) and a self-admitted nonlinear intensity scale; static-image supervision that cannot capture dynamics; ad-hoc manual label corrections that hurt reproducibility; partial LibreFace circularity (AffectNet half labeled and evaluated by LibreFace) — though their manual DISFA half mitigates this better than any of the other three papers studied.
