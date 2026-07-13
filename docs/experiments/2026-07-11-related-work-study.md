# Related-work study: EmojiDiff, MagicFace, PixelSmile, FineFace (July 11, 2026)

Study note (no run). The four papers in `related_work/` were read in full and compared against our approach. (Conducted in two passes the same day: the first three papers were provided; FineFace was pulled and read after EmojiDiff's baseline table surfaced it as the closest unexamined competitor.)

**Bottom line:** EmojiDiff, MagicFace, and PixelSmile are all single-image *editing/transfer* systems — none does text-to-sequence generation with a parametric intensity axis. **FineFace, however, is prior art for AU-intensity-conditioned text-to-image generation** (July 2024, explicit primacy claim). Our novelty therefore rests not on "AU-conditioned generation" but on what none of the four has:

> **Revised claim:** *first identity-consistent, temporally coherent expression-intensity ramps as a first-class sequence-generation parameter, with measured dose–response calibration.*

## The four papers

### EmojiDiff (arXiv 2412.01254v3, Ant Group)

Expression *transfer* with identity preservation: the control signal is an **RGB exemplar image** (masked face region, CLIP-encoded) injected via IP-Adapter-style parallel decoupled cross-attention into frozen SD1.5; identity comes from a reference photo. Trains only projection/attention branches. Data machinery is heavy: a Base adaptor trained on same-identity pairs manufactures 100k cross-identity triplets via FaceFusion face-swap from ~10k in-house images. Adds an ArcFace-cosine identity loss computed on one-step x̂₀ decodes with timestep truncation (ANI). Metrics: Antelopev2 ID cosine, LIQE quality, L1 over 52 MediaPipe blendshapes, landmark-movement score. Headline (realistic): ID 0.666 / Exp 0.054, beating FineFace and LivePortrait on their transfer benchmark.

**Relation to us:** adjacent, not competing — it cannot take "smile = 0.5" as input; a user must find an exemplar per intensity level. No sequence/temporal story. Their Sec. 1 critique of signal-conditioned methods (detail loss; dependence on third-party detector accuracy) applies to us and is pre-empted by our independent-AU-measurement checklist item.

### MagicFace (arXiv 2501.02260v1, Univ. of Oulu)

Photo *editing* conditioned on a **12-dim AU variation vector** (target − source AUs, LibreFace-labeled), injected globally through a single linear layer added to the time embedding. **Full fine-tune** of the denoising UNet plus a full UNet-copy ID encoder (~2×860 M params, 4×A100); background/pose preserved via pixel-aligned conditioning. Trained on 30k same-identity pairs mined from Aff-Wild. Uses AU dropout (10%) + classifier-free guidance on the AU condition (best α≈3.0; improves AU MSE 0.360→0.261). Metrics: AU-intensity MSE (LibreFace), ID embedding distance, background/head-pose RMSE. Their lab-vs-wild ablation shows a DISFA-(lab-)trained model fails on natural images. Unacknowledged circularity: LibreFace produces both training labels and the evaluation metric.

**Relation to us:** closest in **control semantics** (continuous AU intensity on SD1.5) but editing, not generation; relative not absolute targets (needs source-AU estimation at inference); no temporal mechanism; 16× our trainable params. Their lab-data failure mode is the likely root of our MEAD-domain look and male baseline-smile bias.

### PixelSmile (arXiv 2603.25728v1, Fudan + StepFun)

Photo *editing* on Qwen-Image-Edit (MMDiT) with rank-64 LoRA: intensity α∈[0,1] scales a **text-embedding direction** (e_neu + α·Δe); losses = score-supervised flow matching + symmetric InfoNCE on confusable emotion pairs + ArcFace ID loss. Data: **FFE**, 60k images synthesized by Nano Banana Pro and annotated with continuous 12-dim intensity vectors by Gemini 3 Pro; all benchmark scoring also by Gemini 3 Pro (circularity strictly worse than ours — same VLM builds the data and judges the results, with no objective facial-action measurement anywhere). Headline: CLS-6 (Pearson between commanded α and VLM-scored intensity) 0.808; **zero-shot text-embedding interpolation alone reaches CLS-6 0.689**. Their MEAD-trained ablation underperforms — but they used MEAD crudely (3 discrete levels → {0.5, 0.75, 1.0}), unlike our continuous per-frame AU12 supervision, so the conclusion doesn't transfer.

**Relation to us:** closest in **claim** ("linear, continuous expression-intensity control", Pearson-based evaluation ≈ our metric) but editing a source photo, per-α independent stills, unitless α, closed-model-distilled data, 4×H200.

### FineFace (arXiv 2407.20175v1, Univ. of Oulu + Trento) — the closest prior art

**"Towards Localized Fine-Grained Control for Facial Expression Generation"** — **same Oulu group as MagicFace** (FineFace = their generation paper, MagicFace = their editing sibling). Code + dataset public: `github.com/tvaranka/fineface`.

**Method:** frozen SD 2-1-base; conditioning is a continuous 12-dim AU vector y ∈ [0,5]¹² (AU1,2,4,5,6,9,12,15,17,20,25,26) through an AU encoder + IP-Adapter-style decoupled cross-attention (`Z = Attn(Q,K_text,V_text) + λ_AU·Attn(Q,K_AU,V_AU)`), plus a rank-32 LoRA; CFG on the AU condition (null = all-zero vector). Global token-level injection — no spatial map. Training data: DISFA (~90k lab frames, manual AU labels) + AffectNet (~90k stills, LibreFace auto-labels with ad-hoc manual distribution corrections; intensities < 1 zeroed as unreliable). Distribution smoothing (Gaussian label noise σ²=0.2 + 20% integer quantization) gives large consistency gains (their Table 3: CLIP-I 0.81→0.92).

**Evaluation:** 15 prompts × 12 AUs × 5 intensities + 50 combinations; AU MSE via a LibreFace classifier (4.71 individual / 7.54 combination, beating four self-constructed baselines) and CLIP-I vs the unconditioned sample as "character consistency" (self-admitted ambiguous). **Intensity control is shown qualitatively only** (sweep figures); they concede the intensity scale "is nonlinear"; there is **no commanded-vs-detected correlation metric anywhere** and **no face-identity metric at all**.

**Relation to us: prior art.** AU-intensity T2I generation from text alone, no source image, adapter on a frozen backbone, CFG on the expression condition, qualitative intensity sweeps, out-of-range extrapolation (negative AU12 → frown) — and *broader* control than ours (12 AUs + combinations vs our single AU12). Any "first AU-conditioned generation" phrasing on our side is dead. But their intensity sweep is N **independent stills** (shared seed, nothing binds identity architecturally).

## Comparison table

| Axis | EmojiDiff | MagicFace | PixelSmile | FineFace | **Ours** |
|---|---|---|---|---|---|
| Task | transfer (exemplar→photo) | editing (photo + ΔAU) | editing (photo + α) | generation (independent stills) | **text→5-frame ramp** |
| Intensity semantics | none (exemplar-implicit) | relative 12-AU vector | unitless α on text direction | absolute 12-AU vector [0,5] | **absolute per-frame AU12** |
| Injection | parallel cross-attn (2D UNet) | linear → time embedding | text-embedding interpolation + LoRA | decoupled cross-attn + LoRA-32 | **spatial 6-ch map → zero-init merge in temporal attention** |
| Trained / backbone | adapters on frozen SD1.5 | **full** UNet ×2 | LoRA-64 on MMDiT | adapter + LoRA on frozen SD2.1 | 218 M adaptor on frozen SD1.5+AnimateDiff |
| Compute | ~2 days A100-class | 4×A100, 100k steps | 4×H200, 100 ep | n/s (batch 16) | **1×RTX 3090, 100k steps** |
| Training data | 100k face-swapped triplets (in-house) | 30k Aff-Wild pairs | 60k closed-model synthetic | DISFA + AffectNet stills (~180k) | 4,028 public MEAD clips |
| Identity across intensities | ID loss (single image) | UNet-copy ID encoder | ArcFace loss | **none** (no metric, no mechanism) | temporal attention (architectural; verification pending) |
| Intensity calibration measured? | n/a | AU MSE (aggregate) | CLS Pearson (VLM-judged) | **no** (qualitative only, admits nonlinear) | **planned: CLS + AU MSE, detector-judged** |
| Label/eval circularity | landmarks both sides (unflagged) | LibreFace both sides (unflagged) | Gemini both sides (unflagged) | partial LibreFace (unflagged; DISFA half manual) | py-feat both sides (**flagged, fix scheduled**) |

## The honest novelty ledger (post-FineFace)

**Already in prior art (FineFace, July 2024):** AU-intensity T2I generation without a source image; adapter on frozen backbone; CFG on the expression condition; smooth qualitative intensity sweeps; out-of-range extrapolation; 12-AU + combination control.

**Still ours alone:**

1. **Identity-consistent intensity ramps within one generated sequence** — temporal attention binds the 5 frames to one identity/scene by construction; the per-frame condition commands a *trajectory*, not a value. No paper of the four has any cross-intensity identity mechanism in generation. This is the core defensible contribution.
2. **Measured dose–response calibration** — none of the four quantifies commanded-vs-detected intensity with an objective facial-action detector (PixelSmile's CLS is VLM-judged inside its own circularity; FineFace is qualitative-only with an admitted nonlinear scale). Our CLS/Pearson protocol is an evaluation contribution over the closest prior art.
3. **CCL change-encoding** — direction/magnitude of change *between frames*; meaningless in single-image settings.
4. **Video-derived ordered supervision** — real MEAD smile dynamics vs static stills where FineFace had to discard low intensities as unreliable, precisely the regime where video ramps are richest.
5. **Unified multi-axis framing** — expression as one axis in GenPhoto's camera-parameter family with the identical mechanism (bokeh / focal / shutter / expression).

## Threats to novelty and positioning

1. **FineFace is prior art** — cite it as such (and MagicFace as the same group's editing sibling); drop any "first AU-conditioned generation" phrasing; lead with the revised claim above.
2. **"Why not a big editor zero-shot?"** PixelSmile's no-training row (CLS-6 0.689) proves naive text-direction interpolation on a modern backbone already gives decent linearity. Our prompt-engineering-baseline checklist item is mandatory; the answer is *no source image + identity-consistent sequences + absolute calibrated units*.
3. **"Fine-grained" collision.** EmojiDiff (detailed facial actions), PixelSmile (intensity linearity), and FineFace ("localized fine-grained control") all use the phrase differently — avoid it as a headline term; say *continuous, calibrated, identity-consistent*.
4. **Detector-dependency critique** (EmojiDiff Sec. 1) — the standard attack on AU-conditioned methods. Closing our independent-AU-measurement item pre-empts it; we would then hold a higher evaluation standard than all four papers, each of which is silently circular.

## Mandatory new experiment: FineFace head-to-head

FineFace's code is public. Run their AU12 sweep as independent stills vs our ramps, same prompts, and compare:

1. **AU12 dose–response Pearson r** — they may be competitive here; must be measured, not assumed.
2. **Cross-frame ArcFace identity consistency** — independent stills should lose decisively; this is the measurement that isolates our core contribution.

The single most important external comparison for the paper. On the Evaluation Robustness checklist (Tier 1).

## Concrete borrowings (mapped to our checklists)

1. **Independent AU measurement** → **LibreFace** (different detector family, validated by MagicFace for this signal) or **MediaPipe blendshapes** `mouthSmileLeft/Right` (dependency we already ship, per EmojiDiff's Exp metric). Optionally a VLM judge as a third modality — but never one that produced any labels.
2. **Dose–response calibration** → PixelSmile's **CLS protocol** (uniformly spaced commanded intensities → Pearson vs detected) for comparability, plus MagicFace-style **AU MSE** as the absolute-error complement (Pearson is scale/offset-invariant).
3. **Identity metrics** → EmojiDiff's Antelopev2-cosine frame-to-frame; PixelSmile's **HES** (harmonic mean of expression accuracy and ID similarity) as the single-number trade-off; multi-model ID averaging (ArcFace+AdaFace+FaceNet) for robustness; FineFace-style **CLIP-I(uncond vs cond)** as a cheap character-consistency proxy.
4. **Training-recipe upgrades for a future run** → MagicFace's AU dropout + expression-CFG (AU MSE 0.360→0.261); FineFace's **label distribution smoothing** (σ²=0.2 noise + partial quantization, large evidenced gains); EmojiDiff's one-step-x̂₀ identity loss (ANI) if we add the ArcFace regularizer (expect their control-vs-identity tension: ID +3–6%, Exp slightly worse).
5. **Cheap probes** → FineFace's negative / out-of-range intensity extrapolation (their Fig. 8) as a disentanglement demo — test intensities < 0 and > 1.
6. **Data direction** → MagicFace's lab-vs-wild ablation predicts our MEAD-domain look and male-bias artifact; in-the-wild same-identity pairs (Aff-Wild-style) are the indicated next dataset. PixelSmile's negative MEAD result is *not* evidence against our pipeline (they discarded the continuous signal we extract). FineFace's 12-AU multi-label + smoothing recipe is the blueprint for our Option B (AU-vector) extension.
7. **External baseline idea** → a free expression-transfer model (e.g., LivePortrait) fed graded reference frames as a dose–response baseline.

## Sources

Full per-paper analyses (identity, method, evaluation, headline numbers with section references) were produced from complete reads of the four PDFs in `related_work/`; numbers cited above trace to: EmojiDiff Tables 1–6, Eq. 4–8, Supp. A; MagicFace Tables I–IV, Fig. 4–5, Sec. III-D/IV-A/VI; PixelSmile Tables 1–3, Eq. 4–9, §3.1/5.2/5.5, App. B–C; FineFace Tables 1–3, Eq. 2–5, §4.2–5.2, Fig. 6/8/18–19, supp.
