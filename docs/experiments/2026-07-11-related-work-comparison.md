# Related-work study: EmojiDiff, MagicFace, PixelSmile (July 11, 2026)

Study note (no run). Three papers from `related_work/` were read in full and compared against our approach. Bottom line: **all three are single-image expression *editing/transfer* systems; none does text-to-sequence generation with a parametric intensity axis.** Our niche — expression intensity as a first-class *generation* parameter swept across an identity-consistent sequence on a frozen backbone — is intact, but each paper sharpens how we must position and evaluate.

## The three papers

### EmojiDiff (arXiv 2412.01254v3, Ant Group)

Expression *transfer* with identity preservation: the control signal is an **RGB exemplar image** (masked face region, CLIP-encoded) injected via IP-Adapter-style parallel decoupled cross-attention into frozen SD1.5; identity comes from a reference photo. Trains only projection/attention branches. Data machinery is heavy: a Base adaptor trained on same-identity pairs manufactures 100k cross-identity triplets via FaceFusion face-swap from ~10k in-house images. Adds an ArcFace-cosine identity loss computed on one-step x̂₀ decodes with timestep truncation (ANI). Metrics: Antelopev2 ID cosine, LIQE quality, L1 over 52 MediaPipe blendshapes, landmark-movement score. Headline (realistic): ID 0.666 / Exp 0.054, beating FineFace and LivePortrait on their transfer benchmark.

**Relation to us:** adjacent, not competing — it cannot take "smile = 0.5" as input; a user must find an exemplar per intensity level. No sequence/temporal story. Their Sec. 1 critique of signal-conditioned methods (detail loss; dependence on third-party detector accuracy) applies to us and is pre-empted by our independent-AU-measurement checklist item.

### MagicFace (arXiv 2501.02260v1, Univ. of Oulu)

Photo *editing* conditioned on a **12-dim AU variation vector** (target − source AUs, LibreFace-labeled), injected globally through a single linear layer added to the time embedding. **Full fine-tune** of the denoising UNet plus a full UNet-copy ID encoder (~2×860 M params, 4×A100); background/pose preserved via pixel-aligned conditioning. Trained on 30k same-identity pairs mined from Aff-Wild. Uses AU dropout (10%) + classifier-free guidance on the AU condition (best α≈3.0; improves AU MSE 0.360→0.261). Metrics: AU-intensity MSE (LibreFace), ID embedding distance, background/head-pose RMSE. Their lab-vs-wild ablation shows a DISFA-(lab-)trained model fails on natural images. Unacknowledged circularity: LibreFace produces both training labels and the evaluation metric.

**Relation to us:** closest in **control semantics** (continuous AU intensity on SD1.5) but editing, not generation; relative not absolute targets (needs source-AU estimation at inference); no temporal mechanism; 16× our trainable params. Their lab-data failure mode is the likely root of our MEAD-domain look and male baseline-smile bias.

### PixelSmile (arXiv 2603.25728v1, Fudan + StepFun)

Photo *editing* on Qwen-Image-Edit (MMDiT) with rank-64 LoRA: intensity α∈[0,1] scales a **text-embedding direction** (e_neu + α·Δe); losses = score-supervised flow matching + symmetric InfoNCE on confusable emotion pairs + ArcFace ID loss. Data: **FFE**, 60k images synthesized by Nano Banana Pro and annotated with continuous 12-dim intensity vectors by Gemini 3 Pro; all benchmark scoring also by Gemini 3 Pro (circularity strictly worse than ours — same VLM builds the data and judges the results, with no objective facial-action measurement anywhere). Headline: CLS-6 (Pearson between commanded α and VLM-scored intensity) 0.808; **zero-shot text-embedding interpolation alone reaches CLS-6 0.689**. Their MEAD-trained ablation underperforms — but they used MEAD crudely (3 discrete levels → {0.5, 0.75, 1.0}), unlike our continuous per-frame AU12 supervision, so the conclusion doesn't transfer.

**Relation to us:** closest in **claim** ("linear, continuous expression-intensity control", Pearson-based evaluation ≈ our metric) but editing a source photo, per-α independent stills, unitless α, closed-model-distilled data, 4×H200.

## Comparison table

| Axis | EmojiDiff | MagicFace | PixelSmile | **Ours** |
|---|---|---|---|---|
| Task | transfer (exemplar→photo) | editing (photo + ΔAU) | editing (photo + α) | **text→5-frame ramp, no source image** |
| Intensity semantics | none (exemplar-implicit) | relative 12-AU vector | unitless α on text direction | **absolute per-frame AU12 (FACS-grounded)** |
| Injection | parallel cross-attn (2D UNet) | linear → time embedding | text-embedding interpolation + LoRA | **spatial 6-ch map → zero-init merge in temporal attention** |
| Trained / backbone | adapters on frozen SD1.5 | **full** UNet ×2 | LoRA-64 on MMDiT | 218 M adaptor on frozen SD1.5+AnimateDiff |
| Compute | ~2 days A100-class | 4×A100, 100k steps | 4×H200, 100 ep | **1×RTX 3090, 100k steps** |
| Training data | 100k face-swapped triplets (in-house) | 30k Aff-Wild pairs | 60k closed-model synthetic | 4,028 public MEAD clips |
| Identity across intensities | ID loss (single image) | UNet-copy ID encoder | ArcFace loss | temporal attention (architectural; verification pending) |
| Label/eval circularity | landmarks both sides (unflagged) | LibreFace both sides (unflagged) | Gemini both sides (unflagged) | py-feat both sides (**flagged, fix scheduled**) |

## Threats to novelty and positioning

1. **"Why not a big editor zero-shot?"** PixelSmile's no-training row (CLS-6 0.689) proves naive text-direction interpolation on a modern backbone already gives decent linearity. Our prompt-engineering-baseline checklist item is now mandatory, and the positioning must lead with *generation without a source image* + *identity-consistent sequences* + *absolute calibrated units* — none of which any editor row demonstrates.
2. **"Fine-grained" collision.** EmojiDiff and PixelSmile both use the phrase; EmojiDiff means detailed facial actions, PixelSmile means intensity linearity. Our claim should be phrased: *continuous, calibrated, physiologically-grounded (AU12) scalar control as a first-class conditioning axis of a text-to-image generator, extending GenPhoto's camera-physics axes to facial action units with the same frozen-backbone mechanism.*
3. **Detector-dependency critique** (EmojiDiff Sec. 1) — the standard attack on AU-conditioned methods. Closing our independent-AU-measurement item pre-empts it; notably we'd then hold a higher evaluation standard than all three papers, each of which is silently circular.
4. **FineFace (arXiv 2407.20175)** — AU-intensity-conditioned *generation*, surfaced as an EmojiDiff baseline — may be the closest true competitor to our task. **Action: pull and study it next.**

## Concrete borrowings (mapped to our checklists)

1. **Independent AU measurement** → two ready options: **LibreFace** (different detector family, validated by MagicFace for exactly this) and **MediaPipe blendshapes** `mouthSmileLeft/Right` (dependency we already ship, per EmojiDiff's Exp metric). Optionally a VLM judge as a third modality — but never one that produced any labels.
2. **Dose–response calibration** → adopt PixelSmile's **CLS protocol** (uniformly spaced commanded intensities → Pearson vs detected) for direct comparability, and add MagicFace-style **AU MSE** as the absolute-error complement (Pearson is scale/offset-invariant).
3. **ArcFace identity item** → EmojiDiff's Antelopev2-cosine implementation frame-to-frame; PixelSmile's **HES** (harmonic mean of expression accuracy and ID similarity) as the single-number trade-off metric; multi-model ID averaging (ArcFace+AdaFace+FaceNet) for robustness.
4. **Training-recipe upgrade for a future run** → MagicFace's AU dropout + expression-CFG (their Table IV: AU MSE 0.360→0.261) is cheap and directly applicable to our conditioning channel; EmojiDiff's one-step-x̂₀ identity loss (ANI) is the concrete recipe if we add the ArcFace regularizer (expect their observed control-vs-identity tension: ID +3–6% but Exp slightly worse).
5. **Data direction** → MagicFace's lab-vs-wild ablation predicts our MEAD-domain look and male-bias artifact; in-the-wild same-identity pairs (Aff-Wild-style) are the indicated next dataset. PixelSmile's negative MEAD result is *not* evidence against our pipeline (they discarded the continuous signal we extract).
6. **External baseline idea** → a free expression-transfer model (e.g., LivePortrait) fed graded reference frames as a dose–response baseline.

## Sources

Full per-paper analyses (identity, method, evaluation, headline numbers with section references) were produced from complete reads of the three PDFs in `related_work/`; numbers cited above trace to: EmojiDiff Tables 1–6, Eq. 4–8, Supp. A; MagicFace Tables I–IV, Fig. 4–5, Sec. III-D/IV-A/VI; PixelSmile Tables 1–3, Eq. 4–9, §3.1/5.2/5.5, App. B–C.
