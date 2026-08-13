# Evaluation Methods in Four Facial-Expression-Control Papers

## The papers

| Short name     | arXiv      | Date     | Group                         | Task                                                 | Backbone                 |
| -------------- | ---------- | -------- | ----------------------------- | ---------------------------------------------------- | ------------------------ |
| **FineFace**   | 2407.20175 | Jul 2024 | Oulu + Trento (Varanka, Zhao) | AU-conditioned T2I generation, intensity-adjustable  | SD 2.1-base + adapter    |
| **EmojiDiff**  | 2412.01254 | Dec 2024 | Ant Group / Alipay            | RGB-reference expression transfer w/ ID preservation | SD1.5 / SDXL + E-Adapter |
| **MagicFace**  | 2501.02260 | Jan 2025 | Oulu + Southeast (Wei, Zhao)  | Relative-AU expression *editing*                     | SD + ID-encoder          |
| **PixelSmile** | 2603.25728 | Mar 2026 | Fudan + StepFun               | Fine-grained continuous intensity *editing*          | Flow-matching edit model |

FineFace and MagicFace are from the same lab and share a methodology; PixelSmile is the newest and has by far the most developed evaluation.

## Master comparison of evaluation design

| Dimension                               | FineFace                                                           | EmojiDiff                                              | MagicFace                                                   | PixelSmile                                                          |
| --------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------- |
| **Eval set**                            | 1,650 generated imgs (15 prompts × 12 AU × 5 intensities + combos) | 50 celebs × 30 expressions, ×3 styles                  | 20 identities → 8,940 edits (12 AU × 5 levels + 387 combos) | FFE-Bench: held-out portraits → 12 expression classes, real + anime |
| **Eval imgs are…**                      | model-generated                                                    | real celebrities                                       | real identities                                             | **synthetic** (Nano Banana Pro)                                     |
| **Control-fidelity metric**             | AU MSE (LibreFace)                                                 | Exp = MediaPipe blendshape L1; LMS = landmark-ratio L1 | AU MSE (LibreFace)                                          | Acc (Gemini) + mSCR confusion rate (Gemini)                         |
| **Intensity / linearity metric**        | ⚠️ qualitative only                                                | ⚠️ implicit (blendshape magnitude)                     | ⚠️ qualitative only                                         | ✅ **CLS = Pearson r(α, VLM score)**                                |
| **Identity preservation**               | CLIP-I (gen-w/-cond vs gen-wo-cond)                                | ID = Antelopev2 cosine                                 | ID L2 (`face_recognition` lib)                              | ID Sim = avg cosine of ArcFace+AdaFace+FaceNet                      |
| **Other "should-not-change" axes**      | —                                                                  | IQ (LIQE)                                              | background RMSE, head-pose RMSE (FLAME)                     | HES (harmonic mean of expr × ID)                                    |
| **Image-quality metric (FID/IS/LPIPS)** | ❌ none                                                            | ❌ none (uses blind IQA instead)                       | ❌ none                                                     | ❌ none                                                             |
| **Human study**                         | ❌                                                                 | ❌                                                     | ❌                                                          | ✅ 10 annotators, 2,400 imgs, rank continuity + identity (1–5)      |
| **Label ↔ eval circularity**            | LibreFace both ends                                                | **avoided by design**                                  | LibreFace both ends                                         | Gemini 3 Pro both ends                                              |

## Per-paper evaluation breakdown

**FineFace** — Two metrics only. **AU MSE** (L2 between the prompted AU intensity vector and the AU intensities that LibreFace detects in the output; lower = better) is the control-fidelity workhorse. **CLIP-I** is image similarity between a generation *with* and *without* the AU condition — a proxy for "did identity/scene survive the edit." Notably the authors say CLIP-I should be *moderate*, not maximal: a method that ignores the AU scores deceptively high (they flag LoRA-AU with an asterisk for exactly this). Intensity is tested at 5 discrete levels but folded into AU MSE; monotonicity/continuity is shown only in figures (0→5 ramps). No human study, no FID.

**EmojiDiff** — Four automatic metrics: **ID** (Antelopev2 face-embedding cosine), **IQ** (LIQE blind image-quality net), **Exp.** (L1 over 52 MediaPipe blendshapes vs the reference), **LMS** (L1 over five normalized MediaPipe landmark-amplitude ratios for eyes/pupils/mouth). The distinctive move is **deliberate circularity avoidance**: identity is scored with a *different* model (Antelopev2) than the one used to inject identity in training (ArcFace/IP-Adapter FaceID), and expression is scored with MediaPipe while the expression encoder is CLIP. Plus a generalization check — reusing the expression module as a feature extractor for RAF-DB expression recognition (88.7% vs 86.4% baseline). No explicit intensity metric; fine-grained control is argued qualitatively. No human study.

**MagicFace** — Four automatic metrics, all error-style (lower = better): **AU MSE** (LibreFace, vs relative-AU target), **ID L2** (`face_recognition` embeddings), **background RMSE** (face-masked pixel RMSE), **head-pose RMSE** (FLAME pose coefficients). The design philosophy: one metric says the expression *should* change (AU MSE) and three say identity/background/pose *should not* — disentanglement = winning all four at once. Guidance scale swept over 16 values. Same as FineFace: 5 intensity levels folded into AU MSE, monotonicity shown only in figures, no human study, no FID.

**PixelSmile** — The most complete eval, and the only one that quantifies intensity control directly:
- **CLS (Control Linearity Score)** = Pearson correlation between the input intensity coefficient α (swept uniformly) and the VLM-predicted intensity score. This is an explicit monotonicity/linearity metric — reported as CLS-6/CLS-12. Competitors score near-zero or **negative** (K-Slider CLS-6 = −0.046), which is the whole point of the metric.
- **mSCR (mean Structural Confusion Rate)** — how often edits toward class *i* get classified as a confusing neighbor *j* (Fear↔Surprise, Anger↔Disgust). Measures disentanglement of overlapping expressions.
- **HES** — harmonic mean of expression accuracy and identity similarity.
- **Acc** — categorical editing accuracy.
- **ID Sim** — average cosine over *three* recognition models (ArcFace + AdaFace + FaceNet).
- Plus an **intensity-vs-identity trade-off curve** (sweep α, plot expression score against ID) and a **human study** (10 annotators rank continuity + identity on 2,400 images). All classification/scoring is done by **Gemini 3 Pro**.

## Cross-cutting themes

**1. Nobody uses FID.** All four dropped standard generative-quality metrics (FID/IS/LPIPS). For controlled single-attribute edits on small curated sets, distributional metrics are uninformative; the field has converged on *task-specific* fidelity (did the target AU/expression appear?) + *preservation* (did identity/background/pose survive?). EmojiDiff is the only one adding an image-quality signal, and it uses a blind IQA net (LIQE), not FID.

**2. Two-sided evaluation is universal.** Every paper pairs a "the edit happened" metric against one or more "nothing else changed" metrics. The sophistication of the preservation side grows over time: CLIP-I (FineFace, crude) → single face model (MagicFace, EmojiDiff) → three-model ensemble + explicit head-pose/background terms (MagicFace/PixelSmile).

**3. "Higher identity similarity" is a trap, and three papers say so.** FineFace (CLIP-I should be moderate), EmojiDiff (implicitly), and most explicitly **PixelSmile**: realistic edits sit at ID ≈ 0.6–0.7; **>0.8 means "copy-paste" (the model barely edited)** and **<0.5 means identity distortion**. A naive "maximize identity" reading rewards models that do nothing.

**4. The control-fidelity evaluator evolved: AU classifier → blendshapes/landmarks → VLM.**
- LibreFace AU intensities (FineFace, MagicFace)
- MediaPipe blendshapes + landmark ratios (EmojiDiff)
- Gemini 3 Pro soft scores (PixelSmile)

**5. Intensity/monotonicity is the field's weakest-measured axis — except PixelSmile.** Three of four *claim* smooth intensity control but only show it in ramp figures; PixelSmile is the first to give it a number (CLS = Pearson r) plus a trade-off curve.

**6. Circularity is pervasive and mostly unaddressed.** FineFace and MagicFace use LibreFace to both label training data and score outputs; PixelSmile uses Gemini for both. Only **EmojiDiff** structurally breaks the loop (eval model ≠ training-supervision model). The others either acknowledge the scorer is imperfect (FineFace, MagicFace) or acknowledge VLM bias without naming it as circularity (PixelSmile).

**7. Human studies are rare.** Three of four have none; only PixelSmile (the newest) ran one — a ranking design on continuity + identity that it reports as consistent with the machine metrics.

## What this means for your eval (GenPhoto expression fork)

Your `comp_metrics/expression_au_accuracy.py` uses **py-feat AU12 Pearson r** — which is essentially **PixelSmile's CLS metric** (correlation of predicted intensity against the control input). So you're already using the strongest control-fidelity idea in the set, one that only the most recent paper formalized. Three concrete moves the literature suggests:

- **Break the py-feat→py-feat circularity the way EmojiDiff does.** Your `status.md` already flags this caveat honestly (as do FineFace/MagicFace/PixelSmile about their scorers). The mitigation the field actually uses is EmojiDiff's: **score with a different estimator than the one that produced training labels** — e.g., cross-check your AU12 correlation with LibreFace or a VLM. Even a partial cross-validation on a subset would put you ahead of FineFace/MagicFace/PixelSmile on this axis.
- **Your "ramp control robust, no absolute calibration" finding maps cleanly onto CLS vs AU-MSE.** CLS/Pearson r measures *monotonic correlation* (calibration-free) while LibreFace AU MSE measures *absolute error against a target*. That your ramp correlates but isn't absolutely calibrated is exactly the FineFace/MagicFace distinction — and the reason PixelSmile reports correlation, not MSE, for intensity. It's a defensible framing, and PixelSmile's Fig. 4 (sweep intensity, plot expression-score vs identity) is a good template to reproduce.
- **Cheap wins available:** none of these use FID, so your not relying on it is well-supported; but an identity/consistency proxy (your baseline-consistency CLIP-I analogue) plus the "identity similarity should be *moderate*" caveat is worth stating explicitly, and a small PixelSmile-style ranking study (continuity + identity) would be the single most novel addition, since 3 of 4 papers have no human eval at all.