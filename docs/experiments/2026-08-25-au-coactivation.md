# 2026-08-25 — Whole-face AU co-activation & leakage under scalar AU12 conditioning

What does the *whole face* do when the model is commanded an AU12 (smile) intensity ramp?
The model is trained on a single scalar axis; this experiment characterizes the response of
all 20 py-feat AUs to give a general (whole-face) view, cross-checked against two independent
detectors. Plan: `plans/we-need-to-play-shimmying-forest.md`.

## Setup

- **Re-scoring only** (no regeneration, no retraining): the existing generated clips under
  `inference_output/batch_eval/` were re-scored for all AUs. py-feat's `xgb` detector already
  computes all 20 AUs per call, so `detect_all_aus` (new, in `expression_au_accuracy.py`) is
  a free generalization of `detect_au12`. **AU12 regression guard: the new path reproduces the
  published `au12` exactly (max\|Δ\|=0.0).**
- **Stimulus:** ascending ramp `[0,.25,.5,.75,1]`, 40 prompts × 3 seeds (n=120), model of
  record = trained 100k; frozen backbone = paired control (same clip ids).
- **Three detectors:** (1) py-feat presence probabilities [0,1] (primary, but same family as
  the training labels); (2) **LibreFace** intensities (independent model family, isolated env —
  breaks the py-feat label/eval circularity); (3) offline **FaceMesh geometry** proxies
  (AU45 EAR, AU25/26 lip-gap, AU1/2/4 brow). Scores: `results_allau.jsonl`,
  `results_libreface.jsonl` per set.
- **Statistics** (`analyze_allau.py`): per-clip Spearman of detected-vs-commanded (primary, since
  py-feat gives probabilities not FACS intensities and there are only 5 monotone steps),
  aggregated over clips via Fisher-z with **BCa cluster bootstrap** CI; paired trained−baseline
  Δr with **Wilcoxon** + **Benjamini-Hochberg FDR** across the 20 AUs; per-AU detection-validity
  gating; AU×AU Spearman matrix (baseline-subtracted). Figures:
  `docs/experiments/figures/2026-08-25-au-coactivation-{profile,trajectories,matrix,emergence}.{pdf,png}`.

## Results

**The AU12 axis does not move AU12 in isolation — it drives the whole MEAD-"happy" expression
covariance.** Three coherent groups (py-feat Spearman r, trained; LibreFace r; Δr trained−baseline):

| Group | AUs | py-feat r | LibreFace r | model-induced |
|---|---|---|---|---|
| **Commanded axis** | AU12 | +0.90 | +0.92 | Δr +0.93, q<0.05 |
| **Smile cluster (co-activated)** | AU06 (Duchenne) | +0.88 | +0.71 | Δr +0.96, q<0.05 |
| | AU25 (lips part) | +0.57 | **+0.88** | LF/geo strongly model-induced (below) |
| | AU10 | +0.41 | — | Δr +0.58, q<0.05 |
| **Anti-coupled (released as smile grows)** | AU23 | **−0.72** | — | Δr −0.55 |
| | AU24, AU17, AU15 | −0.41, −0.38, −0.32 | — | negative |
| **Localized on py-feat (but see nuance)** | AU02, AU04, AU05, AU28 | ≈ 0 | AU04 +0.37 | detector-dependent |

1. **AU06 (Duchenne cheek raiser) co-activates robustly across all three detectors and is
   strongly model-induced** (py-feat 0.88, LibreFace 0.71 vs LF-baseline 0.26). This is the
   learned signature of MEAD-happy smiles: a genuine Duchenne smile, not just a lip movement.
2. **AU25/26 (mouth opening): independent detectors show it is strongly model-induced even where
   py-feat's paired Δr is masked.** LibreFace AU25 0.882 (baseline −0.717) and the offline lip-gap
   geometry 0.889 (baseline −0.364) both flip from negative in the baseline to strongly positive
   when trained — a large model-induced effect that py-feat's Δr (≈0) understates because py-feat
   already reads an AU25↔ramp correlation on the baseline. The independent channels correct the
   primary detector here.
3. **Anti-coupling is a real finding:** AU23 (lip tightener, −0.72), AU24, AU17, AU15 *decrease*
   as the smile grows. The model reproduces the full expression covariance of MEAD "happy":
   mouth-closing / frowning actions relax as the smile intensifies.
4. **Localization is only partial, and detector-dependent.** Brow AUs are ≈0 on py-feat (AU01
   +0.34, AU04 +0.08, AU02 +0.07), suggesting the axis leaves the upper face largely alone. But
   **LibreFace and geometry disagree**: LibreFace AU01 0.632 (baseline −0.439), AU04 0.368
   (baseline −0.574); geometry brow-raise −0.537 and inner-brow-gap −0.547 (both model-induced).
   So there is **subtle model-induced brow movement that py-feat underreports** — the axis is
   *not* cleanly localized. Honest verdict: the smile cluster is unambiguous; the degree of
   upper-face entanglement is real but its magnitude is detector-dependent.

**The independent cross-check earned its keep — it caught a py-feat artifact.** AU20 looks
strongly ramp-correlated on py-feat (+0.79) but **LibreFace disagrees (−0.19)**, it is
**not estimable** (valid on only 46% of clips), and its Δr is *negative*. This is py-feat
internal detector coupling, not a real facial movement. Likewise **AU43** (py-feat +0.56) is
**not corroborated by the eye-aspect-ratio geometry** (trained 0.042 ≈ baseline 0.053), so the
apparent "eyes closing with the smile" is largely a py-feat effect. AU07 (+0.95) is not
estimable (33%) and excluded. These are exactly the failures the circularity guard exists to
surface — a methodological win supporting evaluation-contribution C2.

**Emergence (H4): co-activation is learned, and grows with training** — AU06 Spearman by step:
0.70 (1k) → 0.89 (5k) → 0.91 (10k) → 0.93 (25k) → 0.94 (50k) → 0.88 (100k), tracking the AU12
dose-response closely (including the mild 100k past-peak dip). AU25 emerges similarly. The whole
smile cluster is acquired together with the commanded axis.

**Prior-art contrast (FineFace, independent stills):** FineFace matches or exceeds us on the
smile cluster (AU12 0.97, AU06 0.98) — expected, since it commands a 12-AU vector — but shows
**cleaner localization**: AU01 leakage 0.017 vs our 0.337, and AU23 +0.18 vs our −0.72. Because
FineFace controls brow and lower-face AUs independently, the AU12 command does not drag them;
our single scalar axis inherits the full MEAD-happy covariance. This is the empirical case for
the multi-AU extension (see [[2026-08-11-au-vector-proposal]]).

## Hypothesis verdicts

- **H1 (AU06/AU25 co-activate): supported.** AU06 across all three detectors, model-induced;
  AU25 confirmed by LibreFace + geometry.
- **H2 (nominally-inert AUs stay flat / localized): partially refuted.** True for the upper face
  on py-feat, but LibreFace and geometry reveal subtle model-induced brow movement — not cleanly
  localized.
- **H3 (model-induced): supported.** The smile cluster shows large trained−baseline Δr (q<0.05),
  and the independent baseline-subtracted signals confirm it.
- **H4 (emergence with training): supported.** AU06/AU25 co-activation grows with steps,
  plateaus ~50k, mild 100k dip — mirrors the AU12 dose-response.
- **Bonus:** the model reproduces the full happy expression covariance (anti-coupling of
  AU23/24/17/15), and the independent detectors caught a py-feat artifact (AU20) and an
  uncorroborated signal (AU43).

## Caveats

- py-feat outputs are presence *probabilities*, not FACS intensities → association claims only
  (Spearman). LibreFace does give FACS-scale intensities (a partial remedy).
- py-feat's 20 heads share features → AUs co-detect even on real faces; the paired Δr, the
  baseline-subtracted matrix, and the two independent detectors are the mitigations. Where the
  three detectors disagree (AU20, AU25, brow), the independent ones are given more weight.
- Per-AU reliability varies; AU07/AU20 excluded as not estimable.
- MEAD training is happy-only, female-only validation actors → the covariance reported is the
  MEAD-happy prior; do not overclaim generality across emotions.

## Implications for the draft (Cap05 + au-vector)

- Answers the certain banca question "what else does your model change?": the scalar axis
  renders a *coherent happy expression* (smile cluster up, mouth-closing/frown down, mild brow
  entanglement) — a feature (natural-looking) and a limitation (AU12 not isolable). Belongs in
  Cap05 as a generality/threats subsection.
- Strengthens the evaluation-protocol contribution (C2): the independent-detector guard is shown
  doing real work (catching the AU20/AU43 py-feat artifacts, correcting AU25/brow).
- Directly motivates the multi-AU / multi-emotion extension — FineFace's independent control
  localizes cleanly where our single axis does not (append recorded in the au-vector note).

Artifacts: scores under `inference_output/batch_eval/*/results_allau.jsonl` +
`results_libreface.jsonl`; stats `inference_output/batch_eval/trained/au_coactivation_stats.json`;
code `comp_metrics/expression_au_accuracy.py` (`detect_all_aus`), `scripts/rescore_allau.py`,
`scripts/rescore_libreface.py`, `scripts/analyze_allau.py`, `scripts/plot_allau.py`.
