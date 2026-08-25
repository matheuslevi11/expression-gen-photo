# 2026-08-11 — Design note: AU-vector control (plan.md Option B), FineFace-style

**Type:** design note (no experiment run). Question: should the fork move from scalar
AU12 intensity (Option A) to a multi-dimensional **AU vector** (Option B in `plan.md`),
as FineFace does with its 12-AU vector?

**Verdict up front:** worth doing *as a future direction* — it is already Option B and
multi-emotion is already listed as a follow-up — but it is a **large, data-gated** item,
**not** a cheap next experiment, and it should **not** be scoped as a FineFace
replication. Framed correctly it is *multi-AU / multi-emotion temporal **trajectories***
(the sequence-native differentiator applied to a path through AU space), and it should be
sequenced **after** the cheap claim-closing items (training-step dose–response — time
critical — then LPIPS/CLIP and the prompt-engineering baseline).

## Three things that make the naive version a trap

### 1. Data is the binding constraint (dominant cost)

Current training data is MEAD **happy-only** (`--emotion happy`, 4,028 train clips).
Happy ≈ AU6 + AU12, so **every other AU has ~zero variance across all training frames**.
A K-dim AU vector trained on this distribution learns *dead inputs* — it is an AU12 model
with extra constant channels, and the AU-MSE on non-smile AUs would be meaningless.

Real multi-AU control therefore requires **multi-emotion MEAD** (the 8 emotions defined
in `preprocess_mead.py:70`: angry, contempt, disgusted, fear, happy, neutral, sad,
surprised) → re-preprocess the footage + a **full retrain from scratch** (~100k steps,
days on a 3090). This is the opposite of the current Evaluation Robustness checklist,
every item of which is explicitly *no-retrain*.

- **Labels are nearly free.** py-feat's XGBoost head already computes the full AU set;
  `AUEstimator` (`preprocess_mead.py:517`) keeps only AU12 and discards the rest
  (`Annotation.au12`, line 116). Re-running preprocessing to retain the full AU vector
  needs **no new model** — just keep more columns. So the label cost is low; the
  **footage preprocessing + storage + retrain** is the real cost.

### 2. Injection-mechanism mismatch with the fork's own design

FineFace injects its 12-AU vector via **token-level decoupled cross-attention**
(`Z = Attn(Q,K_text,V_text) + λ·Attn(Q,K_AU,V_AU)`) — **no spatial map**. This fork's
entire mechanism is the **6-channel spatial physical map** reused from GenPhoto
(`cin: 384`; `create_intensity_embedding` broadcasts a *scalar* to `[f, 3, H, W]`;
`expression_dataset.py:31`). A K-dim vector does **not** broadcast to a spatial map
naturally. "Like FineFace" therefore means one of:

- **(a)** adopt FineFace's cross-attention injection → abandons the 6-channel
  spatial-map mechanism that *is* the fork's architectural identity, or
- **(b)** pack K AU values into the 3 physical channels (e.g. per-AU spatial regions via
  anatomical priors, as `plan.md` Option B sketches) → keeps the mechanism but is real
  new engineering and untested.

Neither is a drop-in change to the scalar path.

### 3. Strategic: it walks onto FineFace's home turf

The 2026-07-21 head-to-head already found ramp-following a **tie** and FineFace **ahead
on absolute calibration** (CLS 0.50 vs 0.07). Multi-AU + combinations is **exactly**
FineFace's demonstrated strength (12 AUs, 50 combinations, out-of-range extrapolation).
A direct AU-vector replication competes where the fork is at best at parity, and dilutes
the one clean differentiator (identity-consistent temporal sequence). It also does **not**
address the current top limitation (no absolute calibration) and could *spread* that
deficit across more axes under the same relative-only training regime.

## The defensible framing

Not "AU vector like FineFace" but **multi-AU / multi-emotion temporal trajectories**: a
single identity-consistent clip that traverses a *path* through AU space
(e.g. neutral → surprise → smile), with the per-frame condition commanding a trajectory
rather than a static vector. All four related-work systems are either independent stills
(FineFace) or single-image edits (EmojiDiff/MagicFace/PixelSmile) — **none** produces an
identity-consistent multi-AU trajectory. This *extends* the sequence-native contribution
instead of surrendering it, and is the version worth a paper section.

## Minimal scope if pursued (after the cheap items)

1. **Data:** re-preprocess a multi-emotion MEAD subset retaining the full AU vector
   (py-feat columns already computed; keep them). Start with 2–3 emotions that exercise
   distinct AUs (e.g. happy=AU6+12, surprised=AU1+2+5+26, sad=AU1+4+15) before the full 8.
2. **Embedding:** decide injection path — (b) multi-channel/region spatial map to preserve
   the GenPhoto mechanism, vs (a) cross-attention to match FineFace. Prefer (b) for
   novelty coherence; prototype both cheaply on the scalar checkpoint first.
3. **Eval:** FineFace-style **AU-MSE** over the vector (LibreFace / a non-py-feat detector
   to avoid the labeled-AU circularity), plus per-AU CLS, plus the existing cross-frame
   identity metric — the trajectory/identity story is the point of comparison.
4. **Guard:** confirm non-smile AUs actually have training variance before trusting any
   multi-AU number (the trap in §1).

## Relation to plan.md / status.md

`plan.md` Option B ("AU vector, richer control") and its "Next steps … multi-AU control
(Option B)" name this directly; `status.md` Known limitations already flag *"Option A
embedding only … no multi-AU control yet"* and *"Single emotion filter … multi-emotion is
a follow-up."* This note records the **why-not-yet** and the **correct framing** so the
item is actionable rather than aspirational. Backlog row added to the `status.md`
Experiment log (2026-08-11).

## Empirical update (2026-08-25): co-activation strengthens the case

The AU co-activation analysis ([[2026-08-25-au-coactivation]]) gives this proposal an
empirical spine. The single AU12 axis does **not** control AU12 in isolation: it drives the
whole MEAD-happy covariance (AU06/AU25/AU10 up, AU23/24/17/15 down, subtle brow entanglement
that py-feat underreports but LibreFace + geometry reveal). The prior-art contrast is the
key datum: **FineFace's independent 12-AU control localizes cleanly** (AU01 leakage 0.017 vs
our 0.337; AU23 +0.18 vs our −0.72) where our scalar axis inherits the full expression
prior. So multi-AU control is not just "richer" — it is what would let the model
**decouple** axes the current formulation entangles. Note this does not change the §1 data
blocker: exercising non-smile AUs still requires multi-emotion MEAD (happy-only gives no
variance on the very AUs the co-activation analysis shows moving as a locked cluster).
