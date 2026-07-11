# 2026-07-09 — Design note: StyleGAN as a paired-data "renderer" for expression ramps

**Type:** design note (no experiment run). Question: can StyleGAN play the role that
deterministic physics rendering (BokehMe, crop+resize, sensor simulation) plays in the
original GenPhoto data pipeline — i.e. generate ground-truth expression-ramp training
frames — instead of (or alongside) MEAD?

## What made the physics simulators special

GenPhoto's training-data generation has three properties:

1. **Paired by construction** — the base image is untouched except for the one effect;
   "same scene, different parameter" is guaranteed, not measured.
2. **Calibrated by construction** — the ground-truth label (kernel size, focal mm) is the
   *input* to the renderer; labels are exact and free.
3. **Cheap and deterministic** — on-the-fly per-batch generation, arbitrary parameter
   coverage from one photo.

The MEAD pipeline gives up all three (real footage, py-feat-measured labels, fixed
identity/intensity distribution) — which is where the known limitations come from: male
baseline-smile bias, high AU12 at low labeled intensity, py-feat train/eval circularity.

## How StyleGAN latent editing scores

Machinery: sample identity `w` in StyleGAN2-FFHQ latent space, walk a smile direction
(InterFaceGAN boundary, or StyleSpace channels, which are more disentangled). Fixed `w` +
edit magnitude α is deterministic and costs one forward pass, so GenPhoto-style on-the-fly
generation is genuinely feasible.

- **Paired by construction: partially.** Smile directions in W space entangle with other
  attributes (teeth, eyes, age drift at large α). "Only the expression changed" becomes a
  soft guarantee that must be enforced with an ArcFace-similarity filter, unlike BokehMe
  where it holds for free. StyleSpace edits / conditional projection mitigate, don't
  eliminate.
- **Calibrated by construction: no — the crux.** α is a latent step size, not AU12. The
  α→AU12 mapping is nonlinear and *identity-dependent*, so labels must still be measured
  by a detector.
- **Cheap / deterministic / coverage: yes, fully.** Unlimited identities, arbitrary
  intensity spacing, demographic balance by rejection sampling — directly attacks the MEAD
  distribution artifacts.

## Two ways to close the calibration gap

1. **Measure, don't prescribe.** Generate ramps on a dense α grid, label every frame with
   a detector, train on measured values — same philosophy as the MEAD pipeline but with
   perfectly identity-paired frames and controllable coverage. Labeling with MediaPipe
   mouth-corner geometry (already a dependency) instead of py-feat breaks the train/eval
   circularity in one move.
2. **Calibrate per identity.** Generation is fast enough to binary-search α per identity
   until the detector reads exactly AU12 = 0.25 / 0.5 / 0.75… This recovers something close
   to calibration-by-construction: frames at *exact* target intensities, which MEAD cannot
   provide. Also exactly the data the dose–response calibration experiment (Evaluation
   Robustness checklist) wants.

## Risks specific to this fork

- **Domain gap.** FFHQ faces are aligned, frontal, studio-like, with characteristic GAN
  artifacts; the expression encoder + merge layers would partly learn FFHQ statistics.
  MEAD has the mirror-image problem (lab lighting, few identities) → the right use is
  *mixing* both sources, not replacing one with the other.
- **Entanglement is adversarial to the AU12 metric.** If the smile direction also
  brightens/reshapes the face, the model can learn correlated shortcuts — the "global
  brightness instead of geometry" risk from `plan.md`. Cheap insurance: drop generated
  ramps whose frame-to-frame ArcFace similarity dips below threshold.

## Relation to plan.md options

`plan.md` names 3DMM re-rendering (DECA/EMOCA/FLAME) as the most principled analog —
there the expression coefficient *is* a defined parameter. StyleGAN sits between that and
MEAD: more photorealistic than raw 3DMM renders, less calibrated than 3DMM coefficients.
The best-of-both family is 3DMM-rigged generators (StyleRig, GIF: FLAME parameters driving
StyleGAN) — the natural data engine if the fork moves beyond scalar AU12 to Option B/C
(multi-AU or landmark flow).

## Verdict

Viable and well-matched — but framed correctly it is not "deterministic physics rendering
with StyleGAN" (the calibration guarantee does not transfer); it is a *deterministic,
identity-paired frame generator with post-hoc or feedback-loop labeling*. Concrete value:
unlimited paired identities at exact target intensities, demographic balance against the
male-baseline bias, and a clean path to breaking py-feat circularity — at the cost of a
domain gap managed by mixing with MEAD.
