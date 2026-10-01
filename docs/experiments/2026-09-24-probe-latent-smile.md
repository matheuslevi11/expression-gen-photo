# 2026-09-24 — Probe 1: linear SD-VAE smile direction (feasibility of the pure-latent operator)

Cheap feasibility probe for **path A** of the prompt-decoupled expression operator
([[2026-09-24-prompt-decoupled-operator]]): is there a direction in SD-VAE latent space that
decodes to a *graded, realistic, identity-preserving* smile? No diffusion, no retraining.
Script: `scripts/probe_latent_smile_direction.py`.

## Setup

- Fit a global smile direction $d = \text{mean}(z \mid \text{au12}>0.7) - \text{mean}(z \mid
  \text{au12}<0.15)$ over 400 MEAD frames (200 low + 200 high), encoded to SD-VAE latents
  ($\times 0.18215$, `latent_dist.mean`).
- Sweep 12 low-AU12 (neutral-ish) seeds along $z(\alpha)=z_0+\alpha d$, $\alpha\in\{0,.5,1,1.5,2,3\}$;
  VAE-decode each and score detected AU12 (py-feat), identity cosine (facenet-VGGFace2,
  decode$(z_0)$ vs decode$(z(\alpha))$), plus a visual grid (`sweep_grid.png`).

## Results

| $\alpha$ | detected AU12 | identity vs $z_0$ |
|---|---|---|
| 0   | 0.425 | 1.000 |
| 0.5 | 0.536 | 0.958 |
| 1   | 0.591 | 0.882 |
| 1.5 | 0.645 | 0.779 |
| 2   | 0.731 | 0.661 |
| 3   | 0.807 | 0.367 |

latent-CLS (Pearson $\alpha$ vs detected AU12) = **0.504** (moderate). VAE reconstruction ceiling:
encode$\to$decode of a real high-AU12 frame reads AU12 **0.87** vs raw **0.85** — the autoencoder
roundtrip is faithful.

## Decisive finding (visual)

The linear direction produces a real smile, **but it is entangled with a strong global darkening
and skin-tone shift**: across the $\alpha$ sweep the mouth forms a smile *while the whole image
gets darker and contrast-crushed*. By $\alpha\ge 2$ — the range needed to reach a strong smile
(AU12 $\gtrsim 0.7$) — faces are visibly darkened and identity collapses (0.66 $\to$ 0.37). The
degradation is a **photometric confound, not a clean expression edit** — exactly the
"global-brightness shortcut instead of geometry" risk noted in `docs/plan.md`. The faithful VAE
ceiling confirms the failure is the linear edit, not the decoder.

## Verdict

**The pure *linear* latent operator is not viable.** A clean, identity-preserving graded smile is
unreachable: modest $\alpha$ preserves identity but caps AU12 near 0.6, and reaching a strong
smile forces $\alpha$ into the region where the photometric entanglement destroys identity and
realism.

## What this does and does not rule out

- Tests only the **weakest** version of path A (a global *linear* mean-difference direction). A
  **learned nonlinear operator** $g(z_0, s)$ — what path A would actually train — could in
  principle disentangle smile from the brightness/skin-tone drift. The probe raises the bar (any
  latent operator must actively suppress this entanglement) but does not prove a learned one
  impossible.
- The confound is partly a **MEAD-data artifact** (high-AU12 frames correlate with lighting);
  per-identity paired directions or brightness-regressed targets might reduce it.
- The **VAE roundtrip is faithful**, so the autoencoder is not the bottleneck — the linear latent
  arithmetic is.

## Recommendation

Deprioritise the pure-latent operator (path A). Run **probe 2** next (diffusion editing via
DDIM-inversion / SDEdit, path B): it reuses the already-trained model, which produces realistic,
identity-preserving smiles (qualification: cross-frame identity 0.92), so it is the better route
to "animate a given face" and does not rely on the SD-VAE latent being linearly disentangled.
Artifacts: `inference_output/probes/latent_smile/{summary.json,sweep_grid.png}`.
