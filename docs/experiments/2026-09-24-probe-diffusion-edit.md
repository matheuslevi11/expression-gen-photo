# 2026-09-24 — Probe 2: diffusion editing (SDEdit) of a supplied face

Cheap probe for **path B** of the prompt-decoupled expression operator
([[2026-09-24-prompt-decoupled-operator]]): can the *already-trained* model animate an
**arbitrary supplied neutral face** into a graded, identity-preserving smile **without
retraining**, by SDEdit? No inversion training, no new weights.
Script: `scripts/probe_diffusion_edit.py`.

## Setup

- **SDEdit editor.** Encode an input face → SD-VAE latent $z_0$; tile it over $F=5$ frames;
  add noise to an intermediate level $t^\* = \text{strength}\cdot T$; denoise with the trained
  adaptor's UNet under the per-frame S-ramp $(0,0.25,0.5,0.75,1.0)$ and a generic prompt
  ("A portrait photograph of a person, frontal view, neutral background"). Mirrors the
  pipeline's DDIM loop from $t^\*$ instead of $T$. 25 steps, CFG 8.0, seed 42.
- **Strength sweep** $\{0.5, 0.7, 0.9, 1.0\}$ — the SDEdit knob (edit vs. regenerate).
- **6 inputs:** 4 **in-domain** MEAD low-AU12 frames (distinct actors M003/M005/M007/M009)
  + 2 **out-of-domain** faces = frame 0 of frozen-SD1.5 *baseline* clips (SD1.5-domain
  portraits, busy backgrounds; never seen expression conditioning).
- **Metrics:** `ramp_r` = Pearson(detected AU12, commanded S) [py-feat]; `id->input` =
  facenet cosine of each output frame to the *supplied* face; `cross_id_min` = min adjacent
  frame-to-frame identity within the clip (temporal coherence).
  Artifacts: `inference_output/probes/diffusion_edit/{summary.json,edit_grid.png}`.

## Results — in-domain (4 MEAD neutral faces), per strength

| strength | ramp $r$ | id→input | cross-id min |
|---|---|---|---|
| 0.5 | 0.68 | **0.75** | 0.95 |
| 0.7 | **0.85** | 0.68 | 0.93 |
| 0.9 | 0.88 | 0.36 | 0.91 |
| 1.0 | 0.85 | 0.24 | 0.87 |

**A sharp strength ↔ identity trade-off.** SDEdit works, but only in a band. At **S ≈ 0.7**
the model animates the *supplied* face into a graded, realistic smile (ramp $r=0.85$) while
still preserving the input identity ($0.68$); cross-frame identity stays high ($\ge 0.9$),
so the clips are temporally coherent. Push strength to $\ge 0.9$ and identity to the input
**collapses** ($0.36 \to 0.24$): high-strength SDEdit is near-full **regeneration**, so it
ignores the supplied face and re-draws a new one (a recurring "default" generated face
recurs across several distinct male inputs at S=1.0). The usable *editing* regime is the
low-strength part of the knob; the high-strength part is generation.

## Results — out-of-domain (2 SD1.5-baseline faces)

| strength | ramp $r$ | id→input | cross-id min |
|---|---|---|---|
| 0.5 | −0.86 | 0.59 | 0.84 |
| 0.7 | 0.64 | 0.28 | 0.65 |
| 0.9 | 0.76 | −0.10 | 0.74 |
| 1.0 | 0.83 | −0.09 | 0.77 |

**Zero-shot editing does not transfer to out-of-domain faces.** At low strength the ramp is
undefined or **inverted** (py-feat face-detection NaNs on the busy backgrounds; one input
gave $r=-0.86$); by the strength where the ramp is followed ($\ge 0.9$), identity to the
input is **gone** ($\le -0.09$) — the model has denoised the OOD image away and regenerated
an in-domain face. Caveat: these two OOD inputs were poorly chosen (frame-0 of baseline
clips with very busy backgrounds; one is barely a detectable face), so this is *suggestive*
of the failure, not a clean measurement of it — but the direction is unambiguous.

## Visual (`edit_grid.png`, input | 5 ramp frames per row)

- **In-domain, S=0.5–0.7:** each supplied face is animated neutral → graded smile
  (open-mouth, teeth) with the **same identity** and **no darkening** — the exact opposite
  of path A's photometric collapse ([[2026-09-24-probe-latent-smile]]).
- **In-domain, S≥0.9:** the face morphs into a *different* person mid-ramp; S=1.0 rows are a
  regenerated default face, not the input.
- **OOD rows:** busy backgrounds are denoised into a clean in-domain portrait as strength
  rises — the model regenerates rather than edits.

## Verdict

**Path B is the right direction and the editing capability is real — but zero-shot SDEdit is
not itself the deliverable.** Positives: at S≈0.7 the *already-trained* S-operator animates a
*supplied* in-domain neutral face into a graded, identity-preserving, temporally coherent
smile with no retraining and no darkening artifact — directly demonstrating the advisor's
"give me a neutral face, I make it smile 0.2/0.8" on faces from the training domain.
Limits: (i) it works only in a **narrow strength band** (identity breaks by S≈0.9);
(ii) it does **not generalize to out-of-domain faces** zero-shot (they get regenerated, not
edited); (iii) the ramp is still **relative**, not absolute-calibrated. Together these say
the production path B needs a **trained reusable-face-embedding operator**
(IP-Adapter-style decoupled conditioning + inversion for arbitrary faces), with the strength
fixed to a learned operating point and **absolute calibration on the critical path** — not
raw SDEdit. This confirms **path B over path A** and specifies what the trained version must
add over this probe.

## What this does and does not establish

- **Establishes:** the trained S-operator can drive graded expression onto an *externally
  supplied* latent (not only a prompt-generated one) — the decoupling the advisor asked for
  is reachable, in-domain, without retraining.
- **Does not establish:** arbitrary-face (in-the-wild) editing, absolute intensity, or that
  SDEdit is the final architecture. Those require the trained path-B operator + calibration.
- **Next:** on the powerful machine, prototype path B properly — face/identity embedding
  conditioning with inversion for input faces, folding in the calibration remedies
  (constant/permuted-list augmentation, AU dropout + expression-CFG) that
  [[2026-09-24-prompt-decoupled-operator]] flagged as a prerequisite.
