# 2026-09-24 — Design note: prompt-decoupled expression operator (advisor proposal)

Design note (no run). Source: advisor comment at the qualification defence (2026-09-24).
Captures the proposal, its feasible instantiations, the risks, and how it sits in the backlog
now that retraining compute is available (experiments moving to a more powerful machine).

## The proposal (distilled)

Take the text prompt out of the core loop. The prompt is only a *content source* (identity,
scene, background — "use whatever you want, green background, yellow background"); the
contribution is the intensity vector **S**. Restructure the architecture as:

> a **face embedding** (produced from a prompt, or from any source) + a per-frame intensity
> $(s_1,\dots,s_F)$ → a **temporal set of embeddings**, each modulated by its $s_t$ → **VAE-decode**.

Stated endpoint: *"give me an embedding of a neutral face and I make it smile 0.2, 0.8."* i.e. a
**standalone, prompt-decoupled expression operator** that animates any given face. The advisor
asks explicitly whether the architecture can be **isolated** to this form — an architectural
direction, not only a diagram/framing change.

## Why it is compelling

- **Foregrounds the actual contribution.** The S-operator is the novelty; making it a separable
  module that acts on face embeddings expresses that directly.
- **Adds an editing/animation capability the current model lacks.** Today the model generates a
  face from text and cannot take an arbitrary neutral face and animate it. The operator would
  work on *any* supplied face.
- **Potentially cheaper.** A face produced/obtained once, then per-frame S-modulation, avoids
  re-deriving the whole clip through full prompt-conditioned diffusion.

## Relationship to the current architecture

Current pipeline: $(\text{prompt}, S)$ jointly drive the diffusion clip generation; the base face
is re-derived from the prompt each run, and S is merged into temporal attention
(`sec:method-conditioning`). Note the UNet *already* produces a temporal set of latents that the
VAE decodes — so "temporal embeddings, S-modulated, decoded" is partly what happens internally.
The advisor's real asks are (i) make the **content** come from a single reusable embedding so the
source is swappable (prompt **or** image), and (ii) isolate the **S-operator** as a reusable
module.

## Instantiations (honestly weighed)

- **(A) Pure latent operator** — learn $g(z_0, s)\to z_s$ on SD-VAE latents from \gls{mead}
  (encode the neutral frame $\to z_0$; encode the intensity-$s$ frame $\to$ target); at inference
  obtain $z_0$ from a prompt-generated image or an input image, apply $g$ per frame, VAE-decode.
  *Cheapest, editing-capable, strong base-identity, fully prompt-free.* **Central risk:** the
  SD-VAE latent is not semantically disentangled and the VAE decoder is not generative, so
  synthesising a large open-mouth/teeth smile from a neutral latent via a light operator may blur
  or fail. This is the make-or-break feasibility question.
- **(B) Reusable face-embedding conditioning** (most practical) — keep the diffusion generation
  but replace the text content-path with a reusable **face/identity embedding** (IP-Adapter-style
  decoupled cross-attention, produced once from a prompt or an input image), keeping the
  S-operator on top. "Prompt out of the game," content becomes a pluggable embedding, and a
  supplied neutral face can be animated. Retains diffusion realism; needs DDIM inversion for
  arbitrary input faces. This blends generation and editing under one S-operator and is likely
  the cleanest realisation.
- **(C) Framing only** — present the diagram with the prompt as interchangeable and S
  foregrounded, and demonstrate prompt-agnosticism. Cheap and honest, but does not deliver
  animation of an arbitrary input face.

## Two connections that govern sequencing

1. **This requires absolute calibration.** "Smile 0.2 vs 0.8" as distinct, correct intensities
   needs the absolute calibration the model currently lacks (constant-list CLS = 0.07). The
   calibration remedies (constant/permuted-list augmentation, AU dropout + expression-CFG, label
   smoothing) are therefore a **prerequisite** for this operator to deliver on the promise, not a
   competing direction.
2. **It moves toward the editing/animation camp** (MagicFace territory) that the qualification
   distinguished from. Re-position deliberately: the distinction is the *scalar graded-intensity*
   operator + *temporal sequence* + *embedding-native* + *prompt-decoupled*, and the strongest
   story frames it as **unifying** generation (face from prompt) and editing (face from image)
   under one S-operator. It is also the face-latent analogue of PixelSmile's text-embedding
   direction, and connects to [[2026-07-09-stylegan-data-note]] (latent editing) and
   [[2026-08-11-au-vector-proposal]] (S becoming a vector).

## Cheap de-risking probes (no retraining, before committing the machine)

1. **Latent-feasibility probe (targets risk of A).** Encode MEAD neutral + graded-intensity frames
   to SD-VAE latents; test whether a linear (or lightly-learned) direction produces graded smiles
   on decode, and specifically whether large expressions survive the VAE decoder. Answers the
   central technical risk before any operator is trained.
2. **Editing probe with the current model (targets B).** DDIM-invert an input neutral face to a
   latent, run the existing S-conditioned generation, and check whether it animates the supplied
   identity. Probes the editing capability without retraining.

## Verdict / placement

High-value and advisor-endorsed, but architecturally larger and riskier than the calibration
remedy. Recommended order: run the two cheap probes first; if the latent operator (A) is viable it
becomes the main new architecture, with calibration folded in; if not, pursue the reusable
face-embedding path (B), which keeps diffusion realism while still decoupling the prompt. Either
way, absolute calibration is on the critical path. Backlog row added to `status.md` (2026-09-24).

**Result — probe 1 (2026-09-24, [[2026-09-24-probe-latent-smile]]):** the *linear* version of path A
is **not viable** — a global SD-VAE smile direction entangles the smile with a darkening/skin-tone
shift, so a strong smile costs identity (0.66→0.37). Path A survives only as a *learned
disentangling* operator (higher bar, uncertain); this shifts priority to **path B** (diffusion
editing via inversion/SDEdit). Probe 2 is the next step.

**Result — probe 2 (2026-09-24, [[2026-09-24-probe-diffusion-edit]]):** SDEdit with the trained
model **confirms path B as the direction** — at strength ≈0.7 it animates a *supplied in-domain*
neutral face into a graded, identity-preserving smile (ramp r=0.85, id→input 0.68) with **no
darkening**, zero-shot. But it works only in a **narrow strength band** (identity collapses to 0.24
by S≥0.9 = pure regeneration) and **does not transfer to out-of-domain faces** zero-shot. So
**zero-shot SDEdit is not the deliverable**: the production path B needs a *trained* reusable
face-embedding operator (IP-Adapter-style conditioning + inversion for arbitrary faces) at a fixed
S operating point, with absolute calibration on the critical path. This selects **B over A** and
specifies what the trained version must add. Both cheap probes are now done; the next step is
building the trained path-B operator on the powerful machine.
