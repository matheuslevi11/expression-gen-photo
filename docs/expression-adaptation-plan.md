# Adapting Generative Photography to Facial Expression Control

This document is about a plan for adapting the Generative Photography codebase from **camera-parameter** control to **facial-expression** control

---

## What the system does (recap)

Before planning the adaptation, it helps to separate three roles that “camera parameter” plays simultaneously—each must be replaced independently.

### 1. The sequence abstraction

A short “video” of 5 frames is synthesized where each frame shows the **same scene** with a **different value** of one scalar parameter. The 3D UNet plus temporal attention enforces **scene consistency** across frames.

### 2. The parameter embedding

Each frame gets a **6-channel spatial tensor** `[f, 6, H, W]`:

- **Channels 0–2:** a **physics-based signal** — the embedding encodes what the parameter *looks like* (e.g. Gaussian blur footprint for bokeh, crop-ratio mask for focal length, exposure scale for shutter speed).
- **Channels 3–5:** the **CCL embedding** — differences between consecutive CLIP text embeddings of prompts like `<bokeh kernel size: 2.44>`, `<bokeh kernel size: 8.3>`. This encodes the **direction and magnitude of change** across frames.

### 3. The training data simulation

Ground-truth training frames are produced by applying **physics-based rendering** to base images (BokehMe for bokeh, crop+resize for focal length, sensor simulation for shutter speed), often **on-the-fly** per batch.

---

## Mapping each component to facial expression

### The sequence abstraction → same identity, different expression intensity

Instead of “same scene at five focal lengths,” you generate “same face at five expression intensities” (e.g. neutral → subtle → moderate → strong → peak smile). The temporal module’s job shifts from **scene consistency** to **identity consistency**. The architecture does not depend on camera semantics—it only enforces that a 5-frame sequence shares latent structure.

**Viability: high.** No fundamental architectural change required.

---

### The physics-based simulation → the hardest problem

Camera training data is cheap and deterministic: from one photo you can derive many variants with simple operations. For facial expressions there is **no equally cheap deterministic simulation** from an arbitrary image.

| Method | Quality | Cost | Identity preservation |
|--------|---------|------|-------------------------|
| **3D Morphable Model (FLAME / EMOCA)** | High | High — synthetic sequences | Near-perfect (same mesh, different expression coefficients) |
| **Face reenactment / warping (FOMM, TPS)** | Medium | Medium | Degrades under large expression changes |
| **Expression datasets (AffectNet, MEAD, CelebV)** | Variable | Low | Often not paired (different identities per class) |
| **AU-labeled FFHQ + retrieval** | Low | Low | Not paired |

The most principled analog to GenPhoto’s pipeline is **3DMM-based rendering**: fit a 3DMM (e.g. EMOCA, DECA) to each base image, then re-render the **same identity** at varying expression coefficients. Tradeoffs: synthetic-looking training data and possible domain gap versus photorealistic SD priors.

A pragmatic alternative: **MEAD** (Multimodal Emotional Audio-Driven) or **RAVDESS** — same actor, multiple emotions and intensities — closer to real paired expression data without full synthesis.

---

### The parameter embedding → expression embedding design

Camera embeddings exploit crisp physical meaning. Expressions can be represented at several levels:

**Option A — scalar expression intensity (closest to shutter speed / bokehK)**

- One scalar per frame, e.g. Action Unit **AU12** (lip corner puller, smile proxy) in `[0, 1]`.
- Physical channel: scalar broadcast to `[f, 3, H, W]` (like shutter speed).
- CCL: prompts such as `<smile intensity: 0.2>`, `<smile intensity: 0.7>` → CLIP differences.
- **Fastest path to a working experiment.**

**Option B — AU vector (richer control)**

- FACS Action Units (e.g. 17 AUs): expression as a vector in ℝ¹⁷.
- One axis per experiment (e.g. “smile axis” = AU6 + AU12), or one model per AU — mirrors GenPhoto’s one-parameter-per-model design.
- Physical channel: project AU activation onto **spatial maps** (e.g. mouth region for AU12) using anatomical priors.

**Option C — landmark displacement map (spatially rich)**

- Landmark displacement from neutral to target (MediaPipe, dlib) as a **displacement / flow-like map** for the physical channel — analogous to focal-length spatial signals.
- CCL text differences unchanged in spirit.

Option C (or B+C) is theoretically strong; **Option A** is the most tractable first step.

---

### The training objective

The diffusion loss and attention injection are **parameter-agnostic**. You would train the camera encoder (conceptually an **ExpressionEncoder**) plus LoRA from scratch on face data, with the frozen SD1.5 backbone.

---

## Adaptation map (components)

| GenPhoto component | Expression analog |
|--------------------|-------------------|
| `CameraBokehK` (etc.) dataset | `ExpressionIntensity` dataset (MEAD, DECA/FLAME renders, etc.) |
| `create_bokehK_embedding()` (etc.) | `create_expression_embedding()` (scalar, AU, or flow) |
| CCL text prompts | e.g. `<smile intensity: X.X>` |
| BokehMe / crop / sensor simulation | 3DMM re-render, AU-conditioned warp, or paired video frames |
| `CameraCameraEncoder` | Same architecture, new weights (rename conceptually to ExpressionEncoder) |
| `camera_adaptor_ckpt` | `expression_adaptor_ckpt` (trained for expressions) |
| Metrics: correlation vs. ground truth | AU accuracy (e.g. OpenFace), **ArcFace** identity similarity across frames |

**Likely file touch points:**

- `genphoto/data/dataset.py` — new dataset class
- `train_bokehK.py` → adapted `train_expression.py` (pattern)
- `inference_bokehK.py` → adapted `inference_expression.py` (pattern)
- `configs/` — new YAML
- `comp_metrics/` — AU / identity metrics as needed

**Often unchanged:** UNet structure, attention processor, motion module, pipeline skeleton, core training loss.

---

## Viability

**Reasons it can work**

- The camera encoder is **generic**: it consumes a spatial tensor; it does not assume blur kernels specifically.
- Temporal attention can learn **identity** consistency on face sequences the way it learned **scene** consistency on camera sequences.
- SD1.5 already encodes substantial face knowledge (e.g. LAION), so a frozen backbone is a reasonable base.

**Risks**

1. **Data:** Camera effects are cheap to simulate at scale; expression-varied **paired** sequences are scarce or synthetic. Narrow training data → overfitting to identities or expression types.
2. **Entanglement:** Identity and expression are not disentangled in SD1.5. The temporal path must preserve hair, skin, background, and face shape while changing only expression-related geometry—you may need an **identity regularizer** (e.g. ArcFace consistency across frames).
3. **Scalar approximation:** Real expressions are multi-muscle. A single scalar (e.g. AU12) simplifies the problem but may encourage shortcuts (e.g. global brightness) instead of true geometry.

---

## Suggested first experiment

1. **Data:** MEAD (or similar) — same actor, multiple intensities per emotion; start with one class (e.g. smile) and three–five intensity levels as frames.
2. **Embedding:** **Option A** — scalar AU12 (or smile proxy) + CCL prompts; minimal new engineering.
3. **Training:** Train **expression encoder + LoRA** as in GenPhoto; keep SD1.5 UNet frozen.
4. **Evaluation:** OpenFace (or similar) for AU agreement, ArcFace for **identity consistency** across the 5 frames, plus standard image quality metrics (e.g. FID).

**Next steps if it works:** landmark / flow-based physical channel (Option C), 3DMM-expanded training data, or multi-AU control (Option B).
