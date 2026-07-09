# Fair frozen-backbone baseline verification (July 9, 2026)

Closes the "Fair SD1.5 baseline" readiness item and corrects the diagnosis recorded in [`2026-07-08-trained-vs-untrained.md`](2026-07-08-trained-vs-untrained.md).

## Why the July 8 diagnosis was wrong

The July 8 write-up claimed the "untrained" output was corrupted because `set_all_attn_processor()` installs "randomly-initialised merge weights". Reading `genphoto/models/attention_processor.py` shows every `merge` layer (`qkv_merge` / `q_merge` / `kv_merge`) is **zero-initialised at construction** (`init.zeros_` on weight and bias), and each is used as `merge(h + camera_feature) * scale + h` — so with no adaptor checkpoint loaded the camera path is mathematically the identity: the (random) expression encoder's features cannot influence any attention layer. The RealEstate10K LoRA checkpoint was also verified to contain **no** merge keys (256 keys, all `to_*_lora`), so nothing overwrites the zeros.

## Experiment design

Three inference variants with the same prompt ("A portrait photograph of a young woman, frontal view, neutral background."), the same intensity list `[0.0, 0.25, 0.5, 0.75, 1.0]`, and the same seed (42, re-applied immediately before sampling — see code changes below):

| Variant | Attention processors | Adaptor ckpt | Output |
|---------|---------------------|--------------|--------|
| **Bypass baseline** | `add_temporal: false` — no camera-conditioned processors installed anywhere | none | `inference_output/fair_baseline/baseline_bypass/` |
| **Zero-merge "untrained"** | camera processors installed (as in all prior runs) | none | `inference_output/fair_baseline/baseline_zeromerge/` |
| **Trained** | camera processors installed | 100 k ckpt | `inference_output/fair_baseline/trained_asc/` |

## Results

- **Bypass ≡ zero-merge: max pixel diff = 0** across all 5 frames. The camera-conditioned attention path with unloaded (zero-initialised) merge weights is *bit-exactly* inert — installing the processors without a checkpoint is already a fair baseline, not "broken attention".
- Both are also **bit-identical to the July 8 `untrained.gif`**, retroactively validating the earlier ablation as fair.
- The trained run is bit-identical to the July 8 `trained_asc.gif` (deterministic reproduction of the ablation).

| Variant | Detected AU12 | Pearson *r* vs target | AU12 span |
|---------|---------------|----------------------|-----------|
| Frozen backbone (fair baseline) | [0.031, 0.032, 0.030, 0.018, 0.020] | −0.85 | ~0.01 (flat) |
| Trained (100 k) | [0.098, 0.072, 0.157, 0.771, 0.949] | **+0.91** | **~0.85** |

The frozen backbone renders a coherent (if oversaturated, psychedelic-background) portrait with **zero** expression response to the conditioning; the trained adaptor produces a photorealistic MEAD-domain portrait with a strong monotonic smile ramp. Combined with the descending-list reversal (*r* = −0.98), this is clean evidence that the 100 k training run — and nothing else in the pipeline — is responsible for nuanced expression control.

A labeled side-by-side figure is at `inference_output/fair_baseline/comparison.jpg`.

## Code / config changes

1. **`inference_expression.py`** — added `--seed` (default 42), re-applied via `torch.manual_seed` + `torch.cuda.manual_seed_all` immediately before the pipeline call so model variants that consume different numbers of RNG draws during construction still sample identical initial noise; logs a "Baseline mode" notice when no camera-conditioned processors are requested.
2. **`configs/inference_genphoto/expression_baseline.yaml`** — new template: `expression_adaptor_ckpt: null` + `attention_processor_kwargs.add_temporal: false` (true bypass; equivalent to zero-merge, verified above).
