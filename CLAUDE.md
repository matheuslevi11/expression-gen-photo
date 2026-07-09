# Generative Expressions (Generative Photography fork)

Research fork adapting GenPhoto's camera-parameter control to facial-expression (smile / AU12) intensity control. SD1.5 + AnimateDiff backbone, frozen; only the expression encoder + attention `merge` weights train. See `docs/status.md` for current state and `docs/plan.md` for design rationale.

## Environment

- Conda env: `genphoto` (torch 2.1, numpy 1.x pins — see `environment.yaml`).
- Always set `CUDA_VISIBLE_DEVICES=<free GPU>` for training/inference; the scripts map `local_rank=0` to physical GPU 0 otherwise. Check with `nvidia-smi --query-gpu=index,memory.free --format=csv`.
- Large artifacts live on `/databases-4tb` (checkpoints, `MEAD_processed/`, `inference_output/`) — never write training outputs to `/` (it is nearly full). `MEAD*`/outputs are gitignored.
- Pretrained backbones: HF cache snapshot `models--pandaphd--generative_photography` (SD1.5 + `unet_merged`, `RealEstate10K_LoRA.ckpt`, `v3_sd15_mm.ckpt`).

## Documentation convention (MUST follow)

Docs are split into a *living state* doc and an *append-only lab notebook*:

```
docs/
  plan.md          # design rationale — frozen, do not extend
  status.md        # living doc: exec summary, experiment log, checklists, commands, file map
  experiments/     # lab notebook: one file per experiment, YYYY-MM-DD[-]slug.md
```

Rules:

1. **`docs/status.md` is edited destructively.** Keep it slim (~150 lines). Check items off in place; delete claims that turn out wrong. There is exactly **one** readiness checklist and **one** Evaluation Robustness checklist — never paste a new cumulative copy of a checklist.
2. **`docs/experiments/` files are append-only.** Never rewrite a past entry's measurements or conclusions. If a finding is later refuted, add a short **correction banner** at the top of the old file linking to the new date-stamped entry that supersedes it (see `2026-07-08-trained-vs-untrained.md` for the pattern), and record the corrected analysis in the new entry.
3. **Every new experiment = one new date-stamped file + one row in the Experiment log table in `status.md`** (+ tick any checklist items it closes). Do not append experiment sections to `status.md` itself.
4. Config templates committed under `configs/` keep `/path/to/...` placeholders; configs with real local paths belong in scratch space, not in git.

## Verification habits for this repo

- Inference ablations must be seed-controlled: `inference_expression.py --seed` re-seeds immediately before sampling, so variants with different construction-time RNG draws still get identical initial noise. Same seed + same config ⇒ bit-identical GIFs.
- The fair no-training baseline is `configs/inference_genphoto/expression_baseline.yaml` (no camera-conditioned processors). It is bit-equivalent to running with processors installed but no adaptor checkpoint (merge layers are zero-initialised).
- Quantitative expression eval: `comp_metrics/expression_au_accuracy.py` (py-feat AU12 Pearson *r*). Note the circularity caveat in `docs/status.md` — training labels also come from py-feat.
- ⚠️ Do not delete intermediate checkpoints under `output/expression/` until the training-step dose–response experiment (Evaluation Robustness checklist) has run.
