# Full training run — 100 k steps (May 21–26, 2026)

Config: `configs/train_genphoto/expression.yaml` (effective batch 4 via grad accumulation, single RTX 3090).

## Run history

| Run directory | Steps | Note |
|---------------|-------|------|
| `expression-2026-05-21T21-40-07` | 50 | Initial test, stopped manually. |
| `expression-2026-05-21T22-07-08` | 23 250 | First real run, stopped manually. |
| `expression-2026-05-26T21-43-56` | 100 000 | **Resumed from step 23 000** and ran to completion. |

**Final checkpoint:**
`output/expression/expression-2026-05-26T21-43-56/checkpoints/checkpoint-step-100000.ckpt`

## Training stability

- Loss remained very low and stationary for most of the run (frequently < 0.01) with occasional spikes up to ~0.2. No divergence or OOMs.
- GPU memory stable at ~21.4 GiB (RTX 3090, 24 GiB).
- Validation GIFs emitted every 5 000 steps (capped at 8 clips).

## Checkpoint housekeeping note

The final run produced ~77 checkpoints (~2.6 GB each, ~200 GB total). Before deleting any of them, run the **training-step dose–response** experiment from the Evaluation Robustness checklist in [`../status.md`](../status.md) — the intermediate checkpoints are the raw material for it. After that, it is safe to keep only `checkpoint-step-100000.ckpt` plus one mid-run fallback (e.g. `checkpoint-step-50000.ckpt`).
