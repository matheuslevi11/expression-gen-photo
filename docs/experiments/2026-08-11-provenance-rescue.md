# 2026-08-11 — Provenance rescue: three method facts for Cap04

Rescue note (no run). Step 0 of the qualifying-draft plan
(`plans/qualifying_draft_plan.md`): record three method/provenance facts that lived only
in code or were unrecorded, so Cap04 can be drafted **from the code** (not the stale
`docs/plan.md`) and can answer the certain banca question "what is your test set?".
All three verified 2026-08-11 against `train_expression.py` and the on-disk annotations.

## Fact 1 — the "100 k-step run" is three segments, and AdamW moments reset at the resume

The single "100 k" trajectory is actually: a 50-step manual test, a 23,250-step run
stopped manually, then a **resume from step 23,000** that ran to 100 k (see
[2026-05-26-training-run-100k](2026-05-26-training-run-100k.md) run-history table). The
resume is a **weights-only resume**:

- **Saved** each checkpoint (`train_expression.py:450-459`): `global_step`,
  `camera_encoder_state_dict`, `attention_processor_state_dict`, **and**
  `optimizer_state_dict` (line 457).
- **Restored** on resume (`train_expression.py:321-335`): only `global_step`,
  `camera_encoder_state_dict`, and `attention_processor_state_dict`. The
  `optimizer_state_dict` is **never read** — the resume logs "Loaded encoder + attention
  processor weights for resume."

⇒ **AdamW first/second-moment estimates reset to zero at step 23,000** (and the LR
schedule is re-derived from `global_step`, not restored). Reportable and previously
undocumented. Practical impact is likely small (moments re-warm within tens–hundreds of
steps and the loss was already low and stationary by 23 k), but it must be stated in
Cap04's training-details paragraph rather than implying one clean optimizer trajectory.

## Fact 2 — `docs/plan.md` is stale in three ways; draft Cap04 from the code

`plan.md` is the *design rationale* (frozen), and it predates the implementation. Do
**not** quote it for these three; the code is authoritative:

1. **Trainable set.** plan.md says "train the camera encoder … **plus LoRA**"
   (lines 83, 131). The code **re-freezes LoRA**: `spatial_attn_proc_modules` are set
   trainable, then every param whose name contains `"lora"` is set back to
   `requires_grad = False` (`train_expression.py:204-209`). The actual trainable set is
   the **expression encoder + the attention `merge` params** (`:217-224`) — 218 M params
   (199.26 M encoder + 18.96 M merge), matching the 2026-05-21 validation note. LoRA
   (RealEstate10K) is a frozen backbone component, not trained.
2. **Metrics.** plan.md proposes **OpenFace** for AU agreement and **FID** for image
   quality (lines 97, 132). The implementation uses **py-feat** for AU12
   (`comp_metrics/expression_au_accuracy.py`), and the evaluation-methods study argues
   *against* FID (none of the four closest works use it; it is uninformative for
   controlled single-attribute edits on small sets). Cap04 §4.6 uses py-feat + MediaPipe
   + facenet identity, not OpenFace/FID.
3. **Filenames.** plan.md names `genphoto/data/dataset.py` (line 101); the real dataset
   class `ExpressionMEAD` lives in **`genphoto/data/expression_dataset.py`**.

## Fact 3 — the train/val split is a clean 2-actor holdout; no MEAD in the headline eval loop

Audited `MEAD_processed/annotations/{train,validation}.json` (read-only):

| | clips | actors | emotion | view |
|---|---|---|---|---|
| train | 4,028 | 45 (27 M / 18 W) | happy | front |
| val | 176 | 2 (**W009, W011**) | happy | front |

- **Actor overlap = 0.** The 4,028/176 (~95.8/4.2) split is a **clean actor holdout**
  (two held-out identities), *not* a random or `--val-fraction 0.1` split — so there is
  **no identity leakage** into the training-time validation loss/samples. This resolves
  the plan's "matches neither" uncertainty: it is an actor holdout, just an uneven one.
- **Caveat:** both validation actors are **female**, so the training-time val signal does
  not cover the documented male-baseline-smile bias.
- **The point for the banca:** this split governs only *training-time* validation
  sampling. **No MEAD data is in the headline evaluation loop** — all reported results
  generate from the 40-prompt text bank (`configs/eval_prompts.txt`) and are scored by
  external detectors (py-feat, MediaPipe, facenet). Raw MEAD intensity is 3 discrete
  levels (1/2/3, ~balanced: 1345/1338/1345); the dataset composes sorted-frame ramps from
  these. Cap04 §4.1/§4.5 must state this explicitly.

## For the draft

- Cap04 training-details: report the three-segment run + optimizer reset (Fact 1) and the
  encoder-+-merge trainable set (Fact 2.1); cite line numbers, not plan.md.
- Cap04 §4.1/§4.5: state the actor-holdout split and the "no MEAD in eval loop" fact
  (Fact 3) — pre-empts "what is your test set?".
- Cap04 §4.6 / Cap02 §metrics: py-feat + MediaPipe + facenet, not OpenFace/FID (Fact 2.2).
