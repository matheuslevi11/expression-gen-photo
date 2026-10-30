# Longer training: design and pre-registration (October 29, 2026)

This entry follows [2026-10-29 low-end retrain results](2026-10-29-low-end-retrain-results.md). There, the treatment's calibration was still rising at its last checkpoint: CLS MediaPipe on the original units went 0.27 → 0.30 → 0.40 → 0.46 over EMA 10k → 25k.

**Question:** does a longer cosine schedule raise calibration further without costing ramps, permuted following or identity?

This entry fixes the design and the readings before any result; the results will go in a new entry.

## Runs

| | config | change vs `expression_lowend_treat.yaml` | job |
|---|---|---|---|
| treatment 50k | `expression_lowend_treat_50k.yaml` | `max_train_steps` 50,000 (cosine horizon 50k); checkpoints every 5k | 752058 (1 node × 4 H100, 6 h limit) |
| seed replicate | `expression_lowend_treat_s142.yaml` | `global_seed` 142 | 752059 (1 node × 4 H100) |
| reference (main model) | the existing treatment run, seed 42, 25k | — | 750648 |

- **Why a fresh 50k run.** The 25k run's LR is already 0, so restarting the cosine from its end would confound length with a warm restart.
- **What changes in the 50k run.** It keeps seed 42. Data sampling and diffusion noise do not depend on the weights, so it should see the same data and noise sequence as the reference for its first 25k steps; only the LR differs. This is expected from the code, not verified.
- **Why seed 142, not 43.** Per-rank seeds are `global_seed + rank`, so seed 43 would reuse three of seed 42's four per-rank streams (43–45). Seed 142 uses 142–145.
- **Why a replicate.** The paired bootstrap covers generation noise (prompts and seeds), not training-run noise, and [2026-10-05](2026-10-05-cluster-control-discrepancy.md) showed that two runs of one recipe can differ. The replicate gives one draw of run-to-run variation.
- **Code.** Unchanged since the 10-29 runs (HEAD `2c70575`); only the two configs are new.

## Evaluation

**Fresh units**, used by none of the models' earlier evaluations:
- `calibration_jitter` and `permuted`: prompts 10–39 × seeds 48–50 (90 units). Prompts 30–39 were never used in these protocols, and seeds 48–50 were never used anywhere.
- `scaled`: 40 prompts × seeds 48–50 (120 units).

**Models:**
- treatment 25k, EMA (eval job 752060);
- treatment 50k, EMA-50k;
- replicate, EMA-25k.

The last two are submitted when their training ends. Statistics are a paired 2,000-draw bootstrap over units.

## Pre-registered readings

**Checkpoint:** the EMA checkpoint at each run's last step (50k or 25k), fixed now.

1. **Calibration (primary).** The CLS MediaPipe Δ (50k − 25k) has a CI lower bound > 0, and a point estimate larger than the replicate's |Δ| (seed 142 − seed 42) on the same units.
2. **Guardrails** (point estimates, 50k − 25k):
   - ramp MediaPipe r ≥ −0.05;
   - permuted AU12 r and permuted MediaPipe r ≥ −0.05;
   - ramp worst-case adjacent identity ≥ −0.02. Identity fell with steps in the 25k run, from 0.903 to 0.888.

**Also reported:**
- CLS py-feat;
- c ≤ 0.1 and frame 0 on both detectors;
- fail rate, permuted identity, calibration curves;
- 50k − seed 142;
- the replicate's Δ on every metric. That is the run-to-run noise, and also a robustness check of the 10-29 numbers.

**Decision rules:**
- If 1 and 2 hold, the treatment 50k becomes the main model.
- If 1 holds but 2 fails, longer training trades the guarded axis for calibration. The 25k stays the main model and the trade-off is reported.
- If 1 fails, the 25k stays. Calibration has plateaued for this recipe, and the next lever is not training length (e.g. a magnitude-aware CCL, or data).

**Analysis code.** `longer_analysis.py` implements these readings. It was written, and dry-run on stand-in directories, before any result. It is saved in `inference_output/batch_eval/fresh/longer_analysis/` with the eval submitter, and writes `inference_output/batch_eval/fresh/longer_report.json`.
