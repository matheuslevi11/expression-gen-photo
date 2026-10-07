# Why the cluster control scored ramp r 0.65 vs 0.85 (October 5, 2026)

Follow-up to the open discrepancy in [2026-10-02](2026-10-02-calibration-augmentation.md) (Reading §3): the ramp-only control trained on SDumont2nd scored `scaled` AU12 r **0.646** at 25k, vs **0.850** for the original RTX-3090 run at 25k ([2026-08-11](2026-08-11-training-step-dose-response.md)), same protocol.

## 1. Evaluation environment: ruled out

The original `checkpoint-step-25000.ckpt` was copied from the workstation (sha256 `4cb74576…ce0de`, matched on arrival) and scored on the cluster with the cluster pipeline. Its inference config differs from the cluster control's only in the checkpoint path (job 675317):

| `scaled`, n=120 | Aug 2026 (workstation) | cluster |
|---|---|---|
| AU12 r | 0.850 | **0.855** |
| MediaPipe r | 0.854 | 0.856 |
| AU12 MSE | 0.083 | 0.084 |
| id mean-adj | 0.937 | 0.936 |

H100 inference, freshly downloaded py-feat/facenet weights and the cluster generation path all reproduce the workstation numbers. Paired, original − cluster control = **+0.209 [+0.145, +0.281]** AU12 r, +0.176 [+0.114, +0.242] MediaPipe r. The gap is in the trained weights.

## 2. Code drift since May: ruled out

The May code is not in git (the expression port landed in a single 2026-07-01 commit), so it was checked indirectly:

- `train_ds[0]` with today's code reproduces the 2026-05-21 validator fingerprint exactly: pixel range [−0.933, 1.000], `phys_mean=0.842`, `ccl_mean=2.29e-05`, intensities [0.640, 0.815, 0.869, 0.918, 0.970].
- Trainable sizes match: encoder 199.26 M, `merge` 18.96 M.
- The grad-accumulation fix predates the original run (applied 2026-05-21 before the 22:07 launch; ~2 s/step on the 3090 confirms 4 real micro-batches).
- Resolved-config diff, original vs cluster control: only the intended differences (accum 4 → 1 with 4 processes, launcher, workers, step budget, checkpoint interval). LR, seed, precision and architecture are identical.

## 3. The two trajectories agree to 10k, then the cluster run regresses

Control checkpoints scored with the same `scaled` protocol (jobs 675328–675331):

| step | cluster ctrl AU12 r | MediaPipe r | original AU12 r | original MP r |
|---|---|---|---|---|
| 5k | 0.728 ± 0.246 | 0.858 | 0.746 | 0.834 |
| 10k | **0.789** ± 0.238 | 0.836 | 0.779 | 0.798 |
| 15k | 0.757 ± 0.268 | 0.794 | – | – |
| 20k | 0.727 ± 0.335 | 0.789 | – | – |
| 25k | 0.646 ± 0.387 | 0.680 | 0.850 | 0.854 |

Up to 10k the cluster run matches the original on both detectors, so the cluster setup (DDP over 4 GPUs, H100 numerics) learns the same thing at the same rate. After 10k it regresses while variance grows. The regression is a **failure tail, not a uniform drop**:

| run | r < 0 | r < 0.5 | r > 0.8 | median |
|---|---|---|---|---|
| cluster ctrl 10k | 1% | 6% | 66% | 0.860 |
| cluster ctrl 25k | 8% | 24% | 50% | 0.803 |
| cluster aug 25k | 7% | 24% | 45% | 0.779 |
| original 25k | 0% | 3% | 77% | 0.883 |

The failing ramps already smile at the commanded-0 frame (detected AU12 0.6–0.9 where the original starts at 0.08–0.16 on the same prompt and seed): the model loses its neutral end on a subset of samples and falls back to MEAD-happy's high-smile prior. The failing sets of the two cluster arms overlap little (4 of 16–17), so this is stochastic instability, not specific hard prompts.

## Reading

- Not a cluster bug and not code drift. Late-training ramp control is **unstable on a single run**: the original also regressed later (50k 0.85 → 100k 0.77, std 0.13 → 0.21, [2026-08-11](2026-08-11-training-step-dose-response.md)), and this run started earlier. Whether the timing difference is seed-to-seed variance or helped by the original's AdamW-moment reset at 23k (2k post-reset steps before its 25k checkpoint, [provenance](2026-08-11-provenance-rescue.md)) is not resolved; the original's 15k/20k/23k checkpoints (on the workstation) would show whether it also dipped before the reset.
- **Consequence for 2026-10-02:** both calibration arms were compared at 25k, inside the degraded regime. The aug-vs-ctrl comparison is still paired and fair, but a single late checkpoint is a fragile operating point. Re-evaluating both arms at 10k, where the control is healthy, is the cleaner test (jobs 675354 / 675355).
- **Practical:** for cluster runs of this recipe, select checkpoints by evaluation (≈10k here) rather than assuming 25k is the plateau, and treat any single-run 25k number as having run-to-run spread on the order of ±0.1–0.2 r until a second seed measures it.

## Addendum: both calibration arms re-evaluated at 10k

Full protocols on each arm's `checkpoint-step-10000` (jobs 675354, 675355 + 675863), paired bootstrap as in 2026-10-02:

| Protocol | Metric | ctrl 10k | aug 10k | aug − ctrl (95% CI) | same Δ at 25k |
|---|---|---|---|---|---|
| calibration | CLS py-feat | 0.146 | 0.221 | +0.075 [−0.016, +0.158] | +0.124 [+0.041, +0.215] |
| | CLS MediaPipe | 0.109 | 0.185 | **+0.077 [+0.015, +0.142]** | +0.156 [+0.085, +0.237] |
| scaled | AU12 r | 0.789 | 0.789 | −0.000 [−0.045, +0.049] | −0.011 |
| | MediaPipe r | 0.836 | 0.763 | **−0.072 [−0.127, −0.016]** | +0.116 |
| | id mean-adj | 0.946 | 0.947 | +0.002 | +0.001 |
| permuted | AU12 r | 0.239 | 0.491 | **+0.252 [+0.145, +0.351]** | +0.331 |
| | MediaPipe r | 0.480 | 0.553 | +0.073 [−0.045, +0.178] | +0.264 |
| | id mean-adj | 0.961 | 0.938 | **−0.023 [−0.031, −0.016]** | +0.015 |

MediaPipe smile (×100) by commanded constant level, ctrl → aug: 0.0 1.63→1.54 · 0.1 1.62→0.97 · 0.2 1.97→1.28 · 0.3 1.97→1.39 · 0.4 1.97→1.55 · 0.5 2.00→1.64 · 0.6 2.11→1.69 · 0.7 2.10→1.92 · 0.8 2.19→2.10 · 0.9 2.29→2.21 · 1.0 2.36→2.32.

### Reading (revises 2026-10-02's conclusion)

1. **Part of the 25k "calibration gain" was the control degrading.** At the healthy 10k checkpoint the ramp-only control already has some absolute calibration (CLS 0.15 py-feat / 0.11 MediaPipe), far above the 0.07 measured on the original 100k model in July. The near-zero CLS that motivated this work is at least partly a late-training symptom (the same high-smile drift that produces the 25k failure tail), not purely a property of ramp-only training.
2. **The augmentation's calibration effect at 10k is small:** +0.08 CLS on both detectors, significant only on MediaPipe. Against the pre-registered criterion ("clearly above control on both detectors") it **fails at 10k** and passes at 25k. The aug response is still the same shape: lower floor at c = 0.1 (MediaPipe 0.97 vs 1.62) and a steeper rise (2.4× vs 1.5× from c = 0.1 to 1.0), but flat-ish through 0.1–0.5 and the c = 0.0 bump remains.
3. **Robust effect: non-monotonic following.** Permuted AU12 r +0.25 at 10k and +0.33 at 25k (both CIs exclude 0); MediaPipe agrees in sign but is significant only at 25k.
4. **Costs at 10k that 25k hid:** ramp MediaPipe r −0.07 and permuted-list identity −0.02 (both CIs exclude 0). Ramp AU12 r and ramp identity, the pre-registered guardrails, are unchanged.
5. Net: constant/permuted augmentation is a **modest calibration lever and a strong trajectory-flexibility lever**. It does not deliver FineFace-level absolute calibration (0.50). Checkpoint choice moves CLS as much as the augmentation does, so future calibration work should compare arms over several checkpoints (or seeds), not a single step.

### Incident: silent CPU fallback in evaluation

The first ctrl-10k calibration run failed at CUDA init on one node ("CUDA-capable device(s) is/are busy or unavailable"); its resubmission on another node ran **on CPU** (29 s per denoising step vs 0.12 s), because `inference_expression.load_models` silently picks CPU when CUDA is unusable, and hit the time limit after 7 samples. Those 7 CPU samples, plus 15 from a cancelled retry, were deleted and the protocol regenerated cleanly on GPU (job 675863). All other evaluation logs were checked: every sample was generated at GPU speed (median 3 s/sample). `scripts/slurm/batch_eval_sdumont2nd.srm` now aborts if CUDA is unusable, logs the node's GPU state at start, and appends rather than overwrites per-experiment logs.
