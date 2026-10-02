# SDumont2nd distributed smoke test (October 1, 2026)

First runs of the training code on the LNCC SDumont2nd cluster (H100 nodes), validating the SLURM launch path — not the model graph (covered by [2026-05-16 smoke test](2026-05-16-smoke-test.md)).

Script: `scripts/slurm/smoke_expression_sdumont2nd.srm`, partition `h100_dev`, account `crono`.
Config: filled-in copy of `configs/train_genphoto/expression_sdumont2nd_smoke.yaml` (6 steps, grad-accum 4, validation at step 3 with 2 clips, checkpoint at step 6).

| Job | Layout | Result | Wall | Effective batch |
|-----|--------|--------|------|-----------------|
| 665303 | 1 node × 2 GPU | COMPLETED | 1m25s | 8 |
| 665334 | 2 nodes × 2 GPU | COMPLETED | 1m36s | 16 |

Both runs: 6 optimizer steps (loss 0.003–0.034), ~0.2–0.25 s/iter, 19.0 GB GPU memory per rank, validation GIFs under `samples/sample-3/`, `checkpoint-step-6.ckpt` (~2.6 GB) containing encoder (150 tensors) + attention `merge` (40) + optimizer state.

Confirmed:
- `init_dist(launcher="slurm")` maps ranks to distinct GPUs (1-node: cudaDev 0/1 on different bus IDs).
- Intra-node NCCL uses P2P/IPC; cross-node uses InfiniBand with GPUDirect RDMA (`NET/IB/*/GDRDMA`) with `NCCL_SOCKET_IFNAME` unset.
- The `genphoto` env (2023 pins, torch 2.1.1 / CUDA 12.1) runs on H100 (`sm_90` in arch list; xformers attention OK).

Problems found and fixed in the scripts before these passed:
1. **`h100_dev` QOS (`sh_h100_dev`) caps 48 CPU / 2 GPU / 500 GB per node.** A full-node request pends forever with `Reason=QOSMaxCpuPerNode`; `sbatch --test-only` does not check QOS. Smoke script now requests 2 GPUs/node. `petrobr-h100` (QOS `petrobr_h100`) has no cap, so the production script keeps 4 GPUs/node.
2. **`set -u` breaks `conda activate`** (MKL activate hook reads unset `MKL_INTERFACE_LAYER`; job failed in 2 s). Activation is wrapped in `set +u … set -u`.
3. **Submitting from a shell with `genphoto` active made every task run base-conda python** (`No module named 'omegaconf'`): sbatch exports the submit env, `module load anaconda3` re-prepends base `bin`, and `conda activate genphoto` is then a no-op. Reproduced with a debug job; scripts now deactivate any inherited env first and fail fast if `python` is not `$CONDA_PREFIX/bin/python`. Job 665334 was deliberately submitted from an activated shell.

Benign noise in the logs: c10d IPv6 socket warnings (falls back to IPv4), `accelerate` not installed (slower model load only), `terminate called without an active exception` at process exit (exit code 0).

Not yet exercised: 4 GPUs/node layout (production only), long runs, resume.
