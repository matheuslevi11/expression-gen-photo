"""Generate FineFace AU12 sweeps for the head-to-head comparison.

Runs in the ISOLATED `fineface` conda env (torch 2.2 / diffusers 0.27), NOT
`genphoto`. Emits stills + a manifest.jsonl whose entries carry `"frames": [png,…]`,
which `scripts/batch_eval.py score` (run afterwards in the `genphoto` env) consumes
exactly like our own GIFs — so FineFace numbers are directly comparable to ours.

FineFace produces one still per (prompt, AU vector); it has no temporal model. We
approximate a "sequence" by generating N independent stills that vary only AU12,
holding the seed fixed within a sweep (identical initial noise) — the most
charitable identity-consistency setting for the baseline.

Two experiments mirror our 2026-07-14 eval:
  scaled       40 prompts × 3 seeds × 5 stills at commanded [0,.25,.5,.75,1.0]
               (FineFace AU12 = commanded × 5). Targets = commanded values, so
               AU12 Pearson r / identity are comparable to our `scaled`.
  calibration  10 prompts × 3 seeds × 11 single stills at commanded {0,.1,…,1.0}.
               One-frame entries; feeds the pooled-CLS dose-response summary.

Env / invocation: see docs/experiments/2026-07-21-fineface-setup.md.

    python scripts/fineface_generate.py --experiment scaled \
        --out-dir inference_output/batch_eval/fineface
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from fineface import FineFacePipeline

AU_MAX = 5.0  # FineFace AU intensities are on a 0–5 scale; commanded c∈[0,1] → c×5.
SCALED_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
CALIBRATION_LEVELS = [round(c, 1) for c in [i / 10 for i in range(11)]]


def load_prompts(path: Path, limit: int | None) -> list[str]:
    prompts = [ln.strip() for ln in path.read_text().splitlines()
               if ln.strip() and not ln.startswith("#")]
    return prompts[:limit] if limit else prompts


def build_jobs(experiment: str, prompts: list[str], seeds: list[int]) -> list[dict]:
    jobs = []
    if experiment == "scaled":
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                jobs.append(dict(id=f"ff_scaled_p{pi:03d}_s{seed}", prompt=prompt,
                                 prompt_idx=pi, seed=seed, intensities=SCALED_LEVELS))
    elif experiment == "calibration":
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                for c in CALIBRATION_LEVELS:
                    tag = f"{c:.1f}".replace(".", "")
                    jobs.append(dict(id=f"ff_calib_p{pi:03d}_s{seed}_c{tag}", prompt=prompt,
                                     prompt_idx=pi, seed=seed, level=c, intensities=[c]))
    else:
        raise ValueError(experiment)
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=["scaled", "calibration"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompts", default="configs/eval_prompts.txt")
    p.add_argument("--num-prompts", type=int, default=None,
                   help="cap prompts (default: 40 for scaled, 10 for calibration)")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--steps", type=int, default=50, help="FineFace default is 50")
    p.add_argument("--limit", type=int, default=None, help="cap jobs (smoke tests)")
    args = p.parse_args()

    n = args.num_prompts or (40 if args.experiment == "scaled" else 10)
    prompts = load_prompts(Path(args.prompts), n)
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = build_jobs(args.experiment, prompts, seeds)
    if args.limit:
        jobs = jobs[: args.limit]

    out_dir = Path(args.out_dir)
    frames_root = out_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        done = {json.loads(l)["id"] for l in manifest_path.read_text().splitlines() if l.strip()}
    pending = [j for j in jobs if j["id"] not in done]
    print(f"{len(jobs)} jobs, {len(jobs) - len(pending)} done, {len(pending)} to run", flush=True)
    if not pending:
        return

    pipe = FineFacePipeline()
    print("PIPELINE_LOADED", flush=True)

    with manifest_path.open("a") as mf:
        for i, job in enumerate(pending):
            sweep_dir = frames_root / job["id"]
            sweep_dir.mkdir(parents=True, exist_ok=True)
            frame_paths = []
            for fi, c in enumerate(job["intensities"]):
                # Re-seed identically before every still in the sweep so only AU12
                # differs (shared initial noise = best-case identity for FineFace).
                torch.manual_seed(job["seed"])
                torch.cuda.manual_seed_all(job["seed"])
                img = pipe(job["prompt"], {"AU12": c * AU_MAX},
                           num_inference_steps=args.steps).images[0]
                fp = sweep_dir / f"frame_{fi}_c{c:.2f}.png"
                img.save(fp)
                frame_paths.append(str(fp))
            entry = dict(id=job["id"], experiment=args.experiment, frames=frame_paths,
                         intensities=job["intensities"], prompt=job["prompt"],
                         prompt_idx=job["prompt_idx"], seed=job["seed"],
                         au_scale_max=AU_MAX, steps=args.steps)
            if "level" in job:
                entry["level"] = job["level"]
            mf.write(json.dumps(entry) + "\n")
            mf.flush()
            print(f"[{i + 1}/{len(pending)}] {job['id']}", flush=True)


if __name__ == "__main__":
    main()
