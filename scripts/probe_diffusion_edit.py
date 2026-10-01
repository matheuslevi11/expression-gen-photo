"""Probe 2 (diffusion editing, path B): can the *trained* model animate a SUPPLIED face?

Tests the reusable-face path of the prompt-decoupled expression operator
([[2026-09-24-prompt-decoupled-operator]]) WITHOUT retraining, via SDEdit: encode an input
neutral face, add noise to an intermediate level (strength), then denoise with the S-ramp
conditioning of the trained adaptor. If the output preserves the *input* identity while
following the S-ramp, editing works without retraining.

Reuses the trained GenPhotoPipeline components (unet, scheduler, camera_encoder, text
encoder, vae) and mirrors the pipeline's denoising loop, starting from t* instead of T.

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=<free> conda run -n genphoto python scripts/probe_diffusion_edit.py \
      --config inference_output/batch_eval/dose_response/configs/step_100000.yaml \
      --strengths 0.5,0.7,0.9,1.0 --out-dir inference_output/probes/diffusion_edit
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MEAD = REPO_ROOT / "MEAD_processed"
H, W, SCALE = 256, 384, 0.18215


def load_frame(path) -> np.ndarray:
    p = Path(path)
    if not p.is_absolute():
        p = MEAD / p
    return np.asarray(Image.open(p).convert("RGB").resize((W, H)))


def to_vae_input(rgb, device):
    return (torch.from_numpy(rgb).float().permute(2, 0, 1) / 127.5 - 1.0).unsqueeze(0).to(device)


@torch.no_grad()
def sdedit(pipeline, z0_video, cam_emb, text_emb, strength, num_steps, guidance, device, seed):
    sched = pipeline.scheduler
    sched.set_timesteps(num_steps, device=device)
    init_ts = min(int(num_steps * strength), num_steps)
    t_start = max(num_steps - init_ts, 0)
    timesteps = sched.timesteps[t_start:]
    feats = pipeline.camera_encoder(cam_emb)
    feats = [rearrange(x, "(b f) c h w -> b c f h w", b=1) for x in feats]
    feats = [torch.cat([x, x], 0) for x in feats]  # CFG double
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z0_video.shape, generator=gen, device=device, dtype=z0_video.dtype)
    latents = sched.add_noise(z0_video, noise, timesteps[:1])
    for t in timesteps:
        lmi = sched.scale_model_input(torch.cat([latents] * 2), t)
        pred = pipeline.unet(lmi, t, encoder_hidden_states=text_emb,
                             camera_embedding_features=feats).sample
        pu, pt = pred.chunk(2)
        pred = pu + guidance * (pt - pu)
        latents = sched.step(pred, t, latents).prev_sample
    video = pipeline.decode_latents(latents)  # [1, c, F, H, W] in [0,1]
    return [(video[0, :, f].transpose(1, 2, 0) * 255).astype(np.uint8) for f in range(video.shape[2])]


def pick_neutral_inputs(n, lo=0.15):
    pool = []
    for rec in json.loads((MEAD / "annotations/train.json").read_text()):
        for fr in rec["frames"]:
            if float(fr["au12"]) < lo:
                pool.append(fr["path"])
    random.seed(1); random.shuffle(pool)
    return pool[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "inference_output/batch_eval/dose_response/configs/step_100000.yaml")
    ap.add_argument("--inputs", type=str, default=None, help="comma-separated image paths; default 3 MEAD neutral frames")
    ap.add_argument("--prompt", type=str, default="A portrait photograph of a person, frontal view, neutral background.")
    ap.add_argument("--intensity-list", type=str, default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--strengths", type=str, default="0.5,0.7,0.9,1.0")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "inference_output/probes/diffusion_edit")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from inference_expression import load_models, IntensityEmbedding
    from comp_metrics.expression_au_accuracy import _load_pyfeat, detect_au12
    from batch_eval import IdentityScorer, pearson

    cfg = OmegaConf.load(args.config)
    pipeline, device = load_models(cfg)
    intens = [float(x) for x in args.intensity_list.split(",")]
    strengths = [float(x) for x in args.strengths.split(",")]
    F = len(intens)

    cam = IntensityEmbedding(intensity_values=torch.tensor(intens, dtype=torch.float32).unsqueeze(1),
                             tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder, device=device).load()
    cam = rearrange(cam.unsqueeze(0), "b f c h w -> b c f h w")  # [1,6,F,H,W]
    text_emb = pipeline._encode_prompt([args.prompt], device, 1, args.guidance > 1.0, None)

    Detector = _load_pyfeat()
    detector = Detector(au_model="xgb", emotion_model="resmasknet", identity_model=None,
                        face_model="retinaface", landmark_model="mobilefacenet", facepose_model="img2pose")
    ident = IdentityScorer(device)

    inputs = args.inputs.split(",") if args.inputs else pick_neutral_inputs(3)
    results, grid_rows = [], []
    for inp in inputs:
        raw = load_frame(inp)
        with torch.no_grad():
            z0 = pipeline.vae.encode(to_vae_input(raw, device)).latent_dist.mean * SCALE
        z0_video = z0.unsqueeze(2).repeat(1, 1, F, 1, 1)  # [1,4,F,h,w]
        for s in strengths:
            frames = sdedit(pipeline, z0_video, cam, text_emb, s, args.steps, args.guidance, device, args.seed)
            au = detect_au12(detector, frames)
            ramp_r = pearson(au, intens)
            id_to_input = np.nanmean([ident([raw, f]).get("id_mean_adjacent", np.nan) for f in frames])
            cross_id = ident(frames).get("id_min_adjacent", np.nan)
            results.append(dict(input=str(inp), strength=s, au12=[round(a, 3) for a in au],
                                ramp_r=round(float(ramp_r), 3), id_to_input=round(float(id_to_input), 3),
                                cross_id_min=round(float(cross_id), 3)))
            grid_rows.append(np.concatenate([raw] + frames, axis=1))
            print(f"input={Path(inp).name[:18]:18} strength={s}  ramp_r={ramp_r:.3f}  "
                  f"id->input={id_to_input:.3f}  cross_id_min={cross_id:.3f}")
    Image.fromarray(np.concatenate(grid_rows, axis=0)).save(args.out_dir / "edit_grid.png")
    (args.out_dir / "summary.json").write_text(json.dumps(
        {"prompt": args.prompt, "intensity_list": intens, "strengths": strengths, "results": results}, indent=2))
    print(f"\ngrid (input | 5 output frames per row) -> {args.out_dir/'edit_grid.png'}")
    print(f"summary -> {args.out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
