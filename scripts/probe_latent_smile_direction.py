"""Probe 1 (latent-feasibility): is there an SD-VAE latent direction that decodes to a
graded, realistic smile with identity preserved?

Tests the make-or-break question for the "pure latent operator" (path A) of the
prompt-decoupled expression operator (docs/experiments/2026-09-24-prompt-decoupled-operator.md).
No diffusion, no retraining: uses only the frozen SD-VAE, MEAD frames, py-feat, and facenet.

Method:
  1. Encode MEAD frames to VAE latents (x0.18215), binned by their own per-frame au12.
  2. Fit a global smile direction d = mean(latent | au12>HI) - mean(latent | au12<LO).
  3. For neutral seeds z0 (low au12), sweep z(a) = z0 + a*d, VAE-decode, and measure
     detected AU12, identity cosine vs decode(z0), and realism (visual grid).
  4. Report the VAE encode->decode reconstruction ceiling as a baseline.

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=4 conda run -n genphoto python scripts/probe_latent_smile_direction.py \
      --n-fit 400 --n-seed 12 --out-dir inference_output/probes/latent_smile
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MEAD = REPO_ROOT / "MEAD_processed"
SNAP = "/home/levi/.cache/huggingface/hub/models--pandaphd--generative_photography/snapshots/92c29567186da6c7f8ada09eab8b5bfc7c998314/stable-diffusion-v1-5"
H, W = 256, 384
SCALE = 0.18215


def load_frame(rel_path: str) -> np.ndarray:
    img = Image.open(MEAD / rel_path).convert("RGB").resize((W, H))
    return np.asarray(img)


def to_vae_input(rgb: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


def decode_latent(vae, z: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        img = vae.decode(z / SCALE).sample[0]
    img = ((img.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-fit", type=int, default=400, help="frames to fit the smile direction")
    ap.add_argument("--n-seed", type=int, default=12, help="neutral seed frames to sweep")
    ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--hi", type=float, default=0.70)
    ap.add_argument("--alphas", type=str, default="0,0.5,1,1.5,2,3")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "inference_output/probes/latent_smile")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(SNAP, subfolder="vae").to(device).eval()
    from comp_metrics.expression_au_accuracy import _load_pyfeat, detect_au12
    Detector = _load_pyfeat()
    detector = Detector(au_model="xgb", emotion_model="resmasknet", identity_model=None,
                        face_model="retinaface", landmark_model="mobilefacenet", facepose_model="img2pose")
    from batch_eval import IdentityScorer  # facenet InceptionResnetV1 / VGGFace2
    ident = IdentityScorer(device)

    # flat pool of (path, au12) from MEAD, shuffled
    pool = []
    for rec in json.loads((MEAD / "annotations/train.json").read_text()):
        for fr in rec["frames"]:
            pool.append((fr["path"], float(fr["au12"])))
    random.seed(0); random.shuffle(pool)
    lo_frames = [p for p, a in pool if a < args.lo]
    hi_frames = [p for p, a in pool if a > args.hi]
    print(f"pool={len(pool)}  low(<{args.lo})={len(lo_frames)}  high(>{args.hi})={len(hi_frames)}")

    def enc_mean(paths):
        zs = []
        for p in paths:
            with torch.no_grad():
                z = vae.encode(to_vae_input(load_frame(p), device)).latent_dist.mean * SCALE
            zs.append(z)
        return torch.cat(zs, 0)

    k = args.n_fit // 2
    z_lo = enc_mean(lo_frames[:k]); z_hi = enc_mean(hi_frames[:k])
    d = z_hi.mean(0, keepdim=True) - z_lo.mean(0, keepdim=True)   # [1,4,32,48] smile direction
    print(f"fit direction from {z_lo.shape[0]} low + {z_hi.shape[0]} high latents; |d|={d.norm().item():.2f}")

    # sweep from neutral seeds
    seeds = lo_frames[k:k + args.n_seed]
    curve = {f"{a:g}": {"au12": [], "id": []} for a in alphas}
    recon_au12, recon_gap = [], []
    grid_rows = []
    for si, sp in enumerate(seeds):
        with torch.no_grad():
            z0 = vae.encode(to_vae_input(load_frame(sp), device)).latent_dist.mean * SCALE
        base_img = decode_latent(vae, z0)
        row = []
        for a in alphas:
            img = decode_latent(vae, z0 + a * d)
            au = detect_au12(detector, [img])[0]
            idc = ident([base_img, img]).get("id_mean_adjacent", float("nan"))
            curve[f"{a:g}"]["au12"].append(au)
            curve[f"{a:g}"]["id"].append(idc)
            row.append(img)
        grid_rows.append(np.concatenate(row, axis=1))
        # reconstruction ceiling: a real high-au12 frame encode->decode vs its raw au12
        if si < 6:
            hp = hi_frames[si]
            raw = load_frame(hp); rec = decode_latent(vae, vae.encode(to_vae_input(raw, device)).latent_dist.mean * SCALE)
            recon_au12.append(detect_au12(detector, [rec])[0])
            recon_gap.append(detect_au12(detector, [raw])[0])
    Image.fromarray(np.concatenate(grid_rows, axis=0)).save(args.out_dir / "sweep_grid.png")

    def ms(xs):
        v = np.array([x for x in xs if x is not None and np.isfinite(x)], float)
        return (round(float(v.mean()), 3), round(float(v.std()), 3), int(v.size)) if v.size else (None, None, 0)

    summary = {"alphas": alphas, "n_seed": len(seeds),
               "au12_by_alpha": {a: ms(c["au12"]) for a, c in curve.items()},
               "identity_by_alpha": {a: ms(c["id"]) for a, c in curve.items()},
               "recon_ceiling_au12": ms(recon_au12), "raw_high_au12": ms(recon_gap)}
    # latent-CLS: pooled correlation of alpha vs detected au12
    A = [a for a in alphas for _ in curve[f"{a:g}"]["au12"]]
    Y = [y for a in alphas for y in curve[f"{a:g}"]["au12"]]
    m = np.isfinite(A) & np.isfinite(Y)
    summary["latent_cls_pearson"] = round(float(np.corrcoef(np.array(A)[m], np.array(Y)[m])[0, 1]), 3) if m.sum() > 2 else None
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== RESULTS ===")
    print(f"latent-CLS (Pearson alpha vs detected AU12): {summary['latent_cls_pearson']}")
    print(f"{'alpha':>6}{'AU12(mean±sd)':>18}{'identity':>12}")
    for a in alphas:
        au = summary["au12_by_alpha"][f"{a:g}"]; idc = summary["identity_by_alpha"][f"{a:g}"]
        print(f"{a:>6g}   {au[0]} ± {au[1]}      {idc[0]}")
    print(f"VAE recon ceiling AU12 {summary['recon_ceiling_au12'][0]} vs raw {summary['raw_high_au12'][0]}")
    print(f"grid -> {args.out_dir/'sweep_grid.png'}  | summary -> {args.out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
