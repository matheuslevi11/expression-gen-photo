"""Batch evaluation harness for the expression adaptor.

One tool for the Evaluation Robustness checklist (docs/status.md): generates a
manifest-tracked sweep of samples (loading the pipeline once), then scores them
with three independent measurements:

  - py-feat AU12          (same detector family as the training labels)
  - MediaPipe smile proxy (mouth-corner elevation from FaceMesh landmarks —
                           independent of the training labels)
  - identity cosine       (facenet-pytorch InceptionResnetV1/VGGFace2 embeddings,
                           frame-to-frame — the "same face, different expression" check)

Subcommands::

    # 1. Generate (pipeline loaded once; resumable — existing GIFs are skipped)
    python scripts/batch_eval.py generate \
        --config /path/to/inference_config.yaml \
        --experiment calibration --out-dir inference_output/batch_eval/trained

    # 2. Score (GPU for py-feat/facenet; also resumable per sample)
    python scripts/batch_eval.py score \
        --manifest inference_output/batch_eval/trained/manifest.jsonl

    # 3. Aggregate into summary.json + printed tables
    python scripts/batch_eval.py summarize \
        --results inference_output/batch_eval/trained/results.jsonl

Experiments (--experiment):
    scaled         prompts x seeds x ascending [0,.25,.5,.75,1]         (Tier 2)
    calibration    prompts x constant [c]*5, c in 0..1 step .1          (Tier 1, CLS)
    permuted       prompts x seeds x 3 fixed non-monotonic lists        (Tier 2)
    extrapolation  prompts x constant lists outside [0,1]               (Tier 3)

External samples (e.g. a FineFace sweep for the head-to-head) can be scored by
hand-writing a manifest whose entries use "frames": [png paths] instead of
"gif": path — the scorer treats the frame list as the sequence.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger("batch_eval")

ASCENDING = [0.0, 0.25, 0.5, 0.75, 1.0]
PERMUTATIONS = [
    [0.0, 1.0, 0.5, 0.25, 0.75],
    [1.0, 0.0, 0.75, 0.25, 0.5],
    [0.5, 0.0, 1.0, 0.75, 0.25],
]
CALIBRATION_LEVELS = [round(c, 1) for c in np.arange(0.0, 1.01, 0.1)]
EXTRAPOLATION_LEVELS = [-0.5, -0.25, 1.25, 1.5]


def load_prompts(path: Path, limit: int | None) -> list[str]:
    prompts = [ln.strip() for ln in path.read_text().splitlines()
               if ln.strip() and not ln.startswith("#")]
    return prompts[:limit] if limit else prompts


def build_jobs(experiment: str, prompts: list[str], seeds: list[int]) -> list[dict]:
    jobs = []
    if experiment == "scaled":
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                jobs.append(dict(prompt_idx=pi, prompt=prompt, seed=seed, intensities=ASCENDING))
    elif experiment == "calibration":
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                for c in CALIBRATION_LEVELS:
                    jobs.append(dict(prompt_idx=pi, prompt=prompt, seed=seed,
                                     intensities=[c] * 5, level=c))
    elif experiment == "permuted":
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                for li, lst in enumerate(PERMUTATIONS):
                    jobs.append(dict(prompt_idx=pi, prompt=prompt, seed=seed,
                                     intensities=lst, perm_idx=li))
    elif experiment == "extrapolation":
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                for c in EXTRAPOLATION_LEVELS:
                    jobs.append(dict(prompt_idx=pi, prompt=prompt, seed=seed,
                                     intensities=[c] * 5, level=c))
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return jobs


def job_name(experiment: str, job: dict) -> str:
    tag = "_".join(f"{v:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
                   for v in job["intensities"])
    return f"{experiment}_p{job['prompt_idx']:03d}_s{job['seed']}_{tag}"


# --------------------------------------------------------------------------- generate

def cmd_generate(args):
    import torch
    from einops import rearrange
    from omegaconf import OmegaConf

    from genphoto.utils.util import save_videos_grid
    from inference_expression import IntensityEmbedding, load_models

    cfg = OmegaConf.load(args.config)
    prompts = load_prompts(Path(args.prompts), args.num_prompts)
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = build_jobs(args.experiment, prompts, seeds)
    if args.limit:
        jobs = jobs[: args.limit]

    out_dir = Path(args.out_dir)
    gif_dir = out_dir / args.experiment
    gif_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    already = set()
    if manifest_path.exists():
        already = {json.loads(ln)["id"] for ln in manifest_path.read_text().splitlines() if ln.strip()}

    pending = [j for j in jobs if job_name(args.experiment, j) not in already]
    log.info("%d jobs total, %d already done, %d to run", len(jobs), len(jobs) - len(pending), len(pending))
    if not pending:
        return

    log.info("Loading pipeline once from %s ...", args.config)
    pipeline, device = load_models(cfg)

    with manifest_path.open("a") as mf:
        for i, job in enumerate(pending):
            name = job_name(args.experiment, job)
            gif_path = gif_dir / f"{name}.gif"
            intensities = torch.tensor(job["intensities"], dtype=torch.float32).unsqueeze(1)
            camera_embedding = IntensityEmbedding(
                intensity_values=intensities,
                tokenizer=pipeline.tokenizer,
                text_encoder=pipeline.text_encoder,
                device=device,
            ).load()
            camera_embedding = rearrange(camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")

            torch.manual_seed(job["seed"])
            torch.cuda.manual_seed_all(job["seed"])
            with torch.no_grad():
                sample = pipeline(
                    prompt=job["prompt"], camera_embedding=camera_embedding,
                    video_length=5, height=256, width=384,
                    num_inference_steps=25, guidance_scale=8.0,
                ).videos[0]
            save_videos_grid(sample[None, ...], str(gif_path))

            entry = dict(
                id=name, experiment=args.experiment, gif=str(gif_path),
                config=str(args.config),
                adaptor_ckpt=cfg.get("expression_adaptor_ckpt"),
                **job,
            )
            mf.write(json.dumps(entry) + "\n")
            mf.flush()
            log.info("[%d/%d] %s", i + 1, len(pending), name)


# --------------------------------------------------------------------------- scoring backends

def load_frames(entry: dict) -> list[np.ndarray]:
    from PIL import Image, ImageSequence
    if entry.get("gif"):
        with Image.open(entry["gif"]) as gif:
            return [np.array(f.convert("RGB")) for f in ImageSequence.Iterator(gif)]
    return [np.array(Image.open(p).convert("RGB")) for p in entry["frames"]]


class PyFeatScorer:
    """AU12 per frame — same detector family as the training labels."""

    def __init__(self):
        from comp_metrics.expression_au_accuracy import _load_pyfeat
        Detector = _load_pyfeat()
        self.detector = Detector(au_model="xgb", emotion_model="resmasknet", identity_model=None,
                                 face_model="retinaface", landmark_model="mobilefacenet",
                                 facepose_model="img2pose")

    def __call__(self, frames) -> list[float]:
        from comp_metrics.expression_au_accuracy import detect_au12
        return detect_au12(self.detector, frames)


class MediaPipeSmileScorer:
    """Label-independent smile proxy: mouth-corner elevation above the lip
    center, normalized by face height, from FaceMesh landmarks. Positive and
    increasing = broader smile. Only correlation/ordering is meaningful, not
    the absolute scale. NaN when no face is found."""

    CORNERS, LIP_TOP, LIP_BOTTOM, FOREHEAD, CHIN = (61, 291), 13, 14, 10, 152

    def __init__(self):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False)

    def __call__(self, frames) -> list[float]:
        out = []
        for rgb in frames:
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                out.append(float("nan"))
                continue
            lm = res.multi_face_landmarks[0].landmark
            lip_center_y = (lm[self.LIP_TOP].y + lm[self.LIP_BOTTOM].y) / 2
            corner_y = (lm[self.CORNERS[0]].y + lm[self.CORNERS[1]].y) / 2
            face_h = abs(lm[self.CHIN].y - lm[self.FOREHEAD].y)
            out.append(float((lip_center_y - corner_y) / face_h) if face_h > 1e-6 else float("nan"))
        return out


class IdentityScorer:
    """Frame-to-frame face-embedding cosine similarity (facenet-pytorch
    InceptionResnetV1, VGGFace2). Returns (mean_adjacent, min_adjacent,
    mean_vs_first); NaNs when fewer than 2 faces are detected."""

    def __init__(self, device):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
        self.torch = torch
        self.mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=True)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self.device = device

    def __call__(self, frames):
        from PIL import Image
        embs = []
        with self.torch.no_grad():
            for rgb in frames:
                face = self.mtcnn(Image.fromarray(rgb))
                if face is None:
                    embs.append(None)
                    continue
                emb = self.resnet(face.unsqueeze(0).to(self.device))[0]
                embs.append((emb / emb.norm()).cpu().numpy())
        valid = [(i, e) for i, e in enumerate(embs) if e is not None]
        if len(valid) < 2:
            return dict(id_mean_adjacent=float("nan"), id_min_adjacent=float("nan"),
                        id_mean_vs_first=float("nan"), id_faces_found=len(valid))
        adjacent = [float(np.dot(a[1], b[1])) for a, b in zip(valid, valid[1:])]
        first = valid[0][1]
        vs_first = [float(np.dot(first, e)) for _, e in valid[1:]]
        return dict(id_mean_adjacent=float(np.mean(adjacent)),
                    id_min_adjacent=float(np.min(adjacent)),
                    id_mean_vs_first=float(np.mean(vs_first)),
                    id_faces_found=len(valid))


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or a[mask].std() < 1e-8 or b[mask].std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


# --------------------------------------------------------------------------- score

def cmd_score(args):
    entries = [json.loads(ln) for ln in Path(args.manifest).read_text().splitlines() if ln.strip()]
    results_path = Path(args.results or Path(args.manifest).parent / "results.jsonl")
    done = set()
    if results_path.exists():
        done = {json.loads(ln)["id"] for ln in results_path.read_text().splitlines() if ln.strip()}
    pending = [e for e in entries if e["id"] not in done]
    log.info("%d samples, %d already scored, %d to score", len(entries), len(entries) - len(pending), len(pending))
    if not pending:
        return

    device = "cuda" if not args.cpu else "cpu"
    aus = PyFeatScorer() if not args.skip_au else None
    smile = MediaPipeSmileScorer() if not args.skip_mediapipe else None
    ident = IdentityScorer(device) if not args.skip_identity else None

    with results_path.open("a") as rf:
        for i, entry in enumerate(pending):
            frames = load_frames(entry)
            targets = entry["intensities"]
            rec = dict(id=entry["id"], experiment=entry["experiment"], targets=targets,
                       prompt_idx=entry.get("prompt_idx"), seed=entry.get("seed"),
                       level=entry.get("level"), perm_idx=entry.get("perm_idx"))
            if aus:
                au12 = aus(frames)
                rec.update(au12=au12, au12_pearson=pearson(au12, targets),
                           au12_mse=float(np.nanmean((np.asarray(au12) - np.clip(targets, 0, 1)) ** 2)))
            if smile:
                mp_s = smile(frames)
                rec.update(mp_smile=mp_s, mp_pearson=pearson(mp_s, targets))
            if ident:
                rec.update(ident(frames))
            rf.write(json.dumps(rec) + "\n")
            rf.flush()
            log.info("[%d/%d] %s  r_au12=%s", i + 1, len(pending), entry["id"],
                     f"{rec.get('au12_pearson', float('nan')):.3f}" if aus else "-")


# --------------------------------------------------------------------------- summarize

def _stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    return dict(n=int(v.size), mean=float(v.mean()) if v.size else None,
                std=float(v.std(ddof=1)) if v.size > 1 else None)


def cmd_summarize(args):
    recs = [json.loads(ln) for ln in Path(args.results).read_text().splitlines() if ln.strip()]
    by_exp = defaultdict(list)
    for r in recs:
        by_exp[r["experiment"]].append(r)

    summary = {}
    for exp, rows in by_exp.items():
        s = dict(n_samples=len(rows))
        for key in ("au12_pearson", "mp_pearson", "au12_mse",
                    "id_mean_adjacent", "id_min_adjacent", "id_mean_vs_first"):
            vals = [r.get(key) for r in rows if key in r]
            if vals:
                s[key] = _stats(vals)
        if exp in ("calibration", "extrapolation"):
            # dose–response: mean detected AU12 per commanded level, and CLS =
            # Pearson over all pooled (level, per-frame detected) pairs.
            curve, pooled_c, pooled_d = {}, [], []
            for r in rows:
                if r.get("au12") is None or r.get("level") is None:
                    continue
                curve.setdefault(r["level"], []).extend(r["au12"])
                pooled_c += [r["level"]] * len(r["au12"])
                pooled_d += r["au12"]
            s["curve_au12_by_level"] = {
                str(lvl): _stats(vals) for lvl, vals in sorted(curve.items())}
            s["cls_pearson_pooled"] = pearson(pooled_c, pooled_d)
        summary[exp] = s

    out_path = Path(args.out or Path(args.results).parent / "summary.json")
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out_path)

    for exp, s in summary.items():
        print(f"\n== {exp}  (n={s['n_samples']})")
        for key in ("au12_pearson", "mp_pearson", "au12_mse", "id_mean_adjacent", "id_min_adjacent"):
            if key in s and s[key]["mean"] is not None:
                std = f" ± {s[key]['std']:.4f}" if s[key]["std"] is not None else ""
                print(f"   {key:18s} {s[key]['mean']:.4f}{std}  (n={s[key]['n']})")
        if "cls_pearson_pooled" in s:
            print(f"   {'CLS (pooled)':18s} {s['cls_pearson_pooled']:.4f}")
        for lvl, st in s.get("curve_au12_by_level", {}).items():
            print(f"     level {lvl:>5}: detected AU12 {st['mean']:.3f} ± {st['std'] or 0:.3f}")


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a sweep of samples (one pipeline load)")
    g.add_argument("--config", required=True, help="inference YAML (trained or baseline)")
    g.add_argument("--experiment", required=True,
                   choices=["scaled", "calibration", "permuted", "extrapolation"])
    g.add_argument("--prompts", default=str(REPO_ROOT / "configs/eval_prompts.txt"))
    g.add_argument("--num-prompts", type=int, default=None,
                   help="use only the first N prompts (default: all)")
    g.add_argument("--seeds", default="42", help="comma-separated, e.g. 42,43,44")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--limit", type=int, default=None, help="cap total jobs (smoke tests)")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score", help="score generated samples")
    s.add_argument("--manifest", required=True)
    s.add_argument("--results", default=None, help="output JSONL (default: alongside manifest)")
    s.add_argument("--skip-au", action="store_true")
    s.add_argument("--skip-mediapipe", action="store_true")
    s.add_argument("--skip-identity", action="store_true")
    s.add_argument("--cpu", action="store_true")
    s.set_defaults(func=cmd_score)

    z = sub.add_parser("summarize", help="aggregate results into summary.json")
    z.add_argument("--results", required=True)
    z.add_argument("--out", default=None)
    z.set_defaults(func=cmd_summarize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
