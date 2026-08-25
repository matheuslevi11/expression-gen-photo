"""Re-score already-generated clips for the FULL py-feat AU set + independent
FaceMesh geometry proxies.

RE-SCORING ONLY. This reads existing ``manifest.jsonl`` files and the frames they
reference; it never regenerates clips and never touches the existing
``results.jsonl`` / ``summary.json``. Output goes to a new ``results_allau.jsonl``
keyed by clip ``id`` (resumable: already-scored ids are skipped).

py-feat ``xgb`` outputs are AU *presence probabilities* in [0,1], not FACS
intensities -- use for ordering/correlation, not absolute scale. Geometry proxies
are dimensionless ratios (independent of py-feat); their sign is documented per key.

Usage (run from the repo root)::

    conda run -n genphoto python scripts/rescore_allau.py \
        --manifest inference_output/batch_eval/trained/manifest.jsonl
    # add --limit 2 for a smoke test; --skip-geo / --skip-au to isolate a scorer
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))   # allow `import batch_eval`
sys.path.insert(0, str(REPO_ROOT))     # allow `from comp_metrics...`

from batch_eval import load_frames, pearson  # reuse (module-level imports are light)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger("rescore_allau")


def spearman(a, b) -> float:
    """Rank-correlation with NaN masking (mirrors batch_eval.pearson semantics)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    ar, br = a[mask], b[mask]
    if ar.std() < 1e-8 or br.std() < 1e-8:
        return float("nan")
    rank = lambda x: np.argsort(np.argsort(x))
    return float(np.corrcoef(rank(ar), rank(br))[0, 1])


# --------------------------------------------------------------------------- scorers

class PyFeatAllAUScorer:
    """All 20 py-feat AUs per frame (same detector config as batch_eval.PyFeatScorer)."""

    def __init__(self):
        from comp_metrics.expression_au_accuracy import _load_pyfeat
        Detector = _load_pyfeat()
        self.detector = Detector(au_model="xgb", emotion_model="resmasknet", identity_model=None,
                                 face_model="retinaface", landmark_model="mobilefacenet",
                                 facepose_model="img2pose")

    def __call__(self, frames):
        from comp_metrics.expression_au_accuracy import detect_all_aus
        return detect_all_aus(self.detector, frames)


class FaceMeshGeoScorer:
    """Label-independent geometric AU proxies from MediaPipe FaceMesh (468 landmarks).

    Returns dimensionless ratios per frame (NaN when no face). Sign key:
      * AU45_eye_aspect_ratio   -- LOWER = eyes closing (AU45/AU7 orbital tightening)
      * AU25_26_inner_lip_gap   -- HIGHER = lips parted / jaw drop
      * AU1_2_brow_raise        -- HIGHER = brows raised
      * AU4_inner_brow_gap      -- LOWER = brows drawn together (furrow)
    """

    def __init__(self):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False)

    @staticmethod
    def _safe_div(n, d):
        return float(n / d) if abs(d) > 1e-6 else float("nan")

    def __call__(self, frames):
        keys = ["AU45_eye_aspect_ratio", "AU25_26_inner_lip_gap",
                "AU1_2_brow_raise", "AU4_inner_brow_gap"]
        out = {k: [] for k in keys}
        for rgb in frames:
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                for k in keys:
                    out[k].append(float("nan"))
                continue
            lm = res.multi_face_landmarks[0].landmark
            face_h = abs(lm[152].y - lm[10].y)
            # AU45: mean eye-aspect-ratio (vertical lid gap / horizontal eye width)
            ear_l = self._safe_div(abs(lm[159].y - lm[145].y), abs(lm[33].x - lm[133].x))
            ear_r = self._safe_div(abs(lm[386].y - lm[374].y), abs(lm[362].x - lm[263].x))
            ear = np.nanmean([ear_l, ear_r]) if np.isfinite([ear_l, ear_r]).any() else float("nan")
            out["AU45_eye_aspect_ratio"].append(float(ear))
            # AU25/26: inner-lip vertical gap, normalized by face height
            out["AU25_26_inner_lip_gap"].append(self._safe_div(abs(lm[14].y - lm[13].y), face_h))
            # AU1/2: mean (eye-top - brow) vertical distance, normalized
            braise = np.nanmean([(lm[159].y - lm[105].y), (lm[386].y - lm[334].y)])
            out["AU1_2_brow_raise"].append(self._safe_div(braise, face_h))
            # AU4: inner-brow horizontal separation, normalized
            out["AU4_inner_brow_gap"].append(self._safe_div(abs(lm[55].x - lm[285].x), face_h))
        return out


# --------------------------------------------------------------------------- driver

def resolve_entry_paths(entry: dict) -> dict:
    """Return a copy of ``entry`` with gif/frames paths made absolute against REPO_ROOT.

    Handles both the repo-root-relative manifests (baseline/, trained/) and the
    absolute ones (dose_response/, fineface/), so the script is CWD-independent.
    """
    e = dict(entry)
    if e.get("gif"):
        p = Path(e["gif"])
        if not p.is_absolute():
            e["gif"] = str(REPO_ROOT / p)
    if e.get("frames"):
        e["frames"] = [f if Path(f).is_absolute() else str(REPO_ROOT / f) for f in e["frames"]]
    return e


def cmd_rescore(args):
    entries = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    results_path = Path(args.results or Path(args.manifest).parent / "results_allau.jsonl")
    done = set()
    if results_path.exists():
        done = {json.loads(l)["id"] for l in results_path.read_text().splitlines() if l.strip()}
    pending = [e for e in entries if e["id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    log.info("%s: %d entries, %d done, %d to score -> %s",
             args.manifest, len(entries), len(entries) - len(pending), len(pending), results_path)
    if not pending:
        return

    au_scorer = PyFeatAllAUScorer() if not args.skip_au else None
    geo_scorer = FaceMeshGeoScorer() if not args.skip_geo else None

    with results_path.open("a") as rf:
        for i, entry in enumerate(pending):
            e = resolve_entry_paths(entry)
            frames = load_frames(e)
            targets = entry.get("intensities", entry.get("targets"))
            if targets is not None and len(frames) != len(targets):
                log.warning("%s: frame count %d != target count %d; skipping",
                            entry["id"], len(frames), len(targets))
                continue
            rec = dict(id=entry["id"], experiment=entry.get("experiment"), targets=targets,
                       prompt_idx=entry.get("prompt_idx"), seed=entry.get("seed"),
                       level=entry.get("level"), perm_idx=entry.get("perm_idx"))
            if au_scorer:
                aus = au_scorer(frames)
                rec["aus"] = aus
                rec["au_ramp_pearson"] = {au: pearson(v, targets) for au, v in aus.items()}
                rec["au_ramp_spearman"] = {au: spearman(v, targets) for au, v in aus.items()}
            if geo_scorer:
                geo = geo_scorer(frames)
                rec["facemesh_geo"] = geo
                rec["geo_ramp_pearson"] = {k: pearson(v, targets) for k, v in geo.items()}
                rec["geo_ramp_spearman"] = {k: spearman(v, targets) for k, v in geo.items()}
            rf.write(json.dumps(rec) + "\n")
            rf.flush()
            r12 = rec.get("au_ramp_pearson", {}).get("AU12", float("nan"))
            r06 = rec.get("au_ramp_pearson", {}).get("AU06", float("nan"))
            log.info("[%d/%d] %s  r(AU12)=%.3f r(AU06)=%.3f", i + 1, len(pending), entry["id"], r12, r06)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--results", default=None, help="output JSONL (default: <manifest_dir>/results_allau.jsonl)")
    p.add_argument("--limit", type=int, default=None, help="cap entries (smoke tests)")
    p.add_argument("--skip-au", action="store_true")
    p.add_argument("--skip-geo", action="store_true")
    args = p.parse_args()
    cmd_rescore(args)


if __name__ == "__main__":
    main()
