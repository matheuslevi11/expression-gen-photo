"""Independent AU cross-check with LibreFace (runs in the isolated `libreface` conda env).

Re-scores already-generated clips for LibreFace AU *intensities* (a different model family
from py-feat, so it breaks the py-feat label/eval circularity — and, unlike py-feat's
presence probabilities, LibreFace returns FACS-scale intensities). Self-contained: does not
import batch_eval (that env is separate). Writes results_libreface.jsonl keyed by clip id.

LibreFace covers 12 AUs: AU01,02,04,05,06,09,12,15,17,20,25,26.

Usage (run from repo root)::
    conda run -n libreface python scripts/rescore_libreface.py \
        --manifest inference_output/batch_eval/trained/manifest.jsonl --experiment scaled
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent

# LibreFace intensity key (au_N_intensity) -> canonical AU name
LF_AUS = {1: "AU01", 2: "AU02", 4: "AU04", 5: "AU05", 6: "AU06", 9: "AU09",
          12: "AU12", 15: "AU15", 17: "AU17", 20: "AU20", 25: "AU25", 26: "AU26"}


def _corr(a, b, spearman=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2 or a[m].std() < 1e-8 or b[m].std() < 1e-8:
        return float("nan")
    x, y = a[m], b[m]
    if spearman:
        rank = lambda z: np.argsort(np.argsort(z))
        x, y = rank(x), rank(y)
    return float(np.corrcoef(x, y)[0, 1])


def load_frames(entry):
    if entry.get("gif"):
        p = entry["gif"]
        p = p if os.path.isabs(p) else str(REPO_ROOT / p)
        with Image.open(p) as gif:
            return [np.array(f.convert("RGB")) for f in ImageSequence.Iterator(gif)]
    frames = []
    for f in entry["frames"]:
        f = f if os.path.isabs(f) else str(REPO_ROOT / f)
        frames.append(np.array(Image.open(f).convert("RGB")))
    return frames


def score_frames(frames):
    """Return {AU_name: [per-frame intensity]} via LibreFace; NaN on failure."""
    import libreface
    out = {au: [] for au in LF_AUS.values()}
    for rgb in frames:
        vals = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            Image.fromarray(rgb).save(tmp_path)
            res = libreface.get_au_intensities_and_detect_aus(tmp_path, device="cpu")
            os.unlink(tmp_path)
            inten = res[1] if isinstance(res, tuple) else res
            vals = {}
            for n, au in LF_AUS.items():
                k = f"au_{n}_intensity"
                v = inten.get(k, float("nan")) if isinstance(inten, dict) else float("nan")
                vals[au] = float(v) if v is not None and np.isfinite(v) else float("nan")
        except Exception as e:  # noqa: BLE001
            print(f"  libreface failed on a frame: {e!r}", file=sys.stderr)
            vals = None
        for au in LF_AUS.values():
            out[au].append(float("nan") if vals is None else vals.get(au, float("nan")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--experiment", default="scaled", help="only score this experiment type")
    ap.add_argument("--results", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    entries = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.experiment:
        entries = [e for e in entries if e.get("experiment") == args.experiment]
    results_path = Path(args.results or Path(args.manifest).parent / "results_libreface.jsonl")
    done = set()
    if results_path.exists():
        done = {json.loads(l)["id"] for l in results_path.read_text().splitlines() if l.strip()}
    pending = [e for e in entries if e["id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{args.manifest} [{args.experiment}]: {len(entries)} entries, "
          f"{len(entries) - len(pending)} done, {len(pending)} to score -> {results_path}")
    if not pending:
        return

    with results_path.open("a") as rf:
        for i, entry in enumerate(pending):
            frames = load_frames(entry)
            targets = entry.get("intensities", entry.get("targets"))
            aus = score_frames(frames)
            rec = dict(id=entry["id"], experiment=entry.get("experiment"), targets=targets,
                       prompt_idx=entry.get("prompt_idx"), seed=entry.get("seed"),
                       aus=aus,
                       au_ramp_pearson={au: _corr(v, targets) for au, v in aus.items()},
                       au_ramp_spearman={au: _corr(v, targets, spearman=True) for au, v in aus.items()})
            rf.write(json.dumps(rec) + "\n")
            rf.flush()
            r12 = rec["au_ramp_pearson"].get("AU12", float("nan"))
            r06 = rec["au_ramp_pearson"].get("AU06", float("nan"))
            print(f"[{i+1}/{len(pending)}] {entry['id']}  r(AU12)={r12:.3f} r(AU06)={r06:.3f}")


if __name__ == "__main__":
    main()
