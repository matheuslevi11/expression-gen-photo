"""Score preprocessed MEAD frames with an independent AU detector (multi-AU E0, 2026-11-15).

Reads ``preprocess_mead.py`` annotation files and scores every saved face crop with one scorer:

    --scorer geo        MediaPipe FaceMesh geometry: the brow / eye / lip-gap proxies of
                        ``rescore_allau.FaceMeshGeoScorer`` plus the mouth-corner smile proxy of
                        ``batch_eval.MediaPipeSmileScorer`` (run in the ``genphoto`` env).
    --scorer libreface  LibreFace AU intensities (12 AUs, 0-5 FACS scale; run in the isolated
                        ``libreface`` env, CPU). Weights must be cached first (compute nodes have no
                        internet): pass the same --weights-dir used for the one-off download.

Writes one JSON line per frame, ``{"path": <record path>, "<scorer>": {name: value | null}}``,
where null = no face / detector failure. Resumable (paths already in the output are skipped)
and shardable over frames (``--shard k --nshards n``) for parallel CPU processes.

    python scripts/score_mead_frames.py --root <processed root> \
        --annotations <root>/shard_0/annotations/train.json ... --scorer geo --out geo_0.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

LF_AUS = {1: "AU01", 2: "AU02", 4: "AU04", 5: "AU05", 6: "AU06", 9: "AU09",
          12: "AU12", 15: "AU15", 17: "AU17", 20: "AU20", 25: "AU25", 26: "AU26"}


def frame_list(annotation_files, root_of):
    """(record path, absolute path) for every frame, in a stable order."""
    out = []
    for a in annotation_files:
        root = root_of(a)
        for rec in json.loads(Path(a).read_text()):
            for f in rec["frames"]:
                out.append((f["path"], root / f["path"]))
    return out


def _clean(v):
    return float(v) if v is not None and math.isfinite(float(v)) else None


def make_geo():
    import cv2
    sys.path.insert(0, str(SCRIPTS_DIR))
    from rescore_allau import FaceMeshGeoScorer
    from batch_eval import MediaPipeSmileScorer
    geo, smile = FaceMeshGeoScorer(), MediaPipeSmileScorer()

    def score(path):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        g = geo([rgb])
        vals = {k: _clean(v[0]) for k, v in g.items()}
        vals["AU12_mouth_corner"] = _clean(smile([rgb])[0])
        return vals
    return score


def make_libreface(weights_dir):
    import libreface

    def score(path):
        try:
            res = libreface.get_au_intensities_and_detect_aus(str(path), device="cpu",
                                                                weights_download_dir=weights_dir)
            inten = res[1] if isinstance(res, tuple) else res
            return {au: _clean(inten.get(f"au_{n}_intensity")) for n, au in LF_AUS.items()}
        except Exception as e:  # noqa: BLE001 - no face / alignment failure
            print(f"  libreface failed on {path}: {e!r}", file=sys.stderr)
            return {au: None for au in LF_AUS.values()}
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", nargs="+", required=True, help="preprocess_mead.py annotation files")
    ap.add_argument("--root", type=Path, default=None,
                    help="root the record paths are relative to (default: the annotation file's grandparent)")
    ap.add_argument("--scorer", choices=["geo", "libreface"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights-dir", default=None, help="LibreFace weights dir (cached)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root_of = (lambda a: args.root) if args.root else (lambda a: Path(a).resolve().parent.parent)
    frames = frame_list(args.annotations, root_of)[args.shard::args.nshards]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["path"] for l in args.out.read_text().splitlines() if l.strip()}
    pending = [f for f in frames if f[0] not in done]
    todo = pending[: args.limit]
    print(f"{args.scorer} shard {args.shard}/{args.nshards}: {len(frames)} frames, "
          f"{len(frames) - len(pending)} done, {len(todo)} to score -> {args.out}", flush=True)
    if not todo:
        return
    if args.scorer == "libreface":
        assert args.weights_dir and Path(args.weights_dir).is_dir(), "pass a cached --weights-dir"
        score = make_libreface(args.weights_dir)
    else:
        score = make_geo()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        for i, (rel, path) in enumerate(todo, 1):
            fh.write(json.dumps({"path": rel, args.scorer: score(path)}) + "\n")
            if i % 500 == 0:
                fh.flush()
                print(f"  {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
