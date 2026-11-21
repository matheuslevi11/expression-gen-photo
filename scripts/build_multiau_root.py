"""Build the multi-AU training root (E1, 2026-11-19) from ``preprocess_mead.py --all-aus`` output.

Inputs are one or more preprocess roots laid out as ``<src>/<emotion>/shard_*/`` (each shard has
``frames/`` and ``annotations/{train,validation}.json``). The script

1. hard-links every shard's ``frames/`` tree into ``<dst>/frames/`` (no copy; record paths are
   relative to the shard dir, i.e. ``frames/<actor>/<emotion>/...``, so they stay valid);
2. derives three label channels per frame from the py-feat AU vector (the E0 choices,
   docs/experiments/2026-11-18-multi-au-e0-results.md):
       au12        smile        = the ``au12`` field (the existing AU12 training label)
       brow_raise  brow raise   = mean(AU01, AU02)
       brow_lower  brow lower   = AU04
   and drops frames without a face (``aus`` null; their ``au12`` is py-feat's 0.0 fallback);
3. re-expresses each channel per actor with the piecewise-linear map of
   ``normalize_labels_per_actor.py`` (actor's neutral median -> ``--floor``, 1 -> 1), keeping the
   raw value as ``<channel>_raw``;
4. writes ``annotations/{train,validation}_<tag>.json`` (records of ``--emotions`` only; the split
   is the preprocess one: validation = W009, W011) and ``annotations/actor_baselines_<tag>.json``.

    python scripts/build_multiau_root.py \
        --src /petrobr/crono/.../MEAD_multiau_processed /petrobr/crono/.../MEAD_multiau_processed_37 \
        --dst /petrobr/crono/.../MEAD_multiau --tag e1 \
        --emotions neutral happy surprised fear angry sad
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_labels_per_actor import normalize  # noqa: E402

CHANNELS = {
    "au12": lambda f: float(f["au12"]),
    "brow_raise": lambda f: (float(f["aus"]["AU01"]) + float(f["aus"]["AU02"])) / 2.0,
    "brow_lower": lambda f: float(f["aus"]["AU04"]),
}


def usable(frame) -> bool:
    aus = frame.get("aus")
    return bool(aus) and all(aus.get(k) is not None for k in ("AU01", "AU02", "AU04"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", nargs="+", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--emotions", nargs="+", required=True)
    p.add_argument("--floor", type=float, default=0.02)
    p.add_argument("--skip-link", action="store_true", help="annotations only (frames already linked)")
    args = p.parse_args()

    shards = sorted(s for src in args.src for s in src.glob("*/shard_*") if (s / "annotations").is_dir())
    assert shards, f"no shards under {args.src}"
    (args.dst / "annotations").mkdir(parents=True, exist_ok=True)
    outs = {split: args.dst / "annotations" / f"{split}_{args.tag}.json" for split in ("train", "validation")}
    base_path = args.dst / "annotations" / f"actor_baselines_{args.tag}.json"
    for path in [*outs.values(), base_path]:
        assert not path.exists(), f"{path} exists"

    if not args.skip_link:
        for i, s in enumerate(shards, 1):
            # hard links merged into dst/frames; --no-clobber keeps a path that is already there
            # (reruns), since one path is one clip in every source.
            subprocess.run(["cp", "-al", "--no-clobber", f"{s}/frames/.", f"{args.dst}/frames/"], check=True)
            if i % 20 == 0 or i == len(shards):
                print(f"linked {i}/{len(shards)} shards", flush=True)

    keep = {e.lower() for e in args.emotions}
    records = {"train": [], "validation": []}
    dropped = 0
    seen = set()
    for s in shards:
        for split in ("train", "validation"):
            a = s / "annotations" / f"{split}.json"
            if not a.exists():
                continue
            for r in json.loads(a.read_text()):
                if r["emotion"].lower() not in keep:
                    continue
                key = (r["actor"], r["clip_id"])
                assert key not in seen, f"duplicate clip {key}"
                seen.add(key)
                frames = [f for f in r["frames"] if usable(f)]
                dropped += len(r["frames"]) - len(frames)
                for f in frames:
                    for c, get in CHANNELS.items():
                        f[f"{c}_raw"] = get(f)
                r["frames"] = frames
                records[split].append(r)

    # per-actor neutral medians per channel, from every split (as normalize_labels_per_actor.py)
    vals = defaultdict(lambda: defaultdict(list))
    for recs in records.values():
        for r in recs:
            if r["emotion"].lower() == "neutral":
                for f in r["frames"]:
                    for c in CHANNELS:
                        vals[r["actor"]][c].append(f[f"{c}_raw"])
    base = {a: {c: float(np.median(v)) for c, v in d.items()} for a, d in vals.items()}
    actors = {r["actor"] for recs in records.values() for r in recs}
    assert actors == set(base), f"actors without neutral frames: {sorted(actors - set(base))}"
    bad = {a: b for a, b in base.items() if not all(0.0 < x < 1.0 for x in b.values())}
    assert not bad, f"baselines outside (0, 1): {bad}"

    for split, recs in records.items():
        for r in recs:
            b = base[r["actor"]]
            for f in r["frames"]:
                for c in CHANNELS:
                    f[c] = normalize(f[f"{c}_raw"], b[c], args.floor)
        outs[split].write_text(json.dumps(recs))
        n_frames = sum(len(r["frames"]) for r in recs)
        by_emo = defaultdict(int)
        for r in recs:
            by_emo[r["emotion"]] += 1
        print(f"wrote {outs[split]}: {len(recs)} clips, {n_frames} frames, "
              f"{len({r['actor'] for r in recs})} actors, {dict(sorted(by_emo.items()))}")
    base_path.write_text(json.dumps({
        "floor": args.floor, "emotion": "neutral", "statistic": "median",
        "channels": {c: "au12 field" if c == "au12" else ("mean(AU01, AU02)" if c == "brow_raise" else "AU04")
                     for c in CHANNELS},
        "actors": base}, indent=2))
    print(f"dropped {dropped} no-face frames; wrote {base_path}")
    for c in CHANNELS:
        b = np.array([base[a][c] for a in base])
        print(f"  {c:10s} baseline min {b.min():.3f} median {np.median(b):.3f} max {b.max():.3f}")


if __name__ == "__main__":
    main()
