"""Merge sharded preprocess_mead.py output for one emotion into an existing MEAD_processed root.

Moves every shard's ``frames/<actor>/<emotion>/`` into ``<dst-root>/frames/<actor>/<emotion>/``
(record paths are relative to the root, so they stay valid) and writes combined annotation
files next to the existing ones, which are left untouched:

    <dst-root>/annotations/train_<tag>.json       = train.json      + shard train records
    <dst-root>/annotations/validation_<tag>.json  = validation.json + shard validation records

    python scripts/merge_mead_emotion.py \
        --src-root /petrobr/crono/.../MEAD_neutral_processed --emotion neutral \
        --dst-root /petrobr/crono/.../MEAD_processed --tag happy_neutral
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", required=True, type=Path, help="dir holding shard_*/ from the preprocess launcher")
    p.add_argument("--dst-root", required=True, type=Path, help="existing MEAD_processed root")
    p.add_argument("--emotion", required=True)
    p.add_argument("--tag", required=True, help="suffix of the combined annotation files")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    shards = sorted(d for d in args.src_root.glob("shard_*") if d.is_dir())
    assert shards, f"no shard_* dirs under {args.src_root}"
    new = {"train": [], "validation": []}
    moves = []
    for shard in shards:
        for split in new:
            recs = json.loads((shard / "annotations" / f"{split}.json").read_text())
            for r in recs:
                assert r["emotion"] == args.emotion, f"{r['clip_id']}: emotion {r['emotion']}"
                assert all((shard / f["path"]).is_file() for f in r["frames"]), f"{r['clip_id']}: missing frames"
            new[split] += recs
        for actor_dir in sorted((shard / "frames").iterdir()):
            src = actor_dir / args.emotion
            if src.is_dir():
                moves.append((src, args.dst_root / "frames" / actor_dir.name / args.emotion))

    out = {}
    for split, recs in new.items():
        base = json.loads((args.dst_root / "annotations" / f"{split}.json").read_text())
        ids = [r["clip_id"] for r in base + recs]
        assert len(ids) == len(set(ids)), f"{split}: duplicate clip_id"
        out[split] = base + recs
        print(f"{split}: {len(base)} existing + {len(recs)} {args.emotion} = {len(out[split])}")
    train_actors = {r["actor"] for r in out["train"]}
    val_actors = {r["actor"] for r in out["validation"]}
    assert not train_actors & val_actors, f"actors in both splits: {train_actors & val_actors}"

    paths = {split: args.dst_root / "annotations" / f"{split}_{args.tag}.json" for split in out}
    for path in paths.values():
        assert not path.exists(), f"{path} exists"
    for src, dst in moves:
        assert not dst.exists(), f"{dst} already exists"
    print(f"{len(moves)} actor dirs to move into {args.dst_root / 'frames'}")
    if args.dry_run:
        return
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    for split, recs in out.items():
        paths[split].write_text(json.dumps(recs, indent=2))
        print(f"wrote {paths[split]}")


if __name__ == "__main__":
    main()
