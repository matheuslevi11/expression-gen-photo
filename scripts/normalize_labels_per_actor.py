"""Re-express AU12 labels relative to each actor's own neutral face.

py-feat AU12 carries an actor-level offset associated with skin tone (see
docs/experiments/2026-10-29-mead-neutral-clips-label-bias.md), so raw labels tie the intensity
scale to identity. This writes copies of the annotation files where each frame's ``au12`` is
mapped through a per-actor monotone piecewise-linear curve. The curve sends the actor's neutral
median ``b`` to ``floor`` and 1 to 1:

    au12 >= b:  floor + (1 - floor) * (au12 - b) / (1 - b)
    au12 <  b:  floor * au12 / b

Below-baseline frames are squeezed into ``[0, floor)`` instead of clipped to 0. Clipping would
make half the neutral frames exactly 0, and identical "0.000" prompts give a zero CCL signal (the
artifact the eval's calibration_jitter dither avoids). The raw value is kept as ``au12_raw``.

    python scripts/normalize_labels_per_actor.py --root /petrobr/crono/.../MEAD_processed \
        --inputs annotations/train_happy_neutral.json annotations/validation_happy_neutral.json \
        --suffix actnorm

writes ``annotations/{train,validation}_happy_neutral_actnorm.json`` and
``annotations/actor_baselines_actnorm.json``. Every actor needs neutral clips in the inputs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def actor_baselines(records, emotion="neutral"):
    """Median AU12 over all of each actor's ``emotion`` frames."""
    values = defaultdict(list)
    for r in records:
        if r["emotion"] == emotion:
            values[r["actor"]].extend(float(f["au12"]) for f in r["frames"])
    return {a: (float(np.median(v)), len(v)) for a, v in sorted(values.items())}


def normalize(au12: float, baseline: float, floor: float) -> float:
    if au12 >= baseline:
        return floor + (1.0 - floor) * (au12 - baseline) / (1.0 - baseline)
    return floor * au12 / baseline


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path, help="MEAD_processed root")
    p.add_argument("--inputs", required=True, nargs="+", help="annotation files relative to --root")
    p.add_argument("--suffix", default="actnorm")
    p.add_argument("--floor", type=float, default=0.02, help="label given to the actor's neutral median")
    args = p.parse_args()

    data = {name: json.loads((args.root / name).read_text()) for name in args.inputs}
    records = [r for recs in data.values() for r in recs]
    base = actor_baselines(records)
    missing = sorted({r["actor"] for r in records} - set(base))
    assert not missing, f"actors without neutral frames: {missing}"
    assert all(0.0 < b < 1.0 for b, _ in base.values()), base

    outputs = {name: args.root / name.replace(".json", f"_{args.suffix}.json") for name in args.inputs}
    base_path = args.root / "annotations" / f"actor_baselines_{args.suffix}.json"
    for path in [*outputs.values(), base_path]:
        assert not path.exists(), f"{path} exists"

    for name, recs in data.items():
        for r in recs:
            b = base[r["actor"]][0]
            for f in r["frames"]:
                f["au12_raw"] = float(f["au12"])
                f["au12"] = normalize(f["au12_raw"], b, args.floor)
        outputs[name].write_text(json.dumps(recs, indent=2))
        print(f"wrote {outputs[name]} ({len(recs)} clips)")
    base_path.write_text(json.dumps(
        {"floor": args.floor, "emotion": "neutral", "statistic": "median",
         "actors": {a: {"baseline": b, "frames": n} for a, (b, n) in base.items()}}, indent=2))
    b = np.array([v for v, _ in base.values()])
    print(f"wrote {base_path}: {len(base)} actors, baseline min {b.min():.3f} median {np.median(b):.3f} max {b.max():.3f}")


if __name__ == "__main__":
    main()
