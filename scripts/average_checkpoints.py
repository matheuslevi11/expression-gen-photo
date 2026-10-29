"""Uniform weight average of expression-adaptor checkpoints (encoder + attention merge).

Writes a checkpoint that inference_expression.py loads like any other; the optimizer
state is dropped. Every input must share keys and shapes (same run / architecture).

    python scripts/average_checkpoints.py \
        --ckpts .../checkpoint-step-15000.ckpt .../checkpoint-step-20000.ckpt \
        --out .../avg-step-15000-20000.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

STATE_KEYS = ("camera_encoder_state_dict", "attention_processor_state_dict")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    assert len(args.ckpts) >= 2, "need at least two checkpoints"
    assert not args.out.exists(), f"{args.out} exists"

    ckpts = [torch.load(str(c), map_location="cpu", mmap=True) for c in args.ckpts]
    out = {"averaged_from": [str(c) for c in args.ckpts],
           "averaged_steps": [c.get("global_step") for c in ckpts]}
    for key in STATE_KEYS:
        states = [c[key] for c in ckpts]
        assert all(s.keys() == states[0].keys() for s in states), f"{key}: key mismatch"
        avg = {}
        for name, t in states[0].items():
            assert all(s[name].shape == t.shape for s in states), f"{key}.{name}: shape mismatch"
            if t.is_floating_point():
                avg[name] = (sum(s[name].double() for s in states) / len(states)).to(t.dtype)
            else:
                avg[name] = t.clone()
        out[key] = avg
    torch.save(out, args.out)
    print(f"averaged steps {out['averaged_steps']} -> {args.out}")


if __name__ == "__main__":
    main()
