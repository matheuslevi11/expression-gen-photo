"""Paired comparison of two batch_eval runs (e.g. control vs. a training remedy).

Each run dir holds ``<experiment>/results.jsonl`` as written by
``scripts/slurm/batch_eval_sdumont2nd.srm``. Samples are paired by id (same
prompt, seed and intensity list), and 95% CIs come from a paired bootstrap over
(prompt, seed) units, so prompt/seed difficulty is shared between the arms.

    python scripts/compare_eval.py \
        --a inference_output/batch_eval/calib_ctrl_25k --name-a ctrl \
        --b inference_output/batch_eval/calib_aug_25k  --name-b aug
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from batch_eval import pearson

PER_SAMPLE_KEYS = ("au12_pearson", "mp_pearson", "id_mean_adjacent", "id_min_adjacent")
CLS_DETECTORS = (("au12", "CLS py-feat"), ("mp_smile", "CLS MediaPipe"))


def load(run_dir: Path, experiment: str) -> dict[str, dict]:
    path = run_dir / experiment / "results.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def pooled_cls(rows, key):
    c, d = [], []
    for r in rows:
        if r.get(key) is not None and r.get("level") is not None:
            c += [r["level"]] * len(r[key])
            d += r[key]
    return pearson(c, d)


def mean_finite(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def bootstrap(units, stat, n_boot, rng):
    """Point estimate and 95% CI of stat(selected_units) under unit resampling."""
    keys = sorted(units)
    point = stat([u for k in keys for u in units[k]])
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        draws.append(stat([u for i in pick for u in units[keys[i]]]))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def compare(a: dict, b: dict, experiment: str, n_boot: int, rng) -> dict:
    ids = sorted(set(a) & set(b))
    units = defaultdict(list)
    for i in ids:
        units[(a[i].get("prompt_idx"), a[i].get("seed"))].append((a[i], b[i]))
    out = dict(n_paired=len(ids), n_units=len(units), only_a=len(set(a) - set(b)), only_b=len(set(b) - set(a)))

    if experiment in ("calibration", "extrapolation"):
        for key, label in CLS_DETECTORS:
            out[label] = {
                "a": bootstrap(units, lambda p: pooled_cls([x for x, _ in p], key), n_boot, rng),
                "b": bootstrap(units, lambda p: pooled_cls([y for _, y in p], key), n_boot, rng),
                "b-a": bootstrap(units, lambda p: pooled_cls([y for _, y in p], key)
                                 - pooled_cls([x for x, _ in p], key), n_boot, rng),
            }
        curve = defaultdict(lambda: ([], []))
        for i in ids:
            if a[i].get("au12") and b[i].get("au12"):
                curve[a[i]["level"]][0].extend(a[i]["au12"])
                curve[a[i]["level"]][1].extend(b[i]["au12"])
        out["curve_au12"] = {str(lvl): (mean_finite(x), mean_finite(y)) for lvl, (x, y) in sorted(curve.items())}
    else:
        for key in PER_SAMPLE_KEYS:
            out[key] = {
                "a": bootstrap(units, lambda p: mean_finite([x.get(key) for x, _ in p]), n_boot, rng),
                "b": bootstrap(units, lambda p: mean_finite([y.get(key) for _, y in p]), n_boot, rng),
                "b-a": bootstrap(units, lambda p: mean_finite([(y.get(key) if y.get(key) is not None else np.nan)
                                                               - (x.get(key) if x.get(key) is not None else np.nan)
                                                               for x, y in p]), n_boot, rng),
            }
    return out


def fmt(t):
    point, lo, hi = t
    return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, type=Path)
    p.add_argument("--b", required=True, type=Path)
    p.add_argument("--name-a", default="a")
    p.add_argument("--name-b", default="b")
    p.add_argument("--experiments", default="calibration,scaled,permuted,extrapolation")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="optional JSON output")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    report = {}
    for exp in args.experiments.split(","):
        a, b = load(args.a, exp), load(args.b, exp)
        if not a or not b:
            continue
        r = report[exp] = compare(a, b, exp, args.n_boot, rng)
        print(f"\n== {exp}  paired n={r['n_paired']} over {r['n_units']} (prompt, seed) units"
              + (f"  [unpaired: {args.name_a} {r['only_a']}, {args.name_b} {r['only_b']}]" if r["only_a"] or r["only_b"] else ""))
        print(f"   {'metric':16s} {args.name_a:>26s} {args.name_b:>26s} {args.name_b + '-' + args.name_a:>26s}")
        for metric, v in r.items():
            if isinstance(v, dict) and "b-a" in v:
                print(f"   {metric:16s} {fmt(v['a']):>26s} {fmt(v['b']):>26s} {fmt(v['b-a']):>26s}")
        if "curve_au12" in r:
            print(f"   detected py-feat AU12 by commanded level ({args.name_a} -> {args.name_b}):")
            for lvl, (x, y) in r["curve_au12"].items():
                print(f"     {float(lvl):+.2f}: {x:.3f} -> {y:.3f}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
